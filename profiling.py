"""
GPU profiling and utilization logging for Parameter Golf experiments.

Usage:
    from profiling import TrainingProfiler

    profiler = TrainingProfiler(enabled=True, log_fn=print)
    profiler.start()

    for step in range(num_steps):
        # ... training step ...
        profiler.step(step, train_loss=loss_val)

    profiler.stop()
    profiler.summary()

Env vars:
    PROFILE=1               Enable torch.profiler (TensorBoard flame charts)
    PROFILE_DIR=./logs/prof  Output directory for TensorBoard traces
    PROFILE_WAIT=5           Steps to skip before profiling
    PROFILE_WARMUP=3         Profiler warmup steps
    PROFILE_ACTIVE=5         Steps to actively record
    GPU_LOG_EVERY=100        Log GPU memory stats every N steps
    CUDA_TIMING=1            Per-phase CUDA event timing (forward/backward/optimizer)
"""

from __future__ import annotations

import os
import subprocess
import time
from contextlib import nullcontext
from typing import Any, Callable

import torch


def _get_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_gpu_info_nvidia() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": parts[0],
                    "gpu_util": parts[1],
                    "mem_util": parts[2],
                    "mem_used_mb": parts[3],
                    "mem_total_mb": parts[4],
                    "temp_c": parts[5],
                    "power_w": parts[6],
                })
        return gpus
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


class StepTimer:
    """CUDA or CPU event-based timing for training step phases."""

    def __init__(self, device_type: str):
        self.device_type = device_type
        self.use_cuda_events = device_type == "cuda"
        self._marks: dict[str, Any] = {}
        self._order: list[str] = []

    def mark(self, name: str) -> None:
        if self.use_cuda_events:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._marks[name] = ev
        else:
            self._marks[name] = time.perf_counter()
        if name not in self._order:
            self._order.append(name)

    def elapsed_ms(self, start: str, end: str) -> float:
        if start not in self._marks or end not in self._marks:
            return -1.0
        if self.use_cuda_events:
            torch.cuda.synchronize()
            return self._marks[start].elapsed_time(self._marks[end])
        return (self._marks[end] - self._marks[start]) * 1000.0

    def report(self) -> dict[str, float]:
        results = {}
        for i in range(len(self._order) - 1):
            a, b = self._order[i], self._order[i + 1]
            results[f"{a}->{b}"] = self.elapsed_ms(a, b)
        if len(self._order) >= 2:
            results["total"] = self.elapsed_ms(self._order[0], self._order[-1])
        return results

    def reset(self) -> None:
        self._marks.clear()
        self._order.clear()


class TrainingProfiler:
    """Unified profiling interface for Parameter Golf training runs.

    Each run_name gets its own subdirectory under profile_dir so that
    TensorBoard can display all experiments side-by-side for comparison.
    Directory layout:
        logs/profiler/
            EXP-001_baseline/
                worker0.pt.trace.json
            EXP-002_engram/
                worker0.pt.trace.json
    View all with: tensorboard --logdir=./logs/profiler
    """

    def __init__(
        self,
        enabled: bool | None = None,
        profile_dir: str | None = None,
        run_name: str | None = None,
        gpu_log_every: int | None = None,
        cuda_timing: bool | None = None,
        log_fn: Callable[[str], None] | None = None,
        rank: int = 0,
    ):
        self.rank = rank
        self.is_master = rank == 0
        self.log_fn = log_fn or print
        self.device_type = _get_device_type()

        self.profile_enabled = (
            enabled if enabled is not None
            else os.environ.get("PROFILE", "0") == "1"
        )
        base_dir = profile_dir or os.environ.get("PROFILE_DIR", "./logs/profiler")
        run_name = run_name or os.environ.get("RUN_ID") or os.environ.get("PROFILE_RUN_NAME")
        if run_name:
            self.profile_dir = os.path.join(base_dir, run_name)
        else:
            self.profile_dir = base_dir
        self.profile_wait = int(os.environ.get("PROFILE_WAIT", "5"))
        self.profile_warmup = int(os.environ.get("PROFILE_WARMUP", "3"))
        self.profile_active = int(os.environ.get("PROFILE_ACTIVE", "5"))

        self.gpu_log_every = (
            gpu_log_every if gpu_log_every is not None
            else int(os.environ.get("GPU_LOG_EVERY", "100"))
        )
        self.cuda_timing_enabled = (
            cuda_timing if cuda_timing is not None
            else os.environ.get("CUDA_TIMING", "0") == "1"
        )

        self._profiler: torch.profiler.profile | None = None
        self._step_timer: StepTimer | None = None
        self._peak_mem_gb: float = 0.0
        self._step_times: list[float] = []
        self._t_last_step: float = 0.0

    def _log(self, msg: str) -> None:
        if self.is_master:
            self.log_fn(msg)

    def start(self) -> None:
        if self.profile_enabled and self.is_master:
            os.makedirs(self.profile_dir, exist_ok=True)

            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device_type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            self._profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=self.profile_wait,
                    warmup=self.profile_warmup,
                    active=self.profile_active,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
            )
            self._profiler.__enter__()
            self._log(
                f"profiler:started dir={self.profile_dir} "
                f"schedule=wait{self.profile_wait}/warm{self.profile_warmup}/active{self.profile_active}"
            )

        if self.cuda_timing_enabled:
            self._step_timer = StepTimer(self.device_type)

        self._t_last_step = time.perf_counter()
        self._log(
            f"profiling:config profile={self.profile_enabled} "
            f"gpu_log_every={self.gpu_log_every} cuda_timing={self.cuda_timing_enabled} "
            f"device={self.device_type}"
        )

    def stop(self) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._profiler = None
            self._log(f"profiler:stopped traces_written_to={self.profile_dir}")

    def mark(self, name: str) -> None:
        """Record a timing mark within a step (e.g., 'fwd', 'bwd', 'opt')."""
        if self._step_timer is not None:
            self._step_timer.mark(name)

    def step(self, step_num: int, **metrics: float) -> None:
        """Call at the end of each training step."""
        now = time.perf_counter()
        step_ms = (now - self._t_last_step) * 1000.0
        self._step_times.append(step_ms)
        self._t_last_step = now

        if self._profiler is not None:
            self._profiler.step()

        if self.is_master and self.gpu_log_every > 0 and step_num % self.gpu_log_every == 0:
            self._log_gpu_metrics(step_num, step_ms, metrics)

        if self._step_timer is not None and self.gpu_log_every > 0 and step_num % self.gpu_log_every == 0:
            self._log_step_timing(step_num)

    def _log_gpu_metrics(self, step_num: int, step_ms: float, metrics: dict[str, float]) -> None:
        parts = [f"gpu_stats:step:{step_num}"]

        if self.device_type == "cuda":
            alloc_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            self._peak_mem_gb = max(self._peak_mem_gb, peak_gb)
            parts.append(f"mem_alloc:{alloc_gb:.2f}GB")
            parts.append(f"mem_reserved:{reserved_gb:.2f}GB")
            parts.append(f"mem_peak:{peak_gb:.2f}GB")
        elif self.device_type == "mps":
            alloc_gb = torch.mps.current_allocated_memory() / 1024**3
            parts.append(f"mps_alloc:{alloc_gb:.2f}GB")

        parts.append(f"step_ms:{step_ms:.1f}")

        for k, v in metrics.items():
            parts.append(f"{k}:{v:.4f}")

        nvidia_info = _get_gpu_info_nvidia()
        if nvidia_info and self.rank < len(nvidia_info):
            gpu = nvidia_info[self.rank]
            parts.append(f"sm_util:{gpu['gpu_util']}%")
            parts.append(f"mem_util:{gpu['mem_util']}%")
            parts.append(f"temp:{gpu['temp_c']}C")
            parts.append(f"power:{gpu['power_w']}W")

        self._log(" ".join(parts))

    def _log_step_timing(self, step_num: int) -> None:
        if self._step_timer is None:
            return
        report = self._step_timer.report()
        if not report:
            return
        parts = [f"step_timing:step:{step_num}"]
        for phase, ms in report.items():
            parts.append(f"{phase}:{ms:.1f}ms")
        self._log(" ".join(parts))
        self._step_timer.reset()

    def summary(self) -> None:
        if not self._step_times:
            return
        n = len(self._step_times)
        avg_ms = sum(self._step_times) / n
        parts = [
            f"profiling:summary steps={n}",
            f"avg_step_ms={avg_ms:.1f}",
        ]
        if n > 10:
            trimmed = sorted(self._step_times)[n // 10 : -n // 10 or None]
            parts.append(f"p10-p90_avg_ms={sum(trimmed) / len(trimmed):.1f}")
        if self.device_type == "cuda":
            parts.append(f"peak_mem={self._peak_mem_gb:.2f}GB")
        self._log(" ".join(parts))

    def new_step_timer(self) -> StepTimer:
        """Get a fresh StepTimer for detailed per-phase timing of a single step."""
        return StepTimer(self.device_type)

    @property
    def profiler_context(self) -> Any:
        """Return the active profiler as a context, or nullcontext if disabled."""
        if self._profiler is not None:
            return self._profiler
        return nullcontext()
