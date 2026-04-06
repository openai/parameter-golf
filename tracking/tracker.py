"""
Modular MLflow tracking for Parameter Golf experiments.

Usage:
    from tracking import ParameterGolfTracker

    tracker = ParameterGolfTracker(experiment_name="parameter_golf")
    tracker.start_run(run_name="baseline_sp1024_s1337")
    tracker.log_hyperparams(args)          # Hyperparameters dataclass/object
    tracker.log_system_info()              # GPU, driver, etc.
    tracker.log_data_info(args, n_train_shards=10, n_val_tokens=62_000_000)
    tracker.log_step(step=100, train_loss=4.5, train_time_ms=35000, step_avg_ms=350)
    tracker.log_validation(step=1000, val_loss=2.5, val_bpb=1.25, train_time_ms=350000)
    tracker.log_final(val_loss=2.27, val_bpb=1.22, eval_time_ms=11000,
                       model_bytes=67_000_000, code_bytes=47000,
                       quant_bytes=14_500_000, peak_mem_mib=10240,
                       steps_completed=1691, stopped_reason="wallclock_cap")
    tracker.end_run()

The tracker is a no-op if mlflow is not installed, so it never breaks training.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


def _get_git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["git_branch"] = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_dirty"] = str(bool(dirty))
    except Exception:
        pass
    return info


def _get_gpu_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = str(torch.cuda.device_count())
            mem = torch.cuda.get_device_properties(0).total_mem
            info["gpu_memory_gb"] = f"{mem / 1e9:.1f}"
            info["cuda_version"] = torch.version.cuda or "unknown"
    except Exception:
        pass
    return info


class ParameterGolfTracker:
    """Thin wrapper around MLflow for Parameter Golf experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "parameter_golf",
        tracking_uri: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and MLFLOW_AVAILABLE
        self._run = None
        self._start_time: float | None = None

        if self.enabled:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    @property
    def active(self) -> bool:
        return self.enabled and self._run is not None

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        merged_tags = {
            "competition": "parameter_golf",
            **_get_git_info(),
            **(tags or {}),
        }
        self._run = mlflow.start_run(run_name=run_name, tags=merged_tags)
        self._start_time = time.time()

    def end_run(self, status: str = "FINISHED") -> None:
        if not self.active:
            return
        mlflow.end_run(status=status)
        self._run = None

    # ------------------------------------------------------------------
    # Params: model, training schedule, optimizer, data
    # ------------------------------------------------------------------

    def log_hyperparams(self, args: Any) -> None:
        """Log all fields from a Hyperparameters-like object as MLflow params."""
        if not self.active:
            return

        params: dict[str, Any] = {}
        for attr in dir(args):
            if attr.startswith("_"):
                continue
            val = getattr(args, attr, None)
            if callable(val):
                continue
            params[attr] = val

        mlflow.log_params(params)

    def log_model_info(self, n_params: int, world_size: int, grad_accum_steps: int) -> None:
        if not self.active:
            return
        mlflow.log_params(
            {
                "model_params": n_params,
                "world_size": world_size,
                "grad_accum_steps": grad_accum_steps,
            }
        )

    def log_data_info(
        self,
        data_path: str,
        tokenizer_path: str,
        n_train_shards: int,
        n_val_tokens: int,
        data_variant: str = "sp1024",
    ) -> None:
        if not self.active:
            return
        mlflow.log_params(
            {
                "data_path": data_path,
                "tokenizer_path": tokenizer_path,
                "n_train_shards": n_train_shards,
                "n_val_tokens": n_val_tokens,
                "data_variant": data_variant,
            }
        )

    def log_system_info(self) -> None:
        if not self.active:
            return
        gpu_info = _get_gpu_info()
        tags = {f"sys.{k}": v for k, v in gpu_info.items()}
        mlflow.set_tags(tags)

    # ------------------------------------------------------------------
    # Tags (arbitrary key-value for filtering)
    # ------------------------------------------------------------------

    def set_tags(self, tags: dict[str, str]) -> None:
        if not self.active:
            return
        mlflow.set_tags(tags)

    # ------------------------------------------------------------------
    # Step-level metrics (train loop)
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        train_loss: float,
        train_time_ms: float,
        step_avg_ms: float,
        lr_scale: float | None = None,
    ) -> None:
        if not self.active:
            return
        metrics: dict[str, float] = {
            "train_loss": train_loss,
            "train_time_ms": train_time_ms,
            "step_avg_ms": step_avg_ms,
        }
        if lr_scale is not None:
            metrics["lr_scale"] = lr_scale
        mlflow.log_metrics(metrics, step=step)

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------

    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_bpb: float,
        train_time_ms: float,
    ) -> None:
        if not self.active:
            return
        mlflow.log_metrics(
            {
                "val_loss": val_loss,
                "val_bpb": val_bpb,
                "val_train_time_ms": train_time_ms,
            },
            step=step,
        )

    # ------------------------------------------------------------------
    # Final metrics (end of run)
    # ------------------------------------------------------------------

    def log_final(
        self,
        val_loss: float,
        val_bpb: float,
        eval_time_ms: float,
        model_bytes_raw: int,
        code_bytes: int,
        quant_bytes: int,
        peak_mem_mib: int,
        reserved_mem_mib: int,
        steps_completed: int,
        iterations_requested: int,
        total_train_time_ms: float,
        stopped_reason: str = "completed",
    ) -> None:
        if not self.active:
            return

        total_submission_raw = model_bytes_raw + code_bytes
        total_submission_quant = quant_bytes + code_bytes
        under_budget = int(total_submission_quant <= 16_000_000)

        mlflow.log_metrics(
            {
                "final_val_loss": val_loss,
                "final_val_bpb": val_bpb,
                "final_eval_time_ms": eval_time_ms,
                "model_bytes_raw": float(model_bytes_raw),
                "code_bytes": float(code_bytes),
                "total_submission_bytes_raw": float(total_submission_raw),
                "quant_compressed_bytes": float(quant_bytes),
                "total_submission_bytes_quant": float(total_submission_quant),
                "submission_size_mb": total_submission_quant / 1e6,
                "under_16mb_budget": float(under_budget),
                "peak_memory_mib": float(peak_mem_mib),
                "reserved_memory_mib": float(reserved_mem_mib),
                "steps_completed": float(steps_completed),
                "total_train_time_ms": total_train_time_ms,
                "total_train_time_s": total_train_time_ms / 1000.0,
            }
        )
        mlflow.set_tags(
            {
                "stopped_reason": stopped_reason,
                "iterations_requested": str(iterations_requested),
            }
        )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def log_config_snapshot(self, args: Any, extra: dict[str, Any] | None = None) -> None:
        """Dump resolved hyperparams + extra info as a JSON artifact."""
        if not self.active:
            return

        snapshot: dict[str, Any] = {}
        for attr in dir(args):
            if attr.startswith("_"):
                continue
            val = getattr(args, attr, None)
            if callable(val):
                continue
            snapshot[attr] = val
        if extra:
            snapshot.update(extra)

        path = Path("_mlflow_config_snapshot.json")
        path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path))
        path.unlink(missing_ok=True)

    def log_file_artifact(self, file_path: str, artifact_path: str | None = None) -> None:
        if not self.active:
            return
        if os.path.isfile(file_path):
            mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def log_text_artifact(self, text: str, filename: str) -> None:
        if not self.active:
            return
        path = Path(f"_mlflow_{filename}")
        path.write_text(text, encoding="utf-8")
        mlflow.log_artifact(str(path))
        path.unlink(missing_ok=True)
