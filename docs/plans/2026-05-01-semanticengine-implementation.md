# SemanticEngine Submission Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a runnable `train_gpt.py` for the SemanticEngine ARM submission that trains on 8×H100, builds a ≤16MB artifact, and reports a prequential `val_bpb` score.

**Architecture:** `engine_entry.py` in `chaoscontrol.public` is a thin adapter that (a) exposes a clean `init_arm_topology()` / `build_arm_config()` API for `train_gpt.py` and (b) delegates the full ARM training loop to the existing `run_condition()` in `experiments/23_fast_path/runner_fast_path.py` via a `CHAOSCONTROL_ROOT`-based sys.path injection. `train_gpt.py` is the env-var hyperparameter layer + main() that calls these functions and logs the score.

**Tech Stack:** Python 3.11+, PyTorch 2.11+cu130, chaoscontrol (installed from GitHub), sentencepiece SP16384, HuggingFace shards from `Natooka/parameter-golf-sp-tokenizers`, native extensions (`_lm_head_loss`, `_cpu_ssm_controller`, `_ssm_scan`).

**Key design decisions recorded in:** `docs/plans/2026-05-01-semanticengine-submission-design.md`

---

## Dependency Order

```
Task 1 (scaffold) → Task 2 (topology fn) → Task 3 (config builder) → Task 4 (engine_entry complete)
                                                                              ↓
Task 5 (submission folder) → Task 6 (train_gpt.py Section 1) → Task 7 (train_gpt.py Section 2) → Task 8 (requirements.txt) → Task 9 (smoke test)
```

Tasks 2 and 3 can run in parallel after Task 1. Task 4 depends on 2 and 3. Tasks 5–9 are sequential.

---

## Repo targets

- **chaoscontrol repo** (`/Users/kennethmalloy/Local Documents/Developer/chaoscontrol/`): Tasks 1–4
- **parameter-golf repo** (`/Users/kennethmalloy/Local Documents/Developer/parameter-golf/`): Tasks 5–9

---

## Task 1: Scaffold `chaoscontrol/public/` module

**Files:**
- Create: `src/chaoscontrol/public/__init__.py`
- Create: `src/chaoscontrol/public/engine_entry.py` (stub)

**Step 1: Create `__init__.py`**

```python
# src/chaoscontrol/public/__init__.py
from chaoscontrol.public.engine_entry import (
    RoleInfo,
    init_arm_topology,
    build_arm_config,
    run_arm_submission,
)

__all__ = ["RoleInfo", "init_arm_topology", "build_arm_config", "run_arm_submission"]
```

**Step 2: Create stub `engine_entry.py`**

```python
# src/chaoscontrol/public/engine_entry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class RoleInfo:
    rank: int
    world_size: int
    packet_rank: int
    maintenance_rank: int
    is_train_rank: bool
    is_packet_rank: bool
    is_maintenance_rank: bool
    split_memory_ranks: bool

def init_arm_topology(rank: int, world_size: int) -> RoleInfo:
    raise NotImplementedError

def build_arm_config(hyperparams: Any) -> dict[str, Any]:
    raise NotImplementedError

def run_arm_submission(
    config: dict[str, Any],
    *,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    output_json: str | None,
    val_cache_dir: str | None,
    world_size_override: int | None = None,
) -> dict[str, Any]:
    raise NotImplementedError
```

**Step 3: Verify import works**

```bash
cd /Users/kennethmalloy/Local\ Documents/Developer/chaoscontrol
.venv/bin/python -c "from chaoscontrol.public import RoleInfo; print('ok')"
```

Expected: `ok`

**Step 4: Commit**

```bash
git add src/chaoscontrol/public/__init__.py src/chaoscontrol/public/engine_entry.py
git commit -m "feat: scaffold chaoscontrol.public module for SemanticEngine submission"
```

---

## Task 2: Implement `init_arm_topology()`

**Files:**
- Modify: `src/chaoscontrol/public/engine_entry.py`
- Create: `tests/public/test_engine_entry.py`

**Reference:** `runner_fast_path.py` lines 1848–1876 — the function starting around line 1848 that returns a dict with `packet_rank`, `maintenance_rank`, `memory_ranks`, `train_ranks`, `split_memory_ranks`. Port its logic into `init_arm_topology()`.

**Step 1: Write the failing tests**

```python
# tests/public/test_engine_entry.py
import pytest
from chaoscontrol.public.engine_entry import init_arm_topology

def test_8gpu_splits_6_2():
    role6 = init_arm_topology(rank=6, world_size=8)
    role7 = init_arm_topology(rank=7, world_size=8)
    role0 = init_arm_topology(rank=0, world_size=8)
    assert role6.is_packet_rank
    assert not role6.is_maintenance_rank
    assert role7.is_maintenance_rank
    assert not role7.is_packet_rank
    assert role0.is_train_rank
    assert role6.split_memory_ranks
    assert role7.split_memory_ranks

def test_4gpu_shares_memory():
    role3 = init_arm_topology(rank=3, world_size=4)
    role0 = init_arm_topology(rank=0, world_size=4)
    assert role3.is_packet_rank
    assert role3.is_maintenance_rank  # shared on 4 GPU
    assert not role3.is_train_rank
    assert role0.is_train_rank
    assert not role3.split_memory_ranks

def test_1gpu_all_train():
    role = init_arm_topology(rank=0, world_size=1)
    assert role.is_train_rank
    assert not role.is_packet_rank
    assert not role.is_maintenance_rank

def test_packet_rank_value_8gpu():
    role = init_arm_topology(rank=0, world_size=8)
    assert role.packet_rank == 6
    assert role.maintenance_rank == 7
```

**Step 2: Run to verify failure**

```bash
cd /Users/kennethmalloy/Local\ Documents/Developer/chaoscontrol
.venv/bin/python -m pytest tests/public/test_engine_entry.py -v
```

Expected: `NotImplementedError`

**Step 3: Implement**

Replace the `init_arm_topology` stub in `engine_entry.py`:

```python
def init_arm_topology(rank: int, world_size: int) -> RoleInfo:
    """Assign GPU role. On 8+ GPUs with replay_eviction: 6+2 split.
    On 4 GPUs: 3+1 (GPU3 owns both packet-serving and maintenance).
    On 1 GPU: everything on rank 0.
    """
    world = int(world_size)
    split = world >= 8  # split memory ranks only at 8+ GPUs
    packet_rank = world - (2 if split else 1)
    maintenance_rank = world - 1
    is_packet = rank == packet_rank
    is_maintenance = rank == maintenance_rank
    is_train = not is_packet and not is_maintenance
    # On 4 GPU (split=False), packet_rank == maintenance_rank == 3,
    # so that rank is both. is_train is False for it.
    return RoleInfo(
        rank=rank,
        world_size=world_size,
        packet_rank=packet_rank,
        maintenance_rank=maintenance_rank,
        is_train_rank=is_train,
        is_packet_rank=is_packet,
        is_maintenance_rank=is_maintenance,
        split_memory_ranks=split,
    )
```

**Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/public/test_engine_entry.py -v
```

Expected: all 4 tests pass.

**Step 5: Commit**

```bash
git add src/chaoscontrol/public/engine_entry.py tests/public/test_engine_entry.py
git commit -m "feat: implement init_arm_topology() with 6+2 / 3+1 / 1-gpu routing"
```

---

## Task 3: Implement `build_arm_config()`

**Files:**
- Modify: `src/chaoscontrol/public/engine_entry.py`
- Modify: `tests/public/test_engine_entry.py`

**Context:** `build_arm_config()` takes the hyperparameter object from `train_gpt.py` (any object with attributes) and returns the config dict that `run_condition()` in `runner_fast_path.py` expects. The required keys are a superset of what `exp26._crct_lock()`, `exp26._fast_slow_lock()`, `exp26._replay_eviction_pipeline_lock()`, and `exp26._artifact_size_lock()` return — plus model/training keys like `vocab_size`, `seq_len`, `batch_size`, `budget_seconds`, `seed`, `optimizer`, etc.

**Reference:** Read `experiments/26_arm/exp26.py` for the four lock functions. Read `experiments/23_fast_path/runner_fast_path.py:run_condition` (line 14055–14130 approx.) for what keys it reads from config.

**Telemetry-tuned defaults (from profiling — these override exp26 lock values):**

| Key | Value | Rationale |
|---|---|---|
| `crct_memory_write_tokens_per_step` | 192 | Up from 128/32; per-step cap headroom |
| `online_episodic_write_tokens_per_chunk` | 64 | Up from 16; first meaningful step without being reckless |
| `crct_target_write_rate` | 0.20 | Matches observed adaptive smoke ~0.219 |
| `async_teacher_max_lag_steps` | leave at current | Lag is 3–4 steps; pipe not bottleneck |
| `crct_async_teacher_pending_batches` | leave at current | No ring drops observed |

**Step 1: Write the failing test**

```python
# append to tests/public/test_engine_entry.py

from chaoscontrol.public.engine_entry import build_arm_config

class _FakeHyperparams:
    vocab_size = 16384
    model_dim = 384
    ssm_delta_rank = 32
    seq_len = 512
    batch_size = 1024
    budget_seconds = 600.0
    seed = 42
    base_lr = 0.064
    weight_decay = 0.01
    grad_clip_norm = 1.0
    log_a_beta_coupling = True
    log_a_beta_ema = 0.99
    log_a_beta_min = 0.5
    lm_head_tile_size = 4096

def test_build_arm_config_required_keys():
    cfg = build_arm_config(_FakeHyperparams())
    required = [
        "vocab_size", "model_dim", "seq_len", "batch_size",
        "budget_seconds", "seed", "optimizer",
        "crct_enabled", "replay_eviction_enabled",
        "fast_slow_enabled", "fast_slow_alpha",
        "crct_memory_write_tokens_per_step",
        "online_episodic_write_tokens_per_chunk",
        "crct_target_write_rate",
    ]
    for key in required:
        assert key in cfg, f"missing key: {key}"

def test_build_arm_config_telemetry_tuned_defaults():
    cfg = build_arm_config(_FakeHyperparams())
    assert cfg["crct_memory_write_tokens_per_step"] == 192
    assert cfg["online_episodic_write_tokens_per_chunk"] == 64
    assert abs(cfg["crct_target_write_rate"] - 0.20) < 1e-6
    assert cfg["model_dim"] == 384
    assert cfg["optimizer"] == "muon"
    assert cfg["optimizer_log_a_beta_coupling"] is True
```

**Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/public/test_engine_entry.py::test_build_arm_config_required_keys -v
```

Expected: `NotImplementedError`

**Step 3: Implement**

Replace the `build_arm_config` stub. The function builds the config dict by merging the four exp26 lock dicts over a base config derived from hyperparams. Import the lock functions directly from exp26 (they are pure Python dict builders):

```python
import sys
import os
from pathlib import Path

def _exp26_locks() -> dict[str, Any]:
    """Import lock dicts from exp26 without triggering exp26's dist init."""
    cc_root = os.environ.get("CHAOSCONTROL_ROOT", "/workspace/chaoscontrol")
    exp26_dir = str(Path(cc_root) / "experiments" / "26_arm")
    exp24_dir = str(Path(cc_root) / "experiments" / "24_training_time_bundle")
    for d in (exp26_dir, exp24_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    from exp26 import _crct_lock, _fast_slow_lock, _replay_eviction_pipeline_lock, _artifact_size_lock
    return {
        **_artifact_size_lock(),
        **_fast_slow_lock(),
        **_crct_lock(),
        **_replay_eviction_pipeline_lock(),
    }


def build_arm_config(hp: Any) -> dict[str, Any]:
    """Build the run_condition config dict from train_gpt.py hyperparams."""
    locks = _exp26_locks()
    cfg: dict[str, Any] = {
        # --- model ---
        "vocab_size": int(hp.vocab_size),
        "seq_len": int(hp.seq_len),
        "batch_size": int(hp.batch_size),
        "dtype": "bf16",
        "device": "auto",
        # --- training ---
        "budget_seconds": float(hp.budget_seconds),
        "seed": int(hp.seed),
        "base_lr": float(hp.base_lr),
        "weight_decay": float(hp.weight_decay),
        "grad_clip_norm": float(hp.grad_clip_norm),
        # --- optimizer: Muon + SemanticOptimizer channel coupling ---
        "optimizer": "muon",
        "optimizer_log_a_beta_coupling": bool(hp.log_a_beta_coupling),
        "optimizer_log_a_beta_ema": float(hp.log_a_beta_ema),
        "optimizer_log_a_beta_min": float(hp.log_a_beta_min),
        # --- calc_types: use packet_online_cache for official BPB ---
        "calc_types": ["packet_online_cache"],
        "headline_calc_type": "packet_online_cache",
    }
    # Merge ARM locks (crct, fast_slow, replay_eviction, artifact_size)
    cfg.update(locks)
    # Apply telemetry-tuned overrides (supersede lock defaults)
    cfg.update({
        "crct_memory_write_tokens_per_step": int(getattr(hp, "crct_memory_write_tokens_per_step", 192)),
        "online_episodic_write_tokens_per_chunk": int(getattr(hp, "online_episodic_write_tokens_per_chunk", 64)),
        "crct_target_write_rate": float(getattr(hp, "crct_target_write_rate", 0.20)),
        "lm_head_tile_size": int(getattr(hp, "lm_head_tile_size", 4096)),
    })
    return cfg
```

**Step 4: Run to verify pass**

```bash
.venv/bin/python -m pytest tests/public/test_engine_entry.py -v
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add src/chaoscontrol/public/engine_entry.py tests/public/test_engine_entry.py
git commit -m "feat: implement build_arm_config() with telemetry-tuned CRCT defaults"
```

---

## Task 4: Implement `run_arm_submission()` and complete `engine_entry.py`

**Files:**
- Modify: `src/chaoscontrol/public/engine_entry.py`

**Context:** `run_arm_submission()` adds `experiments/23_fast_path` (and any other needed experiment dirs) to `sys.path` using `CHAOSCONTROL_ROOT`, then imports and calls `run_condition` from `runner_fast_path.py`. This delegates the full 14,850-line ARM training+eval loop to the existing production implementation without reinventing it.

**Step 1: Implement**

```python
def _ensure_runner_path() -> None:
    """Add experiment dirs to sys.path so runner_fast_path is importable.

    Requires the chaoscontrol repo to be cloned at CHAOSCONTROL_ROOT
    (default /workspace/chaoscontrol). The pod bootstrap script puts it there.
    On local dev, set CHAOSCONTROL_ROOT to your clone path.
    """
    cc_root = os.environ.get("CHAOSCONTROL_ROOT", "/workspace/chaoscontrol")
    root = Path(cc_root)
    dirs = [
        root / "experiments" / "23_fast_path",
        root / "experiments" / "24_training_time_bundle",
        root / "experiments" / "26_arm",
        root / "src",  # in case chaoscontrol isn't installed, fall back to src
    ]
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def run_arm_submission(
    config: dict[str, Any],
    *,
    data_path: str,
    sp_model_path: str,
    budget_seconds: float,
    output_json: str | None,
    val_cache_dir: str | None,
    world_size_override: int | None = None,
) -> dict[str, Any]:
    """Delegate to run_condition() in runner_fast_path.py.

    runner_fast_path.py is the production ARM training + eval loop
    (experiments/23_fast_path/runner_fast_path.py, ~14,850 lines).
    We call it directly rather than reimplementing it. The config dict
    produced by build_arm_config() is what run_condition() expects.
    """
    _ensure_runner_path()
    from runner_fast_path import run_condition  # type: ignore[import]
    return run_condition(
        config,
        data_path=data_path,
        sp_model_path=sp_model_path,
        budget_seconds=budget_seconds,
        output_json=output_json,
        output_ckpt=None,
        world_size_override=world_size_override,
        val_cache_dir=val_cache_dir,
    )
```

**Step 2: Smoke-test the import chain on CPU (no GPU required)**

```bash
CHAOSCONTROL_ROOT="/Users/kennethmalloy/Local Documents/Developer/chaoscontrol" \
  .venv/bin/python -c "
from chaoscontrol.public.engine_entry import _ensure_runner_path
_ensure_runner_path()
from runner_fast_path import run_condition
print('run_condition importable:', callable(run_condition))
"
```

Expected: `run_condition importable: True`

**Step 3: Commit**

```bash
git add src/chaoscontrol/public/engine_entry.py
git commit -m "feat: implement run_arm_submission() delegating to runner_fast_path.run_condition"
```

---

## Task 5: Create submission folder structure

**Repo:** parameter-golf  
**Files:**
- Create dir: `records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/`
- Create dir: `records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/tokenizers/`
- Note: `submission.json`, `README.md`, and `train_seed*.log` are **not created yet** — they fill in after the actual run.

**Step 1: Create the skeleton**

```bash
mkdir -p "records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/tokenizers"
touch "records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/.gitkeep"
touch "records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/tokenizers/.gitkeep"
```

**Step 2: Verify**

```bash
ls records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/
```

Expected: `.gitkeep  tokenizers/`

**Step 3: Note on tokenizer**

The SP16384 tokenizer (`fineweb_16384_bpe.model`) lives at `Natooka/parameter-golf-sp-tokenizers` on HuggingFace and at `baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model` on the pod after bootstrap. Copy it into the submission folder on the pod before submitting:

```bash
cp baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
   records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/tokenizers/
```

**Step 4: Commit**

```bash
git add records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/
git commit -m "feat: create SemanticEngine submission folder skeleton"
```

---

## Task 6: Write `train_gpt.py` — Section 1 (Hyperparameters)

**Files:**
- Create: `records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/train_gpt.py`

**Step 1: Write the Hyperparameters class (heavily commented)**

The file starts with standard imports and a `Hyperparameters` class configurable entirely via env vars. Comments explain the architectural motivation for every non-obvious value. Here is the complete Section 1:

```python
#!/usr/bin/env python3
"""SemanticEngine — CareSSM with live episodic memory.

Entry point: torchrun --standalone --nproc_per_node=8 train_gpt.py

SemanticEngine is a CareSSM trunk with a live episodic memory substrate
(CRCT evidence + streaming Adaptive Residual Memory maintenance). Unlike
every other top submission, this is a pure SSM architecture. The memory
substrate runs on dedicated GPUs (GPU6 packet-serving, GPU7 maintenance)
and never blocks the trunk step.

Dependencies:
  - chaoscontrol installed from https://github.com/KenMalloy/chaoscontrol
  - CHAOSCONTROL_ROOT set to the cloned repo root (default /workspace/chaoscontrol)
  - Native extensions built: see chaoscontrol/scripts/pod_build_native_extensions.sh
  - SP16384 shards: Natooka/parameter-golf-sp-tokenizers on HuggingFace
  - ValCache pre-built from the first 50k val docs (scripts/pod_bootstrap.sh)

Components called out in the README:
  - CareSSM: the recurrent SSM trunk blocks (CareSSMCore/CareSSMBlock)
  - ChaosSsm: the CPU SSM controller (off-path evidence/scheduling plane)
  - SemanticOptimizer: Muon with SSM-channel-coupled momentum β, so optimizer
    time constants match each channel's forward-pass recurrence time constant
  - GPU6/GPU7: dedicated memory ranks — never share compute with the trunk
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


class Hyperparameters:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # SP16384 pre-tokenized shards, fetched from Natooka/parameter-golf-sp-tokenizers.
    data_path: str = os.environ.get(
        "DATA_PATH",
        "/workspace/chaoscontrol/baselines/parameter_golf/datasets/fineweb10B_sp16384",
    )
    # SP16384 SentencePiece model. Shipped in tokenizers/ inside this submission folder.
    tokenizer_path: str = os.environ.get(
        "TOKENIZER_PATH",
        str(Path(__file__).parent / "tokenizers" / "fineweb_16384_bpe.model"),
    )
    # ValCache directory — pre-built from the first 50,000 FineWeb validation documents.
    # Required for prequential eval. Built by scripts/pod_bootstrap.sh.
    val_cache_dir: str = os.environ.get(
        "VAL_CACHE_DIR",
        "/workspace/chaoscontrol/experiments/27_ttt_headline/val_cache",
    )
    # Path where the chaoscontrol repo is cloned (for experiment runner import).
    chaoscontrol_root: str = os.environ.get("CHAOSCONTROL_ROOT", "/workspace/chaoscontrol")
    # JSON file to write final result into (optional).
    output_json: str | None = os.environ.get("OUTPUT_JSON", None)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    vocab_size: int = 16384  # SP16384 vocabulary
    # dim=384 is the largest artifact-safe trunk width at int6/LZMA compression:
    #   384 → ~13.71 MB, 416 → ~15.19 MB, 448 → ~16.73 MB (budget exceeded).
    model_dim: int = int(os.environ.get("MODEL_DIM", 384))
    # Low-rank delta projection rank inside each CareSSMCore block.
    ssm_delta_rank: int = int(os.environ.get("SSM_DELTA_RANK", 32))
    seq_len: int = int(os.environ.get("SEQ_LEN", 512))
    batch_size: int = int(os.environ.get("BATCH_SIZE", 1024))
    # Fused LM-head tile size. 4096 keeps the fused backward path while avoiding
    # OOM on the cu130 pod stack after model activations at B=1024/T=512.
    lm_head_tile_size: int = int(os.environ.get("LM_HEAD_TILE_SIZE", 4096))

    # -------------------------------------------------------------------------
    # Training budget
    # -------------------------------------------------------------------------
    # Hard wallclock cap. Checked at the top of each training step so the loop
    # always exits at a complete-step boundary — never mid-step.
    budget_seconds: float = float(os.environ.get("BUDGET_SECONDS", 600.0))
    seed: int = int(os.environ.get("SEED", 42))

    # -------------------------------------------------------------------------
    # Optimizer — SemanticOptimizer (Muon + channel-coupled β)
    # -------------------------------------------------------------------------
    # Muon (Newton-Schulz orthogonalized momentum) on matrix params;
    # AdamW fallback on embeddings and scalars.
    base_lr: float = float(os.environ.get("BASE_LR", 0.064))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.01))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    # SemanticOptimizer: per-channel momentum β coupled to log_a decay.
    # Slow-recurrence channels (log_a near 0 → decay near 1) get high β so
    # gradients integrate over long horizons. Fast channels get lower β.
    log_a_beta_coupling: bool = bool(int(os.environ.get("LOG_A_BETA_COUPLING", 1)))
    log_a_beta_ema: float = float(os.environ.get("LOG_A_BETA_EMA", 0.99))
    log_a_beta_min: float = float(os.environ.get("LOG_A_BETA_MIN", 0.5))

    # -------------------------------------------------------------------------
    # CRCT evidence substrate (telemetry-tuned from profiling on 4×H100)
    # -------------------------------------------------------------------------
    # Per-step write cap. 192 gives the per-step cap meaningful headroom above
    # 128 without entering noisy territory. (exp26 default was 32.)
    crct_memory_write_tokens_per_step: int = int(
        os.environ.get("CRCT_MEMORY_WRITE_TOKENS_PER_STEP", 192)
    )
    # Per-chunk write budget for the online episodic cache. 64 is the first real
    # step up from the profiled 16 without being reckless.
    online_episodic_write_tokens_per_chunk: int = int(
        os.environ.get("ONLINE_EPISODIC_WRITE_TOKENS_PER_CHUNK", 64)
    )
    # Target write rate. 0.20 matches observed adaptive smoke behavior
    # (payload rate ~14/64 = 0.219); previous lock value was 0.10.
    crct_target_write_rate: float = float(
        os.environ.get("CRCT_TARGET_WRITE_RATE", 0.20)
    )
    # Async teacher lag and pending-batch limits are left at exp26 defaults.
    # Profiling shows max lag 3–4 steps and no ring drops — the pipe is not
    # the bottleneck, so no change warranted.
```

**Step 2: Verify the file is valid Python**

```bash
cd "records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM"
python3 -c "import ast; ast.parse(open('train_gpt.py').read()); print('syntax ok')"
```

Expected: `syntax ok`

**Step 3: Write a unit test for hyperparameter defaults and env-var overrides**

```python
# tests/submission/test_train_gpt_hyperparams.py
import importlib.util
import os
import sys
from pathlib import Path

TRAIN_GPT = (
    Path(__file__).parents[2]
    / "records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/train_gpt.py"
)

def _load_hp():
    spec = importlib.util.spec_from_file_location("train_gpt", TRAIN_GPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Hyperparameters

def test_defaults():
    HP = _load_hp()
    assert HP.vocab_size == 16384
    assert HP.model_dim == 384
    assert HP.budget_seconds == 600.0
    assert HP.crct_memory_write_tokens_per_step == 192
    assert HP.online_episodic_write_tokens_per_chunk == 64
    assert abs(HP.crct_target_write_rate - 0.20) < 1e-6
    assert HP.log_a_beta_coupling is True

def test_env_override(monkeypatch):
    monkeypatch.setenv("SEED", "1337")
    monkeypatch.setenv("BUDGET_SECONDS", "300.0")
    monkeypatch.setenv("MODEL_DIM", "256")
    HP = _load_hp()
    assert HP.seed == 1337
    assert HP.budget_seconds == 300.0
    assert HP.model_dim == 256
```

**Step 4: Run tests**

```bash
cd "/Users/kennethmalloy/Local Documents/Developer/parameter-golf"
# Use system python3 or a local venv if available
python3 -m pytest tests/submission/test_train_gpt_hyperparams.py -v
```

Expected: both tests pass.

**Step 5: Commit**

```bash
git add records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/train_gpt.py
git add tests/submission/test_train_gpt_hyperparams.py
git commit -m "feat: train_gpt.py Section 1 — heavily commented Hyperparameters class"
```

---

## Task 7: Write `train_gpt.py` — Section 2 (`main()`)

**Files:**
- Modify: `records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/train_gpt.py`

**Step 1: Append the full main() function**

Append below the `Hyperparameters` class. This is a single `main()` that:
1. Inits dist, routes roles
2. Loads data
3. Builds + runs the submission via chaoscontrol.public
4. Logs the score

```python
def main() -> None:
    import torch
    import torch.distributed as dist

    # The chaoscontrol.public module must be importable.
    # Install chaoscontrol from GitHub per requirements.txt.
    os.environ.setdefault("CHAOSCONTROL_ROOT", Hyperparameters.chaoscontrol_root)
    from chaoscontrol.public.engine_entry import (
        init_arm_topology,
        build_arm_config,
        run_arm_submission,
    )

    # --- Distributed init ---
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE in the environment.
    # dist.init_process_group reads them automatically.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # --- Role routing ---
    # On 8 GPUs: GPU0-5 are train ranks; GPU6 is the dedicated packet-serving
    # rank (low-latency episodic residual production); GPU7 is the dedicated
    # maintenance rank (oracle scoring, slot commits). The train ranks never
    # wait on the memory ranks — if no fresh packet is ready, the trunk
    # proceeds with a zero-residual failsafe.
    # On 4 GPUs: GPU3 shares both memory roles (smoke/profile topology).
    role = init_arm_topology(rank=rank, world_size=world_size)
    if rank == 0:
        print(
            f"[semanticengine] topology: world={world_size} "
            f"packet_rank={role.packet_rank} maintenance_rank={role.maintenance_rank} "
            f"split={role.split_memory_ranks}",
            flush=True,
        )

    # --- Build config ---
    # build_arm_config maps the Hyperparameters class → the config dict that
    # runner_fast_path.run_condition() expects. It merges the four exp26 lock
    # dicts (artifact_size, fast_slow, crct, replay_eviction) and applies the
    # telemetry-tuned overrides.
    config = build_arm_config(Hyperparameters)

    # --- Training + eval ---
    # run_arm_submission delegates to run_condition() in runner_fast_path.py,
    # the production ARM training + prequential eval loop (~14,850 lines).
    #
    # Training: trunk updates weights; memory/controller stack generates
    # evidence and maintains the cache. Wallclock is checked at the top of
    # each step — the loop always exits at a complete-step boundary.
    #
    # Eval: same memory substrate is live, but the run is prequential.
    # Score each chunk under the current state first, accumulate loss/BPB,
    # then optionally update from already-scored tokens. The trunk never
    # sees validation tokens before they are scored. Enforced at the Python
    # level: packet_online_cache raises if the slot count changes between
    # cue read and score accumulation.
    t_start = time.perf_counter()
    result = run_arm_submission(
        config,
        data_path=Hyperparameters.data_path,
        sp_model_path=Hyperparameters.tokenizer_path,
        budget_seconds=Hyperparameters.budget_seconds,
        output_json=Hyperparameters.output_json,
        val_cache_dir=Hyperparameters.val_cache_dir,
        world_size_override=world_size,
    )
    elapsed = time.perf_counter() - t_start

    # --- Score summary (rank 0 only) ---
    if rank == 0:
        eval_r = result.get("eval") or {}
        calc_types = eval_r.get("calc_types") or {}
        poc = calc_types.get("packet_online_cache") or {}
        train_r = result.get("train") or {}
        print(
            f"\n[semanticengine] === SCORE SUMMARY ===\n"
            f"  val_bpb:          {poc.get('bpb', float('nan')):.6f}\n"
            f"  val_loss:         {poc.get('loss', float('nan')):.6f}\n"
            f"  docs_scored:      {poc.get('docs_scored', 0)}\n"
            f"  train_steps:      {train_r.get('steps', 0)}\n"
            f"  train_elapsed_s:  {train_r.get('elapsed_s', 0.0):.1f}\n"
            f"  total_elapsed_s:  {elapsed:.1f}\n"
            f"  artifact_bytes:   {result.get('artifact_bytes', 'N/A')}\n"
            f"  code_bytes:       {result.get('code_bytes', 'N/A')}\n"
            f"[semanticengine] === END SUMMARY ===",
            flush=True,
        )
        if Hyperparameters.output_json:
            Path(Hyperparameters.output_json).write_text(json.dumps(result, indent=2, default=str))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('train_gpt.py').read()); print('syntax ok')"
```

Expected: `syntax ok`

**Step 3: Test the score-summary log parsing**

```python
# append to tests/submission/test_train_gpt_hyperparams.py

def test_score_summary_keys_present():
    """Regression: the summary block must contain val_bpb and artifact_bytes."""
    source = TRAIN_GPT.read_text()
    assert "val_bpb" in source
    assert "artifact_bytes" in source
    assert "packet_online_cache" in source
    assert "score-before-write" in source.lower() or "score each chunk" in source.lower()
```

**Step 4: Run**

```bash
python3 -m pytest tests/submission/ -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/train_gpt.py
git add tests/submission/test_train_gpt_hyperparams.py
git commit -m "feat: train_gpt.py Section 2 — main() with role routing, ARM training, prequential eval, score summary"
```

---

## Task 8: Write `requirements.txt`

**Files:**
- Create: `records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/requirements.txt`

**Step 1: Write the file**

```
# Core — exact versions used on the submission pod
torch==2.11.0
sentencepiece>=0.2.0
numpy>=1.24
huggingface-hub>=0.22

# SemanticEngine / ChaosControl library
chaoscontrol @ git+https://github.com/KenMalloy/chaoscontrol.git

# TransformerEngine (CUDA 13 build). Must be installed before building native extensions.
# transformer_engine[pytorch]==2.13.0
# Install with:
#   pip install transformer_engine[pytorch]==2.13.0 \
#     --extra-index-url https://pypi.nvidia.com \
#     --only-binary=:all: \
#     nvidia-cublas==13.4.0.1
#
# See chaoscontrol/scripts/pod_setup_cuda13.sh for the full idempotent install.

# Native extensions — built from the chaoscontrol repo, not pip-installed.
# After cloning to CHAOSCONTROL_ROOT, run:
#   bash scripts/pod_build_native_extensions.sh
# Extensions: _lm_head_loss, _cpu_ssm_controller, _ssm_scan
```

**Step 2: Commit**

```bash
git add records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/requirements.txt
git commit -m "feat: add requirements.txt with chaoscontrol + TE install notes"
```

---

## Task 9: Pod smoke test (requires 8×H100)

**This task runs on the RunPod pod, not locally.**

**Step 1: Bootstrap the pod**

```bash
cd /workspace
git clone https://github.com/KenMalloy/chaoscontrol.git
cd chaoscontrol
bash scripts/pod_bootstrap.sh
```

Expected: smoke check prints `torch X.Y.Z  cuda=True  GPUs=8`

**Step 2: Copy the submission folder to the pod**

Either push to git and pull, or rsync. Ensure `tokenizers/fineweb_16384_bpe.model` is present:

```bash
cp baselines/parameter_golf/tokenizers/fineweb_16384_bpe.model \
   /path/to/submission/tokenizers/
```

**Step 3: Dry-run check (no actual training, just verifies imports + config build)**

```bash
CHAOSCONTROL_ROOT=/workspace/chaoscontrol \
DATA_PATH=/workspace/chaoscontrol/baselines/parameter_golf/datasets/fineweb10B_sp16384 \
TOKENIZER_PATH=/path/to/submission/tokenizers/fineweb_16384_bpe.model \
VAL_CACHE_DIR=/workspace/chaoscontrol/experiments/27_ttt_headline/val_cache \
BUDGET_SECONDS=30 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed42_smoke.log
```

**Step 4: Check smoke log**

Look for:
- `[semanticengine] topology: world=8 packet_rank=6 maintenance_rank=7 split=True`
- `=== SCORE SUMMARY ===` block with numeric `val_bpb`
- `artifact_bytes` that is ≤ 16,000,000

**Step 5: Full run — 3 seeds**

```bash
for SEED in 42 1337 1234; do
  SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee train_seed${SEED}.log
done
```

Extract the three `val_bpb` values and compute the mean. This is the number that goes into `submission.json`.

---

## Post-run: Fill submission.json and README.md

After the 3-seed run, fill:
- `submission.json` — follow the format of `2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/submission.json`
- `README.md` — headline, component table (SemanticEngine / CareSSM / ChaosSsm / GPU6 / GPU7 / SemanticOptimizer), results table, architecture section, reproducing command

---

## Optional (nice-to-have): ChaosSsm alias

If time permits: in `src/chaoscontrol/public/engine_entry.py`, add:

```python
from chaoscontrol.episodic.cpu_ssm_controller import CpuSsmControllerRuntime as ChaosSsm
```

And export it from `__init__.py`. This gives reviewers a clean name to reference without touching internal class names.
