# SemanticEngine Submission Design

**Date:** 2026-05-01  
**Track:** `track_10min_16mb`  
**Submission folder:** `records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/`

---

## 1. System Overview

The submission presents **SemanticEngine** — a CareSSM trunk with live episodic memory. Unlike every other top submission (transformer-based), this is a pure SSM architecture whose memory substrate is active during both training and prequential eval.

### Named Components

| Name | Role | Code location |
|---|---|---|
| **SemanticEngine** | Overall system | this submission |
| **CareSSM** | SSM trunk blocks | `chaoscontrol.core`, `chaoscontrol.model` |
| **ChaosSsm** | CPU SSM controller (nice-to-have rename from `CpuSsmController*`) | `chaoscontrol.episodic.cpu_ssm_controller` |
| **Episodic memory** | CRCT evidence substrate + MultiSlotOuterModel + replay eviction pipeline | `chaoscontrol.memory`, `chaoscontrol.replay_eviction` |
| **SemanticOptimizer** | Muon with SSM-channel-coupled momentum β | `chaoscontrol.optim.muon` (via `log_a_beta_coupling=True`) |

**Note on episodic memory:** The live memory substrate (CRCT + MultiSlotOuterModel + streaming maintenance) is architecturally compatible with any Mamba-style SSM. CareSSM is built with it in mind, not the other way around.

**Note on SemanticOptimizer:** The concept (per-channel momentum β coupled to each channel's `log_a` decay so optimizer time constants match recurrence time constants) is implemented as the `log_a_beta_coupling` extension in the `Muon` class. The standalone `SemanticOptimizer` class in `optim/semantic.py` is the fuller future version. The submission uses `Muon(log_a_beta_coupling=True)`.

---

## 2. File Structure

### Submission folder (`records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/`)

```
train_gpt.py                        # ~700-900 lines, orchestrating driver (see §4)
requirements.txt                    # chaoscontrol @ git+..., torch, sentencepiece, etc.
submission.json                     # filled after run
README.md                           # filled after run
train_seed<N>.log                   # filled after run (3 seeds)
tokenizers/
  fineweb_16384_bpe.model           # SP16384 tokenizer, shipped in submission folder
```

### New chaoscontrol module (`src/chaoscontrol/public/`)

```
src/chaoscontrol/public/
  __init__.py
  engine_entry.py                   # init_arm_topology(), run_training(), build_artifact(), run_eval()
```

`public/` is the name: it signals this is the stable public-facing interface, not internal experiment scaffolding.

All heavy machinery (distributed loop, CRCT, replay eviction topology, GPTQ, prequential eval) stays in existing chaoscontrol modules. `engine_entry.py` (~150–200 lines) connects them under a stable interface that `train_gpt.py` calls.

---

## 3. Data and Dependencies

### Data

- **Tokenizer:** SP16384 (`fineweb_16384_bpe.model`, 455 KB), shipped inside the submission folder
- **Train/val shards:** `Natooka/parameter-golf-sp-tokenizers` on HuggingFace — 133 train shards (~25 GB) + 1 val shard (~84 MB, 42,266,034 tokens)
- **ValCache:** Pre-built from the first 50,000 validation documents; used by the prequential eval. Built via `scripts/build_exp20_val_cache.py` on pod setup.

### Native extensions (must be built before running)

| Extension | Purpose |
|---|---|
| `_lm_head_loss` | Fused chunked LM head backward (8× VRAM reduction at V=16384) |
| `_cpu_ssm_controller` | ChaosSsm CPU controller (C++ with optional CUDA write-event pack) |
| `_ssm_scan` | Chunked parallel SSM scan CUDA kernel |

Built via `scripts/pod_build_native_extensions.sh`. The full pod setup (CUDA 13 + TE 2.13 + extensions + data) runs via `scripts/pod_bootstrap.sh`.

### Requirements

- PyTorch 2.11.0+cu130 (CUDA 13)
- TransformerEngine 2.13.0
- `chaoscontrol @ git+https://github.com/KenMalloy/chaoscontrol.git`
- `sentencepiece`, `huggingface-hub`, `numpy`

No network calls inside `train_gpt.py` during training or eval. The `chaoscontrol` package is pip-installed before the script runs.

---

## 4. `train_gpt.py` Internal Structure

Entry point: `torchrun --standalone --nproc_per_node=8 train_gpt.py`  
All config via env vars. Matches the interface of every other submission.

### Section 1 — Hyperparameters (heavily commented, ~100 lines)

An env-var-configurable class. Comments explain the architectural motivation for each setting, not just the value.

Key groups:
- **Paths:** `DATA_PATH`, `VAL_CACHE_DIR`, `TOKENIZER_PATH`
- **Model:** `model_dim=384` (artifact-safe at int6/LZMA; next size up at 416 is 15.19 MB, 448 exceeds budget), `ssm_delta_rank=32`
- **CRCT:** `crct_memory_write_tokens_per_step=32`, `crct_target_read_rate=0.25`, `crct_target_write_rate=0.10`, `outer_max_slots=4096`, and the full locked CRCT config from `exp26._crct_lock()`
- **Replay eviction:** `replay_eviction_memory_streams=8`, `replay_eviction_commit_policy="learned"`, and the full pipeline config from `exp26._replay_eviction_pipeline_lock()`
- **Fast/slow:** `fast_slow_alpha=0.25`, `fast_slow_eval_copy="slow"`, controller settings from `exp26._fast_slow_lock()`
- **Training:** `BUDGET_SECONDS=600`, `WARMUP_STEPS=20`, warmdown schedule, `GRAD_CLIP_NORM`
- **Optimizer:** SemanticOptimizer flags — `log_a_beta_coupling=True`, `log_a_beta_ema=0.99`, `log_a_beta_min=0.5`; Muon for matrix params, AdamW fallback for embeddings/scalars
- **Quantization:** GPTQ int6 for matrices, int7 for tied embeddings
- **Eval:** `CHUNK_TOKENS`, `WRITE_TOKENS_PER_CHUNK`, `DECAY` for `packet_online_cache`

### Section 2 — `main()` (heavily commented, ~600-800 lines)

Comments in this section explain the training/eval distinction clearly for reviewers:

> During training, the trunk updates weights and the memory/controller stack generates evidence and maintains the cache. During eval, the same memory substrate is live but the run is prequential: score each chunk under the current state first, accumulate loss, then optionally update from those already-scored tokens. The trunk never sees validation tokens before they are scored.

**Dist init + role routing** (~25 lines)  
Calls `chaoscontrol.public.engine_entry.init_arm_topology(world_size)`. On 8 GPUs: GPU 0–5 are train ranks, GPU 6 is the packet-serving rank, GPU 7 is the maintenance rank. On 4 GPUs: GPU 3 shares both memory roles. Role routing is encapsulated here because it can't be described readably inline.

**Data + ValCache load** (~30 lines)  
Shards from `DATA_PATH`. ValCache from `VAL_CACHE_DIR` (pre-built, not constructed at runtime).

**Model + optimizer** (~60 lines)  
Build `ChaosControlConfig` from the hyperparameter block. Instantiate `CareStudentLM`. Construct the SemanticOptimizer (Muon with `log_a_beta_coupling=True`) on matrix params; AdamW on embeddings and scalars.

**Training loop** (~200 lines)  
```
while True:
    if time.perf_counter() - t_start >= BUDGET_SECONDS:
        break  # always exits at a complete step boundary
    step += 1
    <forward, loss, backward, optimizer step, fast/slow consolidation>
    if step % 100 == 0:
        log(step, loss, tokens_per_sec, elapsed_s)
```

Wallclock check is the first thing in each iteration. When it fires, the loop exits at a complete-step boundary — no partial state enters the artifact. Log message: `"training stopped at step N (wallclock), artifact built from step N state"`.

**Artifact build** (~80 lines)  
Calls `chaoscontrol.artifact.serialize_artifact(model, ...)`. GPTQ int6 + int7 embed + LZMA compression. Logs `code_bytes`, `model_bytes`, `total_bytes` explicitly.

**Prequential eval** (~100 lines)  
Loads the serialized artifact. Calls `evaluate_with_calc_types(model, val_cache, calc_types=["packet_online_cache"], config=eval_config)`. The `packet_online_cache` calc type enforces score-before-write at the Python level (raises `RuntimeError` if the cache slot count changes between cue read and score accumulation). Iterates all 50,000 validation documents. Returns `val_bpb`, `val_loss`.

**Score summary** (~20 lines)  
Rank 0 prints a parseable summary: `val_bpb`, `val_loss`, `artifact_bytes`, `train_steps`, `train_time_s`, `eval_time_s`.

---

## 5. New Chaoscontrol Code — `public/engine_entry.py`

~150–200 lines. Three functions:

**`init_arm_topology(world_size) -> RoleInfo`**  
GPU role assignment. Returns the local process's role (train / packet-serving / maintenance) and associated NCCL group handles. Single source of truth for the 6+2 topology.

**`run_training(model, optimizer, data, config) -> TrainingResult`**  
Thin wrapper over the existing training loop in `chaoscontrol.training`. Returns `steps`, `elapsed_s`, `final_loss`. Called by `train_gpt.py` after model/optimizer construction.

**`run_eval(artifact_path, val_cache, config) -> EvalResult`**  
Loads artifact, calls `evaluate_with_calc_types` with `packet_online_cache`. Returns `bpb`, `loss`, `docs_scored`, `elapsed_s`.

---

## 6. Training / Eval Distinction

The prequential eval contract, stated explicitly for reviewers:

- **Score first:** Each chunk is scored under the model's current memory state. Loss is accumulated before the cache is updated.
- **Write after:** The just-scored hidden states and token NLLs are committed to the episodic cache only after scoring. Future chunks may read them.
- **Trunk weights frozen:** The trunk does not update during eval. Only the episodic cache grows.
- **Enforced:** `packet_online_cache.py` checks `_outer_slot_count(model)` before and after scoring each chunk; a count change before score accumulation raises immediately.

---

## 7. Implementation Plan

The following tasks (in order) produce a runnable train_gpt.py and a score:

1. Create `src/chaoscontrol/public/__init__.py` and `engine_entry.py` with the three functions
2. Write `records/track_10min_16mb/2026-05-01_SemanticEngine_CareSSM/train_gpt.py`
3. Write `requirements.txt`
4. Spin up 8xH100 pod, run `scripts/pod_bootstrap.sh`
5. Run `torchrun --standalone --nproc_per_node=8 train_gpt.py` for seed 42
6. Capture log, verify `val_bpb` in output
7. Repeat for seeds 1337 and 1234 (3-seed mean)
8. Fill `submission.json` and `README.md`

---

## 8. Open Items

- **ChaosSsm rename:** Nice-to-have. `CpuSsmController*` classes can be aliased or renamed in `chaoscontrol/public/` without touching internal code. Not blocking implementation.
- **ScOpt:** Not used in this submission. `ScarcityAwareOptimizer` is the parent concept that birthed `SemanticOptimizer`; noted for future work.
- **Folder name:** `2026-05-01_SemanticEngine_CareSSM` — may shift to a date after the actual run if we slip past May 1.
