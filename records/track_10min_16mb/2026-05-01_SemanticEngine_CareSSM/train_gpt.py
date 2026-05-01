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
import sys  # noqa: F401  # reserved for future sys.exit paths
import time
from pathlib import Path


def _env_bool(key: str, default: int) -> bool:
    return bool(int(os.environ.get(key, default)))


# 75.92 GiB of PyTorch-allocated tensors leaves only ~522 MiB contiguous free
# on an 80 GiB H100 when the SSM scan tries to allocate a 768 MiB (B×T×D
# float32) tensor. The remaining 728 MiB is reserved but fragmented.
# expandable_segments lets the allocator compose non-contiguous segments
# instead of requiring a single 768 MiB contiguous block.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    # JSON / checkpoint files to write final result into (optional).
    output_json: str | None = os.environ.get("OUTPUT_JSON", None)
    output_ckpt: str | None = os.environ.get("OUTPUT_CKPT", None)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    vocab_size: int = 16384  # SP16384 vocabulary
    # dim=384 is the largest artifact-safe trunk width at int6/LZMA compression:
    #   384 → ~13.71 MB, 416 → ~15.19 MB, 448 → ~16.73 MB (budget exceeded).
    model_dim: int = int(os.environ.get("MODEL_DIM", 384))
    # 8 layers is the scaling-law-validated depth for dim=384 (see exp10).
    num_layers: int = int(os.environ.get("NUM_LAYERS", 8))
    # Keep the final path uncheckpointed so the trunk uses the fastest SSM
    # backward. On 8xH100, B=1024/960 OOM before step 1 and B=832 OOMed late;
    # B=800 completed three 600s seeds without activation checkpointing.
    activation_checkpoint: bool = _env_bool("ACTIVATION_CHECKPOINT", 0)
    # Low-rank delta projection rank inside each CareSSMCore block.
    ssm_delta_rank: int = int(os.environ.get("SSM_DELTA_RANK", 32))
    seq_len: int = int(os.environ.get("SEQ_LEN", 512))
    batch_size: int = int(os.environ.get("BATCH_SIZE", 800))
    # Fused LM-head tile size. Scratch = B*T*tile*2 bytes.
    # tile=4096 → 4 GiB; combined with 74.66 GiB of other allocations this leaves
    # only 1.34 GiB free on an 80 GiB H100, causing fragmentation OOM in the SSM
    # scan. tile=2048 → 2 GiB scratch, freeing 2 GiB and giving 3.34 GiB headroom.
    lm_head_tile_size: int = int(os.environ.get("LM_HEAD_TILE_SIZE", 2048))

    # -------------------------------------------------------------------------
    # Training budget
    # -------------------------------------------------------------------------
    # Hard wallclock cap. Checked at the top of each training step so the loop
    # always exits at a complete-step boundary — never mid-step.
    budget_seconds: float = float(os.environ.get("BUDGET_SECONDS", 600.0))
    # Stop the gradient-bearing loop early enough for complete-step shutdown,
    # DDP bookkeeping, and checkpoint serialization while the submission still
    # advertises the true 600s training budget. Hardware/compiler warmup must
    # remain weight/state-free and outside this timed training loop.
    stop_margin_seconds: float = float(os.environ.get("STOP_MARGIN_SECONDS", 32.0))
    seed: int = int(os.environ.get("SEED", 42))
    max_steps: int | None = (
        None
        if os.environ.get("MAX_STEPS") in (None, "")
        else int(os.environ["MAX_STEPS"])
    )
    train_only: bool = _env_bool("TRAIN_ONLY", 0)
    eval_only: bool = _env_bool("EVAL_ONLY", 0)
    checkpoint_path: str | None = os.environ.get("CHECKPOINT_PATH", None)
    eval_max_docs: int = int(os.environ.get("EVAL_MAX_DOCS", 0))
    # Eval is prequential, but causal does not mean read-only. Score each
    # microbatch first, then let already-scored evidence enter the episodic
    # cache for future microbatches. These defaults are the live-writing path;
    # set PACKET_EVAL_WRITE_TOKENS_PER_CHUNK=0 only as an emergency budget
    # fallback.
    packet_eval_batch_docs: int = int(os.environ.get("PACKET_EVAL_BATCH_DOCS", 48))
    packet_eval_batch_token_budget: int = int(
        os.environ.get("PACKET_EVAL_BATCH_TOKEN_BUDGET", 49152)
    )
    packet_eval_write_tokens_per_chunk: int = int(
        os.environ.get("PACKET_EVAL_WRITE_TOKENS_PER_CHUNK", 1)
    )
    packet_eval_controller_read_enabled: bool = _env_bool(
        "PACKET_EVAL_CONTROLLER_READ", 0
    )
    packet_eval_controller_topk_k: int = int(
        os.environ.get("PACKET_EVAL_CONTROLLER_TOPK_K", 16)
    )
    packet_eval_controller_score_mode: str = os.environ.get(
        "PACKET_EVAL_CONTROLLER_SCORE_MODE", "cosine_survival"
    )

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
    log_a_beta_coupling: bool = _env_bool("LOG_A_BETA_COUPLING", 1)
    log_a_beta_ema: float = float(os.environ.get("LOG_A_BETA_EMA", 0.99))
    log_a_beta_min: float = float(os.environ.get("LOG_A_BETA_MIN", 0.5))

    # -------------------------------------------------------------------------
    # CRCT evidence substrate (telemetry-tuned from profiling on 4×H100)
    # -------------------------------------------------------------------------
    # Per-step write cap. 256 gives the per-step cap meaningful headroom above
    # 128 without entering noisy territory. (exp26 default was 32.)
    crct_memory_write_tokens_per_step: int = int(
        os.environ.get("CRCT_MEMORY_WRITE_TOKENS_PER_STEP", 256)
    )
    # Per-chunk write budget for the online episodic cache. 64 is the first real
    # step up from the profiled 16 without being reckless.
    online_episodic_write_tokens_per_chunk: int = int(
        os.environ.get("ONLINE_EPISODIC_WRITE_TOKENS_PER_CHUNK", 64)
    )
    # Target write rate. 0.25 is slightly above the observed adaptive smoke
    # behavior (~0.219) and is a clean round number. Previous lock was 0.10.
    crct_target_write_rate: float = float(
        os.environ.get("CRCT_TARGET_WRITE_RATE", 0.25)
    )
    # Weight mirroring is latest-complete, not every-step truth. On the final
    # 8×H100 B=800 run, physical snapshot throughput was ~1 publish per
    # 6.1 steps (293 writes / 1784 steps) while every-step attempts caused
    # ~1490 latest overwrites. Match the cadence to the hardware instead of
    # queueing stale mirror work.
    crct_teacher_param_sync_interval_steps: int = int(
        os.environ.get("CRCT_TEACHER_PARAM_SYNC_INTERVAL_STEPS", 6)
    )
    # Async teacher lag and pending-batch limits are left at exp26 defaults.
    # The observed pipe had no ring drops; stale snapshot churn, not ring
    # capacity, was the bottleneck.


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
    # The runner sizes ARM buffers and token-budget computations using
    # world_size_override. GPU6 (packet-serving) and GPU7 (maintenance) are
    # not train ranks — passing the full world_size=8 causes the runner to
    # over-allocate by a factor of 8/6, pushing peak VRAM past 80 GiB.
    # Pass num_train_ranks (6 for 8-GPU, 3 for 4-GPU, 1 for 1-GPU) so
    # allocations are sized for the ranks that actually do gradient steps.
    if world_size <= 1:
        num_train_ranks = world_size
    else:
        num_train_ranks = world_size - (2 if role.split_memory_ranks else 1)
    if rank == 0:
        print(
            f"[semanticengine] topology: world={world_size} "
            f"train_ranks={num_train_ranks} "
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
    config["stop_margin_seconds"] = float(Hyperparameters.stop_margin_seconds)
    if Hyperparameters.max_steps is not None:
        config["max_steps"] = int(Hyperparameters.max_steps)
    if Hyperparameters.train_only:
        config["calc_types"] = []
        config["headline_calc_type"] = None
    if Hyperparameters.eval_only:
        if not Hyperparameters.checkpoint_path:
            raise ValueError("EVAL_ONLY=1 requires CHECKPOINT_PATH")
        config["eval_only"] = True
        config["checkpoint_path"] = Hyperparameters.checkpoint_path
        config["calc_types"] = ["packet_online_cache"]
        config["headline_calc_type"] = "packet_online_cache"
    if Hyperparameters.eval_max_docs > 0:
        config.setdefault("calc_type_configs", {})
        config["calc_type_configs"].setdefault("packet_online_cache", {})
        config["calc_type_configs"]["packet_online_cache"]["max_docs"] = int(
            Hyperparameters.eval_max_docs
        )
    if Hyperparameters.eval_only:
        config.setdefault("calc_type_configs", {})
        packet_cfg = config["calc_type_configs"].setdefault("packet_online_cache", {})
        packet_cfg["batch_docs"] = int(Hyperparameters.packet_eval_batch_docs)
        packet_cfg["batch_token_budget"] = int(
            Hyperparameters.packet_eval_batch_token_budget
        )
        packet_cfg["write_tokens_per_chunk"] = int(
            Hyperparameters.packet_eval_write_tokens_per_chunk
        )
        packet_cfg["controller_read_enabled"] = bool(
            Hyperparameters.packet_eval_controller_read_enabled
        )
        packet_cfg["controller_topk_k"] = int(
            Hyperparameters.packet_eval_controller_topk_k
        )
        packet_cfg["controller_score_mode"] = str(
            Hyperparameters.packet_eval_controller_score_mode
        )

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
        output_ckpt=Hyperparameters.output_ckpt,
        val_cache_dir=Hyperparameters.val_cache_dir,
        world_size_override=num_train_ranks,
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


if __name__ == "__main__":
    main()
