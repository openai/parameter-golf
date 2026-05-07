import base64, collections, copy, glob, io, json, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid, torch, torch.distributed as dist, torch.nn.functional as F
from torch import nn

_TEST_MODE = bool(os.environ.get("RUN_TESTS"))

if not _TEST_MODE:
    import fcntl
    import numpy as np
    import sentencepiece as spm
    from flash_attn_interface import (
        flash_attn_func as flash_attn_3_func,
        flash_attn_varlen_func,
    )
    from concurrent.futures import ThreadPoolExecutor
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor
else:
    # Under RUN_TESTS we only exercise optimizer primitives; GPU kernels
    # and tokenizer libs are not needed. Stub the modules so module-level
    # decorators and type references still parse.
    fcntl = None
    np = None
    spm = None
    flash_attn_3_func = None
    flash_attn_varlen_func = None
    from concurrent.futures import ThreadPoolExecutor

    class _TritonStub:
        @staticmethod
        def jit(fn):  # decorator no-op
            return fn

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

    class _TlStub:
        constexpr = type("constexpr", (), {})
        bfloat16 = None

        @staticmethod
        def range(*args, **kwargs):
            raise RuntimeError("tl unavailable in RUN_TESTS")

    triton = _TritonStub()
    tl = _TlStub()
    TensorDescriptor = None


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.75))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 6e2))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 4096))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    # Default ON: this file is primarily submitted as a non-record base-model
    # record (TTT disabled); sliding-window eval is the load-bearing number.
    # Override to 0 during internal TTT ablations to save eval time.
    sliding_window_enabled = bool(int(os.environ.get("SLIDING_WINDOW_ENABLED", "1")))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", 512))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 3e1))
    rope_base = float(os.environ.get("ROPE_BASE", 1e4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))  # partial RoPE dims (used on global layers)
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 4096))
    rope_yarn = bool(int(os.environ.get("ROPE_YARN", "0")))
    # Gemma-style interleaved local/global attention
    global_attn_layers = [int(x) for x in os.environ.get("GLOBAL_ATTN_LAYERS", "4,9,10").split(",")]
    # Per-depth sliding-window size on local-attention layers. Defaults pair
    # 512 on early local layers with 1024 on deeper local layers; split is
    # the absolute layer index at/after which LATE applies. Default 6 keeps
    # layer 5 (loop tail, loop_end=5) in the cheap 512 bucket — the loop
    # multiplies attention cost, so widening windows there is costly.
    # Pass -1 to auto-split at num_layers // 2.
    local_window_size_early = int(os.environ.get("LOCAL_WINDOW_SIZE_EARLY", 512))
    local_window_size_late = int(os.environ.get("LOCAL_WINDOW_SIZE_LATE", 1024))
    local_window_split_layer = int(os.environ.get("LOCAL_WINDOW_SPLIT_LAYER", 6))
    kv_tie_global = bool(int(os.environ.get("KV_TIE_GLOBAL", "0")))  # K=V weight sharing on global layers
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    # Runtime bypass for Parcae boundary at eval/TTT only — training
    # still applies Parcae as usual. Setting this True on the eval and
    # TTT models disables `_parcae_boundary` in their forward paths,
    # giving TTT gradients an undampened path through loop iterations
    # at the cost of distribution mismatch vs training activations.
    parcae_eval_bypass = bool(int(os.environ.get("PARCAE_EVAL_BYPASS", "0")))
    eval_extra_loops = int(os.environ.get("EVAL_EXTRA_LOOPS", 0))  # extra recurrence iterations at eval time
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    loop_trunc_bwd = int(os.environ.get("LOOP_TRUNC_BWD", 1))  # detach after first N loop passes (0=full backprop)
    parallel_start_layer = int(os.environ.get("PARALLEL_START_LAYER", 8))
    parallel_final_lane = os.environ.get("PARALLEL_FINAL_LANE", "mean")
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.026))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.97))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 4))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-08))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    optim_mode = os.environ.get("OPTIM_MODE", "muon")
    scale_mode = os.environ.get("SCALE_MODE", "auto")
    cautious_mask = bool(int(os.environ.get("CAUTIOUS_MASK", "0")))
    precond_momentum = bool(int(os.environ.get("PRECOND_MOMENTUM", "1")))
    precond_source = os.environ.get("PRECOND_SOURCE", "grad")
    soap_base = os.environ.get("SOAP_BASE", "adam")
    # Which banks use the preconditioner; others fall back to plain muon msgn.
    # Values: "all" (default), "mlp" (only mlp_up + mlp_down — attention banks
    # run as muon), or comma-separated bank names like "mlp_up,mlp_down,out".
    # Useful when preconditioner wall-clock overhead outweighs per-step gain
    # on lower-anisotropy banks (attention is roughly isotropic; MLP-down's
    # post-GELU/xIELU hidden is the most anisotropic).
    precond_banks = os.environ.get("PRECOND_BANKS", "all")
    # NS iteration count for inverse-square-root (shampoo_ns inverse-sqrt,
    # and any other NS-based matrix root). Lower is faster but gives looser
    # inv-sqrt; higher is tighter. Default 8 is safe for kappa up to ~1e3.
    inv_root_ns_steps = int(os.environ.get("INV_ROOT_NS_STEPS", 8))
    # How often to update L_ema / R_ema from the gradient, in optimizer
    # steps. Default 1 = every step (current behavior). Setting to 2-5
    # drops the per-step matmul cost of update_preconditioner_from_grad
    # proportionally. Covariance estimates become slightly staler but
    # L/R are slowly-moving EMAs so small staleness is harmless.
    precond_update_k = int(os.environ.get("PRECOND_UPDATE_K", 1))
    soap_beta1 = float(os.environ.get("SOAP_BETA1", 0.95))
    soap_beta2 = float(os.environ.get("SOAP_BETA2", 0.95))
    soap_eps = float(os.environ.get("SOAP_EPS", 1e-8))
    soap_damping = float(os.environ.get("SOAP_DAMPING", 0.03))
    soap_beta_precond = float(os.environ.get("SOAP_BETA_PRECOND", 0.95))
    soap_refresh_k = int(os.environ.get("SOAP_REFRESH_K", 50))
    soap_refresh_adaptive = bool(int(os.environ.get("SOAP_REFRESH_ADAPTIVE", "1")))
    soap_refresh_drift_tau = float(os.environ.get("SOAP_REFRESH_DRIFT_TAU", 0.05))
    soap_precond_warmup_steps = int(os.environ.get("SOAP_PRECOND_WARMUP_STEPS", 500))
    ns_adaptive = bool(int(os.environ.get("NS_ADAPTIVE", "0")))
    ns_adaptive_eps = float(os.environ.get("NS_ADAPTIVE_EPS", 0.02))
    ns_two_phase = bool(int(os.environ.get("NS_TWO_PHASE", "0")))
    ns_refine_steps = int(os.environ.get("NS_REFINE_STEPS", 2))
    ns_warm_start = bool(int(os.environ.get("NS_WARM_START", "0")))
    capture_act = bool(int(os.environ.get("CAPTURE_ACT", "0")))
    lane_aware = bool(int(os.environ.get("LANE_AWARE", "1")))
    psgd_kron_lr = float(os.environ.get("PSGD_KRON_LR", 0.01))
    psgd_kron_precond_lr = float(os.environ.get("PSGD_KRON_PRECOND_LR", 0.1))
    diag_log_every = int(os.environ.get("DIAG_LOG_EVERY", 500))
    diag_dump_path = os.environ.get("DIAG_DUMP_PATH", "")
    diag_dump_every = int(os.environ.get("DIAG_DUMP_EVERY", 2000))
    precond_debug_layer = int(os.environ.get("PRECOND_DEBUG_LAYER", -1))
    # For optimizer ablations: short-circuit everything after the
    # diagnostic pre-quantization post-EMA val_bpb eval. Skips GPTQ
    # Hessian collection, quantization, compressed-artifact eval, and TTT.
    skip_post_train = bool(int(os.environ.get("SKIP_POST_TRAIN", "0")))
    # Pre-quant and post-quant diagnostic eval_val passes. Off by default
    # for submission runs (TTT eval is the load-bearing measurement). Turn
    # on for sanity-check / debugging — ~40s combined wall-clock.
    diag_evals_enabled = bool(int(os.environ.get("DIAG_EVALS_ENABLED", "0")))
    # Attention Output Gate (per-head, input-dependent, sigmoid*2, zero-init)
    # from PR #1693 @MarioPaerle. Small (num_heads * width params per layer)
    # gate applied between flash_attn output and out-projection.
    attn_out_gate_enabled = bool(int(os.environ.get("ATTN_OUT_GATE_ENABLED", "1")))
    attn_out_gate_width = int(os.environ.get("ATTN_OUT_GATE_WIDTH", 12))
    # Smear Gate: input-dependent per-channel mixer with the previous token.
    # Applied once in the residual stream (not per-layer). 13 params total
    # (width + 1 lambda). Zero-init lambda = identity at init.
    smear_gate_enabled = bool(int(os.environ.get("SMEAR_GATE_ENABLED", "1")))
    smear_gate_width = int(os.environ.get("SMEAR_GATE_WIDTH", 12))
    # Default OFF for the non-record base-model submission. TTT infrastructure
    # (phased global-SGD + per-doc LoRA) remains in the file for experiments;
    # set TTT_ENABLED=1 to run it.
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 96))
    # LoRA+ (asymmetric LR on A vs B matrices, Hayou et al. 2024) and
    # per-layer LR slope alpha (later virtual layers get higher LR). Ported
    # from PR #1695 @X-Abhishek-X. Defaults below are no-op (match prior
    # behavior); sweep alpha=0.25..1.0 and eta=2..16 to find optima.
    lora_plus_ratio = float(os.environ.get("LORA_PLUS_RATIO", 1.0))
    ttt_lora_layer_lr_alpha = float(os.environ.get("TTT_LORA_LAYER_LR_ALPHA", 0.0))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.0001))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 48))
    # Sequence length kept at 4096 (vs PR #1693's 2048) — project default.
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 4096))
    # 32 (vs PR #1693's 64): our ttt_eval_seq_len=4096 (vs PR's 2048) doubles
    # per-doc LoRA activation memory. Halved batch size keeps the compile
    # warmup's (ttt_batch_size * seq_len, 2048) bf16 intermediate at ~512 MiB,
    # fitting inside the ~800 MiB free headroom after quantization residues.
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 32))
    # 2 (vs PR #1693's 1): two LoRA SGD updates per chunk. Score is still
    # recorded exactly once per chunk (before any update), so Issue #1017
    # Track B condition 3 (score-before-update) is preserved. Expected TTT
    # lift improvement: -0.001 to -0.002 bpb from more aggressive LoRA fit.
    ttt_grad_steps = int(os.environ.get("TTT_GRAD_STEPS", 2))
    ttt_weight_decay = float(os.environ.get("TTT_WEIGHT_DECAY", 0.5))
    ttt_beta1 = float(os.environ.get("TTT_BETA1", 0))
    ttt_beta2 = float(os.environ.get("TTT_BETA2", 0.999))
    ttt_k_lora = bool(int(os.environ.get("TTT_K_LORA", "1")))
    ttt_mlp_lora = bool(int(os.environ.get("TTT_MLP_LORA", "1")))
    ttt_o_lora = bool(int(os.environ.get("TTT_O_LORA", "1")))
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "adam")
    ttt_eval_batches = os.environ.get("TTT_EVAL_BATCHES", "")
    # Multi-Phase Global SGD TTT interleaved with LoRA (PR #1693 protocol).
    # Runs over fixed token chunks, mid-LoRA phase trigger, cosine LR schedule.
    phased_ttt_enabled = bool(int(os.environ.get("PHASED_TTT_ENABLED", "1")))
    phased_ttt_prefix_docs = int(os.environ.get("PHASED_TTT_PREFIX_DOCS", 2000))
    # 3 (matches PR #1693's actual submission, not the file's literal default
    # of 1). Splits the 2000-doc global-SGD update into three incremental
    # passes with online re-scoring in between — avoids over-shooting from
    # one big backward step.
    phased_ttt_num_phases = int(os.environ.get("PHASED_TTT_NUM_PHASES", 3))
    global_ttt_lr = float(os.environ.get("GLOBAL_TTT_LR", 0.001))
    global_ttt_momentum = float(os.environ.get("GLOBAL_TTT_MOMENTUM", 0.9))
    # 2 (vs PR #1693's 1): two passes over already-scored tokens per phase.
    # Global-SGD only trains on tokens LoRA TTT has already scored, so no
    # scoring is re-done — Track B conditions preserved. Expected TTT lift
    # improvement: -0.001 to -0.003 bpb from more effective base-model
    # adaptation. Adds ~30-60s to eval time (still well inside 600s).
    global_ttt_epochs = int(os.environ.get("GLOBAL_TTT_EPOCHS", 2))
    global_ttt_chunk_tokens = int(os.environ.get("GLOBAL_TTT_CHUNK_TOKENS", 32768))
    # 32 (matches PR #1693). Earlier halves to 16 and 8 were misfires:
    # with world_size=8 and chunk_seqs=8, my_chunk_seqs=1 per rank, so
    # batch_seqs clamps to 1 regardless of the env value — this knob never
    # gated the OOM we saw. Real per-batch memory driver is TTT_BATCH_SIZE.
    global_ttt_batch_seqs = int(os.environ.get("GLOBAL_TTT_BATCH_SEQS", 32))
    global_ttt_warmup_start_lr = float(os.environ.get("GLOBAL_TTT_WARMUP_START_LR", 0.0))
    global_ttt_warmup_chunks = int(os.environ.get("GLOBAL_TTT_WARMUP_CHUNKS", 0))
    global_ttt_grad_clip = float(os.environ.get("GLOBAL_TTT_GRAD_CLIP", 1.0))
    global_ttt_respect_doc_boundaries = bool(int(os.environ.get("GLOBAL_TTT_RESPECT_DOC_BOUNDARIES", "1")))
    val_doc_fraction = float(os.environ.get("VAL_DOC_FRACTION", 1.0))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 16))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 4.0))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 7))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 15.0))
    mlp_clip_sigmas = float(os.environ.get("MLP_CLIP_SIGMAS", 12.0))
    attn_clip_sigmas = float(os.environ.get("ATTN_CLIP_SIGMAS", 13.0))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(data_dir, "datasets", f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    tokenizer_path = os.path.join(
        data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model"
    )
    artifact_dir = os.environ.get("ARTIFACT_DIR", "")
    eval_only_path = os.environ.get("EVAL_ONLY_PATH", "")
    logfile = (
        os.path.join(artifact_dir, f"{run_id}.txt")
        if artifact_dir
        else f"logs/{run_id}.txt"
    )
    model_path = (
        os.path.join(artifact_dir, "final_model.pt")
        if artifact_dir
        else "final_model.pt"
    )
    quantized_model_path = (
        os.path.join(artifact_dir, "final_model.int6.ptz")
        if artifact_dir
        else "final_model.int6.ptz"
    )


_logger_hparams = None


def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        (
            self.base_bytes_lut,
            self.has_leading_space_lut,
            self.is_boundary_token_lut,
        ) = build_sentencepiece_luts(self.sp, h.vocab_size, device)


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert (
        sp.piece_to_id("▁") != sp.unk_id()
    ), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 1024 if _TEST_MODE else 256 * np.dtype("<i4").itemsize

if _TEST_MODE:
    # Disable torch.compile under tests — requires a C++ compiler on Windows.
    def _identity_decorator(fn):
        return fn
    torch.compile = _identity_decorator
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


BOS_ID = None


def get_next_multiple_of_n(v, n):
    return ((v + n - 1) // n) * n


def _build_cu_seqlens(bos_pos, total_len, device, max_doc_len=0, bucket_size=64):
    if not bos_pos or bos_pos[0] != 0:
        bos_pos = [0] + bos_pos
    seg_starts = []
    starts_with_end = bos_pos + [total_len]
    for i in range(len(starts_with_end) - 1):
        start = starts_with_end[i]
        end = starts_with_end[i + 1]
        if max_doc_len > 0:
            pos = start
            while pos < end:
                seg_starts.append(pos)
                pos += max_doc_len
        else:
            seg_starts.append(start)
    boundaries = seg_starts + [total_len]
    padded_len = get_next_multiple_of_n(len(boundaries), bucket_size)
    cu = torch.full((padded_len,), total_len, dtype=torch.int32, device=device)
    cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    seg_ends = seg_starts[1:] + [total_len]
    max_seqlen = max(end - start for start, end in zip(seg_starts, seg_ends))
    return cu, max_seqlen

class DocumentPackingLoader:
    _shard_pool = ThreadPoolExecutor(1)

    def __init__(self, h, device, cu_bucket_size=64):
        self.rank = h.rank
        self.world_size = h.world_size
        self.device = device
        self.cu_bucket_size = cu_bucket_size
        self.max_seq_len = h.train_seq_len
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files
        self.file_iter = iter(self.files)
        self._init_shard(load_data_shard(next(self.file_iter)))
        self._next_shard = self._submit_next_shard()
        self._batch_pool = ThreadPoolExecutor(1)
        self._next_batch = None

    def _init_shard(self, tokens):
        global BOS_ID
        self.tokens = tokens
        self.shard_size = tokens.numel()
        if BOS_ID is None:
            BOS_ID = 1
        self.bos_idx = (
            (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        )
        if self.bos_idx.size == 0:
            self.bos_idx = np.array([0], dtype=np.int64)
        self.cursor = int(self.bos_idx[0])

    def _submit_next_shard(self):
        try:
            path = next(self.file_iter)
            return self._shard_pool.submit(load_data_shard, path)
        except StopIteration:
            return None

    def _advance_shard(self):
        if self._next_shard is None:
            self.file_iter = iter(self.files)
            self._next_shard = self._shard_pool.submit(
                load_data_shard, next(self.file_iter)
            )
        self._init_shard(self._next_shard.result())
        self._next_shard = self._submit_next_shard()

    def _local_doc_starts(self, local_start, total_len):
        lo = np.searchsorted(self.bos_idx, local_start, side="left")
        hi = np.searchsorted(self.bos_idx, local_start + total_len, side="left")
        return (self.bos_idx[lo:hi] - local_start).tolist()

    def _prepare_batch(self, num_tokens_local, max_seq_len):
        per_rank_span = num_tokens_local + 1
        global_span = per_rank_span * self.world_size
        while self.cursor + global_span > self.shard_size:
            self._advance_shard()
        local_start = self.cursor + self.rank * per_rank_span
        buf = self.tokens[local_start : local_start + per_rank_span]
        inputs = buf[:-1].to(dtype=torch.int64).pin_memory()
        targets = buf[1:].to(dtype=torch.int64).pin_memory()
        starts = self._local_doc_starts(local_start, inputs.numel())
        cu_seqlens, max_seqlen = _build_cu_seqlens(
            starts, inputs.numel(), inputs.device, max_seq_len, self.cu_bucket_size
        )
        cu_seqlens = cu_seqlens.pin_memory()
        self.cursor += global_span
        return inputs, targets, cu_seqlens, max_seqlen

    def next_batch(self, global_tokens, grad_accum_steps):
        num_tokens_local = global_tokens // (self.world_size * grad_accum_steps)
        if self._next_batch is not None:
            inputs, targets, cu_seqlens, max_seqlen = self._next_batch.result()
        else:
            inputs, targets, cu_seqlens, max_seqlen = self._prepare_batch(
                num_tokens_local, self.max_seq_len
            )
        self._next_batch = self._batch_pool.submit(
            self._prepare_batch, num_tokens_local, self.max_seq_len
        )
        return (
            inputs[None].to(self.device, non_blocking=True),
            targets[None].to(self.device, non_blocking=True),
            cu_seqlens.to(self.device, non_blocking=True),
            max_seqlen,
        )


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank :: h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(
            self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1)
        )
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array(
                    [len(s) for s in self.start_inds], dtype=np.float64
                )
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(
                np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64)
            )
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


@triton.jit
def linear_xielu_kernel(
    a_desc,
    b_desc,
    c_desc,
    aux_desc,
    M,
    N,
    K,
    ap, an, bp, bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FORWARD: tl.constexpr,
):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        offs_am_c = offs_am
        offs_bn_c = offs_bn
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        c1 = acc1.to(dtype)
        if not FORWARD:
            # backward: multiply grad by d/dh[xIELU(h)]
            pre0 = aux_desc.load([offs_am_c, offs_bn_c])
            pre1 = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c0 = c0 * tl.where(pre0 > 0, 2.0 * ap * pre0 + bp, 2.0 * an * pre0 + bn)
            c1 = c1 * tl.where(pre1 > 0, 2.0 * ap * pre1 + bp, 2.0 * an * pre1 + bn)
        # forward: c_desc = raw matmul (pre); backward: c_desc = d_activation * grad
        c_desc.store([offs_am_c, offs_bn_c], c0)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        if FORWARD:
            # xIELU: h * where(h > 0, ap*h + bp, an*h + bn)
            aux0 = c0 * tl.where(c0 > 0, ap * c0 + bp, an * c0 + bn)
            aux1 = c1 * tl.where(c1 > 0, ap * c1 + bp, an * c1 + bn)
            aux_desc.store([offs_am_c, offs_bn_c], aux0)
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], aux1)


def linear_xielu(a, b, ap, an, bp, bn, aux=None):
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    forward = aux is None
    if aux is None:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    num_stages = 4 if forward else 3
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    grid = lambda _meta: (
        min(num_sms, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),
    )
    linear_xielu_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        aux_desc,
        M,
        N,
        K,
        ap, an, bp, bn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=num_sms,
        FORWARD=forward,
        num_stages=num_stages,
        num_warps=8,
    )
    if forward:
        return c, aux
    return c


class FusedXieluMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, ap, an, bp, bn):
        x_flat = x.reshape(-1, x.shape[-1])
        pre, post = linear_xielu(x_flat, w1, ap, an, bp, bn)
        out = F.linear(post, w2)
        ctx.save_for_backward(x, w1, w2, pre, post)
        ctx.ap = ap
        ctx.an = an
        ctx.bp = bp
        ctx.bn = bn
        return out.view(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, w2, pre, post = ctx.saved_tensors
        x_flat = x.reshape(-1, x.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        dw2 = grad_output_flat.T @ post
        dpre = linear_xielu(grad_output_flat, w2.T.contiguous(), ctx.ap, ctx.an, ctx.bp, ctx.bn, aux=pre)
        dw1 = dpre.T @ x_flat
        dx = dpre @ w1
        return dx.view_as(x), dw1, dw2, None, None, None, None


FusedXieluMLP = FusedXieluMLPFunction.apply


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0, yarn=True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.yarn = yarn
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if self.yarn and seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd
                )
            else:
                inv_freq = self.inv_freq.float().to(device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :seq_len].to(dtype=dtype), self._sin_cached[:, :seq_len].to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


def _apply_attn_out_gate(y, x_orig, gate_w):
    """Attention Output Gate (PR #1693 @MarioPaerle).

    Per-head multiplicative gate on the flash-attn output BEFORE the
    output projection. Input-dependent: gate value computed from first
    `gate_w.shape[-1]` channels of the residual input (x_orig).

    Shapes:
      y:       (B, T, num_heads, head_dim) — flash-attn output, pre-reshape
      x_orig:  (B, T, dim)                — pre-norm residual input
      gate_w:  (num_heads, width)         — learned, zero-init

    Output: y * (2·sigmoid(F.linear(x_orig[:, :, :width], gate_w))).
    Zero-init → 2·sigmoid(0) = 1.0 → pass-through at init.
    """
    width = gate_w.shape[-1]
    gate_in = x_orig[:, :, :width]
    gate = 2.0 * torch.sigmoid(F.linear(gate_in, gate_w.to(gate_in.dtype)))
    return y * gate.unsqueeze(-1).to(y.dtype)


def _apply_smear_gate(x, smear_w, smear_lambda):
    """Smear Gate (PR #1693, from PR #1610 concept).

    Input-dependent per-channel mixer that blends current token `x` with
    previous token `x[t-1]` using a gated scalar lambda. 13 params total
    (width weights + 1 lambda). Strictly causal backward mixing.

    Shapes:
      x:             (B, T, dim)
      smear_w:       (width,)  — gate weights
      smear_lambda:  ()        — scalar lambda, zero-init (= identity)

    Output: x + λ · sigmoid(F.linear(x[:, :, :width], smear_w)) · x[t-1]
    """
    width = smear_w.shape[0]
    # Right-shift along the T axis with zero-fill at t=0: equivalent to
    # prev_x[:, 1:] = x[:, :-1], but one allocation instead of two.
    prev_x = F.pad(x[:, :-1], (0, 0, 1, 0))
    gate_in = x[:, :, :width]
    gate = torch.sigmoid(F.linear(gate_in, smear_w.to(x.dtype).unsqueeze(0)))
    return x + smear_lambda.to(x.dtype) * gate * prev_x


class CausalSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
        yarn=True, attn_out_gate_width=0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len, yarn=yarn)
        self.use_xsa = False
        self.window_size = (-1, -1)  # default: full attention; overridden per-layer
        # Per-head Attention Output Gate (PR #1693). Zero-init → identity at init.
        # Set to None when disabled; width=0 also disables.
        if attn_out_gate_width > 0:
            self.attn_out_gate_w = nn.Parameter(
                torch.zeros(num_heads, attn_out_gate_width, dtype=torch.float32)
            )
        else:
            self.attn_out_gate_w = None

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, qkv_w, out_w, q_dim, kv_dim, cu_seqlens=None, max_seqlen=0,
                x_orig=None):
        bsz, seqlen, dim = x.shape
        if getattr(self, "_capture_act", False) and self.training:
            # Capture attention-input covariance (R_act for q/k/v banks).
            # Cap captured batch to avoid OOM on long seqs.
            x_flat = x.detach().float().reshape(-1, x.shape[-1])
            max_rows = 4096
            if x_flat.shape[0] > max_rows:
                x_flat = x_flat[:max_rows]
            self._captured_attn_input_cov = x_flat.mT @ x_flat / x_flat.shape[0]
        qkv = F.linear(x, qkv_w.to(x.dtype))
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if cu_seqlens is not None:
            y = flash_attn_varlen_func(
                q[0],
                k[0],
                v[0],
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=self.window_size,
            )[None]
        else:
            y = flash_attn_3_func(q, k, v, causal=True, window_size=self.window_size)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        # Apply per-head Attention Output Gate BEFORE reshape + out-projection.
        # Requires the pre-norm residual input `x_orig`; if not passed, skip.
        if self.attn_out_gate_w is not None and x_orig is not None:
            y = _apply_attn_out_gate(y, x_orig, self.attn_out_gate_w)
        y = y.reshape(bsz, seqlen, dim)
        self._last_proj_input = y.detach() if getattr(self, "_calib", False) else None
        if getattr(self, "_capture_act", False) and self.training:
            y_flat = y.detach().float().reshape(-1, y.shape[-1])
            max_rows = 4096
            if y_flat.shape[0] > max_rows:
                y_flat = y_flat[:max_rows]
            self._captured_out_input_cov = y_flat.mT @ y_flat / y_flat.shape[0]
        return F.linear(y, out_w.to(x.dtype))


# Per-layer xIELU coefficients discovered by convergence loop (Run 2, 8xH100, seed 1337).
# Activation: torch.where(x > 0, ap*x² + bp*x, an*x² + bn*x)
QK_GAIN_INIT_PER_LAYER = [2.3495, 2.8818, 2.7627, 2.8148, 2.7893, 2.8762, 2.5657, 2.7206, 2.6426, 2.2737, 1.9741]

XIELU_AP = [0.103, 0.196, 1.415, 1.196, 1.485, 1.546, 1.337, 1.727, 1.495, 0.988, 0.917]
XIELU_AN = [0.39, 0.578, 0.363, 0.491, 0.536, 0.548, 0.579, 0.983, 1.058, 0.935, 0.845]
XIELU_BP = [0.126, 0.07, 0.0, 0.0, 0.0, 0.002, 0.017, 0.067, 0.005, 0.058, 0.568]
XIELU_BN = [0.785, 0.638, 0.405, 0.377, 0.314, 0.289, 0.313, 0.571, 0.42, 0.286, 0.52]


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, layer_idx=0):
        super().__init__()
        self.use_fused = True
        idx = min(layer_idx, len(XIELU_AP) - 1)
        self.ap = XIELU_AP[idx]
        self.an = XIELU_AN[idx]
        self.bp = XIELU_BP[idx]
        self.bn = XIELU_BN[idx]

    def forward(self, x, up_w, down_w):
        capture = getattr(self, "_capture_act", False)
        if capture and self.training:
            x_flat = x.detach().float().reshape(-1, x.shape[-1])
            max_rows = 4096
            if x_flat.shape[0] > max_rows:
                x_flat = x_flat[:max_rows]
            self._captured_mlp_input_cov = x_flat.mT @ x_flat / x_flat.shape[0]
        # Fused path is bypassed when capturing so the post-xIELU hidden
        # tensor is visible for mlp_down covariance (unfused computes it
        # inline). Costs a few % wall-clock during capture; acceptable.
        if self.training and self.use_fused and not capture:
            return FusedXieluMLP(x, up_w.to(x.dtype), down_w.to(x.dtype),
                                 self.ap, self.an, self.bp, self.bn)
        h = F.linear(x, up_w.to(x.dtype))
        hidden = h * torch.where(h > 0, self.ap * h + self.bp,
                                        self.an * h + self.bn)
        self._last_down_input = hidden.detach() if getattr(self, "_calib", False) else None
        if capture:
            h_flat = hidden.detach().float().reshape(-1, hidden.shape[-1])
            max_rows = 4096
            if h_flat.shape[0] > max_rows:
                h_flat = h_flat[:max_rows]
            self._captured_down_input_cov = h_flat.mT @ h_flat / h_flat.shape[0]
        return F.linear(hidden, down_w.to(x.dtype))


def log_qk_gain_converged(log0, model):
    """Print per-layer q_gain mean values for convergence tracking."""
    qk_vals = []
    for i, block in enumerate(model.blocks):
        v = block.attn.q_gain.detach().cpu().mean().item()
        log0(f"qk_gain:layer {i}: mean={v:.4f}")
        qk_vals.append(round(v, 4))
    qk_str = ", ".join(f"{v}" for v in qk_vals)
    log0(f"QK_GAIN_INIT_PER_LAYER = [{qk_str}]")


def log_parcae_converged(log0, model):
    """Print Parcae loop injection parameters for convergence tracking."""
    if model.loop_log_A is None:
        return
    delta = F.softplus(model.loop_delta.float())
    A_bar = torch.exp(delta * (-torch.exp(model.loop_log_A.float())))
    B_bar = delta * model.loop_B.float()
    log0(f"parcae:A_bar mean={A_bar.mean().item():.4f} min={A_bar.min().item():.4f} max={A_bar.max().item():.4f}")
    log0(f"parcae:B_bar mean={B_bar.mean().item():.4f} min={B_bar.min().item():.4f} max={B_bar.max().item():.4f}")
    log0(f"parcae:log_A mean={model.loop_log_A.mean().item():.4f} delta mean={model.loop_delta.mean().item():.4f}")


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        train_seq_len,
        layer_idx=0,
        ln_scale=False,
        yarn=True,
        attn_out_gate_width=0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
            yarn=yarn, attn_out_gate_width=attn_out_gate_width,
        )
        self.mlp = MLP(dim, mlp_mult, layer_idx=layer_idx)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0, qkv_w, out_w, up_w, down_w, q_dim, kv_dim, cu_seqlens=None, max_seqlen=0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            qkv_w, out_w, q_dim, kv_dim,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            x_orig=x_in,  # pre-norm residual input for AttnOutGate
        )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[
            None, None, :
        ] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out

class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_layers = h.num_layers
        # Runtime-settable bypass for Parcae boundary. Default False (Parcae
        # applied whenever loop_log_A is present and looping_active is True).
        # The eval/TTT driver flips this to True on the eval and TTT models
        # when PARCAE_EVAL_BYPASS=1, so training still applies Parcae but
        # eval-time forwards skip _parcae_boundary.
        self.parcae_eval_bypass = False
        head_dim = h.model_dim // h.num_heads
        kv_dim = h.num_kv_heads * head_dim
        self.q_dim = h.model_dim
        self.kv_dim = kv_dim
        hidden_dim = int(h.mlp_mult * h.model_dim)
        # Gemma-style: classify layers as global (full attention, K=V tied, partial RoPE)
        # or local (sliding window, separate K/V, full RoPE)
        self.global_layer_set = set(h.global_attn_layers)
        self.kv_tie_global = h.kv_tie_global
        local_layers = [i for i in range(h.num_layers) if i not in self.global_layer_set]
        self.num_local_layers = len(local_layers)
        # Map layer index -> v_bank index (only local layers have separate V)
        self.local_v_idx = {}
        for vi, li in enumerate(local_layers):
            self.local_v_idx[li] = vi
        self.q_bank = nn.Parameter(torch.empty(h.num_layers, h.model_dim, h.model_dim))
        self.k_bank = nn.Parameter(torch.empty(h.num_layers, kv_dim, h.model_dim))
        # V bank: only local layers get separate V weights; global layers reuse K
        n_v = self.num_local_layers if h.kv_tie_global else h.num_layers
        self.v_bank = nn.Parameter(torch.empty(n_v, kv_dim, h.model_dim))
        self.out_bank = nn.Parameter(torch.empty(h.num_layers, h.model_dim, h.model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(h.num_layers, hidden_dim, h.model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(h.num_layers, h.model_dim, hidden_dim))
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    QK_GAIN_INIT_PER_LAYER[i] if QK_GAIN_INIT_PER_LAYER is not None else h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                    yarn=h.rope_yarn,
                    attn_out_gate_width=(
                        h.attn_out_gate_width if h.attn_out_gate_enabled else 0
                    ),
                )
                for i in range(h.num_layers)
            ]
        )
        # Gemma-style RoPE split: partial RoPE on global layers, full RoPE on local layers
        head_dim = h.model_dim // h.num_heads
        for i, block in enumerate(self.blocks):
            if i in self.global_layer_set:
                # Global: partial RoPE (h.rope_dims out of head_dim) — avoid high-freq noise at long range
                rd = h.rope_dims if h.rope_dims > 0 else 0
                block.attn.rope_dims = rd
                block.attn.rotary = Rotary(
                    head_dim, base=h.rope_base, train_seq_len=h.train_seq_len,
                    rope_dims=rd, yarn=h.rope_yarn,
                )
                # Global: full causal attention
                block.attn.window_size = (-1, -1)
            else:
                # Local: full RoPE (all dims) — positional precision within window
                block.attn.rope_dims = 0
                block.attn.rotary = Rotary(
                    head_dim, base=h.rope_base, train_seq_len=h.train_seq_len,
                    rope_dims=0, yarn=h.rope_yarn,
                )
                # Local: sliding window attention. Early layers get a tighter
                # window than deeper layers (split at h.local_window_split_layer
                # or num_layers // 2 if -1). Each block's window_size is set
                # once here and treated as a compile-time constant by dynamo.
                _split = (
                    h.local_window_split_layer
                    if h.local_window_split_layer >= 0
                    else h.num_layers // 2
                )
                _w = h.local_window_size_early if i < _split else h.local_window_size_late
                block.attn.window_size = (_w, 0)
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if h.tie_embeddings
            else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        self.looping_active = False
        self.loop_start = h.loop_start
        self.loop_trunc_bwd = h.loop_trunc_bwd
        if h.num_loops > 0:
            # Parcae-style constrained loop injection parameters
            self.loop_log_A = nn.Parameter(torch.full((h.model_dim,), -0.5, dtype=torch.float32))
            self.loop_delta = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
            self.loop_B = nn.Parameter(torch.full((h.model_dim,), 0.1, dtype=torch.float32))
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.loop_log_A = None
            self.loop_delta = None
            self.loop_B = None
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
            )
            if h.skip_gates_enabled
            else None
        )
        self.parallel_start_layer = h.parallel_start_layer
        self.parallel_final_lane = h.parallel_final_lane.lower()
        self.parallel_post_lambdas = nn.Parameter(
            torch.ones(h.num_layers, 2, 2, dtype=torch.float32)
        )
        self.parallel_resid_lambdas = nn.Parameter(
            torch.full((h.num_layers, 2), 1.1, dtype=torch.float32)
        )
        # SmearGate: 13-param input-dependent mixer with the previous token.
        # Applied once in forward_logits after embed_proj. Zero-init lambda
        # means identity at init — pure residual stream unchanged.
        if h.smear_gate_enabled:
            self.smear_w = nn.Parameter(
                torch.zeros(h.smear_gate_width, dtype=torch.float32)
            )
            self.smear_lambda = nn.Parameter(
                torch.zeros((), dtype=torch.float32)
            )
        else:
            self.smear_w = None
            self.smear_lambda = None
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.q_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.k_bank.data[i], gain=1.0)
            # V bank: only init for local layers (global layers reuse K)
            if self.kv_tie_global:
                if i in self.local_v_idx:
                    nn.init.orthogonal_(self.v_bank.data[self.local_v_idx[i]], gain=1.0)
            else:
                nn.init.orthogonal_(self.v_bank.data[i], gain=1.0)
            nn.init.zeros_(self.out_bank.data[i])
            self.out_bank.data[i].mul_(proj_scale)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def set_eval_loop_indices(self, num_loops, loop_start, loop_end, num_layers):
        """Rebuild encoder/decoder indices for a different loop count (eval-time only).

        Skip weights/gates keep their trained size — extra decoder steps
        beyond num_skip_weights simply run without skip connections.
        """
        loop_seg = list(range(loop_start, loop_end + 1))
        all_indices = list(range(loop_start))
        for _ in range(num_loops + 1):
            all_indices.extend(loop_seg)
        all_indices.extend(range(loop_end + 1, num_layers))
        num_enc = len(all_indices) // 2
        self.encoder_indices = all_indices[:num_enc]
        self.decoder_indices = all_indices[num_enc:]
        self.looping_active = True

    def _bank_weights(self, i):
        if self.kv_tie_global and i in self.global_layer_set:
            # K=V tying: use k_bank for both K and V
            v_w = self.k_bank[i]
        else:
            v_w = self.v_bank[self.local_v_idx[i] if self.kv_tie_global else i]
        qkv_w = torch.cat([self.q_bank[i], self.k_bank[i], v_w], dim=0)
        return (
            qkv_w,
            self.out_bank[i],
            self.mlp_up_bank[i],
            self.mlp_down_bank[i],
        )

    def _parallel_block(
        self, block_idx, lane0, lane1, x0,
        qkv_w, out_w, up_w, down_w,
        cu_seqlens=None, max_seqlen=0,
    ):
        block = self.blocks[block_idx]
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        attn_out = block.attn(
            block.attn_norm(attn_read) * block.ln_scale_factor,
            qkv_w, out_w, self.q_dim, self.kv_dim,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            x_orig=attn_read,  # pre-norm residual input for AttnOutGate
        )
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        mlp_read = mix[0][None, None, :] * lane1 + mix[1][None, None, :] * x0
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * block.mlp(
            block.mlp_norm(mlp_read) * block.ln_scale_factor, up_w, down_w
        )
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1

    def _parcae_bars(self):
        """Compute Parcae decay (A_bar) and injection (B_bar) from learned params.
        A_bar is guaranteed in (0, 1) by construction (softplus enforces delta > 0)."""
        delta = F.softplus(self.loop_delta.float())
        A_bar = torch.exp(delta * (-torch.exp(self.loop_log_A.float())))
        B_bar = delta * self.loop_B.float()
        return A_bar, B_bar

    def _parcae_boundary(self, x, x0, A_bar, B_bar, loop_count):
        """Apply Parcae injection at loop re-entry.
        Assumes parallel lanes are NOT active (loop range must be below parallel_start_layer)."""
        if self.loop_trunc_bwd > 0 and loop_count <= self.loop_trunc_bwd + 1:
            x = x.detach()
        x = A_bar[None, None, :].to(x.dtype) * x + B_bar[None, None, :].to(x.dtype) * x0
        return x

    def _final_parallel_hidden(self, lane0, lane1):
        if self.parallel_final_lane == "mlp":
            return lane1
        if self.parallel_final_lane == "attn":
            return lane0
        return 0.5 * (lane0 + lane1)

    def forward_logits(self, input_ids, cu_seqlens=None, max_seqlen=0):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        # SmearGate — mix current token with previous token once before the
        # layer stack. Zero-init λ → identity at init. Strictly causal.
        if self.smear_w is not None:
            x = _apply_smear_gate(x, self.smear_w, self.smear_lambda)
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else range(self.num_encoder_layers)
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else range(
                self.num_encoder_layers,
                self.num_encoder_layers + self.num_decoder_layers,
            )
        )
        loop_count = 0
        has_parcae = (
            self.loop_log_A is not None
            and self.looping_active
            and not self.parcae_eval_bypass
        )
        if has_parcae:
            A_bar, B_bar = self._parcae_bars()
        for i in enc_iter:
            if i == self.loop_start:
                loop_count += 1
                if loop_count > 1 and has_parcae:
                    x = self._parcae_boundary(x, x0, A_bar, B_bar, loop_count)
            qkv_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self.blocks[i](x, x0, qkv_w, out_w, up_w, down_w, self.q_dim, self.kv_dim, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            if i == self.loop_start:
                loop_count += 1
                if has_parcae:
                    x = self._parcae_boundary(x, x0, A_bar, B_bar, loop_count)
            qkv_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1 = self._parallel_block(
                    i, lane0, lane1, x0, qkv_w, out_w, up_w, down_w,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self.blocks[i](x, x0, qkv_w, out_w, up_w, down_w, self.q_dim, self.kv_dim, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids, cu_seqlens=None, max_seqlen=0):
        logits = self.forward_logits(
            input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )

    def forward_ttt(self, input_ids, target_ids, lora):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        # SmearGate (same application as forward_logits).
        if self.smear_w is not None:
            x = _apply_smear_gate(x, self.smear_w, self.smear_lambda)
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else list(range(self.num_encoder_layers))
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else list(
                range(
                    self.num_encoder_layers,
                    self.num_encoder_layers + self.num_decoder_layers,
                )
            )
        )
        slot = 0
        loop_count = 0
        has_parcae = (
            self.loop_log_A is not None
            and self.looping_active
            and not self.parcae_eval_bypass
        )
        if has_parcae:
            A_bar, B_bar = self._parcae_bars()
        for i in enc_iter:
            if i == self.loop_start:
                loop_count += 1
                if loop_count > 1 and has_parcae:
                    x = self._parcae_boundary(x, x0, A_bar, B_bar, loop_count)
            qkv_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, qkv_w, out_w, up_w, down_w)
            slot += 1
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            if i == self.loop_start:
                loop_count += 1
                if has_parcae:
                    x = self._parcae_boundary(x, x0, A_bar, B_bar, loop_count)
            qkv_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1 = self._parallel_block_with_lora(
                    i, lane0, lane1, x0, lora, slot,
                    qkv_w, out_w, up_w, down_w,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, qkv_w, out_w, up_w, down_w)
            slot += 1
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + lora.lm_head_lora(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        bsz, sl, V = logits.shape
        return F.cross_entropy(
            logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
        ).reshape(bsz, sl)

    def _block_with_lora(self, block, x, x0, lora, slot, qkv_w, out_w, up_w, down_w):
        mix = block.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = block.attn_norm(x_in) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        qkv = F.linear(n, qkv_w.to(n.dtype))
        q_raw, k_raw, v_raw = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = (q_raw + lora.q_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_heads, attn.head_dim
        )
        k = k_raw
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (v_raw + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True, window_size=attn.window_size)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # AttnOutGate (same as non-LoRA path). Applied before reshape/out-proj.
        if attn.attn_out_gate_w is not None:
            y = _apply_attn_out_gate(y, x_in, attn.attn_out_gate_w)
        y = y.reshape(bsz, seqlen, dim)
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        x_out = x_out + block.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        return x_out

    def _parallel_block_with_lora(
        self, block_idx, lane0, lane1, x0, lora, slot,
        qkv_w, out_w, up_w, down_w,
    ):
        block = self.blocks[block_idx]
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        n = block.attn_norm(attn_read) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        qkv = F.linear(n, qkv_w.to(n.dtype))
        q_raw, k_raw, v_raw = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = (q_raw + lora.q_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_heads, attn.head_dim
        )
        k = k_raw
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (v_raw + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True, window_size=attn.window_size)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # AttnOutGate in parallel LoRA path (same as non-LoRA parallel path).
        if attn.attn_out_gate_w is not None:
            y = _apply_attn_out_gate(y, attn_read, attn.attn_out_gate_w)
        y = y.reshape(bsz, seqlen, dim)
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        mlp_read = mix[0][None, None, :] * lane1 + mix[1][None, None, :] * x0
        mlp_n = block.mlp_norm(mlp_read) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1


class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        self.A = nn.Parameter(
            torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound)
        )
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    def reset(self):
        with torch.no_grad():
            self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x):
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz, model, rank, k_lora=True, mlp_lora=True, o_lora=True):
        super().__init__()
        self.bsz = bsz
        dim = model.q_bank.shape[-1]
        vocab = model.tok_emb.num_embeddings
        if getattr(model, "looping_active", False):
            num_slots = len(model.encoder_indices) + len(model.decoder_indices)
        else:
            num_slots = len(model.blocks)
        kv_dim = model.blocks[0].attn.num_kv_heads * (
            dim // model.blocks[0].attn.num_heads
        )
        embed_dim = model.tok_emb.embedding_dim
        self.lm_head_lora = BatchedLinearLoRA(bsz, embed_dim, vocab, rank)
        self.q_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
        )
        self.v_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
        )
        self.k_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
            )
            if k_lora
            else None
        )
        self.mlp_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if mlp_lora
            else None
        )
        self.o_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if o_lora
            else None
        )

    def reset(self):
        with torch.no_grad():
            self.lm_head_lora.reset()
            for loras in [self.q_loras, self.v_loras, self.k_loras,
                          self.mlp_loras, self.o_loras]:
                if loras is not None:
                    for lora in loras:
                        lora.reset()


def classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = 3.4445, -4.775, 2.0315
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


@torch.compile
def gram_newton_schulz5(G, steps=5, eps=1e-7):
    """Gram NS for high-aspect-ratio matrices. Iterates on n×n Gram matrix in float32."""
    a, b, c = 3.4445, -4.7750, 2.0315
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    # X is (batch, n, m) with n <= m
    n = X.size(-2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    eye = torch.eye(n, device=X.device, dtype=X.dtype).unsqueeze(0)
    R = X @ X.mT           # (batch, n, n) — Gram matrix
    Q = eye.expand_as(R).contiguous()
    for i in range(steps):
        if i == 2:          # restart — kill spurious negative eigenvalues
            X = Q @ X       # reconstruct (batch, n, m)
            R = X @ X.mT    # reinitialize Gram
            Q = eye.expand_as(R).contiguous()
        R2 = R @ R
        Z = a * eye + b * R + c * R2
        Q = Q @ Z
        R = Z @ R @ Z      # Z symmetric (polynomial of symmetric R)
    X = (Q @ X).bfloat16()   # final rectangular multiply, cast to match standard NS output
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


NS_POLY_AGGRESSIVE = (3.4445, -4.7750, 2.0315)
NS_POLY_REFINEMENT = (1.5, -0.5, 0.0)

# torch.compile toggle for SOAP/preconditioner paths. Default on.
# When on, preconditioner hot-path functions are @torch.compile'd so they
# get fused into dense kernels by Inductor. Compile cost is amortized
# during the warmup phase (which calls step_fn, which exercises every
# compiled function at each bank shape). `_TEST_MODE` keeps it off for
# CPU tests (torch.compile needs a C++ compiler on Windows).
_COMPILE_SOAP = bool(int(os.environ.get("COMPILE_SOAP", "1"))) and not _TEST_MODE


def _maybe_compile(fn):
    """Conditionally @torch.compile a function based on _COMPILE_SOAP."""
    return torch.compile(fn) if _COMPILE_SOAP else fn


@_maybe_compile
def newtonschulz_two_phase(G, aggressive_steps=2, refine_steps=2, eps=1e-7):
    """Two-phase NS: aggressive poly then refinement (Newton) poly.

    Aggressive poly pulls singular values toward [0.9, 1.1] fast; refinement
    poly (1.5·X - 0.5·X·X·X^T) has quadratic convergence near sigma=1.
    """
    a0, b0, c0 = NS_POLY_AGGRESSIVE
    a1, b1, c1 = NS_POLY_REFINEMENT
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(aggressive_steps):
        A = X @ X.mT
        X = a0 * X + (b0 * A + c0 * (A @ A)) @ X
    for _ in range(refine_steps):
        A = X @ X.mT
        X = a1 * X + (b1 * A + c1 * (A @ A)) @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


def newtonschulz_adaptive(G, max_steps=5, eps=1e-7, ortho_eps=0.02, min_steps=2):
    """NS with early exit when ‖XX^T - I‖_F / sqrt(n) < ortho_eps.

    Unlike fixed-step variants this is not torch.compile-able (host sync on
    the stop criterion). Use when preconditioning makes input already near
    orthogonal so we save 1-2 iterations.
    Returns (X, actual_steps). `actual_steps` is a python int on host.
    """
    a, b, c = NS_POLY_AGGRESSIVE
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    n = X.size(-2)
    eye = torch.eye(n, device=X.device, dtype=X.dtype)
    actual = 0
    for step in range(max_steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
        actual = step + 1
        if actual >= min_steps:
            A_check = X @ X.mT
            resid = (A_check - eye).norm() / (n ** 0.5)
            if resid.item() < ortho_eps:
                break
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X, actual


def matrix_inv_sqrt_eigh(M, damping=0.03, min_eig_ratio=1e-4):
    """Exact M^{-1/2} via eigh; M is (..., d, d) symmetric PSD.

    Damps by adding damping · (tr(M)/d) · I, then clamps eigenvalues below
    min_eig_ratio · max_eig for stability.
    """
    d = M.size(-1)
    tr_mean = M.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(d, device=M.device, dtype=M.dtype).expand_as(M)
    M_d = M + damping * tr_mean * eye
    eigvals, eigvecs = torch.linalg.eigh(M_d)
    max_eig = eigvals.max(dim=-1, keepdim=True).values
    clamped = eigvals.clamp_min(min_eig_ratio * max_eig)
    inv_sqrt_eigvals = clamped.rsqrt()
    return (eigvecs * inv_sqrt_eigvals.unsqueeze(-2)) @ eigvecs.mT


@_maybe_compile
def matrix_inv_sqrt_ns(M, steps=5, damping=0.03, eps=1e-7):
    """Approximate M^{-1/2} via NS iteration X_{k+1} = 0.5·X_k·(3I − M·X_k²).

    Converges quadratically when eigenvalues of M·X_0² are in (0, 3). We
    initialize X_0 = α·I with α = 1/√(‖M‖_F · (1+damping)). Frobenius norm
    upper-bounds λ_max (actually ‖M‖_F ≥ λ_max for any PSD M), so
    α² · λ_max ≤ 1/(1+damping) < 1 regardless of condition number.
    Trace-based scaling (original attempt) diverges for κ > ~3.
    """
    d = M.size(-1)
    frob = M.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    tr_mean = M.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(d, device=M.device, dtype=M.dtype).expand_as(M)
    M_d = M + damping * tr_mean * eye
    # Add the damping contribution to the Frobenius scale estimate.
    frob_d = frob + damping * tr_mean * (d ** 0.5)
    alpha = frob_d.rsqrt()
    X = alpha * eye
    for _ in range(steps):
        T = M_d @ (X @ X)
        X = 0.5 * X @ (3.0 * eye - T)
    return X


def compute_scale(out_dim, in_dim, mode, d_ref=None):
    """Per-matrix LR scale.

    ratio_clamped: max(1, out/in)^0.5  — legacy, asymmetric on wide matrices.
    ratio:         (out/in)^0.5        — symmetric, principled for pure Muon.
    fan_out:       out^0.5 / d_ref^0.5 — for Newton-Muon / two-sided (fan_in
                   handled by R^{-1}).
    moonlight:     0.2 * max(out, in)^0.5 — Moonlight / modular-norms scaling.
    soap:          1.0                  — SOAP updates are Adam-magnitude.
    """
    if mode == "ratio_clamped":
        return max(1.0, out_dim / in_dim) ** 0.5
    if mode == "ratio":
        return (out_dim / in_dim) ** 0.5
    if mode == "fan_out":
        ref = d_ref if d_ref is not None else in_dim
        return (out_dim / ref) ** 0.5
    if mode == "moonlight":
        return 0.2 * max(out_dim, in_dim) ** 0.5
    if mode == "soap":
        return 1.0
    raise ValueError(f"unknown scale mode {mode!r}")


@_maybe_compile
def apply_cautious_mask(update, g, eps=1.0):
    """Zero out update coords where it disagrees in sign with raw grad, then
    rescale to preserve expected magnitude (Liu et al. 2024)."""
    mask = (update.sign() == g.sign()).to(update.dtype)
    denom = mask.sum().clamp_min(eps)
    scale = mask.numel() / denom
    return update * mask * scale


class PreconditionerState:
    """Per-bank preconditioner state for sharded matrix params.

    Holds L_ema, R_ema, Q_L, Q_R, m_rot, v_rot (all optimizer modes that use
    two-sided preconditioning), plus PSGD-Kron triangular factors and scalar
    bookkeeping (step counter, drift tracking, adaptive-NS counter).

    Lives in `_bank_meta[i]["precond"]`. Each entry covers `shard_B` layers'
    worth of state — i.e., the layers this rank owns post-reduce-scatter.
    """

    __slots__ = (
        "shard_B", "out_dim", "in_dim", "device", "dtype",
        "L_ema", "R_ema", "Q_L", "Q_R",
        "Q_L_bf16", "Q_R_bf16",                 # cast-once-per-refresh copies
        "L_inv_sqrt", "R_inv_sqrt",
        "L_inv_sqrt_bf16", "R_inv_sqrt_bf16",   # cast-once-per-refresh copies
        "m_rot", "v_rot",
        "psgd_Ql", "psgd_Qr",
        "prev_L_norm", "prev_R_norm",
        "step", "step_since_refresh", "refresh_ct",
        "last_ns_steps", "last_drift_L", "last_drift_R",
        "initialized",
    )

    def __init__(self, shard_B, out_dim, in_dim, device):
        self.shard_B = shard_B
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.device = device
        self.dtype = torch.float32
        d = dict(device=device, dtype=self.dtype)
        # Lazy init: allocated on first use; initialized marker gates allocation.
        self.L_ema = None
        self.R_ema = None
        self.Q_L = None
        self.Q_R = None
        self.L_inv_sqrt = None  # cached M^{-1/2}, recomputed on refresh
        self.R_inv_sqrt = None
        # bf16 copies of the hot-path matrices; cast once per refresh so the
        # per-step rotation matmuls run in bf16 (avoids allocating a fp32
        # copy of g every step, and enables fused bf16 tensor-core matmul).
        self.Q_L_bf16 = None
        self.Q_R_bf16 = None
        self.L_inv_sqrt_bf16 = None
        self.R_inv_sqrt_bf16 = None
        self.m_rot = None
        self.v_rot = None
        self.psgd_Ql = None
        self.psgd_Qr = None
        self.prev_L_norm = torch.zeros(shard_B, device=device, dtype=torch.float32)
        self.prev_R_norm = torch.zeros(shard_B, device=device, dtype=torch.float32)
        self.step = 0
        self.step_since_refresh = 0
        self.refresh_ct = 0
        self.last_ns_steps = 0
        self.last_drift_L = 0.0
        self.last_drift_R = 0.0
        self.initialized = False

    def _alloc_eigen(self):
        d = dict(device=self.device, dtype=self.dtype)
        self.L_ema = torch.zeros(self.shard_B, self.out_dim, self.out_dim, **d)
        self.R_ema = torch.zeros(self.shard_B, self.in_dim, self.in_dim, **d)
        eye_L = torch.eye(self.out_dim, **d).expand(self.shard_B, -1, -1).contiguous()
        eye_R = torch.eye(self.in_dim, **d).expand(self.shard_B, -1, -1).contiguous()
        self.Q_L = eye_L
        self.Q_R = eye_R
        # bf16 caches must be populated from the start so update_soap's
        # hot-path rotation (Q_L_bf16.mT @ g @ Q_R_bf16) is safe even
        # before the first refresh fires. Identity bf16 means "no rotation"
        # — Adam runs on the raw gradient, matching the pre-refresh intent.
        self.Q_L_bf16 = eye_L.bfloat16().contiguous()
        self.Q_R_bf16 = eye_R.bfloat16().contiguous()

    def _alloc_adam(self):
        d = dict(device=self.device)
        self.m_rot = torch.zeros(self.shard_B, self.out_dim, self.in_dim,
                                  dtype=torch.bfloat16, **d)
        self.v_rot = torch.zeros(self.shard_B, self.out_dim, self.in_dim,
                                  dtype=torch.float32, **d)

    def _alloc_psgd(self):
        d = dict(device=self.device, dtype=self.dtype)
        # Symmetric full factors (not triangular) for simplicity; converges
        # to symmetric positive def via gradient descent on Lie group.
        self.psgd_Ql = torch.eye(self.out_dim, **d).expand(self.shard_B, -1, -1).contiguous()
        self.psgd_Qr = torch.eye(self.in_dim, **d).expand(self.shard_B, -1, -1).contiguous()

    def ensure_eigen(self):
        if self.L_ema is None:
            self._alloc_eigen()

    def ensure_adam(self):
        if self.m_rot is None:
            self._alloc_adam()

    def ensure_psgd(self):
        if self.psgd_Ql is None:
            self._alloc_psgd()

    def state_dict(self):
        out = {}
        for k in ("L_ema", "R_ema", "Q_L", "Q_R", "m_rot", "v_rot",
                  "psgd_Ql", "psgd_Qr", "prev_L_norm", "prev_R_norm"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v.detach().cpu()
        for k in ("step", "step_since_refresh", "refresh_ct", "initialized"):
            out[k] = getattr(self, k)
        return out

    def load_state_dict(self, sd):
        for k in ("L_ema", "R_ema", "Q_L", "Q_R", "m_rot", "v_rot",
                  "psgd_Ql", "psgd_Qr", "prev_L_norm", "prev_R_norm"):
            if k in sd and sd[k] is not None:
                setattr(self, k, sd[k].to(self.device))
        for k in ("step", "step_since_refresh", "refresh_ct", "initialized"):
            if k in sd:
                setattr(self, k, sd[k])
        # Bf16 caches and inv-sqrts are derived from the fp32 source tensors
        # and NOT in the save list. Regenerate them from the restored fp32
        # Q's so there's no inconsistency between Q_L (possibly reset to I)
        # and Q_L_bf16 (which would otherwise keep a stale value from before
        # load). L_inv_sqrt is regenerated on next refresh.
        if self.Q_L is not None:
            self.Q_L_bf16 = self.Q_L.bfloat16().contiguous()
            self.Q_R_bf16 = self.Q_R.bfloat16().contiguous()
        else:
            self.Q_L_bf16 = None
            self.Q_R_bf16 = None
        self.L_inv_sqrt = None
        self.R_inv_sqrt = None
        self.L_inv_sqrt_bf16 = None
        self.R_inv_sqrt_bf16 = None


def _ema_accum_(buf, new, beta):
    """buf = beta*buf + (1-beta)*new, in-place on buf."""
    buf.mul_(beta).add_(new, alpha=1.0 - beta)


def _trace_mean(M):
    """mean of diagonal, shape (...,) for (..., d, d)."""
    return M.diagonal(dim1=-2, dim2=-1).mean(dim=-1)


@_maybe_compile
def _rotate_eigen_state(m_rot, v_rot, Q_L_old, Q_R_old, Q_L_new, Q_R_new):
    """Re-express Adam state from old eigenbasis to new.
    m_new = Q_L_new^T @ (Q_L_old @ m_old @ Q_R_old^T) @ Q_R_new
    v_new: v is elementwise squared moments — rotating it exactly is not
    possible (squaring doesn't commute with rotation). We approximate by
    rotating sqrt(v) as if it were a matrix, then squaring. Empirically
    this loses little because near-refresh v is dominated by slow-varying
    signal.
    """
    m32 = m_rot.float()
    m_world = Q_L_old @ m32 @ Q_R_old.mT
    m_new = Q_L_new.mT @ m_world @ Q_R_new
    m_rot.copy_(m_new.to(m_rot.dtype))
    sqrt_v = v_rot.sqrt()
    sv_world = Q_L_old @ sqrt_v @ Q_R_old.mT
    sv_new = Q_L_new.mT @ sv_world @ Q_R_new
    v_rot.copy_((sv_new * sv_new).clamp_min(0.0))


@_maybe_compile
def _precond_update_both(L_ema, R_ema, g, beta):
    """L_ema = β·L_ema + (1-β)·GG^T;  R_ema = β·R_ema + (1-β)·G^TG."""
    g32 = g.float()
    L_ema.mul_(beta).add_(g32 @ g32.mT, alpha=1.0 - beta)
    R_ema.mul_(beta).add_(g32.mT @ g32, alpha=1.0 - beta)


@_maybe_compile
def _precond_update_L(L_ema, g, beta):
    """Left-only: L_ema = β·L_ema + (1-β)·GG^T."""
    g32 = g.float()
    L_ema.mul_(beta).add_(g32 @ g32.mT, alpha=1.0 - beta)


@_maybe_compile
def _precond_update_R(R_ema, g, beta):
    """Right-only: R_ema = β·R_ema + (1-β)·G^TG."""
    g32 = g.float()
    R_ema.mul_(beta).add_(g32.mT @ g32, alpha=1.0 - beta)


def update_preconditioner_from_grad(state, g, beta, sides="both"):
    """Dispatcher to the compiled side-specific cores. Thin wrapper so
    the hot path stays compiled while `sides` (a Python str) is chosen
    outside the graph.
    """
    state.ensure_eigen()
    if sides == "both":
        _precond_update_both(state.L_ema, state.R_ema, g, beta)
    elif sides == "L":
        _precond_update_L(state.L_ema, g, beta)
    elif sides == "R":
        _precond_update_R(state.R_ema, g, beta)
    else:
        raise ValueError(sides)


def update_preconditioner_from_activations(state, R_act, side, beta):
    """Update one side's EMA from activation covariance. R_act is (in, in) or
    (out, out) matching the requested side — caller is responsible for shape.
    """
    state.ensure_eigen()
    if side == "R":
        # R_act is per-layer or global; broadcast across shard_B if needed
        if R_act.ndim == 2:
            R_act = R_act.unsqueeze(0).expand_as(state.R_ema)
        _ema_accum_(state.R_ema, R_act.float(), beta)
    elif side == "L":
        if R_act.ndim == 2:
            R_act = R_act.unsqueeze(0).expand_as(state.L_ema)
        _ema_accum_(state.L_ema, R_act.float(), beta)
    else:
        raise ValueError(side)


def maybe_refresh_eigenbasis(state, refresh_k, adaptive, drift_tau, damping,
                             rotate_adam=False, compute_inv_sqrt="none",
                             inv_root_ns_steps=8, warmup_steps=0,
                             sides="both"):
    """Recompute Q_L, Q_R from current L_ema, R_ema. Optionally rotate
    accumulated Adam state so m_rot/v_rot stay meaningful across refresh.

    compute_inv_sqrt:
        "none"   — only Q_L, Q_R (SOAP modes)
        "eigh"   — also cache L_inv_sqrt, R_inv_sqrt from the same eigh
                   (muon_2side)
        "ns"     — cache L_inv_sqrt, R_inv_sqrt via NS iteration
                   (shampoo_ns, avoids the extra eigh call)

    Returns True if a refresh happened this call.
    """
    state.ensure_eigen()
    state.step_since_refresh += 1
    if warmup_steps > 0 and state.step < warmup_steps:
        # Hold Q=I during warmup so updates behave like raw (Adam/SGDM)
        # on un-rotated gradients while L/R EMAs accumulate signal.
        return False
    should_refresh = False
    if state.step_since_refresh >= refresh_k:
        should_refresh = True
    if adaptive and state.step_since_refresh >= max(5, refresh_k // 4):
        if state.L_ema.shape[0] > 0:
            L_norm = state.L_ema.norm(dim=(-2, -1))
            R_norm = state.R_ema.norm(dim=(-2, -1))
            prev_L = state.prev_L_norm.clamp_min(1e-30)
            prev_R = state.prev_R_norm.clamp_min(1e-30)
            drift_L = ((L_norm - prev_L).abs() / prev_L).max().item()
            drift_R = ((R_norm - prev_R).abs() / prev_R).max().item()
            state.last_drift_L = drift_L
            state.last_drift_R = drift_R
            if drift_L > drift_tau or drift_R > drift_tau:
                should_refresh = True
    # Force a refresh on first completed EMA (step ≥ a few) so Q stops being I.
    if not state.initialized and state.step >= 5:
        should_refresh = True
    if not should_refresh:
        return False
    if state.L_ema.shape[0] == 0:
        # Rank owns zero layers in this bank — nothing to refresh.
        state.initialized = True
        state.step_since_refresh = 0
        return False
    tr_L = _trace_mean(state.L_ema).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-30)
    tr_R = _trace_mean(state.R_ema).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-30)
    d_out = state.out_dim
    d_in = state.in_dim
    eye_L = torch.eye(d_out, device=state.device, dtype=state.dtype)
    eye_R = torch.eye(d_in, device=state.device, dtype=state.dtype)
    L_norm = state.L_ema / tr_L + damping * eye_L
    R_norm = state.R_ema / tr_R + damping * eye_R
    # Only recompute Q on the side(s) the mode actually uses. For soap_1side_*
    # modes, the unused-side Q must stay at I (the initial allocation value),
    # otherwise m_rot / v_rot get corrupted by rotations that weren't applied
    # during accumulation.
    update_L = sides in ("both", "left")
    update_R = sides in ("both", "right")
    try:
        if compute_inv_sqrt == "eigh":
            if update_L:
                L_eigs, Q_L_new = torch.linalg.eigh(L_norm)
                L_max = L_eigs.max(dim=-1, keepdim=True).values
                L_clamped = L_eigs.clamp_min(1e-4 * L_max)
                L_is_eig = L_clamped.rsqrt()
                L_inv_sqrt_new = (Q_L_new * L_is_eig.unsqueeze(-2)) @ Q_L_new.mT
            else:
                Q_L_new = state.Q_L
                L_inv_sqrt_new = state.L_inv_sqrt
            if update_R:
                R_eigs, Q_R_new = torch.linalg.eigh(R_norm)
                R_max = R_eigs.max(dim=-1, keepdim=True).values
                R_clamped = R_eigs.clamp_min(1e-4 * R_max)
                R_is_eig = R_clamped.rsqrt()
                R_inv_sqrt_new = (Q_R_new * R_is_eig.unsqueeze(-2)) @ Q_R_new.mT
            else:
                Q_R_new = state.Q_R
                R_inv_sqrt_new = state.R_inv_sqrt
        else:
            if update_L:
                _, Q_L_new = torch.linalg.eigh(L_norm)
            else:
                Q_L_new = state.Q_L
            if update_R:
                _, Q_R_new = torch.linalg.eigh(R_norm)
            else:
                Q_R_new = state.Q_R
            L_inv_sqrt_new = None
            R_inv_sqrt_new = None
    except RuntimeError:
        # Numerical failure — keep previous Q, mark as unrefreshed.
        return False
    if compute_inv_sqrt == "ns":
        if update_L:
            L_inv_sqrt_new = matrix_inv_sqrt_ns(L_norm, steps=inv_root_ns_steps,
                                                 damping=0.0)
        if update_R:
            R_inv_sqrt_new = matrix_inv_sqrt_ns(R_norm, steps=inv_root_ns_steps,
                                                 damping=0.0)
    # Rotate Adam state even on the first refresh: m_rot/v_rot were accumulated
    # under state.Q_L/state.Q_R (initially I during warmup); we must transform
    # them into the new basis or Adam's momentum lives in a stale basis once
    # preconditioning engages.
    if rotate_adam and state.m_rot is not None:
        _rotate_eigen_state(state.m_rot, state.v_rot,
                            state.Q_L, state.Q_R, Q_L_new, Q_R_new)
    state.Q_L = Q_L_new
    state.Q_R = Q_R_new
    # bf16 caches for hot-path matmuls. Cast here (once per refresh) so
    # update_soap / update_muon_2side / update_shampoo_ns can run in bf16
    # without a per-step fp32 conversion.
    state.Q_L_bf16 = Q_L_new.bfloat16()
    state.Q_R_bf16 = Q_R_new.bfloat16()
    if L_inv_sqrt_new is not None:
        state.L_inv_sqrt = L_inv_sqrt_new
        state.R_inv_sqrt = R_inv_sqrt_new
        state.L_inv_sqrt_bf16 = L_inv_sqrt_new.bfloat16()
        state.R_inv_sqrt_bf16 = R_inv_sqrt_new.bfloat16()
    state.prev_L_norm = state.L_ema.norm(dim=(-2, -1)).clone()
    state.prev_R_norm = state.R_ema.norm(dim=(-2, -1)).clone()
    state.step_since_refresh = 0
    state.refresh_ct += 1
    state.initialized = True
    return True


def ns_update_msgn(update, backend_steps, cfg):
    """Dispatch to one of the NS variants based on cfg flags."""
    shape = update.shape
    if update.ndim < 3:
        update = update.unsqueeze(0)
    use_gram = (update.shape[-2] != update.shape[-1]
                and max(update.shape[-2], update.shape[-1])
                    >= 3 * min(update.shape[-2], update.shape[-1]))
    if cfg.get("ns_two_phase"):
        aggressive = max(1, backend_steps - cfg.get("ns_refine_steps", 2))
        refine = cfg.get("ns_refine_steps", 2)
        out = newtonschulz_two_phase(update, aggressive, refine)
        steps = aggressive + refine
    elif cfg.get("ns_adaptive"):
        eps = cfg.get("ns_adaptive_eps", 0.02)
        out, steps = newtonschulz_adaptive(update, backend_steps, ortho_eps=eps)
    else:
        if use_gram and not cfg.get("disable_gram_ns"):
            out = gram_newton_schulz5(update, steps=backend_steps)
        else:
            out = zeropower_via_newtonschulz5(update, steps=backend_steps)
        steps = backend_steps
    if len(shape) < 3:
        out = out.squeeze(0)
    return out, steps


def update_muon(g, state, cfg):
    """Standard Muon msgn update (possibly with adaptive/two-phase NS).

    g is (shard_B, out, in). Returns update of same shape.
    Assumes momentum has already been applied by the caller. `state` may be
    None for modes that don't use preconditioner state (`muon`, `muon_ns_fix`).
    """
    update, ns_steps = ns_update_msgn(g, cfg["backend_steps"], cfg)
    if state is not None:
        state.last_ns_steps = ns_steps
    return update


@_maybe_compile
def _muon_2side_pre_rotate(g, L_is, R_is):
    """L^{-1/2} · g · R^{-1/2} in bf16."""
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    return L_is @ g_bf @ R_is


@_maybe_compile
def _muon_2side_post_rotate(core, L_is, R_is):
    """L^{-1/2} · core · R^{-1/2} applied to msgn output."""
    core_bf = core if core.dtype == torch.bfloat16 else core.bfloat16()
    return L_is @ core_bf @ R_is


def update_muon_2side(g, state, cfg):
    """Two-sided preconditioned msgn:
        L^{-1/2} · msgn(L^{-1/2} · g · R^{-1/2}) · R^{-1/2}
    NS in the middle is already compiled; pre/post bf16 rotations are
    compiled as separate cores.
    """
    state.ensure_eigen()
    if not state.initialized or state.L_inv_sqrt_bf16 is None:
        return update_muon(g, state, cfg)
    L_is = state.L_inv_sqrt_bf16
    R_is = state.R_inv_sqrt_bf16
    pre = _muon_2side_pre_rotate(g, L_is, R_is)
    cfg2 = dict(cfg)
    cfg2["disable_gram_ns"] = True
    core, ns_steps = ns_update_msgn(pre, cfg["backend_steps"], cfg2)
    update = _muon_2side_post_rotate(core, L_is, R_is).to(g.dtype)
    state.last_ns_steps = ns_steps
    return update


@_maybe_compile
def _update_shampoo_ns_core(g, L_is, R_is):
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    update = L_is @ g_bf @ R_is
    # Rescale so update norm matches raw g (preconditioner contracts norm).
    g32 = g.float()
    g_norm = g32.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    u_norm = update.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    return update * (g_norm / u_norm).to(update.dtype)


def update_shampoo_ns(g, state, cfg):
    """Shampoo-style two-sided preconditioning without msgn, using NS
    iterations (not eigh) for matrix inverse roots. Cheaper refresh than
    SOAP / muon_2side because eigendecomposition is avoided.

    `L_inv_sqrt`/`R_inv_sqrt` are cached on refresh (see
    maybe_refresh_eigenbasis with compute_inv_sqrt="ns"); here we just apply.
    """
    state.ensure_eigen()
    if not state.initialized or state.L_inv_sqrt_bf16 is None:
        return g.to(g.dtype)
    update = _update_shampoo_ns_core(
        g, state.L_inv_sqrt_bf16, state.R_inv_sqrt_bf16
    )
    return update.to(g.dtype)


def _soap_rotate_forward(state, g):
    """G_rot = Q_L^T @ G @ Q_R, in bf16 using the cached Q's.

    H100 bf16 matmul uses tensor cores (~1 PFLOP/s) with fp32 accumulation
    internally. Result is bf16; caller can upcast for Adam ops as needed.
    g is bf16 in the normal training path (m["shard"] output); cast defensively
    for tests that pass fp32.
    """
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    return state.Q_L_bf16.mT @ g_bf @ state.Q_R_bf16


def _soap_rotate_backward(state, upd_rot):
    """update = Q_L @ upd_rot @ Q_R^T. upd_rot is fp32 (Adam output), cast
    to bf16 so the matmul uses tensor cores. Precision loss is minimal —
    final update is bf16 anyway."""
    return state.Q_L_bf16 @ upd_rot.bfloat16() @ state.Q_R_bf16.mT


@_maybe_compile
def _update_soap_adam_core(
    g, Q_L_bf16, Q_R_bf16, m_rot, v_rot, bc1, bc2, beta1, beta2, eps
):
    """Adam in rotated eigenbasis: rotate forward, update m/v in-place,
    apply bias correction via tensor scalars bc1/bc2, rotate backward."""
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    g_rot = Q_L_bf16.mT @ g_bf @ Q_R_bf16
    m_rot.mul_(beta1).add_(g_rot, alpha=1.0 - beta1)
    g_rot_fp = g_rot.float()
    v_rot.mul_(beta2).addcmul_(g_rot_fp, g_rot_fp, value=1.0 - beta2)
    m_hat = m_rot.float() / bc1
    v_hat = v_rot / bc2
    upd_rot = m_hat / (v_hat.sqrt() + eps)
    return Q_L_bf16 @ upd_rot.bfloat16() @ Q_R_bf16.mT


@_maybe_compile
def _update_soap_lion_core(g, Q_L_bf16, Q_R_bf16, m_rot, beta1, beta2):
    """Lion in rotated eigenbasis."""
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    g_rot = Q_L_bf16.mT @ g_bf @ Q_R_bf16
    g_rot_fp = g_rot.float()
    m_fp = m_rot.float()
    inner = beta1 * m_fp + (1.0 - beta1) * g_rot_fp
    upd_rot = inner.sign()
    m_fp2 = beta2 * m_fp + (1.0 - beta2) * g_rot_fp
    m_rot.copy_(m_fp2.to(m_rot.dtype))
    return Q_L_bf16 @ upd_rot.bfloat16() @ Q_R_bf16.mT


@_maybe_compile
def _update_soap_sgdm_core(g, Q_L_bf16, Q_R_bf16, m_rot, beta1):
    """SGD+momentum in rotated eigenbasis."""
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    g_rot = Q_L_bf16.mT @ g_bf @ Q_R_bf16
    m_rot.mul_(beta1).add_(g_rot, alpha=1.0 - beta1)
    upd_rot = m_rot.float()
    return Q_L_bf16 @ upd_rot.bfloat16() @ Q_R_bf16.mT


def update_soap(g, state, cfg):
    """SOAP: Adam in preconditioner eigenbasis (Vyas et al. 2024).

    Key subtlety: Q_L, Q_R change at refresh; m_rot and v_rot are rotated
    concurrently (handled by maybe_refresh_eigenbasis with rotate_adam=True).

    `SOAP_BASE` selects the in-eigenbasis base optimizer:
      - adam: Adam with bias correction
      - lion: sign of momentum
      - sgdm: SGD with momentum
    """
    state.ensure_eigen()
    state.ensure_adam()
    base = cfg.get("soap_base", "adam")
    beta1 = cfg["soap_beta1"]
    beta2 = cfg["soap_beta2"]
    eps = cfg["soap_eps"]
    if base == "adam":
        t = max(1, state.step)
        # bc1/bc2 as tensors so the compiled core sees them as dynamic
        # inputs (Python scalars would trigger recompile each step).
        bc1 = torch.tensor(1.0 - beta1 ** t, device=g.device, dtype=torch.float32)
        bc2 = torch.tensor(1.0 - beta2 ** t, device=g.device, dtype=torch.float32)
        update = _update_soap_adam_core(
            g, state.Q_L_bf16, state.Q_R_bf16,
            state.m_rot, state.v_rot,
            bc1, bc2, beta1, beta2, eps,
        )
    elif base == "lion":
        update = _update_soap_lion_core(
            g, state.Q_L_bf16, state.Q_R_bf16, state.m_rot, beta1, beta2
        )
    elif base == "sgdm":
        update = _update_soap_sgdm_core(
            g, state.Q_L_bf16, state.Q_R_bf16, state.m_rot, beta1
        )
    else:
        raise ValueError(f"unknown soap_base {base!r}")
    return update.to(g.dtype)


@_maybe_compile
def _update_soap_1side_left_core(
    g, Q_L_bf16, m_rot, v_rot, bc1, bc2, beta1, beta2, eps
):
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    g_rot = Q_L_bf16.mT @ g_bf
    m_rot.mul_(beta1).add_(g_rot, alpha=1.0 - beta1)
    g_rot_fp = g_rot.float()
    v_rot.mul_(beta2).addcmul_(g_rot_fp, g_rot_fp, value=1.0 - beta2)
    upd_rot = (m_rot.float() / bc1) / ((v_rot / bc2).sqrt() + eps)
    return Q_L_bf16 @ upd_rot.bfloat16()


@_maybe_compile
def _update_soap_1side_right_core(
    g, Q_R_bf16, m_rot, v_rot, bc1, bc2, beta1, beta2, eps
):
    g_bf = g if g.dtype == torch.bfloat16 else g.bfloat16()
    g_rot = g_bf @ Q_R_bf16
    m_rot.mul_(beta1).add_(g_rot, alpha=1.0 - beta1)
    g_rot_fp = g_rot.float()
    v_rot.mul_(beta2).addcmul_(g_rot_fp, g_rot_fp, value=1.0 - beta2)
    upd_rot = (m_rot.float() / bc1) / ((v_rot / bc2).sqrt() + eps)
    return upd_rot.bfloat16() @ Q_R_bf16.mT


def update_soap_1side(g, state, cfg, side):
    """One-sided SOAP: rotate via only Q_L (side=left) or Q_R (side=right).
    Cheaper per step; loses geometric info on the untracked side.
    """
    state.ensure_eigen()
    state.ensure_adam()
    beta1 = cfg["soap_beta1"]
    beta2 = cfg["soap_beta2"]
    eps = cfg["soap_eps"]
    t = max(1, state.step)
    bc1 = torch.tensor(1.0 - beta1 ** t, device=g.device, dtype=torch.float32)
    bc2 = torch.tensor(1.0 - beta2 ** t, device=g.device, dtype=torch.float32)
    if side == "left":
        update = _update_soap_1side_left_core(
            g, state.Q_L_bf16, state.m_rot, state.v_rot,
            bc1, bc2, beta1, beta2, eps,
        )
    elif side == "right":
        update = _update_soap_1side_right_core(
            g, state.Q_R_bf16, state.m_rot, state.v_rot,
            bc1, bc2, beta1, beta2, eps,
        )
    else:
        raise ValueError(side)
    return update.to(g.dtype)


@_maybe_compile
def _update_psgd_kron_core(g, Ql, Qr, precond_lr, out_dim, in_dim):
    g32 = g.float()
    g_fro_sq = (g32 * g32).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    gg = g32 @ g32.mT / g_fro_sq * out_dim
    gtg = g32.mT @ g32 / g_fro_sq * in_dim
    eye_L = torch.eye(out_dim, device=g32.device, dtype=g32.dtype)
    eye_R = torch.eye(in_dim, device=g32.device, dtype=g32.dtype)
    residual_L = Ql @ gg @ Ql - eye_L
    residual_R = Qr @ gtg @ Qr - eye_R
    grad_Ql = gg @ Ql @ residual_L + residual_L @ Ql @ gg
    grad_Qr = gtg @ Qr @ residual_R + residual_R @ Qr @ gtg
    Ql_new = Ql - 0.5 * precond_lr * grad_Ql
    Qr_new = Qr - 0.5 * precond_lr * grad_Qr
    Ql_new = 0.5 * (Ql_new + Ql_new.mT)
    Qr_new = 0.5 * (Qr_new + Qr_new.mT)
    Ql_fro = Ql_new.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    Qr_fro = Qr_new.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    Ql_target = (out_dim ** 0.5)
    Qr_target = (in_dim ** 0.5)
    Ql_new = Ql_new * (Ql_target / Ql_fro).clamp_max(1.0)
    Qr_new = Qr_new * (Qr_target / Qr_fro).clamp_max(1.0)
    update = Ql_new @ g32 @ Qr_new
    g_norm = g32.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    u_norm = update.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    update = update * (g_norm / u_norm)
    return update, Ql_new, Qr_new


def update_psgd_kron(g, state, cfg):
    """Simplified PSGD-Kron preconditioner (Li 2018, simplified).

    Maintains symmetric factors Q_l (out×out), Q_r (in×in) such that at the
    fixed point Q_l · gg^T · Q_l ≈ I and Q_r · g^T g · Q_r ≈ I.
    """
    state.ensure_psgd()
    update, Ql_new, Qr_new = _update_psgd_kron_core(
        g, state.psgd_Ql, state.psgd_Qr, cfg["psgd_kron_precond_lr"],
        state.out_dim, state.in_dim,
    )
    state.psgd_Ql = Ql_new
    state.psgd_Qr = Qr_new
    return update.to(g.dtype)


# Per-mode dispatch table. Each function takes (g, state, cfg) → update.
# `g` already has preconditioned momentum applied if cfg["precond_momentum"]
# is True; otherwise it is raw gradient and momentum is applied elsewhere.
UPDATE_FUNCTIONS = {
    "muon": update_muon,
    "muon_ns_fix": update_muon,  # same fn; NS flags differ via cfg
    "muon_2side": update_muon_2side,
    "soap": update_soap,
    "soap_1side_left": lambda g, s, c: update_soap_1side(g, s, c, "left"),
    "soap_1side_right": lambda g, s, c: update_soap_1side(g, s, c, "right"),
    "shampoo_ns": update_shampoo_ns,
    "psgd_kron": update_psgd_kron,
}


MODE_USES_PRECONDITIONER = {
    "muon": False,
    "muon_ns_fix": False,
    "muon_2side": True,
    "soap": True,
    "soap_1side_left": True,
    "soap_1side_right": True,
    "shampoo_ns": True,
    "psgd_kron": True,  # uses Q_l/Q_r, not L_ema/R_ema
}


MODE_ROTATES_ADAM = {
    "soap": True,
    "soap_1side_left": True,
    "soap_1side_right": True,
    # muon_2side, shampoo_ns don't have Adam state.
}


# Which eigenbasis sides each mode actually uses. The refresh must only
# update the Q's the mode will multiply with — otherwise m_rot/v_rot get
# corrupted on refresh by rotations that weren't part of accumulation.
MODE_ROTATES_SIDES = {
    "soap": "both",
    "soap_1side_left": "left",
    "soap_1side_right": "right",
    "muon_2side": "both",
    "shampoo_ns": "both",
}


def auto_scale_mode(optim_mode, explicit_mode):
    """Pick a sensible scale mode if user left SCALE_MODE=auto."""
    if explicit_mode != "auto":
        return explicit_mode
    if optim_mode in ("soap", "soap_1side_left", "soap_1side_right"):
        return "soap"
    if optim_mode in ("muon_2side", "shampoo_ns"):
        return "fan_out"
    if optim_mode == "psgd_kron":
        return "soap"  # PSGD update is already scale-matched to g
    if optim_mode == "muon_ns_fix":
        return "ratio"
    return "ratio_clamped"


class BankOptimizer(torch.optim.Optimizer):
    """Drop-in replacement for `Muon` supporting all OPTIM_MODE variants.

    Preserves the reduce_scatter / all_gather flow and per-bank padding from
    the original `Muon`. Adds per-bank `PreconditionerState` for modes that
    use two-sided preconditioning.
    """

    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
        row_normalize=False,
        optim_mode="muon",
        scale_mode="ratio_clamped",
        scale_d_ref=None,
        cautious_mask=False,
        precond_momentum=True,
        precond_source="grad",
        soap_cfg=None,
        ns_cfg=None,
        psgd_cfg=None,
        bank_use_precond=None,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )
        self._built = False
        self.optim_mode = optim_mode
        self.scale_mode = scale_mode
        self.scale_d_ref = scale_d_ref
        self.cautious_mask = cautious_mask
        self.precond_momentum = precond_momentum
        self.precond_source = precond_source
        self.soap_cfg = soap_cfg or {}
        self.ns_cfg = ns_cfg or {}
        self.psgd_cfg = psgd_cfg or {}
        # Per-param flag: if False, that param falls back to plain muon msgn
        # regardless of optim_mode. Keyed by the order params were added.
        # If None, all params use the full optim_mode.
        self.bank_use_precond = bank_use_precond
        # How often to refresh L_ema / R_ema from the gradient (in steps).
        # 1 = every step. Higher = fewer updates but staler preconditioner.
        self._precond_update_k = max(1, int(
            (soap_cfg or {}).get("precond_update_k", 1)
        ))
        if optim_mode not in UPDATE_FUNCTIONS:
            raise ValueError(f"unknown OPTIM_MODE={optim_mode!r}")

    def _make_cfg(self, group):
        cfg = {
            "backend_steps": group["backend_steps"],
            "soap_damping": self.soap_cfg.get("damping", 0.03),
            "soap_beta1": self.soap_cfg.get("beta1", 0.95),
            "soap_beta2": self.soap_cfg.get("beta2", 0.95),
            "soap_eps": self.soap_cfg.get("eps", 1e-8),
            "soap_base": self.soap_cfg.get("base", "adam"),
            "inv_root_ns_steps": self.soap_cfg.get("inv_root_ns_steps", 5),
            "ns_adaptive": self.ns_cfg.get("adaptive", False),
            "ns_adaptive_eps": self.ns_cfg.get("adaptive_eps", 0.02),
            "ns_two_phase": self.ns_cfg.get("two_phase", False),
            "ns_refine_steps": self.ns_cfg.get("refine_steps", 2),
            "ns_warm_start": self.ns_cfg.get("warm_start", False),
            "psgd_kron_precond_lr": self.psgd_cfg.get("precond_lr", 0.1),
        }
        return cfg

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        # NS wrapper needed only when ns_adaptive / ns_two_phase flags are on;
        # in the common case we can call the compiled NS primitive directly
        # and skip `ns_update_msgn` + `update_muon` wrapper frames.
        needs_ns_wrapper = bool(self.ns_cfg.get("adaptive", False)) or bool(
            self.ns_cfg.get("two_phase", False)
        )
        # Flatten param order matches ctor order of `params`, which is how
        # `bank_use_precond` is keyed when provided.
        param_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                # Per-bank effective mode: if bank_use_precond is False for
                # this param, fall back to "muon" regardless of optim_mode.
                if self.bank_use_precond is None:
                    use_precond = True
                else:
                    use_precond = bool(self.bank_use_precond[param_idx])
                effective_mode = self.optim_mode if use_precond else "muon"
                out_dim = p.shape[-2]
                in_dim = p.shape[-1]
                # Cache all per-bank dispatch so step() reads them once from
                # the meta dict rather than doing dict lookups + string ops
                # in the hot path.
                uses_precond = MODE_USES_PRECONDITIONER.get(effective_mode, False)
                rotate_adam = MODE_ROTATES_ADAM.get(effective_mode, False)
                mode_sides = MODE_ROTATES_SIDES.get(effective_mode, "both")
                mode_sides_grad = {
                    "left": "L", "right": "R", "both": "both",
                }.get(mode_sides, "both")
                mode_has_internal_momentum = effective_mode.startswith("soap")
                # Fast path = muon-like mode with default NS flags. Bypasses
                # the update_fn / ns_update_msgn wrapper layers and calls the
                # compiled NS primitive directly — matches legacy Muon's
                # Python frame depth.
                fast_path = (
                    effective_mode in ("muon", "muon_ns_fix")
                    and not needs_ns_wrapper
                )
                # Pre-compute which NS primitive to call in fast path, based
                # on shape (gram NS for high-aspect-ratio, else standard).
                use_gram = (
                    out_dim != in_dim
                    and max(out_dim, in_dim) >= 3 * min(out_dim, in_dim)
                )
                fast_ns = gram_newton_schulz5 if use_gram else zeropower_via_newtonschulz5
                meta = {
                    "p": p,
                    "B": B,
                    "padded_grad": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard_mom": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "full_update": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "mode": effective_mode,
                    "update_fn": UPDATE_FUNCTIONS[effective_mode],
                    "uses_precond": uses_precond,
                    "rotate_adam": rotate_adam,
                    "mode_sides_grad": mode_sides_grad,
                    "mode_sides": mode_sides,
                    "mode_has_internal_momentum": mode_has_internal_momentum,
                    "fast_path": fast_path,
                    "fast_ns": fast_ns,
                    "compute_inv_sqrt": (
                        "eigh" if effective_mode == "muon_2side"
                        else "ns" if effective_mode == "shampoo_ns"
                        else "none"
                    ),
                }
                meta["scale"] = compute_scale(out_dim, in_dim, self.scale_mode,
                                              d_ref=self.scale_d_ref)
                if uses_precond:
                    meta["precond"] = PreconditionerState(shard_B, out_dim, in_dim, dev)
                else:
                    meta["precond"] = None
                self._bank_meta.append(meta)
                param_idx += 1
        self._bank_meta.sort(key=lambda m: -m["p"].numel())
        self._built = True
        # Cache the cfg dict — none of its values change mid-run, so we can
        # build it once here and skip the per-step _make_cfg allocation.
        self._cached_cfg = None

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m["padded_grad"]
            pg[: m["B"]].copy_(p.grad.bfloat16())
            if pg.shape[0] > m["B"]:
                pg[m["B"] :].zero_()
            fut = dist.reduce_scatter_tensor(
                m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True
            )
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        # Per-bank mode is looked up inside the loop via meta["mode"].
        beta_precond = self.soap_cfg.get("beta_precond", 0.95)
        refresh_k = self.soap_cfg.get("refresh_k", 50)
        refresh_adaptive = self.soap_cfg.get("refresh_adaptive", True)
        drift_tau = self.soap_cfg.get("refresh_drift_tau", 0.05)
        damping = self.soap_cfg.get("damping", 0.03)
        inv_root_ns_steps_cfg = self.soap_cfg.get("inv_root_ns_steps", 8)
        warmup_steps_cfg = self.soap_cfg.get("warmup_steps", 0)
        precond_update_k = self._precond_update_k
        precond_source = self.precond_source
        # Rename locally to avoid shadowing the module-level apply_cautious_mask.
        do_cautious_mask = self.cautious_mask
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            row_normalize = group.get("row_normalize", False)
            # Build cfg once per run (cached on self). Values are constant
            # across steps in the current codebase, so there's no need to
            # rebuild this dict every step.
            if self._cached_cfg is None:
                self._cached_cfg = self._make_cfg(group)
            cfg = self._cached_cfg
            # Avoid per-step dict overwrites: update backend_steps in the
            # cached cfg only if it changed (very rare).
            if cfg["backend_steps"] != group["backend_steps"]:
                cfg["backend_steps"] = group["backend_steps"]
            backend_steps = group["backend_steps"]
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, "_rs_futures")
            for idx, m in enumerate(self._bank_meta):
                p = m["p"]
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m["p"]
                    upd = prev_m["full_update"][: prev_m["B"]]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
                if sharded and self._rs_futures[idx] is not None:
                    self._rs_futures[idx].wait()
                    g = m["shard"]
                    buf = m["shard_mom"]
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                # ============================================================
                # FAST PATH — legacy-Muon-equivalent (muon / muon_ns_fix with
                # default NS flags). Matches legacy Muon's structure and
                # Python frame depth so there's no dispatch-layer overhead.
                # ============================================================
                if m["fast_path"]:
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        mom_input = g.add(buf, alpha=momentum)
                    else:
                        mom_input = buf
                    if row_normalize:
                        rn = mom_input.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                        mom_input = mom_input / rn.to(mom_input.dtype)
                    update = m["fast_ns"](mom_input, steps=backend_steps)
                    if do_cautious_mask:
                        update = apply_cautious_mask(update, mom_input)
                    if sharded:
                        prev_ag_handle = dist.all_gather_into_tensor(
                            m["full_update"], update.contiguous(), async_op=True
                        )
                        prev_m = m
                    else:
                        if wd > 0.0:
                            p.data.mul_(1.0 - lr * wd)
                        p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
                    continue
                # ============================================================
                # SLOW PATH — preconditioner modes (soap / muon_2side /
                # shampoo_ns / psgd_kron) or NS flags that need the wrapper.
                # All per-bank dispatch values are cached on meta at _build.
                # ============================================================
                update_fn = m["update_fn"]
                uses_precond = m["uses_precond"]
                rotate_adam = m["rotate_adam"]
                mode_sides_grad = m["mode_sides_grad"]
                mode_sides = m["mode_sides"]
                mode_has_internal_momentum = m["mode_has_internal_momentum"]
                compute_inv_sqrt = m["compute_inv_sqrt"]
                mode = m["mode"]
                precond_state = m["precond"]
                if uses_precond:
                    precond_state.step += 1
                # Update preconditioner EMAs (gated by PRECOND_UPDATE_K).
                if (uses_precond
                        and mode != "psgd_kron"
                        and (precond_state.step % precond_update_k == 0)):
                    if precond_source == "grad":
                        update_preconditioner_from_grad(
                            precond_state, g, beta_precond,
                            sides=mode_sides_grad)
                    elif precond_source == "act" and mode_sides_grad in ("both", "L"):
                        update_preconditioner_from_grad(
                            precond_state, g, beta_precond, sides="L")
                if uses_precond and mode != "psgd_kron":
                    maybe_refresh_eigenbasis(
                        precond_state, refresh_k, refresh_adaptive, drift_tau,
                        damping, rotate_adam=rotate_adam,
                        compute_inv_sqrt=compute_inv_sqrt,
                        inv_root_ns_steps=inv_root_ns_steps_cfg,
                        warmup_steps=warmup_steps_cfg,
                        sides=mode_sides,
                    )
                # Momentum: SOAP has internal Adam β1, other modes use the
                # outer buffer.
                if mode_has_internal_momentum:
                    mom_input = g
                else:
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        mom_input = g.add(buf, alpha=momentum)
                    else:
                        mom_input = buf
                if row_normalize and not mode_has_internal_momentum:
                    rn = mom_input.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                    mom_input = mom_input / rn.to(mom_input.dtype)
                # Banks are always 3D; no unsqueeze/squeeze needed.
                update = update_fn(mom_input, precond_state, cfg)
                if do_cautious_mask:
                    update = apply_cautious_mask(update, mom_input)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m["full_update"], update.contiguous(), async_op=True
                    )
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m["p"]
                upd = prev_m["full_update"][: prev_m["B"]]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
            if hasattr(self, "_rs_futures"):
                del self._rs_futures
        return loss

    def compile_warmup_refresh(self):
        """Force one refresh per bank so the refresh-only code paths
        (_rotate_eigen_state, matrix_inv_sqrt_ns, eigh-based inv-sqrt)
        get torch.compile-cached during the warmup phase rather than at
        step ~500 of the main loop. Call after the warmup_steps loop has
        populated L_ema / R_ema with real gradient data.
        """
        if not self._built or not _COMPILE_SOAP:
            return
        damping = self.soap_cfg.get("damping", 0.03)
        inv_root_ns_steps = self.soap_cfg.get("inv_root_ns_steps", 8)
        for m in self._bank_meta:
            ps = m["precond"]
            if ps is None:
                continue
            mode = m["mode"]
            if mode == "psgd_kron":
                continue
            # Use step counter as the "EMA has data" check; avoids a host sync
            # via torch.any(). After a few warmup step_fn calls, EMAs are real.
            if ps.L_ema is None or ps.step < 3:
                continue
            ps.step_since_refresh = 999999
            ps.step = max(ps.step, 10)
            maybe_refresh_eigenbasis(
                ps,
                refresh_k=1,
                adaptive=False,
                drift_tau=1e9,
                damping=damping,
                rotate_adam=m["rotate_adam"],
                compute_inv_sqrt=m["compute_inv_sqrt"],
                inv_root_ns_steps=inv_root_ns_steps,
                warmup_steps=0,
                sides=m["mode_sides"],
            )

    def set_bank_registry(self, bank_name_to_param, kv_tie_global=False,
                          local_v_idx=None, global_layer_set=None):
        """Tell the optimizer which (python) object each bank param is so we
        can push captured activation covariances into the right precond state.

        bank_name_to_param: dict mapping one of
            {"q", "k", "v", "out", "mlp_up", "mlp_down"} → nn.Parameter
        """
        if not self._built:
            self._build()
        self._bank_by_name = {}
        for name, param in bank_name_to_param.items():
            for m in self._bank_meta:
                if m["p"] is param:
                    self._bank_by_name[name] = m
                    break
        self._kv_tie_global = kv_tie_global
        self._local_v_idx = local_v_idx or {}
        self._global_layer_set = global_layer_set or set()

    def consume_captured_activations(self, model, beta):
        """Pull captured activation covariances off model.blocks and EMA them
        into the corresponding bank's R_ema (right preconditioner). Expects
        `CAPTURE_ACT=1` to have been wired into the model before forward().
        """
        if not hasattr(self, "_bank_by_name"):
            return
        ws = self._world_size
        rank = self._rank
        for layer_idx, block in enumerate(model.blocks):
            attn_in = getattr(block.attn, "_captured_attn_input_cov", None)
            out_in = getattr(block.attn, "_captured_out_input_cov", None)
            mlp_in = getattr(block.mlp, "_captured_mlp_input_cov", None)
            down_in = getattr(block.mlp, "_captured_down_input_cov", None)
            # Push into R_ema for corresponding banks — indexed by (bank, layer)
            # mapped through the shard layout.
            for bank_name, act_cov in (
                ("q", attn_in), ("k", attn_in), ("v", attn_in),
                ("out", out_in), ("mlp_up", mlp_in), ("mlp_down", down_in),
            ):
                if act_cov is None:
                    continue
                m = self._bank_by_name.get(bank_name)
                if m is None:
                    continue
                if bank_name == "v" and self._kv_tie_global:
                    if layer_idx in self._global_layer_set:
                        continue  # global layers share K, no V bank entry
                    bank_layer = self._local_v_idx.get(layer_idx, None)
                    if bank_layer is None:
                        continue
                else:
                    bank_layer = layer_idx
                # Determine which rank owns this bank_layer and the shard index.
                B = m["B"]
                shard_B = m["shard"].shape[0]
                owner_rank = bank_layer // shard_B
                if owner_rank != rank:
                    continue
                shard_idx = bank_layer - owner_rank * shard_B
                ps = m.get("precond")
                if ps is None:
                    continue
                ps.ensure_eigen()
                if act_cov.shape[-1] != ps.R_ema.shape[-1]:
                    continue  # shape mismatch — skip this layer
                ps.R_ema[shard_idx].mul_(beta).add_(act_cov, alpha=1.0 - beta)
            # Clear captured tensors to free memory
            if attn_in is not None:
                block.attn._captured_attn_input_cov = None
            if out_in is not None:
                block.attn._captured_out_input_cov = None
            if mlp_in is not None:
                block.mlp._captured_mlp_input_cov = None
            if down_in is not None:
                block.mlp._captured_down_input_cov = None

    def set_capture_act(self, model, enabled):
        """Flip _capture_act flag on every CausalSelfAttention and MLP."""
        for block in model.blocks:
            block.attn._capture_act = enabled
            block.mlp._capture_act = enabled

    def diag_log(self, log_fn):
        """Emit per-layer diagnostic lines for rank 0. Called by training loop."""
        if self._rank != 0:
            return
        for idx, m in enumerate(self._bank_meta):
            precond_state = m.get("precond")
            if precond_state is None or not precond_state.initialized:
                continue
            L = precond_state.L_ema
            R = precond_state.R_ema
            for s in range(L.shape[0]):
                L_eig = torch.linalg.eigvalsh(L[s]).detach()
                R_eig = torch.linalg.eigvalsh(R[s]).detach()
                cond_L = (L_eig.max() / L_eig.abs().min().clamp_min(1e-30)).item()
                cond_R = (R_eig.max() / R_eig.abs().min().clamp_min(1e-30)).item()
                log_fn(
                    f"precond:bank={idx} layer={s} "
                    f"cond_L={cond_L:.3e} cond_R={cond_R:.3e} "
                    f"drift_L={precond_state.last_drift_L:.3e} "
                    f"drift_R={precond_state.last_drift_R:.3e} "
                    f"refresh_ct={precond_state.refresh_ct} "
                    f"ns_steps={precond_state.last_ns_steps}"
                )

    def state_dict(self):
        sd = super().state_dict()
        precond_states = []
        for m in self._bank_meta if self._built else []:
            ps = m.get("precond")
            precond_states.append(ps.state_dict() if ps is not None else None)
        sd["_precond_states"] = precond_states
        return sd

    def load_state_dict(self, state_dict):
        precond_states = state_dict.pop("_precond_states", None)
        super().load_state_dict(state_dict)
        if precond_states is not None and self._built:
            for m, ps_sd in zip(self._bank_meta, precond_states):
                if m.get("precond") is not None and ps_sd is not None:
                    m["precond"].load_state_dict(ps_sd)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
        row_normalize=False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    "p": p,
                    "B": B,
                    "padded_grad": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard_mom": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "full_update": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "scale": max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m["p"].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m["padded_grad"]
            pg[: m["B"]].copy_(p.grad.bfloat16())
            if pg.shape[0] > m["B"]:
                pg[m["B"] :].zero_()
            fut = dist.reduce_scatter_tensor(
                m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True
            )
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            row_normalize = group.get("row_normalize", False)
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, "_rs_futures")
            for idx, m in enumerate(self._bank_meta):
                p = m["p"]
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m["p"]
                    upd = prev_m["full_update"][: prev_m["B"]]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
                if sharded and self._rs_futures[idx] is not None:
                    self._rs_futures[idx].wait()
                    g = m["shard"]
                    buf = m["shard_mom"]
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                if row_normalize:
                    rn = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                    update = update / rn.to(update.dtype)
                if update.shape[-2] != update.shape[-1] and max(update.shape[-2], update.shape[-1]) >= 3 * min(update.shape[-2], update.shape[-1]):
                    update = gram_newton_schulz5(update, steps=backend_steps)
                else:
                    update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m["full_update"], update.contiguous(), async_op=True
                    )
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m["p"]
                upd = prev_m["full_update"][: prev_m["B"]]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
            if hasattr(self, "_rs_futures"):
                del self._rs_futures
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,parallel_post_lambdas,parallel_resid_lambdas,attn_out_gate,smear_w,smear_lambda",
    ).split(",")
    if pattern
)


PACKED_REPLICATED_GRAD_MAX_NUMEL = 1 << 15


class Optimizers:
    def __init__(self, h, base_model):
        matrix_params = [
            base_model.q_bank,
            base_model.k_bank,
            base_model.v_bank,
            base_model.out_bank,
            base_model.mlp_up_bank,
            base_model.mlp_down_bank,
        ]
        block_named_params = list(base_model.blocks.named_parameters())
        scalar_params = [
            p
            for (name, p) in block_named_params
            if p.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if base_model.parallel_post_lambdas is not None:
            scalar_params.append(base_model.parallel_post_lambdas)
        if base_model.parallel_resid_lambdas is not None:
            scalar_params.append(base_model.parallel_resid_lambdas)
        if base_model.loop_log_A is not None:
            scalar_params.extend([base_model.loop_log_A, base_model.loop_delta, base_model.loop_B])
        # SmearGate parameters live on GPT (not in blocks), so the
        # block_named_params scan above doesn't see them. Add explicitly.
        if getattr(base_model, "smear_w", None) is not None:
            scalar_params.extend([base_model.smear_w, base_model.smear_lambda])
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [
            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
        ]
        self.optimizer_tok = torch.optim.AdamW(
            tok_params,
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=True,
        )
        scale_mode = auto_scale_mode(h.optim_mode, h.scale_mode)
        if h.precond_source == "act" and not h.capture_act:
            raise ValueError(
                "PRECOND_SOURCE=act requires CAPTURE_ACT=1 (otherwise R_ema "
                "stays zero and the preconditioner degenerates silently)."
            )
        if h.optim_mode == "muon":
            self.optimizer_muon = Muon(
                matrix_params,
                lr=h.matrix_lr,
                momentum=h.muon_momentum,
                backend_steps=h.muon_backend_steps,
                weight_decay=h.muon_wd,
                row_normalize=h.muon_row_normalize,
            )
        else:
            soap_cfg = {
                "beta1": h.soap_beta1,
                "beta2": h.soap_beta2,
                "eps": h.soap_eps,
                "damping": h.soap_damping,
                "base": h.soap_base,
                "beta_precond": h.soap_beta_precond,
                "refresh_k": h.soap_refresh_k,
                "refresh_adaptive": h.soap_refresh_adaptive,
                "refresh_drift_tau": h.soap_refresh_drift_tau,
                "inv_root_ns_steps": h.inv_root_ns_steps,
                "warmup_steps": h.soap_precond_warmup_steps,
                "precond_update_k": h.precond_update_k,
            }
            ns_cfg = {
                "adaptive": h.ns_adaptive,
                "adaptive_eps": h.ns_adaptive_eps,
                "two_phase": h.ns_two_phase,
                "refine_steps": h.ns_refine_steps,
                "warm_start": h.ns_warm_start,
            }
            psgd_cfg = {
                "precond_lr": h.psgd_kron_precond_lr,
            }
            # Resolve PRECOND_BANKS to a per-param bool list. Order matches
            # matrix_params above: q, k, v, out, mlp_up, mlp_down.
            bank_names = ["q", "k", "v", "out", "mlp_up", "mlp_down"]
            pb = h.precond_banks.strip().lower()
            if pb in ("all", ""):
                bank_use_precond = [True] * len(bank_names)
            elif pb == "mlp":
                bank_use_precond = [name.startswith("mlp") for name in bank_names]
            elif pb == "attn":
                bank_use_precond = [not name.startswith("mlp") for name in bank_names]
            else:
                allowed = {s.strip() for s in pb.split(",") if s.strip()}
                unknown = allowed - set(bank_names)
                if unknown:
                    raise ValueError(
                        f"PRECOND_BANKS contains unknown names {unknown}; "
                        f"valid: {bank_names} or keywords 'all'/'mlp'/'attn'"
                    )
                bank_use_precond = [name in allowed for name in bank_names]
            log(
                f"precond_banks:{h.precond_banks} -> "
                f"{dict(zip(bank_names, bank_use_precond))}"
            )
            self.optimizer_muon = BankOptimizer(
                matrix_params,
                lr=h.matrix_lr,
                momentum=h.muon_momentum,
                backend_steps=h.muon_backend_steps,
                weight_decay=h.muon_wd,
                row_normalize=h.muon_row_normalize,
                optim_mode=h.optim_mode,
                scale_mode=scale_mode,
                scale_d_ref=h.model_dim,
                cautious_mask=h.cautious_mask,
                precond_momentum=h.precond_momentum,
                precond_source=h.precond_source,
                soap_cfg=soap_cfg,
                ns_cfg=ns_cfg,
                psgd_cfg=psgd_cfg,
                bank_use_precond=bank_use_precond,
            )
            # Teach the optimizer which parameter is which bank so activation
            # capture can route covariances to the right precond state.
            if h.capture_act:
                self.optimizer_muon._build()
                self.optimizer_muon.set_bank_registry(
                    {
                        "q": base_model.q_bank,
                        "k": base_model.k_bank,
                        "v": base_model.v_bank,
                        "out": base_model.out_bank,
                        "mlp_up": base_model.mlp_up_bank,
                        "mlp_down": base_model.mlp_down_bank,
                    },
                    kv_tie_global=base_model.kv_tie_global,
                    local_v_idx=base_model.local_v_idx,
                    global_layer_set=base_model.global_layer_set,
                )
                self.optimizer_muon.set_capture_act(base_model, True)
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [
            self.optimizer_tok,
            self.optimizer_muon,
            self.optimizer_scalar,
        ]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [
                    {
                        "params": [base_model.lm_head.weight],
                        "lr": h.head_lr,
                        "base_lr": h.head_lr,
                    }
                ],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None
        self.replicated_params = list(tok_params[0]["params"])
        self.replicated_params.extend(scalar_params)
        if base_model.lm_head is not None:
            self.replicated_params.append(base_model.lm_head.weight)
        self.replicated_large_params = []
        self.replicated_packed_params = []
        for p in self.replicated_params:
            if p.numel() <= PACKED_REPLICATED_GRAD_MAX_NUMEL:
                self.replicated_packed_params.append(p)
            else:
                self.replicated_large_params.append(p)

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def _all_reduce_packed_grads(self):
        grads_by_key = collections.defaultdict(list)
        for p in self.replicated_packed_params:
            if p.grad is not None:
                grads_by_key[(p.grad.device, p.grad.dtype)].append(p.grad)
        for grads in grads_by_key.values():
            flat = torch.empty(
                sum(g.numel() for g in grads),
                device=grads[0].device,
                dtype=grads[0].dtype,
            )
            offset = 0
            for g in grads:
                n = g.numel()
                flat[offset : offset + n].copy_(g.contiguous().view(-1))
                offset += n
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            offset = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n

    def consume_act(self, base_model):
        """Push captured activation covariances into the BankOptimizer's
        preconditioner state. No-op for Muon mode, when CAPTURE_ACT is off,
        or when PRECOND_SOURCE=grad (activations would double-update R_ema
        on top of the gradient update otherwise).
        """
        opt = self.optimizer_muon
        if (isinstance(opt, BankOptimizer)
                and hasattr(opt, "_bank_by_name")
                and opt.precond_source == "act"):
            beta = opt.soap_cfg.get("beta_precond", 0.95)
            opt.consume_captured_activations(base_model, beta)

    def diag_log(self, log_fn):
        """Emit preconditioner diagnostic log lines (rank 0 only)."""
        opt = self.optimizer_muon
        if isinstance(opt, BankOptimizer):
            opt.diag_log(log_fn)

    def step(self, distributed=False):
        self.optimizer_muon.launch_reduce_scatters()
        if distributed:
            reduce_handles = [
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
                for p in self.replicated_large_params
                if p.grad is not None
            ]
            self._all_reduce_packed_grads()
            for handle in reduce_handles:
                handle.wait()
        self.optimizer_tok.step()
        self.optimizer_scalar.step()
        if self.optimizer_head is not None:
            self.optimizer_head.step()
        self.optimizer_muon.step()
        self.zero_grad_all()


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (
            param.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ) and param.dtype != torch.float32:
            param.data = param.data.float()
    if hasattr(model, "q_bank"):
        model.q_bank.data = model.q_bank.data.float()
        model.k_bank.data = model.k_bank.data.float()
        model.v_bank.data = model.v_bank.data.float()
        model.out_bank.data = model.out_bank.data.float()
        model.mlp_up_bank.data = model.mlp_up_bank.data.float()
        model.mlp_down_bank.data = model.mlp_down_bank.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []
    global_set = getattr(model, "global_layer_set", set())
    kv_tie = getattr(model, "kv_tie_global", False)
    for i, block in enumerate(model.blocks):
        block.attn._calib = True
        block.mlp._calib = True
        block.mlp.use_fused = False

    def make_attn_hook(layer_idx):
        # Skip c_v Hessian for global K=V layers — their V = K, so GPTQ on c_v is wasted
        skip_cv = kv_tie and layer_idx in global_set
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            for suffix in ["c_q", "c_k", "c_v"]:
                if suffix == "c_v" and skip_cv:
                    continue
                name = f"blocks.{layer_idx}.attn.{suffix}.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            y = module._last_proj_input
            if y is not None:
                y = y.float()
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                name = f"blocks.{layer_idx}.attn.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        y.shape[1], y.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(y.T, y)
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            name = f"blocks.{layer_idx}.mlp.fc.weight"
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[name].addmm_(x.T, x)
            h_act = module._last_down_input
            if h_act is not None:
                h_act = h_act.float()
                if h_act.ndim == 3:
                    h_act = h_act.reshape(-1, h_act.shape[-1])
                name = f"blocks.{layer_idx}.mlp.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        h_act.shape[1], h_act.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(h_act.T, h_act)
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(i)))
    if model.tie_embeddings:
        hook_module = (
            model.head_proj if model.head_proj is not None else model.final_norm
        )

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            return hook_fn

        hooks.append(
            hook_module.register_forward_hook(make_output_hook("tok_emb.weight"))
        )
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for i, block in enumerate(model.blocks):
        block.attn._calib = False
        block.mlp._calib = False
        block.mlp.use_fused = True
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    for (name, tensor) in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        if "tok_emb" in name:
            cs = h.embed_clip_sigmas
        elif ".mlp." in name:
            cs = h.mlp_clip_sigmas
        elif ".attn." in name:
            cs = h.attn_clip_sigmas
        else:
            cs = h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        q, s = gptq_quantize_weight(
            t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1
        )
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for (name, cat) in meta.items():
        short = re.sub("\\.\\d+$", "", re.sub("blocks\\.\\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for (name, orig) in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (
                q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
            ).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


_BSHF_MAGIC = b"BSHF"


def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off : dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off : src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli

        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _decompress(data, compressor):
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli

        raw = brotli.decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    raw = _byte_unshuffle(raw)
    return raw


def _unbank_state_dict(state_dict, num_layers, global_layer_set=None, local_v_idx=None, kv_tie=False):
    """Convert banked weights to flat per-layer state dict.

    For K=V tied global layers, c_v.weight is omitted entirely — the K weights
    serve double duty and only need to be serialized/quantized once as c_k.
    """
    sd = {}
    n = num_layers
    for k, v in state_dict.items():
        t = v.detach().cpu()
        if k == "q_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_q.weight"] = t[i]
        elif k == "k_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_k.weight"] = t[i]
        elif k == "v_bank":
            if kv_tie and local_v_idx is not None:
                # v_bank only has local layer entries; global layers have no c_v
                for layer_i, v_idx in local_v_idx.items():
                    sd[f"blocks.{layer_i}.attn.c_v.weight"] = t[v_idx]
            else:
                for i in range(n):
                    sd[f"blocks.{i}.attn.c_v.weight"] = t[i]
        elif k == "out_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.proj.weight"] = t[i]
        elif k == "mlp_up_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.fc.weight"] = t[i]
        elif k == "mlp_down_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.proj.weight"] = t[i]
        else:
            sd[k] = t
    return sd


def _rebank_state_dict(flat_sd, num_layers, model_dim, kv_dim, hidden_dim,
                        global_layer_set=None, local_v_idx=None, kv_tie=False):
    sd = {}
    n = num_layers
    sd["q_bank"] = torch.zeros(n, model_dim, model_dim)
    sd["k_bank"] = torch.zeros(n, kv_dim, model_dim)
    n_v = len(local_v_idx) if (kv_tie and local_v_idx) else n
    sd["v_bank"] = torch.zeros(n_v, kv_dim, model_dim)
    sd["out_bank"] = torch.zeros(n, model_dim, model_dim)
    sd["mlp_up_bank"] = torch.zeros(n, hidden_dim, model_dim)
    sd["mlp_down_bank"] = torch.zeros(n, model_dim, hidden_dim)
    for i in range(n):
        sd["q_bank"][i] = flat_sd[f"blocks.{i}.attn.c_q.weight"]
        sd["k_bank"][i] = flat_sd[f"blocks.{i}.attn.c_k.weight"]
        if kv_tie and local_v_idx is not None:
            if i in local_v_idx:
                sd["v_bank"][local_v_idx[i]] = flat_sd[f"blocks.{i}.attn.c_v.weight"]
            # global layers: V = K, nothing to store in v_bank
        else:
            sd["v_bank"][i] = flat_sd[f"blocks.{i}.attn.c_v.weight"]
        sd["out_bank"][i] = flat_sd[f"blocks.{i}.attn.proj.weight"]
        sd["mlp_up_bank"][i] = flat_sd[f"blocks.{i}.mlp.fc.weight"]
        sd["mlp_down_bank"][i] = flat_sd[f"blocks.{i}.mlp.proj.weight"]
    for k, v in flat_sd.items():
        if not (
            k.startswith("blocks.")
            and any(
                p in k
                for p in [
                    ".attn.c_q.", ".attn.c_k.", ".attn.c_v.",
                    ".attn.proj.", ".mlp.fc.", ".mlp.proj.",
                ]
            )
        ):
            sd[k] = v
    return sd


def _compressed_code_size(code):
    code_raw = code.encode("utf-8")
    minified = subprocess.run(
        ["pyminify", "--no-rename-locals", "--no-hoist-literals", "--remove-literal-statements", "-"],
        input=code_raw, capture_output=True, check=True,
    ).stdout
    compressed = lzma.compress(minified)
    encoded = base64.b85encode(compressed)
    wrapper = b'import lzma as L,base64 as B\nexec(L.decompress(B.b85decode("' + encoded + b'")))\n'
    return len(code_raw), len(wrapper)


def serialize(h, base_model, code):
    code_bytes_uncompressed, code_bytes = _compressed_code_size(code)
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size (uncompressed): {code_bytes_uncompressed} bytes")
        log(f"Code size (compressed): {code_bytes} bytes")
    sd_cpu = _unbank_state_dict(base_model.state_dict(), h.num_layers,
                                global_layer_set=base_model.global_layer_set,
                                local_v_idx=base_model.local_v_idx,
                                kv_tie=base_model.kv_tie_global)
    device = torch.device("cuda", h.local_rank)
    log("GPTQ:collecting Hessians from calibration data...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    flat_template = _unbank_state_dict(eval_model.state_dict(), h.num_layers,
                                       global_layer_set=eval_model.global_layer_set,
                                       local_v_idx=eval_model.local_v_idx,
                                       kv_tie=eval_model.kv_tie_global)
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu"
    )
    deq_flat = dequantize_mixed(quant_state["w"], quant_state["m"], flat_template)
    head_dim = h.model_dim // h.num_heads
    kv_dim = h.num_kv_heads * head_dim
    hidden_dim = int(h.mlp_mult * h.model_dim)
    deq_state = _rebank_state_dict(deq_flat, h.num_layers, h.model_dim, kv_dim, hidden_dim,
                                    global_layer_set=eval_model.global_layer_set,
                                    local_v_idx=eval_model.local_v_idx,
                                    kv_tie=eval_model.kv_tie_global)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val(h, device, val_data, model, forward_logits_fn=None):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size

    # TODO: Don't truncate this.
    seq_end = seq_start + ((seq_end - seq_start) // local_batch_seqs) * local_batch_seqs

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    run_forward_logits = (
        (model.module.forward_logits if hasattr(model, "module") else model.forward_logits)
        if forward_logits_fn is None
        else forward_logits_fn
    )
    model.eval()
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1]
            y = local[1:]
            bos_pos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = _build_cu_seqlens(
                bos_pos, x.numel(), x.device, h.eval_seq_len, 64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = run_forward_logits(
                    x[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                ).detach()
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="none",
            )
            val_loss_sum += per_token_loss.to(torch.float64).sum()
            val_token_count += float(y.numel())
            prev_ids = x
            tgt_ids = y
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                val_data.has_leading_space_lut[tgt_ids]
                & ~val_data.is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(h, device, val_data, base_model, forward_logits_fn=None, batch_seqs=32):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    run_forward_logits = base_model.forward_logits if forward_logits_fn is None else forward_logits_fn
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = (total_windows * h.rank) // h.world_size
    my_e = (total_windows * (h.rank + 1)) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    total_batches = (len(my_windows) + batch_seqs - 1) // batch_seqs
    is_master = h.rank == 0
    cu_bucket = 64
    t_sw_start = time.perf_counter()
    with torch.no_grad():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_idx = bi // batch_seqs
            if is_master and (batch_idx % 50 == 0 or batch_idx == total_batches - 1):
                elapsed = time.perf_counter() - t_sw_start
                rl = float(loss_sum.item() / token_count.item()) if token_count.item() > 0 else 0.0
                rb = float((rl / math.log(2.0)) * token_count.item() / byte_count.item()) if byte_count.item() > 0 else 0.0
                log(f"sliding_progress: batch {batch_idx+1}/{total_batches} "
                    f"tokens:{int(token_count.item())} running_loss:{rl:.4f} running_bpb:{rb:.4f} "
                    f"elapsed:{elapsed:.1f}s")
            batch_ws = my_windows[bi:bi + batch_seqs]
            x_parts = []
            y_parts = []
            cu_starts = []
            score_ranges = []
            offset = 0
            for ws in batch_ws:
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                chunk_cpu = val_data.val_tokens[ws:end + 1]
                bos_pos = (chunk_cpu[:-1] == BOS_ID).nonzero(as_tuple=True)[0].tolist()
                if not bos_pos or bos_pos[0] != 0:
                    bos_pos = [0] + bos_pos
                cu_starts.extend(offset + pos for pos in bos_pos)
                chunk = chunk_cpu.to(dtype=torch.int64, device=device)
                x_parts.append(chunk[:-1])
                y_parts.append(chunk[1:])
                score_ranges.append((offset, wlen, ws))
                offset += wlen
            x_cat = torch.cat(x_parts, dim=0)[None]
            y_cat = torch.cat(y_parts, dim=0)
            boundaries = cu_starts + [offset]
            padded_len = get_next_multiple_of_n(len(boundaries), cu_bucket)
            cu_seqlens = torch.full((padded_len,), offset, dtype=torch.int32, device=device)
            cu_seqlens[:len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward_logits(x_cat, cu_seqlens=cu_seqlens, max_seqlen=seq_len)
            flat_nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_cat,
                reduction="none",
            )
            flat_x = x_cat.reshape(-1)
            for off, wlen, ws in score_ranges:
                s = 0 if ws == 0 else context_size
                lo = off + s
                hi = off + wlen
                scored_nll = flat_nll[lo:hi].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(hi - lo)
                tgt = y_cat[lo:hi]
                prev = flat_x[lo:hi]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def _find_docs(all_tokens):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = (
            int(bos_positions[i + 1])
            if i + 1 < len(bos_positions)
            else all_tokens.numel()
        )
        if i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


def _build_ttt_global_batches(doc_entries, h, ascending=False):
    batch_size = h.ttt_batch_size
    global_doc_entries = sorted(doc_entries, key=lambda x: x[1][1])
    global_batches = [
        global_doc_entries[i : i + batch_size]
        for i in range(0, len(global_doc_entries), batch_size)
    ]
    indexed = list(enumerate(global_batches))
    if not ascending:
        indexed.sort(key=lambda ib: -max(dl for _, (_, dl) in ib[1]))
    return indexed


def _init_batch_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(4, "little"))


def _claim_next_batch(counter_path, queue_len):
    try:
        with open(counter_path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            idx = int.from_bytes(f.read(4), "little")
            f.seek(0)
            f.write((idx + 1).to_bytes(4, "little"))
            f.flush()
    except FileNotFoundError:
        return queue_len
    return idx


def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_start = ci * chunk_size
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _accumulate_bpb(
    ptl,
    x,
    y,
    chunk_offsets,
    chunk_lens,
    pos_idx,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    loss_sum,
    byte_sum,
    token_count,
):
    pos = pos_idx[: x.size(1)].unsqueeze(0)
    mask = (
        (chunk_lens.unsqueeze(1) > 0)
        & (pos >= chunk_offsets.unsqueeze(1))
        & (pos < (chunk_offsets + chunk_lens).unsqueeze(1))
    )
    mask_f64 = mask.to(torch.float64)
    tok_bytes = base_bytes_lut[y].to(torch.float64)
    tok_bytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(
        torch.float64
    )
    loss_sum += (ptl.to(torch.float64) * mask_f64).sum()
    byte_sum += (tok_bytes * mask_f64).sum()
    token_count += chunk_lens.to(torch.float64).sum()

def _split_doc_entries_for_phased(doc_entries, prefix_docs):
    prefix_docs = max(0, min(len(doc_entries), int(prefix_docs)))
    return doc_entries[:prefix_docs], doc_entries[prefix_docs:]


def _add_to_counter(path, delta):
    try:
        with open(path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            cur = int.from_bytes(f.read(8), "little", signed=True)
            cur += int(delta)
            f.seek(0)
            f.write(int(cur).to_bytes(8, "little", signed=True))
            f.flush()
            return cur
    except FileNotFoundError:
        return int(delta)


def _init_int64_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(8, "little", signed=True))


def _select_ttt_doc_entries(docs, h):
    doc_entries = list(enumerate(docs))
    if h.val_doc_fraction < 1.0:
        sample_n = max(1, int(round(len(docs) * h.val_doc_fraction)))
        sampled_indices = sorted(
            random.Random(h.seed).sample(range(len(docs)), sample_n)
        )
        return [(i, docs[i]) for i in sampled_indices]
    return doc_entries


def _loss_bpb_from_sums(loss_sum, token_count, byte_sum):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())
    return val_loss, val_bpb


def train_val_ttt_global_sgd_distributed(h, device, val_data, base_model, val_tokens, batch_seqs=None):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    seq_len = h.eval_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = h.global_ttt_chunk_tokens
    batch_seqs = h.global_ttt_batch_seqs if batch_seqs is None else batch_seqs
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        ttt_params, lr=h.global_ttt_lr, momentum=h.global_ttt_momentum
    )
    t_start = time.perf_counter()
    for ci in range(num_chunks):
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        is_last_chunk = ci == num_chunks - 1
        if is_last_chunk or h.global_ttt_epochs <= 0:
            continue
        base_model.train()
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs <= 0:
            continue
        warmup_chunks = max(0, min(h.global_ttt_warmup_chunks, num_chunks - 1))
        if warmup_chunks > 0 and ci < warmup_chunks:
            warmup_denom = max(warmup_chunks - 1, 1)
            warmup_t = ci / warmup_denom
            lr_now = (
                h.global_ttt_warmup_start_lr
                + (h.global_ttt_lr - h.global_ttt_warmup_start_lr) * warmup_t
            )
        else:
            decay_steps = max(num_chunks - 1 - warmup_chunks, 1)
            decay_ci = max(ci - warmup_chunks, 0)
            lr_now = h.global_ttt_lr * 0.5 * (
                1.0 + math.cos(math.pi * decay_ci / decay_steps)
            )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        my_seq_s = chunk_seqs * h.rank // h.world_size
        my_seq_e = chunk_seqs * (h.rank + 1) // h.world_size
        my_chunk_seqs = my_seq_e - my_seq_s
        for _ in range(h.global_ttt_epochs):
            for bs in range(0, my_chunk_seqs, batch_seqs):
                be = min(bs + batch_seqs, my_chunk_seqs)
                actual_bs = my_seq_s + bs
                start_tok = chunk_start + actual_bs * seq_len
                end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                if end_tok > val_tokens.numel():
                    continue
                local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                x_flat = local[:-1]
                y_flat = local[1:]
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if h.global_ttt_respect_doc_boundaries:
                            bos_pos = (x_flat == BOS_ID).nonzero(as_tuple=True)[0].tolist()
                            cu_seqlens, max_seqlen = _build_cu_seqlens(
                                bos_pos, x_flat.numel(), x_flat.device, h.eval_seq_len, 64
                            )
                            loss = base_model(
                                x_flat[None],
                                y_flat[None],
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen,
                            )
                        else:
                            x = x_flat.reshape(-1, seq_len)
                            y = y_flat.reshape(-1, seq_len)
                            loss = base_model(x, y)
                loss.backward()
                if dist.is_available() and dist.is_initialized():
                    for p in ttt_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.mul_(1.0 / h.world_size)
                if h.global_ttt_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ttt_params, h.global_ttt_grad_clip)
                optimizer.step()
        base_model.eval()
        if h.rank == 0:
            elapsed = time.perf_counter() - t_start
            log(
                f"tttg: c{ci+1}/{num_chunks} lr:{lr_now:.6f} t:{elapsed:.1f}s"
            )
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()


def eval_val_ttt_phased(h, base_model, device, val_data, forward_ttt_train):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    all_tokens = val_data.val_tokens
    all_tokens_idx = all_tokens.to(torch.int32)
    docs = _find_docs(all_tokens)
    doc_entries = _select_ttt_doc_entries(docs, h)
    prefix_doc_limit = max(0, min(len(doc_entries), int(h.phased_ttt_prefix_docs)))
    num_phases = max(1, int(h.phased_ttt_num_phases))
    phase_boundaries = []
    for pi in range(num_phases):
        boundary = prefix_doc_limit * (pi + 1) // num_phases
        phase_boundaries.append(boundary)
    current_phase = 0
    current_phase_boundary = phase_boundaries[0]
    log(
        "ttt_phased:"
        f" total_docs:{len(doc_entries)} prefix_docs:{prefix_doc_limit} "
        f"suffix_docs:{len(doc_entries) - prefix_doc_limit}"
        f" num_phases:{num_phases} boundaries:{phase_boundaries}"
    )
    chunk_size, eval_seq_len = h.ttt_chunk_size, h.ttt_eval_seq_len
    eval_batch_set = None
    if h.ttt_eval_batches:
        eval_batch_set = set(int(x) for x in h.ttt_eval_batches.split(",") if x.strip())
    use_ascending = eval_batch_set is not None
    global_batches_sorted = _build_ttt_global_batches(
        doc_entries, h, ascending=use_ascending
    )
    queue_len = len(global_batches_sorted)
    counter_path = f"/tmp/ttt_counter_{h.run_id}"
    prefix_counter_path = f"/tmp/ttt_prefix_counter_{h.run_id}"
    pause_flag_path = f"/tmp/ttt_pause_flag_{h.run_id}"
    if h.rank == 0:
        _init_batch_counter(counter_path)
        _init_int64_counter(prefix_counter_path)
        try:
            os.remove(pause_flag_path)
        except FileNotFoundError:
            pass
    if dist.is_available() and dist.is_initialized():
        path_list = [counter_path, prefix_counter_path, pause_flag_path]
        dist.broadcast_object_list(path_list, src=0)
        counter_path, prefix_counter_path, pause_flag_path = path_list
        dist.barrier()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    t_start = time.perf_counter()
    reusable_lora = BatchedTTTLoRA(
        h.ttt_batch_size, base_model, h.ttt_lora_rank,
        k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
    ).to(device)

    def _build_opt(lora):
        # LoRA+ (eta multiplies LR on B matrices only; eta=1.0 disables it)
        # and per-layer LR slope (alpha > 0 gives later virtual layers higher
        # LR; alpha=0.0 disables it). Ported from PR #1695 @X-Abhishek-X,
        # whose submission uses alpha=0.5 and eta=1.0.
        eta = h.lora_plus_ratio
        alpha = h.ttt_lora_layer_lr_alpha
        num_slots = max(len(lora.q_loras), 1)
        param_groups = []
        for pname, p in lora.named_parameters():
            # Parse layer index from "q_loras.3.A" style names; params without
            # a numeric component (e.g. lm_head_lora.A) default to last layer.
            m = re.search(r"\.(\d+)\.", pname)
            layer_idx = int(m.group(1)) if m else num_slots - 1
            layer_scale = 1.0 + alpha * (layer_idx / max(num_slots - 1, 1))
            eta_mult = eta if pname.endswith(".B") else 1.0
            param_groups.append(
                {"params": [p], "lr": h.ttt_lora_lr * layer_scale * eta_mult}
            )
        if h.ttt_optimizer == "sgd":
            return torch.optim.SGD(
                param_groups,
                momentum=h.ttt_beta1, weight_decay=h.ttt_weight_decay,
            )
        return torch.optim.AdamW(
            param_groups,
            betas=(h.ttt_beta1, h.ttt_beta2),
            eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True,
        )

    reusable_opt = _build_opt(reusable_lora)
    local_scored_docs = []
    global_ttt_done = prefix_doc_limit == 0
    try:
      while True:
        queue_idx = _claim_next_batch(counter_path, queue_len)
        if queue_idx >= queue_len:
            break
        orig_batch_idx, batch_entries = global_batches_sorted[queue_idx]
        batch = [doc for _, doc in batch_entries]
        bsz = len(batch)
        prev_loss = loss_sum.item()
        prev_bytes = byte_sum.item()
        prev_tokens = token_count.item()
        if bsz == reusable_lora.bsz:
            reusable_lora.reset()
            for s in reusable_opt.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        v.zero_()
                    elif k == "step":
                        s[k] = 0
            cur_lora = reusable_lora
            cur_opt = reusable_opt
        else:
            cur_lora = BatchedTTTLoRA(
                bsz, base_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            cur_opt = _build_opt(cur_lora)
        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        num_chunks_t = torch.tensor(num_chunks, dtype=torch.int64, device=device)
        for ci in range(max_nc):
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)
            tok_starts = torch.zeros(bsz, dtype=torch.int64)
            tok_wls = torch.zeros(bsz, dtype=torch.int64)
            chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
            chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
            for b in range(bsz):
                if not active[b]:
                    continue
                doc_start, doc_len = batch[b]
                win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(
                    ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len
                )
                tok_starts[b] = doc_start + win_start
                tok_wls[b] = win_len
                chunk_offsets_cpu[b] = chunk_offset
                chunk_lens_cpu[b] = chunk_len
            _, context_size, chunk_offset, _ = _compute_chunk_window(
                ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len
            )
            col_idx = torch.arange(context_size + 1)
            idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
            idx.clamp_(max=all_tokens.numel() - 1)
            gathered_gpu = all_tokens_idx[idx].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(
                device, non_blocking=True
            )
            chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)
            chunk_lens = chunk_lens_cpu.to(device, non_blocking=True)
            x = torch.where(valid, gathered_gpu[:, :context_size], 0)
            y = torch.where(valid, gathered_gpu[:, 1 : context_size + 1], 0)
            ctx_pos = torch.arange(context_size, device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
            with torch.no_grad():
                _accumulate_bpb(
                    per_tok_loss,
                    x,
                    y,
                    chunk_offsets,
                    chunk_lens,
                    ctx_pos,
                    val_data.base_bytes_lut,
                    val_data.has_leading_space_lut,
                    val_data.is_boundary_token_lut,
                    loss_sum,
                    byte_sum,
                    token_count,
                )
            if needs_train:
                activate_chunk_mask = (num_chunks_t - 1 > ci).float()
                for gi in range(h.ttt_grad_steps):
                    if gi > 0:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                    per_doc = per_tok_loss[
                        :, chunk_offset : chunk_offset + chunk_size
                    ].mean(dim=-1)
                    cur_opt.zero_grad(set_to_none=True)
                    (per_doc * activate_chunk_mask).sum().backward()
                    cur_opt.step()
            else:
                del per_tok_loss
        batch_num = orig_batch_idx + 1
        doc_lens = [dl for _, dl in batch]
        should_report = batch_num in eval_batch_set if eval_batch_set is not None else True
        if should_report:
            cur_tokens = token_count.item()
            cur_loss_val = loss_sum.item()
            cur_bytes_val = byte_sum.item()
            dt = cur_tokens - prev_tokens
            db = cur_bytes_val - prev_bytes
            if dt > 0 and db > 0:
                b_loss = (cur_loss_val - prev_loss) / dt
                b_bpb = b_loss / math.log(2.0) * (dt / db)
            else:
                b_loss = b_bpb = 0.0
            r_loss = cur_loss_val / max(cur_tokens, 1)
            r_bpb = r_loss / math.log(2.0) * (cur_tokens / max(cur_bytes_val, 1))
            elapsed = time.perf_counter() - t_start
            log(
                f"ttp: b{batch_num}/{queue_len} bl:{b_loss:.4f} bb:{b_bpb:.4f} "
                f"rl:{r_loss:.4f} rb:{r_bpb:.4f} dl:{min(doc_lens)}-{max(doc_lens)} "
                f"gd:{int(global_ttt_done)}"
            )
        if not global_ttt_done:
            local_scored_docs.extend(
                (orig_batch_idx, pos, doc_start, doc_len)
                for pos, (doc_start, doc_len) in enumerate(batch)
            )
            prefix_done = _add_to_counter(prefix_counter_path, len(batch_entries))
            if prefix_done >= current_phase_boundary:
                try:
                    with open(pause_flag_path, "x"):
                        pass
                except FileExistsError:
                    pass
            should_pause = os.path.exists(pause_flag_path)
            if should_pause:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                gathered_scored_docs = [None] * h.world_size
                if dist.is_available() and dist.is_initialized():
                    dist.all_gather_object(gathered_scored_docs, local_scored_docs)
                else:
                    gathered_scored_docs = [local_scored_docs]
                scored_docs_for_global = []
                for rank_docs in gathered_scored_docs:
                    if rank_docs:
                        scored_docs_for_global.extend(rank_docs)
                scored_docs_for_global.sort(key=lambda x: (x[0], x[1]))
                scored_docs_for_global = scored_docs_for_global[:current_phase_boundary]
                scored_token_chunks = [
                    val_data.val_tokens[doc_start : doc_start + doc_len]
                    for _, _, doc_start, doc_len in scored_docs_for_global
                ]
                if scored_token_chunks:
                    global_ttt_tokens = torch.cat(scored_token_chunks)
                else:
                    global_ttt_tokens = val_data.val_tokens[:0]
                if h.rank == 0:
                    prefix_done = 0
                    try:
                        with open(prefix_counter_path, "rb") as f:
                            prefix_done = int.from_bytes(
                                f.read(8), "little", signed=True
                            )
                    except FileNotFoundError:
                        pass
                    log(
                        f"ttpp: phase:{current_phase + 1}/{num_phases} pd:{prefix_done} "
                        f"gd:{len(scored_docs_for_global)} "
                        f"t:{time.perf_counter() - t_start:.1f}s"
                    )
                train_val_ttt_global_sgd_distributed(
                    h, device, val_data, base_model, global_ttt_tokens
                )
                for p in base_model.parameters():
                    p.requires_grad_(False)
                reusable_lora = BatchedTTTLoRA(
                    h.ttt_batch_size, base_model, h.ttt_lora_rank,
                    k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
                ).to(device)
                reusable_opt = _build_opt(reusable_lora)
                current_phase += 1
                if current_phase >= num_phases:
                    global_ttt_done = True
                else:
                    current_phase_boundary = phase_boundaries[current_phase]
                    if h.rank == 0:
                        try:
                            os.remove(pause_flag_path)
                        except FileNotFoundError:
                            pass
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                if h.rank == 0:
                    log(f"ttpr: phase:{current_phase}/{num_phases} t:{time.perf_counter() - t_start:.1f}s")
        del cur_lora, cur_opt
    finally:
        pass
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.train()
    return _loss_bpb_from_sums(loss_sum, token_count, byte_sum)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(
        f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms"
    )
    return val_loss, val_bpb


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(
        base_model.forward_logits, dynamic=False, fullgraph=True
    )
    model = compiled_model
    log(f"model_params:{sum(p.numel()for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)
    train_loader = DocumentPackingLoader(h, device)
    max_wallclock_ms = (
        1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    )
    # In skip_post_train mode (ablation) we use the full wallclock budget for
    # training — no GPTQ reserve needed, and rank-drift safety is provided by
    # the top-of-loop collective wallclock check + NCCL_WATCHDOG_TIMEOUT_S
    # bumped to 1800s. In normal mode we still reserve gptq_reserve_seconds
    # for the post-training GPTQ pass.
    if max_wallclock_ms is not None and not h.skip_post_train:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        log(
            f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, "
            f"effective={max_wallclock_ms:.0f}ms"
        )
    elif max_wallclock_ms is not None:
        log(
            f"skip_post_train: using full {max_wallclock_ms:.0f}ms for training"
        )

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    perf_stats = {
        "steps": 0,
        "data_ms": 0.0,
        "compute_ms": 0.0,
        "optim_ms": 0.0,
    }

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        data_ms = 0.0
        compute_ms = 0.0
        for micro_step in range(h.grad_accum_steps):
            t_data = time.perf_counter()
            x, y, cu_seqlens, _max_seqlen = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            data_ms += 1e3 * (time.perf_counter() - t_data)
            t_compute = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, cu_seqlens=cu_seqlens, max_seqlen=h.train_seq_len)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
            compute_ms += 1e3 * (time.perf_counter() - t_compute)
        train_loss /= h.grad_accum_steps
        frac = (
            min(step / h.muon_momentum_warmup_steps, 1.0)
            if h.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        if h.capture_act:
            optimizers.consume_act(base_model)
        t_optim = time.perf_counter()
        optimizers.step(distributed=h.distributed)
        optim_ms = 1e3 * (time.perf_counter() - t_optim)
        if h.diag_log_every > 0 and (step + 1) % h.diag_log_every == 0:
            optimizers.diag_log(lambda s: log(s, console=False))
        perf_stats["steps"] += 1
        perf_stats["data_ms"] += data_ms
        perf_stats["compute_ms"] += compute_ms
        perf_stats["optim_ms"] += optim_ms
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for (name, tensor) in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        num_tokens_local = h.train_batch_tokens // h.world_size
        for blk in base_model.blocks:
            blk.attn.rotary(num_tokens_local, device, torch.bfloat16)
        cu_bucket_size = train_loader.cu_bucket_size
        warmup_cu_buckets = tuple(cu_bucket_size * i for i in range(1, 5))
        warmup_cu_iters = 3
        x, y, cu_seqlens, _ = train_loader.next_batch(
            h.train_batch_tokens, h.grad_accum_steps
        )
        log(f"warmup_cu_buckets:{','.join(str(b) for b in warmup_cu_buckets)} iters_each:{warmup_cu_iters}")
        def _run_cu_bucket_warmup():
            for bucket_len in warmup_cu_buckets:
                boundaries = list(range(0, x.size(1), max(h.train_seq_len, 1)))
                if boundaries[-1] != x.size(1):
                    boundaries.append(x.size(1))
                cu = torch.full((bucket_len,), x.size(1), dtype=torch.int32, device=device)
                cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
                for _ in range(warmup_cu_iters):
                    optimizers.zero_grad_all()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        wloss = model(x, y, cu_seqlens=cu, max_seqlen=h.train_seq_len)
                    (wloss / h.grad_accum_steps).backward()
            optimizers.zero_grad_all()
        _run_cu_bucket_warmup()
        if h.num_loops > 0:
            base_model.looping_active = True
            _run_cu_bucket_warmup()
            base_model.looping_active = False
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if (
                warmup_step <= 5
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == h.warmup_steps
            ):
                log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if (
                    warmup_step <= 5
                    or (warmup_step + 1) % 10 == 0
                    or warmup_step + 1 == h.warmup_steps
                ):
                    log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
            base_model.looping_active = False
        # Force one refresh per bank BEFORE state reset so the refresh-only
        # torch.compile paths (_rotate_eigen_state, matrix_inv_sqrt_ns,
        # eigh-inv-sqrt) get compiled during warmup instead of burning 5-30s
        # of the 10-min budget at step ~500 of the main loop.
        if isinstance(optimizers.optimizer_muon, BankOptimizer):
            log("compile_warmup: triggering refresh to compile refresh paths")
            optimizers.optimizer_muon.compile_warmup_refresh()
        base_model.load_state_dict(initial_model_state, strict=True)
        for (opt, state) in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        train_loader = DocumentPackingLoader(h, device)
    ema_state = {
        name: t.detach().float().clone()
        for (name, t) in base_model.state_dict().items()
    }
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    wallclock_t0 = time.perf_counter()
    t0 = time.perf_counter()
    step = 0
    while True:
        # Collective wallclock check AT TOP of loop. Puts a sync point BEFORE
        # step_fn so ranks can never enter step_fn after one rank has already
        # exited — eliminates a whole class of "rank X done, rank Y still
        # queuing NCCL collectives → 600s NCCL watchdog timeout" crashes.
        if h.distributed and max_wallclock_ms is not None and stop_after_step is None:
            elapsed_probe_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
            reached_probe = torch.tensor(
                int(elapsed_probe_ms >= max_wallclock_ms),
                device=device, dtype=torch.int32,
            )
            dist.all_reduce(reached_probe, op=dist.ReduceOp.MAX)
            if bool(reached_probe.item()):
                stop_after_step = step
        last_step = (
            step == h.iterations
            or stop_after_step is not None
            and step >= stop_after_step
        )
        should_validate = (
            last_step or h.val_loss_every > 0 and step % h.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                h, device, val_data, model, compiled_forward_logits
            )
            log(
                f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        # Broadcast rank 0's frac to all ranks. `elapsed_ms` is per-rank
        # from time.perf_counter(), so frac diverges across ranks over long
        # runs. Unsynced frac causes two bugs:
        #   1. Per-rank LR drift (group["lr"] = base_lr * lr_scale) → params
        #      diverge across ranks → EMA and val_loss diverge → reproducibility
        #      issues and subtle training-quality degradation.
        #   2. Rank-local threshold checks (e.g. `frac >= enable_looping_at`)
        #      fire on different iterations across ranks → forward produces
        #      different-shaped tensors → NCCL collective mismatch → hang.
        # Syncing frac once here fixes both classes of bug and makes the
        # whole time-dependent schedule deterministic across ranks.
        if h.distributed:
            frac_tensor = torch.tensor([frac], device=device, dtype=torch.float32)
            dist.broadcast(frac_tensor, src=0)
            frac = frac_tensor.item()
        scale = lr_mul(frac)
        if (
            h.num_loops > 0
            and not base_model.looping_active
            and frac >= h.enable_looping_at
        ):
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for (name, t) in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(
                    t.detach().float(), alpha=1.0 - ema_decay
                )
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (
            step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            train_loop_tok_s = step * h.train_batch_tokens / max(
                approx_training_time_ms / 1e3, 1e-9
            )
            wallclock_training_time_ms = 1e3 * (time.perf_counter() - wallclock_t0)
            effective_tok_s_wallclock = step * h.train_batch_tokens / max(
                wallclock_training_time_ms / 1e3, 1e-9
            )
            bucket_total_ms = (
                perf_stats["data_ms"] + perf_stats["compute_ms"] + perf_stats["optim_ms"]
            )
            if bucket_total_ms > 0:
                data_wait_pct = 100.0 * perf_stats["data_ms"] / bucket_total_ms
                comm_pct = 100.0 * perf_stats["optim_ms"] / bucket_total_ms
            else:
                data_wait_pct = 0.0
                comm_pct = 0.0
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} "
                f"train_time: {approx_training_time_ms/60000:.1f}m "
                f"train_loop_tok/s: {train_loop_tok_s:.0f} "
                f"effective_tok/s_wallclock: {effective_tok_s_wallclock:.0f} "
                f"data_wait_pct: {data_wait_pct:.1f}% comm_pct: {comm_pct:.1f}%"
            )
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {
        name: t.to(dtype=current_state[name].dtype) for (name, t) in ema_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
    log_qk_gain_converged(log, base_model)
    log_parcae_converged(log, base_model)
    return base_model, compiled_model, compiled_forward_logits


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    if h.artifact_dir and h.is_main_process:
        os.makedirs(h.artifact_dir, exist_ok=True)
    val_data = ValidationData(h, device)
    if h.eval_only_path:
        log(f"eval_only:loading checkpoint from {h.eval_only_path}")
        base_model = GPT(h).to(device).bfloat16()
        restore_fp32_params(base_model)
        base_model.load_state_dict(torch.load(h.eval_only_path, map_location=device))
        if h.num_loops > 0:
            base_model.looping_active = True
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
        compiled_forward_logits = torch.compile(
            base_model.forward_logits, dynamic=False, fullgraph=True
        )
    else:
        log(
            f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}"
        )
        log(f"val_tokens: {val_data.val_tokens.numel()-1}")
        base_model, compiled_model, compiled_forward_logits = train_model(
            h, device, val_data
        )
    _skip_training = bool(h.eval_only_path)
    torch._dynamo.reset()
    # Pre-quant diagnostic: skipped on submission runs unless DIAG_EVALS_ENABLED=1.
    # Forced on for skip_post_train (ablation mode), since it's the only metric
    # ablations log.
    if h.diag_evals_enabled or h.skip_post_train:
        timed_eval(
            "diagnostic pre-quantization post-ema",
            eval_val,
            h,
            device,
            val_data,
            compiled_model,
            compiled_forward_logits,
        )
    if h.skip_post_train:
        # Optimizer ablation mode: we've logged the pre-quant post-EMA
        # val_bpb, which is the only metric the ablation cares about.
        # Skip serialize/quantized-eval/TTT to save wall-clock.
        log("skip_post_train: stopping after pre-quantization post-ema eval")
        return
    if not _skip_training:
        serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    else:
        log("eval_only: skipping serialize (already have quantized model)")
        if not os.path.exists(h.quantized_model_path):
            log("eval_only: no quantized model found, running serialize anyway")
            serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    eval_model.parcae_eval_bypass = h.parcae_eval_bypass
    if h.num_loops > 0:
        eval_model.looping_active = True
    if h.parcae_eval_bypass and eval_model.loop_log_A is not None:
        log("parcae_eval_bypass: skipping _parcae_boundary in eval forward")
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(
        eval_model.forward_logits, dynamic=False, fullgraph=True
    )
    # Post-quant diagnostic: skipped on submission runs unless DIAG_EVALS_ENABLED=1.
    if h.diag_evals_enabled:
        timed_eval(
            "diagnostic quantized",
            eval_val,
            h,
            device,
            val_data,
            compiled_model,
            compiled_forward_logits,
        )
    if h.sliding_window_enabled:
        timed_eval(
            "diagnostic quantized_sliding_window",
            eval_val_sliding,
            h,
            device,
            val_data,
            eval_model,
            forward_logits_fn=compiled_forward_logits,
        )
    # --- Eval with extra recurrence loops ---
    if h.eval_extra_loops > 0 and h.num_loops > 0:
        compiled_model = compiled_forward_logits = None
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        total_loops = h.num_loops + h.eval_extra_loops
        eval_model.set_eval_loop_indices(total_loops, h.loop_start, h.loop_end, h.num_layers)
        log(
            f"eval_extra_loops:{h.eval_extra_loops} total_loops:{total_loops} "
            f"virtual_depth:{len(eval_model.encoder_indices) + len(eval_model.decoder_indices)} "
            f"encoder:{eval_model.encoder_indices} decoder:{eval_model.decoder_indices}"
        )
        compiled_model_extra = torch.compile(eval_model, dynamic=False, fullgraph=True)
        compiled_fwd_extra = torch.compile(eval_model.forward_logits, dynamic=False, fullgraph=True)
        timed_eval(
            f"diagnostic quantized_extraloop{total_loops}",
            eval_val,
            h,
            device,
            val_data,
            compiled_model_extra,
            compiled_fwd_extra,
        )
        if h.sliding_window_enabled:
            timed_eval(
                f"diagnostic quantized_extraloop{total_loops}_sliding_window",
                eval_val_sliding,
                h,
                device,
                val_data,
                eval_model,
                forward_logits_fn=compiled_fwd_extra,
            )
        del compiled_model_extra, compiled_fwd_extra
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        # Note: eval_model still has extra-loop indices; TTT deserializes fresh so this is fine
    if h.ttt_enabled:
        compiled_model = compiled_forward_logits = None
        del eval_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        ttt_model.parcae_eval_bypass = h.parcae_eval_bypass
        if h.parcae_eval_bypass and ttt_model.loop_log_A is not None:
            log("parcae_eval_bypass: skipping _parcae_boundary in TTT forward")
        if h.num_loops > 0:
            if h.eval_extra_loops > 0:
                total_loops = h.num_loops + h.eval_extra_loops
                ttt_model.set_eval_loop_indices(total_loops, h.loop_start, h.loop_end, h.num_layers)
                log(
                    f"ttt_extra_loops:{h.eval_extra_loops} total_loops:{total_loops} "
                    f"virtual_depth:{len(ttt_model.encoder_indices) + len(ttt_model.decoder_indices)} "
                    f"encoder:{ttt_model.encoder_indices} decoder:{ttt_model.decoder_indices}"
                )
            else:
                ttt_model.looping_active = True
        for p in ttt_model.parameters():
            p.requires_grad_(False)

        if h.rope_yarn:
            _yarn_seqlen = h.train_batch_tokens // h.grad_accum_steps
            for block in ttt_model.blocks:
                block.attn.rotary(_yarn_seqlen, device, torch.bfloat16)
        else:
            for block in ttt_model.blocks:
                block.attn.rotary._cos_cached = None
                block.attn.rotary._sin_cached = None
                block.attn.rotary._seq_len_cached = 0
                block.attn.rotary(h.ttt_eval_seq_len, device, torch.bfloat16)

        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

        _fwd_ttt_compiled_inner = None

        def _fwd_ttt(input_ids, target_ids, lora):
            nonlocal _fwd_ttt_compiled_inner
            if _fwd_ttt_compiled_inner is None:
                _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)

        _ttt_debug_bypass = bool(os.environ.get("TTT_DEBUG_BYPASS"))
        if _ttt_debug_bypass:
            def _fwd_ttt_bypass(input_ids, target_ids, lora):
                logits = ttt_model.forward_logits(input_ids)
                dummy = lora.q_loras[0].B.sum() * 0
                logits = logits + dummy
                bsz, sl, V = logits.shape
                return F.cross_entropy(
                    logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
                ).reshape(bsz, sl)
            fwd_ttt_compiled = _fwd_ttt_bypass
            log("ttt_lora:DEBUG BYPASS active - using forward_logits directly (no compile warmup)")
        else:
            fwd_ttt_compiled = _fwd_ttt
            log(f"ttt_lora:warming up compile (random tokens, no val data)")
            global BOS_ID
            if BOS_ID is None:
                BOS_ID = 1
            t_warmup = time.perf_counter()
            warmup_bszes = [h.ttt_batch_size]
            for bsz in warmup_bszes:
                wl = BatchedTTTLoRA(
                    bsz, ttt_model, h.ttt_lora_rank,
                    k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
                ).to(device)
                wo = torch.optim.AdamW(
                    wl.parameters(),
                    lr=h.ttt_lora_lr,
                    betas=(h.ttt_beta1, h.ttt_beta2),
                    eps=1e-10,
                    weight_decay=h.ttt_weight_decay,
                    fused=True,
                )
                for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
                    xw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                    yw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ptl = fwd_ttt_compiled(xw, yw, lora=wl)
                    ptl[:, : min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
                    wo.step()
                    wo.zero_grad(set_to_none=True)
                del wl, wo
            torch.cuda.empty_cache()
            compile_elapsed = time.perf_counter() - t_warmup
            log(f"ttt_lora:compile warmup done ({compile_elapsed:.1f}s)")
        log("\nbeginning TTT eval timer")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_ttt_phased(
            h, ttt_model, device, val_data,
            forward_ttt_train=fwd_ttt_compiled,
        )
        torch.cuda.synchronize()
        ttt_eval_elapsed = time.perf_counter() - t_ttt
        log(
            "quantized_ttt_phased "
            f"val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} "
            f"eval_time:{1e3*ttt_eval_elapsed:.0f}ms"
        )
        log(f"total_eval_time:{ttt_eval_elapsed:.1f}s")
        del ttt_model


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        import datetime
        nccl_timeout_s = int(os.environ.get("NCCL_WATCHDOG_TIMEOUT_S", "1800"))
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=datetime.timedelta(seconds=nccl_timeout_s),
        )
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 64
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs(h.artifact_dir if h.artifact_dir else "logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for (k, v) in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log("Source code:", console=False)
        log("=" * 100, console=False)
        with open(__file__, "r", encoding="utf-8") as _src:
            log(_src.read(), console=False)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            ).stdout,
            console=False,
        )
        log("=" * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


def _run_cpu_tests():
    """CPU-only unit tests for optimizer primitives. Runnable with:
        RUN_TESTS=1 python train_gpt_parcae_soap.py
    No GPU, no data, no model — pure numerical correctness checks.
    """
    import traceback
    torch.manual_seed(0)
    failures = []

    def _check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            failures.append(name)

    print("== NS correctness ==")
    G = torch.randn(12, 8) * 2.0
    X = zeropower_via_newtonschulz5(G, steps=10).float()
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    expected = (U @ Vh).float()
    err = (X - expected).norm() / expected.norm()
    _check("zeropower_via_newtonschulz5 approximates UV^T",
           err < 0.3, f"rel_err={err:.3f}")
    s_X = torch.linalg.svdvals(X)
    sigma_dev = (s_X - 1.0).abs().mean().item()
    # The (3.4445, -4.775, 2.0315) poly oscillates around 1 and only tightens
    # sigma to [0.8, 1.2] even at many iterations in bf16. Check it at least
    # stays bounded (positive, not exploding).
    _check("NS output singular values are bounded",
           sigma_dev < 0.3 and s_X.min() > 0.5 and s_X.max() < 1.5,
           f"mean_dev={sigma_dev:.3f} range=[{s_X.min():.2f},{s_X.max():.2f}]")

    print("== NS two-phase ==")
    G2 = torch.randn(16, 10) * 1.5
    X2 = newtonschulz_two_phase(G2, aggressive_steps=3, refine_steps=3).float()
    s_X2 = torch.linalg.svdvals(X2)
    sigma_dev_2 = (s_X2 - 1.0).abs().mean().item()
    # Refinement phase (Newton poly 1.5x - 0.5x^3) has a true fixed point at
    # sigma=1 with quadratic convergence, so two-phase should tighten more
    # than the aggressive-only poly.
    _check("two-phase NS output singular values cluster near 1",
           sigma_dev_2 < 0.1, f"mean_dev={sigma_dev_2:.3f}")

    print("== NS adaptive ==")
    # Positive early-exit test: use a two-phase result (already sigma ~= 1)
    # and pass it back through adaptive NS with no internal renormalization
    # by pre-scaling Frobenius. We sidestep the internal Frobenius norm by
    # constructing a matrix where ||X||_F equals spectral norm exactly — a
    # scalar multiple of a single orthonormal direction matrix.
    G4 = torch.randn(10, 6) * 10.0
    _, steps4 = newtonschulz_adaptive(G4, max_steps=8, ortho_eps=1e-6, min_steps=2)
    _check("adaptive NS uses all steps when eps unachievable",
           steps4 == 8, f"steps={steps4}")
    _, steps5 = newtonschulz_adaptive(G4, max_steps=8, ortho_eps=1.0, min_steps=2)
    _check("adaptive NS early-exits at min_steps for very permissive eps",
           steps5 == 2, f"steps={steps5}")

    print("== matrix_inv_sqrt_eigh / _ns correctness ==")
    # Construct M with known clean eigenvalues so the damping bias is small.
    U_r = torch.linalg.qr(torch.randn(5, 8, 8))[0]
    eigs = 1.0 + torch.rand(5, 8) * 5.0  # eigenvalues in [1, 6]
    M = U_r @ torch.diag_embed(eigs) @ U_r.mT
    Xe = matrix_inv_sqrt_eigh(M, damping=0.001)
    rec_e = (Xe @ M @ Xe - torch.eye(8)).norm(dim=(-2, -1)).mean()
    _check("matrix_inv_sqrt_eigh: X M X ~= I",
           rec_e < 0.15, f"err={rec_e:.3f}")
    Xn = matrix_inv_sqrt_ns(M, steps=10, damping=0.001)
    rec_n = (Xn @ M @ Xn - torch.eye(8)).norm(dim=(-2, -1)).mean()
    _check("matrix_inv_sqrt_ns: X M X ~= I",
           rec_n < 0.5, f"err={rec_n:.3f}")
    # 5-step NS on moderate-kappa input (baseline for production default=8).
    # Kappa here is ~6; 5 steps should produce reasonable residual (though
    # worse than 8-10 steps). 8-step default gives ~2x margin on our
    # high-kappa MLP-down case.
    Xn_5 = matrix_inv_sqrt_ns(M, steps=5, damping=0.001)
    rec_5 = (Xn_5 @ M @ Xn_5 - torch.eye(8)).norm(dim=(-2, -1)).mean()
    _check("matrix_inv_sqrt_ns at 5 steps converges on moderate kappa",
           rec_5 < 1.5, f"err={rec_5:.3f}")
    Xn_8 = matrix_inv_sqrt_ns(M, steps=8, damping=0.001)
    rec_8 = (Xn_8 @ M @ Xn_8 - torch.eye(8)).norm(dim=(-2, -1)).mean()
    _check("matrix_inv_sqrt_ns at 8 steps (new default) is tighter",
           rec_8 < rec_5, f"err={rec_8:.3f}")
    # High-condition test: build M with kappa ~ 1e3. Old trace-based scaling
    # would have diverged; Frobenius-based scaling must stay bounded.
    eigs_ill = torch.cat([
        torch.tensor([1e3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unsqueeze(0)
    ], dim=0).expand(3, -1)
    U_ill = torch.linalg.qr(torch.randn(3, 8, 8))[0]
    M_ill = U_ill @ torch.diag_embed(eigs_ill) @ U_ill.mT
    Xn_ill = matrix_inv_sqrt_ns(M_ill, steps=12, damping=0.01)
    finite_ill = torch.isfinite(Xn_ill).all().item()
    bounded_ill = Xn_ill.abs().max().item() < 1e6
    _check("matrix_inv_sqrt_ns finite on kappa=1e3 input",
           finite_ill and bounded_ill,
           f"finite={finite_ill} max={Xn_ill.abs().max().item():.2e}")

    print("== compute_scale modes ==")
    _check("ratio_clamped square=1", compute_scale(8, 8, "ratio_clamped") == 1.0)
    _check("ratio_clamped wide=1",
           abs(compute_scale(8, 32, "ratio_clamped") - 1.0) < 1e-9)
    _check("ratio wide<1", compute_scale(8, 32, "ratio") < 1.0)
    _check("ratio_clamped tall=2",
           abs(compute_scale(32, 8, "ratio_clamped") - 2.0) < 1e-9)
    _check("moonlight matches 0.2*sqrt(max)",
           abs(compute_scale(8, 32, "moonlight") - 0.2 * (32 ** 0.5)) < 1e-9)
    _check("soap mode is 1.0", compute_scale(8, 32, "soap") == 1.0)

    print("== cautious mask ==")
    g = torch.tensor([[1.0, -1.0, 2.0], [-3.0, 4.0, -1.0]])
    u_agree = torch.tensor([[0.5, -0.5, 1.0], [-1.5, 2.0, -0.5]])
    u_all = apply_cautious_mask(u_agree, g)
    _check("cautious passes when all signs match",
           torch.allclose(u_all, u_agree, atol=1e-6))
    u_disagree = torch.tensor([[-0.5, 0.5, -1.0], [1.5, -2.0, 0.5]])
    u_masked = apply_cautious_mask(u_disagree, g)
    _check("cautious zeroes out when all signs disagree",
           torch.allclose(u_masked, torch.zeros_like(u_disagree)))

    print("== preconditioner state ==")
    ps = PreconditionerState(shard_B=2, out_dim=4, in_dim=6, device=torch.device("cpu"))
    g_syn = torch.randn(2, 4, 6)
    for _ in range(20):
        update_preconditioner_from_grad(ps, g_syn + 0.1 * torch.randn_like(g_syn), 0.9)
    _check("L_ema symmetric",
           torch.allclose(ps.L_ema, ps.L_ema.mT, atol=1e-5))
    _check("R_ema symmetric",
           torch.allclose(ps.R_ema, ps.R_ema.mT, atol=1e-5))
    # Side-selective update: sides="L" should change L_ema but leave R_ema frozen.
    ps_side = PreconditionerState(shard_B=1, out_dim=4, in_dim=6, device=torch.device("cpu"))
    update_preconditioner_from_grad(ps_side, torch.randn(1, 4, 6), 0.9, sides="both")
    R_before = ps_side.R_ema.clone()
    L_before = ps_side.L_ema.clone()
    update_preconditioner_from_grad(ps_side, torch.randn(1, 4, 6), 0.9, sides="L")
    _check("sides=L updates L_ema",
           not torch.allclose(ps_side.L_ema, L_before))
    _check("sides=L leaves R_ema unchanged",
           torch.allclose(ps_side.R_ema, R_before))
    update_preconditioner_from_grad(ps_side, torch.randn(1, 4, 6), 0.9, sides="R")
    _check("sides=R leaves L_ema unchanged (after next call)",
           torch.allclose(ps_side.L_ema, ps_side.L_ema))  # trivially true; real check below
    L_after_R = ps_side.L_ema.clone()
    update_preconditioner_from_grad(ps_side, torch.randn(1, 4, 6), 0.9, sides="R")
    _check("sides=R keeps L_ema frozen across calls",
           torch.allclose(ps_side.L_ema, L_after_R))
    ps.step = 10
    refreshed = maybe_refresh_eigenbasis(ps, refresh_k=1, adaptive=False,
                                          drift_tau=0.1, damping=0.03,
                                          rotate_adam=False)
    _check("refresh fires when refresh_k=1", refreshed)
    _check("Q_L is orthogonal",
           torch.allclose(ps.Q_L @ ps.Q_L.mT, torch.eye(4).expand(2, 4, 4), atol=1e-4))
    _check("Q_R is orthogonal",
           torch.allclose(ps.Q_R @ ps.Q_R.mT, torch.eye(6).expand(2, 6, 6), atol=1e-4))

    print("== pre-first-refresh safety (Bug 1 regression) ==")
    # update_soap must be callable BEFORE any refresh. Default
    # SOAP_PRECOND_WARMUP_STEPS=500 means refresh is gated off for the first
    # 500 steps; update_soap is called every step. Q_L_bf16 must be
    # initialized (to identity bf16) in _alloc_eigen so the rotation
    # matmul doesn't hit None.mT.
    ps_fresh = PreconditionerState(1, 4, 6, torch.device("cpu"))
    ps_fresh.ensure_eigen()
    ps_fresh.ensure_adam()
    _check("Q_L_bf16 populated after ensure_eigen (pre-refresh)",
           ps_fresh.Q_L_bf16 is not None)
    _check("Q_R_bf16 populated after ensure_eigen (pre-refresh)",
           ps_fresh.Q_R_bf16 is not None)
    cfg_fresh = dict(backend_steps=4, soap_damping=0.03, soap_beta1=0.9,
                     soap_beta2=0.95, soap_eps=1e-8, soap_base="adam",
                     inv_root_ns_steps=5, ns_adaptive=False, ns_two_phase=False,
                     ns_refine_steps=2, ns_warm_start=False,
                     psgd_kron_precond_lr=0.1)
    ps_fresh.step = 1
    try:
        upd_pre = update_soap(torch.randn(1, 4, 6), ps_fresh, cfg_fresh)
        pre_ok = torch.isfinite(upd_pre).all().item()
    except Exception as e:
        pre_ok = False
    _check("update_soap safe to call before any refresh", pre_ok)

    print("== SOAP step does something ==")
    cfg = dict(backend_steps=4, soap_damping=0.03, soap_beta1=0.9, soap_beta2=0.95,
               soap_eps=1e-8, soap_base="adam", inv_root_ns_steps=5,
               ns_adaptive=False, ns_two_phase=False, ns_refine_steps=2,
               ns_warm_start=False, psgd_kron_precond_lr=0.1)
    ps.step = 1
    ps.ensure_adam()
    g_new = torch.randn(2, 4, 6)
    upd_soap = update_soap(g_new, ps, cfg)
    _check("SOAP update finite", torch.isfinite(upd_soap).all())
    _check("SOAP update has matching shape", upd_soap.shape == g_new.shape)

    print("== Each mode produces a finite update ==")
    for mode in ("muon", "muon_ns_fix", "muon_2side", "soap",
                 "soap_1side_left", "soap_1side_right",
                 "shampoo_ns", "psgd_kron"):
        # muon and muon_ns_fix receive state=None in the real flow (no
        # preconditioner); test that behavior too.
        if MODE_USES_PRECONDITIONER.get(mode, False):
            ps_m = PreconditionerState(2, 4, 6, torch.device("cpu"))
            for _ in range(10):
                update_preconditioner_from_grad(ps_m, torch.randn(2, 4, 6), 0.9)
            ps_m.step = 5
            if mode == "muon_2side":
                compute_inv_sqrt = "eigh"
            elif mode == "shampoo_ns":
                compute_inv_sqrt = "ns"
            else:
                compute_inv_sqrt = "none"
            maybe_refresh_eigenbasis(ps_m, 1, False, 0.1, 0.03,
                                      rotate_adam=MODE_ROTATES_ADAM.get(mode, False),
                                      compute_inv_sqrt=compute_inv_sqrt)
            ps_m.step = 10
        else:
            ps_m = None
        fn = UPDATE_FUNCTIONS[mode]
        try:
            upd = fn(torch.randn(2, 4, 6), ps_m, cfg)
            finite = torch.isfinite(upd).all().item()
            _check(f"{mode} finite & correct shape",
                   finite and upd.shape == (2, 4, 6),
                   f"shape={tuple(upd.shape)} finite={finite}")
        except Exception as e:
            _check(f"{mode} runs without exception", False, str(e))

    print("== soap_1side Q-side isolation ==")
    # Under sides="left", Q_R must remain identity after a refresh. Otherwise
    # m_rot (accumulated with Q_R=I) gets corrupted on the next rotation.
    ps_1s = PreconditionerState(shard_B=1, out_dim=4, in_dim=6,
                                 device=torch.device("cpu"))
    for _ in range(10):
        update_preconditioner_from_grad(ps_1s, torch.randn(1, 4, 6), 0.9)
    ps_1s.step = 5
    maybe_refresh_eigenbasis(ps_1s, 1, False, 0.1, 0.03,
                              rotate_adam=False, sides="left")
    eye_R = torch.eye(6).expand(1, 6, 6)
    _check("sides=left leaves Q_R at identity",
           torch.allclose(ps_1s.Q_R, eye_R, atol=1e-5))
    _check("sides=left updates Q_L to orthogonal non-identity",
           not torch.allclose(ps_1s.Q_L, torch.eye(4).expand(1, 4, 4)))
    # Conversely for sides=right.
    ps_1s2 = PreconditionerState(shard_B=1, out_dim=4, in_dim=6,
                                  device=torch.device("cpu"))
    for _ in range(10):
        update_preconditioner_from_grad(ps_1s2, torch.randn(1, 4, 6), 0.9)
    ps_1s2.step = 5
    maybe_refresh_eigenbasis(ps_1s2, 1, False, 0.1, 0.03,
                              rotate_adam=False, sides="right")
    eye_L = torch.eye(4).expand(1, 4, 4)
    _check("sides=right leaves Q_L at identity",
           torch.allclose(ps_1s2.Q_L, eye_L, atol=1e-5))

    print("== first-refresh rotation (Bug 13) ==")
    # After warmup, m_rot is accumulated with Q=I. First refresh must rotate
    # m_rot into the new eigenbasis — otherwise momentum state is stale.
    ps_fr = PreconditionerState(shard_B=1, out_dim=4, in_dim=6,
                                 device=torch.device("cpu"))
    ps_fr.ensure_adam()
    for _ in range(15):
        g_warm = torch.randn(1, 4, 6)
        update_preconditioner_from_grad(ps_fr, g_warm, 0.9)
        ps_fr.m_rot.mul_(0.9).add_(g_warm.to(ps_fr.m_rot.dtype), alpha=0.1)
    ps_fr.step = 10
    m_before = ps_fr.m_rot.clone()
    maybe_refresh_eigenbasis(ps_fr, 1, False, 0.1, 0.03,
                              rotate_adam=True, sides="both")
    # After refresh with Q_L != I, m_rot should have been rotated (changed).
    _check("first refresh rotates m_rot into new basis",
           not torch.allclose(ps_fr.m_rot, m_before, atol=1e-3))

    print("== PSGD-Kron direction sanity ==")
    ps_psgd = PreconditionerState(1, 6, 4, torch.device("cpu"))
    fixed_g = torch.randn(1, 6, 4) * 3.0  # fixed gradient each step
    cfg_psgd = dict(cfg)
    cfg_psgd["psgd_kron_precond_lr"] = 0.05
    # Run many steps with the same gradient. Q_l should move from I toward
    # (gg^T)^{-1/2} in direction, reducing the residual ||Q·gg·Q - I||.
    # Weak convergence — property test is monotone decrease, not strict
    # convergence (full PSGD needs Lie-group trick for that).
    initial_normed_g = fixed_g / fixed_g.norm()
    gg_n = initial_normed_g[0] @ initial_normed_g[0].mT * 6.0
    resid_initial = (gg_n - torch.eye(6)).norm().item()
    for _ in range(500):
        _ = update_psgd_kron(fixed_g, ps_psgd, cfg_psgd)
    Ql = ps_psgd.psgd_Ql[0]
    check_mat = Ql @ gg_n @ Ql
    resid_final = (check_mat - torch.eye(6)).norm().item()
    _check("PSGD-Kron reduces residual from initial Q=I toward fixed point",
           resid_final < resid_initial,
           f"initial={resid_initial:.3f} final={resid_final:.3f}")

    print("== state_dict roundtrip ==")
    ps2 = PreconditionerState(2, 4, 6, torch.device("cpu"))
    for _ in range(8):
        update_preconditioner_from_grad(ps2, torch.randn(2, 4, 6), 0.9)
    ps2.step = 5
    maybe_refresh_eigenbasis(ps2, 1, False, 0.1, 0.03, rotate_adam=False)
    ps2.ensure_adam()
    ps2.m_rot.normal_()
    sd = ps2.state_dict()
    ps3 = PreconditionerState(2, 4, 6, torch.device("cpu"))
    ps3.ensure_adam()
    ps3.load_state_dict(sd)
    _check("state_dict roundtrip preserves L_ema",
           torch.allclose(ps2.L_ema, ps3.L_ema, atol=1e-6))
    _check("state_dict roundtrip preserves Q_L",
           torch.allclose(ps2.Q_L, ps3.Q_L, atol=1e-6))
    _check("state_dict roundtrip preserves m_rot",
           torch.allclose(ps2.m_rot, ps3.m_rot, atol=1e-6))

    print("== auto_scale_mode ==")
    _check("soap mode -> soap scale", auto_scale_mode("soap", "auto") == "soap")
    _check("muon_2side -> fan_out", auto_scale_mode("muon_2side", "auto") == "fan_out")
    _check("muon -> ratio_clamped", auto_scale_mode("muon", "auto") == "ratio_clamped")
    _check("explicit override wins",
           auto_scale_mode("soap", "ratio") == "ratio")

    print("== summary ==")
    total = 0
    if failures:
        print(f"FAILED ({len(failures)} tests): {failures}")
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    if os.environ.get("RUN_TESTS"):
        _run_cpu_tests()
    else:
        main()
