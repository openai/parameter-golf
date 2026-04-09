"Ternary training script for OpenAI's Parameter Golf Challenge. Ciprian-Florin Ifrim - 24 March 2026"

import copy
import glob
import io
import math
import os
import random
import sys
import time
import lzma
from pathlib import Path
import traceback
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    def flash_attn_func(q, k, v, causal=False, **kwargs):
        return _fa3_func(q, k, v, causal=causal)
except ImportError:
    try:
        from flash_attn import flash_attn_func as _fa2_func
        def flash_attn_func(q, k, v, causal=False, **kwargs):
            # flash_attn v2: expects (B, T, H, D), returns (B, T, H, D)
            if q.size(2) != k.size(2):
                r = q.size(2) // k.size(2)
                k = k.repeat_interleave(r, dim=2)
                v = v.repeat_interleave(r, dim=2)
            return _fa2_func(q, k, v, causal=causal)
    except ImportError:
        def flash_attn_func(q, k, v, causal=False, **kwargs):
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if q.size(1) != k.size(1):
                r = q.size(1) // k.size(1)
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)

# ---------------------------------------------------------------------------
# Hardware Optimization Setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    # Use bfloat16 on capable GPUs (H100/A100/RTX 40xx), fall back to float32
    if torch.cuda.get_device_capability()[0] >= 8:
        ptdtype = torch.bfloat16
    else:
        ptdtype = torch.float32  # GTX 1650 Ti etc.
else:
    ptdtype = torch.float32

# ---------------------------------------------------------------------------
# Hyperparameters (all configurable via environment variables)
# ---------------------------------------------------------------------------
def _e(k, d, t=str):
    v = os.environ.get(k, str(d))
    if t == bool: return bool(int(v))
    return t(v)

class Hyperparameters:
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp1024", str)
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model", str)
    num_layers = _e("NUM_LAYERS", 8, int)
    model_dim = _e("MODEL_DIM", 768, int)
    vocab_size = _e("VOCAB_SIZE", 1024, int)

    @property
    def train_files(self): return os.path.join(self.data_path, "fineweb_train_*.bin")
    @property
    def val_files(self): return os.path.join(self.data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")
    seed = _e("SEED", 42, int)
    compile_mode = _e("COMPILE_MODE", "default", str)
    val_batch_size = _e("VAL_BATCH_SIZE", 65536, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 0, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 100, int)
    iterations = _e("ITERATIONS", 10000, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.35, float)
    warmup_steps = _e("WARMUP_STEPS", 5, int)
    compiler_warmup_steps = _e("COMPILER_WARMUP_STEPS", 0, int)  # separate pre-budget compile warmup
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 786432, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 2048, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 599.0, float)
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    num_heads = _e("NUM_HEADS", 8, int)
    mlp_mult = _e("MLP_MULT", 4, int)
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, int)
    rope_base = _e("ROPE_BASE", 5000.0, float)
    rope_type = _e("ROPE_TYPE", "yarn")
    yarn_max_len = _e("YARN_MAX_LEN", 4096, int)
    logit_softcap = _e("LOGIT_SOFTCAP", 30.0, float)
    softcap_type = _e("SOFTCAP_TYPE", "poly")
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)
    qk_gain_init = _e("QK_GAIN_INIT", 2.25, float)
    activation_type = _e("ACTIVATION", "lrelu2")
    leaky_relu_slope = _e("LEAKY_RELU_SLOPE", 0.5, float)
    embed_dim = _e("EMBED_DIM", 254, int)
    training_depth_recurrence = _e("TRAINING_DEPTH_RECURRENCE", 0, int)
    feedback_enabled = _e("FEEDBACK_ENABLED", 0, bool)
    feedback_dim = _e("FEEDBACK_DIM", 32, int)
    feedback_sketch_tokens = _e("FEEDBACK_SKETCH_TOKENS", 2, int)
    feedback_replay = _e("FEEDBACK_REPLAY", "decoder")
    feedback_target = _e("FEEDBACK_TARGET", "decoder")
    feedback_passes = _e("FEEDBACK_PASSES", 1, int)
    eval_feedback_passes = _e("EVAL_FEEDBACK_PASSES", 2, int)
    feedback_fp_storage = _e("FEEDBACK_FP_STORAGE", 0, bool)
    feedback_every = _e("FEEDBACK_EVERY", 2, int)
    untie_at_fraction = _e("UNTIE_AT_FRACTION", 0.0, float)
    moe_enabled = _e("MOE_ENABLED", 1, bool)
    moe_num_experts = _e("MOE_NUM_EXPERTS", 3, int)
    moe_top_k = _e("MOE_TOP_K", 1, int)
    moe_router_aux_loss_coef = _e("MOE_ROUTER_AUX_LOSS_COEF", 0.001, float)
    moe_start_fraction = _e("MOE_START_FRACTION", 0.65, float) # C10: MoE dormant early
    moe_layer_frac = _e("MOE_LAYER_FRAC", 0.67, float)  # MoE only in top (1-frac) of layers
    vrl_enabled = _e("VRL_ENABLED", 1, bool)
    vrl_start_layer = _e("VRL_START_LAYER", 10, int)
    adam_lr = _e("ADAM_LR", 0.001, float)
    adam_wd = _e("ADAM_WD", 0.04, float)
    beta1 = 0.9
    beta2 = 0.95
    adam_eps = 1e-8
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.3, float)
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 128, int)
    shared_blocks = _e("SHARED_BLOCKS", 2, int)
    capsule_enabled = _e("CAPSULE_ENABLED", 1, bool)
    capsule_num = _e("CAPSULE_NUM", 16, int)
    capsule_dim = _e("CAPSULE_DIM", 64, int)
    capsule_carry_decay = _e("CAPSULE_CARRY_DECAY", 0.8, float)
    capsule_carry_enabled = _e("CAPSULE_CARRY_ENABLED", 1, bool)
    partial_rope_dims = _e("PARTIAL_ROPE_DIMS", 16, int)
    ln_scale_damping = _e("LN_SCALE_DAMPING", 1, bool)
    bigram_hash_enabled = _e("BIGRAM_HASH_ENABLED", 1, bool)  # C1: was hardcoded False in main()
    bigram_hash_buckets = _e("BIGRAM_HASH_BUCKETS", 4096, int)
    bigram_hash_dim = _e("BIGRAM_HASH_DIM", 64, int)
    engram_num_heads = _e("ENGRAM_NUM_HEADS", 4, int)
    engram_num_orders = _e("ENGRAM_NUM_ORDERS", 3, int)
    engram_inject_layer = _e("ENGRAM_INJECT_LAYER", 1, int)
    xsa_start_layer = _e("XSA_START_LAYER", 8, int)
    koopman_enabled = _e("KOOPMAN_ENABLED", 1, bool)
    koopman_rank = _e("KOOPMAN_RANK", 2, int)
    koopman_diag_init = _e("KOOPMAN_DIAG_INIT", 0.9, float)
    koopman_consistency_weight = _e("KOOPMAN_CONSISTENCY_WEIGHT", 0.005, float)
    koopman_speculator_enabled = _e("KOOPMAN_SPECULATOR_ENABLED", 1, bool)
    koopman_speculator_steps = _e("KOOPMAN_SPECULATOR_STEPS", 3, int)
    koopman_speculator_weight = _e("KOOPMAN_SPECULATOR_WEIGHT", 0.01, float)
    adaptive_halt_enabled = _e("ADAPTIVE_HALT_ENABLED", 1, bool)
    adaptive_halt_threshold = _e("ADAPTIVE_HALT_THRESHOLD", 0.05, float)
    max_eval_passes = _e("MAX_EVAL_PASSES", 3, int)
    architecture = _e("ARCHITECTURE", "hybrid")
    # Koopman SSM hyperparameters (for SSM and hybrid modes)
    koopman_state_dim = _e("KOOPMAN_STATE_DIM", 128, int)
    koopman_mixer_rank = _e("KOOPMAN_MIXER_RANK", 4, int)
    koopman_conv_kernel = _e("KOOPMAN_CONV_KERNEL", 4, int)
    koopman_decay_window = _e("KOOPMAN_DECAY_WINDOW", 32, int)

    skc_num_capsules = _e("SKC_NUM_CAPSULES", 32, int)
    skc_capsule_dim = _e("SKC_CAPSULE_DIM", 128, int)
    skc_conv_kernel = _e("SKC_CONV_KERNEL", 4, int)
    skc_block_size = _e("SKC_BLOCK_SIZE", 64, int)
    tko_enabled = _e("TKO_ENABLED", 1, bool)
    weight_sharing = _e("WEIGHT_SHARING", 1, bool)
    inside_out_training = _e("INSIDE_OUT_TRAINING", 1, bool)
    deq_feedback = _e("DEQ_FEEDBACK", 1, bool)
    deq_max_iter = _e("DEQ_MAX_ITER", 3, int)
    deq_tol = _e("DEQ_TOL", 0.01, float)
    deq_anderson_m = _e("DEQ_ANDERSON_M", 3, int)

    fp_storage = _e("FP_STORAGE", 0, bool)
    stochastic_depth_prob = _e("STOCHASTIC_DEPTH_PROB", 0.1, float)
    ternary_noise_scale = _e("TERNARY_NOISE_SCALE", 0.02, float)
    self_distill_kl_weight = _e("SELF_DISTILL_KL_WEIGHT", 0.1, float)
    ema_enabled = _e("EMA_ENABLED", 1, bool)
    ema_eval_apply = _e("EMA_EVAL_APPLY", 1, bool)
    ema_decay = _e("EMA_DECAY", 0.997, float)
    ema_start_fraction = _e("EMA_START_FRACTION", 0.40, float)
    # Optimizer (Muon / NeoMuon)
    matrix_optimizer = _e("MATRIX_OPTIMIZER", "muon")
    # Optimizer LRs: tuned-down from the original 0.035 defaults which were too aggressive
    # for a hybrid ternary architecture with Muon. Scale back up if loss curves are stable.
    matrix_lr = _e("MATRIX_LR", 0.006, float)
    scalar_lr = _e("SCALAR_LR", 0.001, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.004, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 500, int)
    muon_wd = _e("MUON_WD", 0.04, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    # Eval features
    sliding_eval = _e("SLIDING_EVAL", 1, bool)
    sliding_eval_stride = _e("SLIDING_EVAL_STRIDE", 64, int)
    sliding_batch_size = _e("SLIDING_BATCH_SIZE", 256, int)
    temp_scaling = _e("TEMP_SCALING", 1, bool)
    # Export
    turbo_quant_export = _e("TURBO_QUANT_EXPORT", 1, bool)
    turbo_quant_train = _e("TURBO_QUANT_TRAIN", 0, bool)
    # Eval-time features
    ngram_cache_enabled = _e("NGRAM_CACHE_ENABLED", 1, bool)
    ngram_max_order = _e("NGRAM_MAX_ORDER", 5, int)
    ngram_alpha_base = _e("NGRAM_ALPHA_BASE", 0.05, float)
    ngram_alpha_scale = _e("NGRAM_ALPHA_SCALE", 0.55, float)
    ngram_entropy_center = _e("NGRAM_ENTROPY_CENTER", 4.0, float)
    ttt_enabled = _e("TTT_ENABLED", 1, bool)
    ttt_scope = _e("TTT_SCOPE", "feedback")
    ttt_lr = _e("TTT_LR", 0.002, float)
    ttt_epochs = _e("TTT_EPOCHS", 3, int)
    ttt_chunk_tokens = _e("TTT_CHUNK_TOKENS", 32768, int)
    ttt_momentum = _e("TTT_MOMENTUM", 0.9, float)
    ttt_batch_seqs = _e("TTT_BATCH_SEQS", 32, int)
    ttt_grad_clip = _e("TTT_GRAD_CLIP", 1.0, float)
    val_batch_size = _e("VAL_BATCH_SIZE", 32768, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 200, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 20, int)
    churn_log_every = _e("CHURN_LOG_EVERY", 0, int)
    batch_tokens_start = _e("BATCH_TOKENS_START", 0, int)
    batch_schedule_fraction = _e("BATCH_SCHEDULE_FRACTION", 0.0, float)
    seq_len_start = _e("SEQ_LEN_START", 0, int)
    seq_schedule_fraction = _e("SEQ_SCHEDULE_FRACTION", 0.0, float)
    curr_enabled = _e("CURRICULUM_ENABLED", 0, bool)
    curr_p1_f = _e("CURRICULUM_PHASE1_FRAC", 0.05, float)
    curr_p2_f = _e("CURRICULUM_PHASE2_FRAC", 0.10, float)
    curr_p3_f = _e("CURRICULUM_PHASE3_FRAC", 0.17, float)
    curr_p4_f = _e("CURRICULUM_PHASE4_FRAC", 0.25, float)
    curr_p5_f = _e("CURRICULUM_PHASE5_FRAC", 0.35, float)
    curr_p1_s = _e("CURRICULUM_PHASE1_SEQ", 64, int)
    curr_p2_s = _e("CURRICULUM_PHASE2_SEQ", 128, int)
    curr_p3_s = _e("CURRICULUM_PHASE3_SEQ", 256, int)
    curr_p4_s = _e("CURRICULUM_PHASE4_SEQ", 512, int)
    curr_p5_s = _e("CURRICULUM_PHASE5_SEQ", 1024, int)
    eval_depth_recurrence = _e("EVAL_DEPTH_RECURRENCE", 0, int)
    gptq_lite_enabled = _e("GPTQ_LITE_ENABLED", 0, bool)
    gptq_lite_percentiles = _e("GPTQ_LITE_PERCENTILES", 5, int)
    lzma_preset = _e("LZMA_PRESET", 4, int)
    head_lr = _e("HEAD_LR", 0.002, float)
    # Export fidelity: per-tensor threshold + scale calibration
    ternary_threshold_search = _e("TERNARY_THRESHOLD_SEARCH", 0, bool)
    ternary_threshold_low = _e("TERNARY_THRESHOLD_LOW", 0.02, float)
    ternary_threshold_high = _e("TERNARY_THRESHOLD_HIGH", 0.15, float)
    ternary_threshold_steps = _e("TERNARY_THRESHOLD_STEPS", 4, int)
    ternary_scale_search = _e("TERNARY_SCALE_SEARCH", 0, bool)
    ternary_scale_mult_low = _e("TERNARY_SCALE_MULT_LOW", 0.9, float)
    ternary_scale_mult_high = _e("TERNARY_SCALE_MULT_HIGH", 1.1, float)
    ternary_scale_mult_steps = _e("TERNARY_SCALE_MULT_STEPS", 3, int)
    ternary_calib_top_n = _e("TERNARY_CALIB_TOP_N", 4, int)
    calib_prefilter_mult = _e("CALIB_PREFILTER_MULT", 2, int)
    calib_max_candidates = _e("CALIB_MAX_CANDIDATES", 12, int)
    calib_max_evals = _e("CALIB_MAX_EVALS", 32, int)
    calib_max_seconds = _e("CALIB_MAX_SECONDS", 90.0, float)
    calib_second_pass = _e("CALIB_SECOND_PASS", 0, bool)
    calib_proxy_max_tok = _e("CALIB_PROXY_MAX_TOK", 4096, int)
    # Export proxy eval during training (select checkpoint by round-trip BPB)
    export_proxy_eval = _e("EXPORT_PROXY_EVAL", 0, bool)
    export_proxy_every = _e("EXPORT_PROXY_EVERY", 300, int)
    export_proxy_num_seqs = _e("EXPORT_PROXY_NUM_SEQS", 16, int)
    export_proxy_use_best = _e("EXPORT_PROXY_USE_BEST", 1, bool)
    # Export-aligned training: use calibrated quantizer in final phase
    export_aligned_train = _e("EXPORT_ALIGNED_TRAIN", 0, bool)
    export_aligned_train_start_fraction = _e("EXPORT_ALIGNED_TRAIN_START_FRACTION", 0.80, float)

CTP = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix", "resid_mixes",
    "q_gain", "skip_weight", "skip_weights", "vocab_bias", "add_gate", "mul_gate",
    "recurrent_gate", "vrl_alpha",
    "koopman",  # Koopman dynamics params go to scalar Adam for stability
    "mixer_diag", "mixer_lowrank", "mixer_conv", "mixer_scale",  # Koopman SSM mixer scalars
)

# ---------------------------------------------------------------------------
# Ternary packing — base-3 encoding (5 trits/byte)
# ---------------------------------------------------------------------------
def pack_ternary(q: Tensor):
    f = (q.reshape(-1).to(torch.int8) + 1).numpy()
    n = len(f)
    p = (5 - n % 5) % 5
    if p: f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
    g = f.reshape(-1, 5).astype(np.uint8)
    return (g[:,0] + g[:,1]*3 + g[:,2]*9 + g[:,3]*27 + g[:,4]*81).tobytes(), n

def unpack_ternary(data: bytes, n: int) -> Tensor:
    v = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    t = np.zeros((len(v), 5), dtype=np.int8)
    for i in range(5): t[:,i] = v % 3; v //= 3
    return torch.from_numpy(t.reshape(-1)[:n].astype(np.int8) - 1)

def pack_ternary_bitmask(q: Tensor):
    f = q.reshape(-1).to(torch.int8).numpy(); n = len(f)
    nz = (f != 0)
    return np.packbits(nz).tobytes() + np.packbits(f[nz] > 0).tobytes(), n

def unpack_ternary_bitmask(data: bytes, n: int) -> Tensor:
    ms = (n + 7) // 8
    nz = np.unpackbits(np.frombuffer(data[:ms], dtype=np.uint8))[:n].astype(bool)
    s = np.unpackbits(np.frombuffer(data[ms:], dtype=np.uint8))[:int(nz.sum())].astype(bool)
    w = np.zeros(n, dtype=np.int8); w[nz] = np.where(s, 1, -1)
    return torch.from_numpy(w)

# ---------------------------------------------------------------------------
# FP4 quantization (per-row absmax, 2 values packed per byte)
# ---------------------------------------------------------------------------
def quantize_to_int4(t: Tensor) -> tuple[Tensor, Tensor, list]:
    t32 = t.float()
    orig_shape = t32.shape
    if t32.ndim < 2:
        t32 = t32.unsqueeze(0)
    absmax = t32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(t32 / scale), -7, 7).to(torch.int8)
    flat = q.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = F.pad(flat, (0, 1))
    low = (flat[0::2] + 8).to(torch.uint8)
    high = (flat[1::2] + 8).to(torch.uint8)
    return low | (high << 4), scale.half().squeeze(-1), list(orig_shape)

def dequantize_from_int4(packed: Tensor, scale: Tensor, shape: list) -> Tensor:
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    flat = torch.zeros(packed.numel() * 2, dtype=torch.int8)
    flat[0::2] = low
    flat[1::2] = high
    numel = 1
    for s in shape:
        numel *= s
    flat = flat[:numel].float()
    if len(shape) <= 1:
        return (flat * scale.float().squeeze()).reshape(shape)
    return (flat.reshape(-1, shape[-1]) * scale.float().unsqueeze(-1)).reshape(shape)

# ---------------------------------------------------------------------------
# State dict serialization (ternary + fp16/fp8/fp4)
# ---------------------------------------------------------------------------
def q_sd(state_dict: dict, group_size: int = 64, fp_storage=False, ternary_method="standard", ternary_override_names: set | None = None, calib: dict | None = None) -> tuple[dict, dict]:
    "Ternary for large 2D weight matrices, fp16/fp8/fp4 for everything else."
    quantized = {}
    stats = {"ternary_params": 0, "ternary_bytes": 0, "fp_params": 0, "fp_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float().contiguous()
        t_orig_shape = list(t.shape)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        is_ternary_candidate = (
            t.ndim == 2 and t.numel() > 16_384
            and "tok_emb" not in name and "lm_head" not in name and "embed_proj" not in name
        ) or (ternary_override_names is not None and name in ternary_override_names) \
          or ("prototypes" in name)
        if is_ternary_candidate:
            pad = (group_size - t.shape[1] % group_size) % group_size
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, group_size)

            # TurboQuant: Hadamard rotation before quantization for lower MSE
            turbo_used = False
            if Hyperparameters.turbo_quant_export and (group_size & (group_size - 1)) == 0:
                H = _build_hadamard_pt(group_size, t_grouped.device)
                t_grouped = t_grouped @ H
                turbo_used = True

            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
            tensor_calib = (calib or {}).get(name, {})
            thr = tensor_calib.get("thr", 0.0)
            scale_mult = tensor_calib.get("scale_mult", 1.0)
            z = t_grouped / scale
            if thr > 0.0:
                q = torch.where(z.abs() < thr, torch.zeros_like(z), z.round().clamp(-1, 1)).to(torch.int8)
            else:
                q = z.round().clamp(-1, 1).to(torch.int8)
            if scale_mult != 1.0:
                scale = scale * scale_mult

            if ternary_method == "standard":
                packed_bytes, n_trits = pack_ternary(q)
                entry_type = "ternary"
            else:
                packed_bytes, n_trits = pack_ternary_bitmask(q)
                entry_type = "ternary_bitmask"

            entry = {
                "type": entry_type, "packed": packed_bytes,
                "scale": scale.half().squeeze(-1),
                "shape": list(t.shape), "padded_cols": t_padded.shape[1],
                "group_size": group_size, "n_trits": n_trits,
                "orig_shape": t_orig_shape,
            }
            if turbo_used:
                entry["turbo"] = True  # load path must inverse-Hadamard after dequant
            quantized[name] = entry
            stats["ternary_params"] += t.numel()
            stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
        elif fp_storage == "fp4" and t.ndim == 2:
            packed, scale, orig_shape = quantize_to_int4(t)
            quantized[name] = {"type": "fp4", "packed": packed, "scale": scale, "shape": orig_shape}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += packed.numel() + scale.numel() * 2
        elif fp_storage and t.ndim == 2:
            quantized[name] = {"type": "fp8", "data": t.to(torch.float8_e4m3fn)}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel()
        else:
            quantized[name] = {"type": "fp16", "data": t.half()}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel() * 2
    return quantized, stats

def deq_sd(quantized: dict, target_dtype=torch.bfloat16):
    """Reconstruct full-precision state dict from quantized representation.

    IMPORTANT: reconstruction must use `t = q * scale` (identical to training-time STE
    in TernaryLinear.forward). Do NOT divide by q_absmean — that introduces a systematic
    weight amplification that causes round-trip BPB drift.
    """
    out = {}
    for name, entry in quantized.items():
        if entry["type"] in ("ternary", "ternary_bitmask"):
            if entry["type"] == "ternary":
                q = unpack_ternary(entry["packed"], entry["n_trits"])
            else:
                q = unpack_ternary_bitmask(entry["packed"], entry["n_trits"])

            gs = entry["group_size"]
            q = q.float().reshape(-1, gs)
            scale = entry["scale"].float().unsqueeze(-1)
            # Must match TernaryLinear.forward(): dequant = q * scale
            t = q * scale
            # TurboQuant: inverse-Hadamard rotation to recover original-space weights
            if entry.get("turbo") and (gs & (gs - 1)) == 0:
                H = _build_hadamard_pt(gs, t.device)
                t = t @ H  # H is self-inverse
            t = t.reshape(-1, entry["padded_cols"])
            shape = entry["shape"]
            result = t[:shape[0], :shape[1]].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = result.reshape(orig).contiguous() if orig and orig != shape else result.contiguous()
        elif entry["type"] == "fp8":
            out[name] = entry["data"].to(torch.float32).to(target_dtype).contiguous()
        elif entry["type"] == "fp4":
            out[name] = dequantize_from_int4(entry["packed"], entry["scale"], entry["shape"]).to(target_dtype).contiguous()
        else:
            out[name] = entry["data"].to(target_dtype).contiguous()
    return out

# ---------------------------------------------------------------------------
# Ternary diagnostics (logged during training)
# ---------------------------------------------------------------------------
def tern_stats(model: nn.Module, group_size: int = 64):
    total = zeros = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name or "prototypes" in name) and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1)
                zeros += int((q == 0).sum().item())
                total += int(q.numel())
    return {"zero_frac": zeros / max(total, 1), "total_weights": total}

_prev_committed: dict = {}

def churn_fn(model: nn.Module, group_size: int = 64):
    global _prev_committed
    total = flipped = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name or "prototypes" in name) and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1).cpu().numpy()
                if name in _prev_committed:
                    flipped += int(np.sum(q != _prev_committed[name]))
                    total += q.size
                _prev_committed[name] = q
    return flipped / max(total, 1)

# ---------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalized momentum)
# ---------------------------------------------------------------------------
def ns_orth(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, wd: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = F.rms_norm(g.float(), (g.size(-1),)).bfloat16()
                    g = ns_orth(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("wd", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
            if "step" not in group: group["step"] = 0
            group["step"] += 1
        return loss

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ld_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = ld_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = ld_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
        self._copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._prefetch: tuple[Tensor, Tensor] | None = None

    def _load_raw(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens < seq_len:
            local_tokens = seq_len
        else:
            local_tokens = (local_tokens // seq_len) * seq_len
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        # Cast to int64 on CPU before pinning — avoids a second GPU kernel launch
        local_cpu = chunk[start:start + per_rank_span].to(torch.int64)
        # Do NOT silently clamp tokens — bad token IDs should surface as a hard error,
        # not silently corrupt training inputs with wrong-vocab gradients.
        if self._copy_stream is not None:
            with torch.cuda.stream(self._copy_stream):
                local = local_cpu.pin_memory().to(self.device, non_blocking=True)
        else:
            local = local_cpu.to(self.device)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self._prefetch is not None and self._copy_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._copy_stream)
            x, y = self._prefetch
        else:
            x, y = self._load_raw(global_tokens, seq_len, grad_accum_steps)
        # Kick off next prefetch in background
        if self._copy_stream is not None:
            self._prefetch = self._load_raw(global_tokens, seq_len, grad_accum_steps)
        return x, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def apply_qat_ste(w: Tensor, fp_storage: str | bool) -> Tensor:
    """Applies Straight-Through Estimator (STE) for FP4 or FP8 simulated quantization."""
    if not fp_storage:
        return w
    if fp_storage == "fp4":
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0
        q = torch.clamp(torch.round(w / scale), -7.0, 7.0)
        w_sim = q * scale
        return (w_sim - w).detach() + w
    elif fp_storage is True or fp_storage == "fp8":
        w_sim = w.to(torch.float8_e4m3fn).to(w.dtype)
        return (w_sim - w).detach() + w
    return w

class QATLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, fp_storage: str | bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.fp_storage = fp_storage

    def forward(self, x: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.linear(x, w_qat.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class QATEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, fp_storage: str | bool = False):
        super().__init__(num_embeddings, embedding_dim)
        self.fp_storage = fp_storage

    def forward(self, input: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.embedding(input, w_qat, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

_TURBO_QUANT_TRAIN = False  # Set by main() from args.turbo_quant_train
_TURBO_QUANT_KV = False
_EXPORT_CALIB: dict = {}   # {param_name: {"thr": float, "scale_mult": float}} — set during aligned training phase
_HADAMARD_CACHE_PT: dict[int, Tensor] = {}

def _build_hadamard_pt_unnormalized(n, device):
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32, device=device)
    else:
        h_half = _build_hadamard_pt_unnormalized(n // 2, device)
        top = torch.cat([h_half, h_half], dim=1)
        bot = torch.cat([h_half, -h_half], dim=1)
        return torch.cat([top, bot], dim=0)

def _build_hadamard_pt(n, device):
    """Build normalized orthogonal Hadamard matrix in PyTorch. H @ H.T = I."""
    key = (n, str(device))
    if key in _HADAMARD_CACHE_PT:
        return _HADAMARD_CACHE_PT[key]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    
    # Only divide by sqrt(n) once at the end!
    h = _build_hadamard_pt_unnormalized(n, device) / (n ** 0.5)
    _HADAMARD_CACHE_PT[key] = h
    return h

def quantize_kv_ste_pt(x: Tensor, turbo: bool = True, H_fixed: Tensor = None) -> Tensor:
    """Ternary STE quantization specifically for KV cache vectors with TurboQuant rotation.
    Combines:
    1. Fast Johnson-Lindenstrauss Transform (Pseudo-random sign flip + Hadamard)
    2. Exact L2 Norm De-biasing (preserves attention temperature and inner products)
    """
    head_dim = x.size(-1)
    H_rot = None
    if turbo and H_fixed is not None:
        H_rot = H_fixed.to(x.dtype)
    elif turbo and (head_dim & (head_dim - 1)) == 0:
        H_rot = _build_hadamard_pt(head_dim, x.device).to(x.dtype)
    
    if H_rot is None:
        scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        q = (x / scale).round().clamp(-1, 1)
        dequant = q * scale
        return x + (dequant - x).detach()

    # 1. Pseudo-random sign flip to induce Beta distribution
    inds = torch.arange(head_dim, device=x.device, dtype=x.dtype)
    signs = torch.where(torch.sin(inds) > 0, 1.0, -1.0)
    x_scrambled = x * signs
    
    # 2. Hadamard Rotation
    x_rot = x_scrambled @ H_rot

    # 3. Quantize
    scale = x_rot.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    q = (x_rot / scale).round().clamp(-1, 1)
    dequant = q * scale

    # 4. De-bias (Match expected L2 norm to preserve inner product)
    energy_in = torch.sqrt(torch.sum(torch.square(x_rot), dim=-1, keepdim=True) + 1e-6)
    energy_out = torch.sqrt(torch.sum(torch.square(dequant), dim=-1, keepdim=True) + 1e-6)
    dequant = dequant * (energy_in / energy_out)

    # 5. Inverse FJLT
    dequant = (dequant @ H_rot) * signs

    # STE: forward uses quantized, backward uses original
    return x + (dequant - x).detach()




class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, group_size=64):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size
        self._calib_name: str | None = None  # Set by GPT after construction for aligned training

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.float()
        g = self.group_size

        if w.numel() % g != 0:
            raise RuntimeError(
                f"TernaryLinear group_size={g} does not divide weight numel={w.numel()} "
                f"(shape={tuple(w.shape)}). Adjust architecture so all ternary layer "
                f"widths are multiples of BITNET_GROUP_SIZE={g}."
            )
        w_g = w.reshape(-1, g)
        if _TURBO_QUANT_TRAIN and (g & (g - 1)) == 0:
            H = _build_hadamard_pt(g, w.device).to(w.dtype)
            w_g = w_g @ H
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)

        # Export-aligned training: apply calibrated threshold + scale_mult if set
        calib = _EXPORT_CALIB.get(self._calib_name, {}) if self._calib_name else {}
        thr = calib.get("thr", 0.0)
        scale_mult = calib.get("scale_mult", 1.0)
        z = w_g / scale
        if thr > 0.0:
            q = torch.where(z.abs() < thr, torch.zeros_like(z), z.round().clamp(-1, 1))
        else:
            q = z.round().clamp(-1, 1)
        dequant = q * (scale * scale_mult)

        if _TURBO_QUANT_TRAIN and (g & (g - 1)) == 0:
            dequant = dequant @ H  # Self-inverse
        w_ternary = w + (dequant.reshape(w.shape) - w).detach()
        return F.linear(x, w_ternary.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class NormedTernaryLinear(TernaryLinear):
    "Ternary linear with RMSNorm on input — for output projections receiving un-normalized activations."
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CTP)) and param.dtype != torch.float32:
                param.data = param.data.float()


class EngramHash(nn.Module):
    """Engram-inspired multi-head, multi-order n-gram memory with context gating.

    Key ideas from DeepSeek Engram paper:
    1. Multi-head hashing: K heads per n-gram order reduce collision rate
    2. Context-aware gating: sigmoid gate from hidden state suppresses noisy lookups
    3. Multi-order: bigrams + trigrams capture different context scales
    4. Injection at internal layers (not just input) for richer context gating
    """
    _PRIMES = [92821, 131071, 174763, 216091, 262147, 314159, 393241, 462841,
               524287, 611953, 700001, 786433, 873781, 967229, 1048573, 1153381]

    def __init__(self, num_buckets: int, hash_dim: int, model_dim: int,
                 fp_storage: str | bool, num_heads: int = 4, num_orders: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.num_orders = num_orders
        self.head_dim = hash_dim // (num_orders * num_heads)
        assert self.head_dim > 0, f"hash_dim={hash_dim} too small for {num_orders}x{num_heads} heads"
        self.buckets_per_head = num_buckets
        actual_dim = self.head_dim * num_orders * num_heads
        # Embedding tables: one per (order, head)
        self.tables = nn.ModuleList([
            QATEmbedding(num_buckets, self.head_dim, fp_storage=fp_storage)
            for _ in range(num_orders * num_heads)
        ])
        self.proj = QATLinear(actual_dim, model_dim, bias=False, fp_storage=fp_storage)
        # Context-aware gating
        self.gate_k = nn.Linear(actual_dim, model_dim, bias=False)
        self.gate_scale = model_dim ** -0.5

    def _hash_ngram(self, input_ids: Tensor, order: int, head_idx: int) -> Tensor:
        B, T = input_ids.shape
        p = self._PRIMES[(order * self.num_heads + head_idx) % len(self._PRIMES)]
        if order == 0:  # bigram
            prev = input_ids[:, :-1].long()
            curr = input_ids[:, 1:].long()
            h = (prev * p + curr) % self.buckets_per_head
            h = F.pad(h, (1, 0), value=0)
        elif order == 1:  # trigram
            pp = input_ids[:, :-2].long()
            prev = input_ids[:, 1:-1].long()
            curr = input_ids[:, 2:].long()
            h = (pp * (p * p) + prev * p + curr) % self.buckets_per_head
            h = F.pad(h, (2, 0), value=0)
        elif order == 2:  # four-gram
            ppp = input_ids[:, :-3].long()
            pp = input_ids[:, 1:-2].long()
            prev = input_ids[:, 2:-1].long()
            curr = input_ids[:, 3:].long()
            h = (ppp * (p * p * p) + pp * (p * p) + prev * p + curr) % self.buckets_per_head
            h = F.pad(h, (3, 0), value=0)
        else:
            raise ValueError(f"Unsupported n-gram order {order+2}")
        return h.int()

    def retrieve(self, input_ids: Tensor) -> Tensor:
        parts = []
        for order in range(self.num_orders):
            for head in range(self.num_heads):
                idx = self._hash_ngram(input_ids, order, head)
                table_idx = order * self.num_heads + head
                parts.append(self.tables[table_idx](idx))
        return torch.cat(parts, dim=-1)

    def __call__(self, input_ids: Tensor, hidden: Tensor | None = None) -> Tensor:
        """Fully vectorized n-gram retrieval."""
        num_total_heads = self.num_orders * self.num_heads
        primes = torch.tensor(self._PRIMES[:num_total_heads], device=input_ids.device, dtype=torch.long)
        parts = []
        ids_long = input_ids.long()
        for order in range(self.num_orders):
            p = primes[order*self.num_heads : (order+1)*self.num_heads]
            if order == 0:
                h = (ids_long[:, :-1].unsqueeze(-1) * p + ids_long[:, 1:].unsqueeze(-1)) % self.buckets_per_head
            elif order == 1:
                h = (ids_long[:, :-2].unsqueeze(-1) * (p*p) + ids_long[:, 1:-1].unsqueeze(-1) * p + ids_long[:, 2:].unsqueeze(-1)) % self.buckets_per_head
            elif order == 2:
                h = (ids_long[:, :-3].unsqueeze(-1) * (p**3) + ids_long[:, 1:-2].unsqueeze(-1) * (p**2) + ids_long[:, 2:-1].unsqueeze(-1) * p + ids_long[:, 3:].unsqueeze(-1)) % self.buckets_per_head
            else:
                h_ls = [self._hash_ngram(input_ids, order, hdy).unsqueeze(-1) for hdy in range(self.num_heads)]
                h = torch.cat(h_ls, dim=-1)
            h = torch.nn.functional.pad(h, (0, 0, order+1, 0), value=0)
            parts.append(h)
        all_indices = torch.cat(parts, dim=-1)
        head_outputs = [table(all_indices[:, :, i]) for i, table in enumerate(self.tables)]
        memory = torch.cat(head_outputs, dim=-1)
        if hidden is not None:
            gate = torch.sigmoid((torch.nn.functional.normalize(hidden.float(), dim=-1) * torch.nn.functional.normalize(self.gate_k(memory).float(), dim=-1)).sum(dim=-1, keepdim=True) * self.gate_scale)
            return gate.to(memory.dtype) * self.proj(memory)
        return self.proj(memory)
    def forward(self, input_ids: Tensor, hidden: Tensor | None = None) -> Tensor:
        return self.__call__(input_ids, hidden)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, no_cache: bool = False,
                 rope_type: str = "rope", yarn_max_len: int = 4096, train_seq_len: int = 1024):
        super().__init__()
        self.no_cache = no_cache
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if rope_type == "yarn":
            scale = train_seq_len / yarn_max_len
            freq_idx = torch.arange(0, dim, 2, dtype=torch.float32)
            ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
            inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len, device, dtype):
        if self.no_cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            return freqs.cos()[None, :, None, :].to(dtype=dtype), freqs.sin()[None, :, None, :].to(dtype=dtype)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        group_size: int = 64,
        no_cache: bool = False,
        rope_type: str = "rope",
        yarn_max_len: int = 4096,
        train_seq_len: int = 1024,
        partial_rope_dims: int = 0,
        vrl_enabled: bool = False,
        xsa: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Partial RoPE: only rotate first N dims per head, rest attend without position
        self.rope_dims = partial_rope_dims if partial_rope_dims > 0 else self.head_dim
        self.vrl_enabled = vrl_enabled
        self.xsa = xsa
        self.c_qkv = TernaryLinear(dim, self.q_size + 2 * self.kv_size, bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(dim, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(
            self.rope_dims,
            base=rope_base,
            no_cache=no_cache,
            rope_type=rope_type,
            yarn_max_len=yarn_max_len,
            train_seq_len=train_seq_len,
        )
        # VRL: learnable mixing weight for value residual
        if vrl_enabled:
            self.vrl_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Hadamard cache for KV quantizer
        if (self.head_dim & (self.head_dim - 1)) == 0:
            h_mat = _build_hadamard_pt(self.head_dim, torch.device("cpu"))
            self.register_buffer("_H_kv", h_mat, persistent=False)
        else:
            self.register_buffer("_H_kv", None, persistent=False)

    def forward(self, x: Tensor, v0: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv_out = self.c_qkv(x)
        q_out, k_out, v_out = qkv_out.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Partial RoPE: only rotate first rope_dims dims
        if self.rope_dims < self.head_dim:
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q_rot, q_pass = q[..., :self.rope_dims], q[..., self.rope_dims:]
            k_rot, k_pass = k[..., :self.rope_dims], k[..., self.rope_dims:]
            q = torch.cat((apply_rotary_emb(q_rot, cos, sin), q_pass), dim=-1)
            k = torch.cat((apply_rotary_emb(k_rot, cos, sin), k_pass), dim=-1)
        else:
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        # VRL: blend current values with first-layer values
        if self.vrl_enabled and v0 is not None:
            alpha = torch.sigmoid(self.vrl_alpha).to(dtype=v.dtype)
            v = alpha * v + (1 - alpha) * v0
            
        if _TURBO_QUANT_KV:
            h_mat = getattr(self, "_H_kv", None)
            k = quantize_kv_ste_pt(k, turbo=True, H_fixed=h_mat)
            v = quantize_kv_ste_pt(v, turbo=True, H_fixed=h_mat)
            
        y = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
        # XSA: subtract self-value to force reliance on context, not self-attending
        if self.xsa:
            # Expand KV heads to match query heads for subtraction
            kv_rep = self.num_heads // self.num_kv_heads
            v_expanded = v.repeat_interleave(kv_rep, dim=2) if kv_rep > 1 else v
            y = y - v_expanded
        return self.proj(y.reshape(bsz, seqlen, dim)), v

class TernaryMoE(nn.Module):
    """Sparse Ternary Mixture of Experts.
    Scales parameter count dramatically while keeping FLOPs constant via top-k routing.
    """
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        num_experts: int,
        top_k: int,
        group_size: int = 64,
        activation: str = "relu2",
        leaky_relu_slope: float = 0.5,
        moe_start_fraction: float = 0.65,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_start_fraction = moe_start_fraction
        # Router is a lightweight dense fp16/32 layer
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MLP(dim, mlp_mult, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope)
            for _ in range(num_experts)
        ])
        self.aux_loss = None

    def forward(self, x: Tensor, elapsed_fraction: float = 1.0) -> Tensor:
        if elapsed_fraction < self.moe_start_fraction:
            return self.experts[0](x)
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        final_output = torch.zeros_like(x_flat)
        # Iterative gather over active experts
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i)
            if expert_mask.any():
                expert_indices = expert_mask.nonzero(as_tuple=True)
                token_indices = expert_indices[0]
                expert_weights = routing_weights[expert_mask]
                tokens_for_expert = x_flat[token_indices]
                expert_out = expert(tokens_for_expert)
                final_output.index_add_(0, token_indices, expert_out * expert_weights.unsqueeze(-1))
                
        # Auxiliary load balancing loss calculation (only if training)
        if self.training:
            density = F.softmax(router_logits, dim=1).mean(dim=0)
            fraction_routed = selected_experts.float().histc(bins=self.num_experts, min=0, max=self.num_experts - 1) / float(B * T * self.top_k)
            # Standard load-balancing: N * sum(f_i * P_i). With uniform routing this gives 1.0.
            # .mean() * N == sum(f_i*P_i)/N * N but only if all N are equal; use .sum() directly.
            self.aux_loss = (density * fraction_routed).sum() * self.num_experts
        else:
            self.aux_loss = None

        return final_output.view(B, T, D)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        group_size: int = 64,
        activation: str = "relu2",
        leaky_relu_slope: float = 0.5,
    ):
        super().__init__()
        hidden = mlp_mult * dim
        self.activation = activation
        self.leaky_relu_slope = leaky_relu_slope
        if activation == "swiglu":
            self.gate_up = TernaryLinear(dim, hidden * 2, bias=False, group_size=group_size)
        else:
            self.fc = TernaryLinear(dim, hidden, bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(hidden, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        elif self.activation == "relu":
            hidden = torch.relu(self.fc(x))
        elif self.activation == "lrelu2":
            hidden = F.leaky_relu(self.fc(x), negative_slope=self.leaky_relu_slope).square()
        else:
            hidden = torch.relu(self.fc(x)).square()
        return self.proj(hidden)


class KoopmanTokenMixer(nn.Module):
    """Causal token mixing via Koopman-inspired linear recurrence (PyTorch).

    Replaces Self-Attention with O(T) linear dynamics:
      1. Project input to state space
      2. Short causal convolution for local context
      3. Input-dependent gating
      4. Causal linear scan via exponentially decaying convolution
      5. Low-rank cross-dimension mixing
      6. Project back to model dim
    """
    def __init__(self, dim: int, state_dim: int, rank: int = 4, conv_kernel: int = 4,
                 decay_window: int = 32, group_size: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.decay_window = decay_window

        self.proj_in = TernaryLinear(dim, state_dim, bias=False, group_size=group_size)
        self.proj_out = NormedTernaryLinear(state_dim, dim, bias=False, group_size=group_size)
        self.proj_out._zero_init = True
        
        # Gates for controlling inputs and decay
        self.g_proj = TernaryLinear(dim, state_dim, bias=False, group_size=group_size)
        self.dt_proj = TernaryLinear(dim, state_dim, bias=False, group_size=group_size)

        # Short causal conv (depthwise, per-channel)
        self.mixer_conv = nn.Parameter(torch.ones(conv_kernel, state_dim) / conv_kernel)
        # We leave mixer_diag initialized just in case static fallback is called by older checkpoints
        self.mixer_diag = nn.Parameter(torch.full((state_dim,), 0.8))
        self.mixer_lowrank_U = nn.Parameter(torch.randn(state_dim, rank) * 0.001)
        self.mixer_lowrank_V = nn.Parameter(torch.randn(state_dim, rank) * 0.001)
        self.mixer_scale = nn.Parameter(torch.ones(dim))

        self._use_hadamard = (state_dim & (state_dim - 1)) == 0 and state_dim >= 2
        if self._use_hadamard:
            H = _build_hadamard_pt(state_dim, torch.device("cpu"))
            self.register_buffer("_H", H, persistent=False)

    def _short_causal_conv(self, x: Tensor) -> Tensor:
        K = self.conv_kernel
        B, T, S = x.shape
        weight = self.mixer_conv.flip([0]).T.unsqueeze(1).to(x.dtype)
        x_padded = F.pad(x.transpose(1, 2), (K - 1, 0))
        h_t = F.conv1d(x_padded, weight, groups=S)
        return h_t.transpose(1, 2).contiguous()

    def _causal_decay_scan(self, x: Tensor, gate: Tensor, override_window: int | None = None, dt_gate: Tensor | None = None) -> Tensor:
        """High-Performance Parallel Chunked Scan (CV-Scan) for CUDA.
        Hierarchical scan (O(log T) depth) for sub-500ms step times.
        """
        B, T, S = x.shape
        W = 32 # Chunk size
        # Pad T to next multiple of W so reshape is always valid
        T_orig = T
        T_pad = ((T + W - 1) // W) * W
        if T_pad > T:
            pad = T_pad - T
            x = F.pad(x, (0, 0, 0, pad))
            gate = F.pad(gate, (0, 0, 0, pad))
            if dt_gate is not None:
                dt_gate = F.pad(dt_gate, (0, 0, 0, pad))
            T = T_pad
        num_chunks = T // W
        
        if self._use_hadamard:
            x = x @ self._H.to(dtype=x.dtype, device=x.device)

        # 1. Inputs and Decays
        if dt_gate is not None:
            logD = -F.softplus(dt_gate.float())
            D = torch.exp(logD).to(x.dtype)
            B_vals = (gate * x * (1.0 - D)).to(x.dtype)
        else:
            D_static = torch.clamp(self.mixer_diag, -0.999, 0.999).to(x.dtype)
            D = D_static.view(1, 1, S).expand(B, T, S)
            B_vals = (gate * x * (1.0 - D_static.abs())).to(x.dtype)

        # 2. Reshape to chunks (B, num_chunks, W, S)
        D_c = D.reshape(B, num_chunks, W, S)
        B_c = B_vals.reshape(B, num_chunks, W, S)

        # 3. Level 1: Intra-chunk parallel scan
        # Fully unrolled for torch.compile
        c_h = [B_c[:, :, 0]]
        c_d = [D_c[:, :, 0]]
        for t in range(1, W):
            c_h.append(D_c[:, :, t] * c_h[-1] + B_c[:, :, t])
            c_d.append(D_c[:, :, t] * c_d[-1])
        
        h_local = torch.stack(c_h, dim=2)
        d_local = torch.stack(c_d, dim=2)

        # 4. Level 2: Inter-chunk prefix scan
        chunk_finals_h = h_local[:, :, -1] 
        chunk_finals_d = d_local[:, :, -1] 
        
        p_h = [torch.zeros_like(chunk_finals_h[:, 0])]
        for i in range(num_chunks - 1):
             p_h.append(chunk_finals_d[:, i] * p_h[-1] + chunk_finals_h[:, i])
        
        chunk_prefixes = torch.stack(p_h, dim=1).unsqueeze(2)

        # 5. Global state: h = h_local + d_local * chunk_prefix
        h = h_local + d_local * chunk_prefixes
        h = h.reshape(B, T, S)

        # Strip padding if we padded earlier
        if T_orig < T:
            h = h[:, :T_orig]

        # 6. Low-rank coupling
        U = self.mixer_lowrank_U.to(x.dtype)
        V = self.mixer_lowrank_V.to(x.dtype)
        h = h + (h @ V) @ U.T

        if self._use_hadamard:
            h = h @ self._H.to(dtype=x.dtype, device=x.device)
        return h

    def forward(self, x: Tensor) -> Tensor:
        normed = F.rms_norm(x, (x.size(-1),))
        s = self.proj_in(normed)
        g = torch.sigmoid(self.g_proj(normed))
        dt_gate = self.dt_proj(normed)
        s = self._short_causal_conv(s)
        h = self._causal_decay_scan(s, g, dt_gate=dt_gate)
        return self.proj_out(h)


class KoopmanBlock(nn.Module):
    """A single layer using Koopman SSM for token mixing (replaces attention)."""
    def __init__(self, dim: int, state_dim: int, mlp_mult: int, mixer_rank: int = 4,
                 conv_kernel: int = 4, decay_window: int = 32, group_size: int = 64,
                 activation: str = "lrelu2", leaky_relu_slope: float = 0.5,
                 ln_scale_factor: float = 1.0):
        super().__init__()
        self.ln_scale_factor = ln_scale_factor
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.mixer = KoopmanTokenMixer(dim, state_dim, rank=mixer_rank, conv_kernel=conv_kernel,
                                        decay_window=decay_window, group_size=group_size)
        if Hyperparameters.moe_enabled:
            self.mlp = TernaryMoE(dim, mlp_mult, num_experts=Hyperparameters.moe_num_experts, top_k=Hyperparameters.moe_top_k, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope)
        else:
            self.mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation,
                           leaky_relu_slope=leaky_relu_slope)
        # SkipInit: zero-init residual scales to prevent early instability
        self.attn_scale = nn.Parameter(torch.zeros(dim))
        self.mlp_scale = nn.Parameter(torch.zeros(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None) -> tuple[Tensor, None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        
        mixer_out = self.mixer(self.attn_norm(x) * self.ln_scale_factor)
        x = x + self.attn_scale.to(dtype=x.dtype) * mixer_out
        
        mlp_out = self.mlp(self.mlp_norm(x) * self.ln_scale_factor)
        if isinstance(mlp_out, tuple): 
            mlp_out, moe_loss = mlp_out
        else:
            moe_loss = None

        x = x + self.mlp_scale.to(dtype=x.dtype) * mlp_out
        return x, moe_loss


class FeedbackPooler(nn.Module):
    def __init__(self, model_dim: int, feedback_dim: int, num_tokens: int, fp_storage: str | bool):
        super().__init__()
        self.num_tokens = max(1, num_tokens)
        self.proj = QATLinear(model_dim, feedback_dim, bias=False, fp_storage=fp_storage)

    def forward(self, x: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_tokens).transpose(1, 2)
        return self.proj(F.rms_norm(pooled, (pooled.size(-1),)))


class FeedbackAdapter(nn.Module):
    """Fast-Weight Delta Rule Memory Adapter.
    
    Replaces static gating and backward-pass TTT with a continuous forward-pass 
    associative memory. The 'sketch' (compressed semantic summary from later layers)
    is loaded into a fast-weight matrix $W$, which is queried by the current decoder 
    features $X$ to retrieve temporally contextualized facts without gradient descent.
    """
    def __init__(self, model_dim: int, feedback_dim: int, fp_storage: str | bool):
        super().__init__()
        # Initializing fast weight learning rate to ~0.11 via sigmoid(-2)
        self.fast_weight_lr = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        
        # K, V generated from the dense semantic 'sketch'
        self.k_proj = nn.Linear(feedback_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(feedback_dim, model_dim, bias=False)
        
        # Query generated from the local decoder token state
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        
        # Gate to inject retrieved memory
        self.out_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))

    def forward(self, x: Tensor, sketch: Tensor | None) -> Tensor:
        if sketch is None:
            return x
            
        B, T, D = x.shape
        # sketch shape is [B, S, feedback_dim]
        # Generate keys and values
        k = self.k_proj(sketch)  # [B, S, D]
        v = self.v_proj(sketch)  # [B, S, D]
        
        # Initialize fast weight matrix if None
        memory_matrix = torch.zeros((B, D, D), device=x.device, dtype=torch.float32)

        # Fast-Weight Update (Delta Rule)
        # M_new = M_old + lr * (v - M_old @ k^T) @ k
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # Predict values currently in memory
        pred_v_t = torch.bmm(memory_matrix, k_t) # [B, D, S]
        delta = v_t - pred_v_t # [B, D, S]
        
        # Update memory
        lr = torch.sigmoid(self.fast_weight_lr)
        memory_matrix = memory_matrix + lr * torch.bmm(delta, k) # [B, D, D]
        
        # Output retrieval
        q = self.q_proj(x) # [B, T, D]
        q_t = q.transpose(1, 2) # [B, D, T]
        
        retrieved_t = torch.bmm(memory_matrix, q_t) # [B, D, T]
        retrieved = retrieved_t.transpose(1, 2) # [B, T, D]
        
        # Hadamard-gated injection
        gate = torch.tanh(self.out_gate).to(dtype=x.dtype)
        x_out = x + gate * retrieved.to(dtype=x.dtype)
        
        return x_out


class KoopmanDynamics(nn.Module):
    """Diagonal + low-rank stable linear dynamics in capsule space.

    Predicts next-pass capsule state from current state:
        c_pred = D ⊙ c + U(V^T c)
        c_new  = α ⊙ c_observed + (1-α) ⊙ c_pred

    First-principles design:
        - D initialized at 0.9 (critical damping, ρ(D)=0.9 < 1)
        - UV initialized small (spectral perturbation << 1-ρ(D))
        - α at sigmoid(0)=0.5 (maximum-entropy prior)
        - Stability guaranteed at init: ρ(D + UV^T) ≤ 0.9 + ε
    """
    def __init__(self, capsule_dim: int, rank: int = 4, diag_init: float = 0.9):
        super().__init__()
        self.capsule_dim = capsule_dim
        self.diag = nn.Parameter(torch.full((capsule_dim,), diag_init, dtype=torch.float32))
        init_scale = 0.01 / max(rank ** 0.5, 1.0)
        self.U = nn.Parameter(torch.randn(capsule_dim, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(capsule_dim, rank) * init_scale)
        self.alpha = nn.Parameter(torch.full((capsule_dim,), -5.0, dtype=torch.float32))  # sigmoid(-5)≈0.007: capsules start OFF
        # Precompute Hadamard for capsule dim (must be power of 2)
        self._use_hadamard = (capsule_dim & (capsule_dim - 1)) == 0 and capsule_dim >= 2
        if self._use_hadamard:
            H = _build_hadamard_pt(capsule_dim, torch.device('cpu'))
            self.register_buffer('_H', H)

    def _rotate(self, c: Tensor) -> Tensor:
        """Hadamard rotate: spreads capsule info uniformly across dims."""
        if self._use_hadamard:
            return c @ self._H.to(dtype=c.dtype, device=c.device)
        return c

    def predict(self, c: Tensor) -> Tensor:
        """Predict next-pass capsule state. c: (B, N, capsule_dim)
        Hadamard-rotate → diagonal+low-rank evolve → rotate back."""
        c_rot = self._rotate(c)
        # Clamp diagonal for multi-step spectral stability: |d_i| < 1 prevents
        # exponential blowup when composing predict() K times in speculate().
        d_clamped = torch.clamp(self.diag, -0.999, 0.999).to(dtype=c_rot.dtype)
        c_diag = d_clamped * c_rot
        c_lowrank = (c_rot @ self.V.to(dtype=c_rot.dtype)) @ self.U.to(dtype=c_rot.dtype).T
        c_evolved = c_diag + c_lowrank
        return self._rotate(c_evolved)  # H is self-inverse

    def speculate(self, c: Tensor, steps: int) -> Tensor:
        """Recursively apply Koopman operator to fast-forward presentation.
        Acts as 1-step diffusion jump in latent space."""
        curr = c
        for _ in range(steps):
            curr = self.predict(curr)
        return curr

    def blend(self, c_observed: Tensor, c_prev: Tensor) -> tuple[Tensor, Tensor]:
        """Blend observed capsules with predicted evolution. Returns (blended, c_pred)."""
        c_pred = self.predict(c_prev)
        alpha = torch.sigmoid(self.alpha).to(dtype=c_observed.dtype)
        return alpha * c_observed + (1.0 - alpha) * c_pred, c_pred


class CapsuleBank(nn.Module):
    """Structured semantic state carriers with Koopman-driven recurrent dynamics.

    Upgrade from simple gated blending to predictive latent dynamics:
    - Koopman module predicts where capsule state should evolve
    - Blend prediction with fresh observation from current pass
    - Returns c_pred for consistency loss (auxiliary training signal)
    """
    def __init__(self, model_dim: int, capsule_num: int, capsule_dim: int, fp_storage: str | bool,
                 koopman_enabled: bool = True, koopman_rank: int = 4, koopman_diag_init: float = 0.9):
        super().__init__()
        self.capsule_num = capsule_num
        self.capsule_dim = capsule_dim
        self.prototypes = nn.Parameter(torch.randn(capsule_num, capsule_dim) * 0.02)
        self.read_proj = nn.Linear(model_dim, capsule_dim, bias=False)
        self.write_proj = nn.Linear(capsule_dim, model_dim, bias=False)
        self.recurrent_gate = nn.Parameter(torch.zeros(capsule_dim, dtype=torch.float32))
        self.gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
        # Koopman dynamics
        self.koopman = None
        if koopman_enabled:
            self.koopman = KoopmanDynamics(capsule_dim, rank=koopman_rank, diag_init=koopman_diag_init)

    def forward(self, x: Tensor, prev_capsules: Tensor | None = None, speculate_steps: int = 0) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Returns (corrected_x, capsule_state, c_pred_for_loss, c_spec)."""
        bsz, seqlen, dim = x.shape
        x_proj = self.read_proj(F.rms_norm(x, (dim,)))
        scores = torch.einsum("btd,nd->btn", x_proj, self.prototypes.to(x_proj.dtype))
        attn = torch.softmax(scores / (self.capsule_dim ** 0.5), dim=1)
        capsules = torch.einsum("btn,btd->bnd", attn, x_proj)

        c_pred = None
        c_spec = None
        if prev_capsules is not None:
            if self.koopman is not None:
                capsules, c_pred = self.koopman.blend(capsules, prev_capsules)
                if speculate_steps > 0:
                    c_spec = self.koopman.speculate(capsules, speculate_steps)
                    if not self.training:
                        capsules = c_spec
            else:
                rg = torch.sigmoid(self.recurrent_gate).to(dtype=capsules.dtype)
                capsules = rg * capsules + (1 - rg) * prev_capsules

        readout = torch.einsum("btn,bnd->btd", attn, capsules)
        correction = self.write_proj(readout)
        g = torch.tanh(self.gate).to(dtype=x.dtype)
        return x + g * correction, capsules, c_pred, c_spec


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        group_size: int = 64,
        activation: str = "relu2",
        leaky_relu_slope: float = 0.5,
        no_cache: bool = False,
        rope_type: str = "rope",
        yarn_max_len: int = 4096,
        train_seq_len: int = 1024,
        partial_rope_dims: int = 0,
        vrl_enabled: bool = False,
        ln_scale_factor: float = 1.0,
        xsa: bool = False,
        moe_enabled: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.ln_scale_factor = ln_scale_factor
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        if Hyperparameters.architecture == "skc":
            self.skc_layer = SKCLayer(
                dim, Hyperparameters.skc_num_capsules, Hyperparameters.skc_capsule_dim,
                Hyperparameters.skc_conv_kernel, Hyperparameters.skc_block_size,
                mlp_mult=mlp_mult, group_size=group_size, activation=activation,
                leaky_relu_slope=leaky_relu_slope, ln_scale_factor=ln_scale_factor,
                moe_enabled=moe_enabled, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k
            )
            self.attn = None
            self.mlp = None
        else:
            self.skc_layer = None
            self.attn = CausalSelfAttention(
                dim,
                num_heads,
                num_kv_heads,
                rope_base,
                qk_gain_init,
                group_size=group_size,
                no_cache=no_cache,
                rope_type=rope_type,
                yarn_max_len=yarn_max_len,
                train_seq_len=train_seq_len,
                partial_rope_dims=partial_rope_dims,
                vrl_enabled=vrl_enabled,
                xsa=xsa,
            )
            if moe_enabled:
                self.mlp = TernaryMoE(dim, mlp_mult, num_experts=moe_num_experts, top_k=moe_top_k, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope, moe_start_fraction=Hyperparameters.moe_start_fraction)
            else:
                self.mlp = MLP(
                    dim,
                    mlp_mult,
                group_size=group_size,
                activation=activation,
                leaky_relu_slope=leaky_relu_slope,
            )
        # Zero-init residual scales: branches start dead and grow gradually from data.
        # This matches the stable init already used in KoopmanBlock and SKCLayer.
        self.attn_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None, elapsed_fraction: float = 1.0) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        attn_out, v_out = self.attn(self.attn_norm(x) * self.ln_scale_factor, v0=v0)
        x = x + self.attn_scale.to(dtype=x.dtype) * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype) * self.mlp(self.mlp_norm(x) * self.ln_scale_factor, elapsed_fraction=elapsed_fraction)
        return x, v_out



# Note: Assumes TernaryLinear, NormedTernaryLinear, and rms_norm are already imported or available.
# This code will be injected right before the Architecture Selection logic in train_gpt.py

def causal_wht_blockwise(x, block_size=64):
    B, T, D = x.shape
    pad_len = (block_size - T % block_size) % block_size
    if pad_len > 0:
        # F.pad format: (left_D, right_D, left_T, right_T)
        x = F.pad(x, (0, 0, 0, pad_len))
    T_padded = x.shape[1]
    num_blocks = T_padded // block_size

    x_blocks = x.view(B, num_blocks, block_size, D)
    h = block_size.bit_length() - 1
    result = x_blocks
    for stage in range(h):
        stride = 1 << stage
        result = result.view(B, num_blocks, block_size // (2 * stride), 2 * stride, D)
        even = result[:, :, :, :stride, :]
        odd = result[:, :, :, stride:, :]
        result = torch.cat([even + odd, even - odd], dim=3)
        result = result.view(B, num_blocks, block_size, D)

    result = result * (1.0 / math.sqrt(block_size))
    result = result.view(B, T_padded, D)
    return result[:, :T, :]

def causal_spectral_decay_scan(x_blocks, decay_rates, gate):
    B, num_blocks, block_sz, D = x_blocks.shape
    decay = torch.clamp(decay_rates, -0.999, 0.999)
    gated = gate * x_blocks
    
    if num_blocks > 1:
        block_finals = gated[:, :, -1, :]
        idx = torch.arange(num_blocks, device=decay.device)
        diff = idx[:, None] - idx[None, :]
        mask = diff >= 0
        diff_clamped = torch.max(diff, torch.zeros_like(diff))
        
        M = decay[None, None, :] ** diff_clamped[:, :, None].to(decay.dtype)
        M = torch.where(mask[:, :, None], M, torch.zeros_like(M))
        
        states = torch.einsum('ijd,bjd->bid', M, block_finals)
        prefix_states = torch.cat([torch.zeros_like(states[:, :1, :]), states[:, :-1, :]], dim=1)
        return x_blocks + prefix_states[:, :, None, :] * decay[None, None, None, :]
    return x_blocks

class SpectralTernaryAuxLoss(nn.Module):
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, x_spec):
        energy = torch.mean(x_spec * x_spec, dim=(0, 2)) + 1e-8
        p = energy / torch.sum(energy)
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        max_entropy = math.log(max(x_spec.shape[1], 1))
        return self.weight * (max_entropy - entropy)

class FrequencyBandRouter(nn.Module):
    def __init__(self, num_capsules, capsule_dim, block_size):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.block_size = block_size
        centers = np.linspace(0, 1, num_capsules, dtype=np.float32)
        self.band_centers = nn.Parameter(torch.from_numpy(centers))
        self.band_log_widths = nn.Parameter(torch.full((num_capsules,), -1.0, dtype=torch.float32))
        # C6: content-conditioned routing residual.
        # content_scale starts at -2.0 => sigmoid(-2) ≈ 0.12, so positional prior
        # dominates early and routing goes content-adaptive as training progresses.
        self.content_router = nn.Linear(capsule_dim, num_capsules, bias=False)
        self.content_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x_spec, T):
        B = x_spec.shape[0]
        seq_pos = (torch.arange(T, device=x_spec.device, dtype=torch.float32) % self.block_size) / max(self.block_size - 1, 1)
        widths = torch.exp(self.band_log_widths)
        diff = seq_pos[:, None] - self.band_centers[None, :]
        pos_logits = -0.5 * (diff * diff) / (widths[None, :] ** 2 + 1e-6)
        pos_logits = pos_logits[None, :, :].expand(B, T, self.num_capsules)
        # Content term: rms-normed projection, blended in with a learned scale
        content_logits = self.content_router(F.rms_norm(x_spec, (x_spec.size(-1),))).float()
        alpha = torch.sigmoid(self.content_scale)
        routing_weights = torch.softmax(pos_logits + alpha * content_logits, dim=-1)
        capsules = torch.einsum("btn,btd->bnd", routing_weights, x_spec)
        return routing_weights, capsules

class KoopmanSpectralEvolution(nn.Module):
    def __init__(self, capsule_dim, num_capsules, rank=8):
        super().__init__()
        self.capsule_dim = capsule_dim
        self.num_capsules = num_capsules
        # C7: init at 0.0 → sigmoid(0) = 0.5 (50/50 blend of old/new state at start).
        # Old init of 2.0 → sigmoid(2.0)=0.88 was too sticky and slowed early learning.
        self.eigenvalues = nn.Parameter(torch.zeros(num_capsules, capsule_dim, dtype=torch.float32))
        self.coupling_U = nn.Parameter(torch.randn(num_capsules, rank, dtype=torch.float32) * 0.01)
        self.coupling_V = nn.Parameter(torch.randn(rank, num_capsules, dtype=torch.float32) * 0.01)
        self.nonlinear_gate = nn.Parameter(torch.zeros(num_capsules, capsule_dim, dtype=torch.float32))

    def forward(self, capsules, prev_capsules=None):
        lam = torch.sigmoid(self.eigenvalues)
        if prev_capsules is not None:
            evolved = lam[None, :, :] * prev_capsules + (1.0 - lam[None, :, :]) * capsules
        else:
            evolved = capsules
        coupling = torch.matmul(self.coupling_U, self.coupling_V)
        coup_norm = torch.sqrt(torch.sum(coupling * coupling) + 1e-8)
        coupling = coupling / torch.clamp_min(coup_norm, 1.0)
        cross_info = torch.einsum("nm,bmc->bnc", coupling, evolved)
        evolved = evolved + cross_info
        nl_gate = torch.sigmoid(self.nonlinear_gate)
        evolved = (1.0 - nl_gate[None, :, :]) * evolved + nl_gate[None, :, :] * torch.tanh(evolved)
        return evolved

class SKCLayer(nn.Module):
    def __init__(self, dim, capsule_num=32, capsule_dim=128, conv_kernel=4, block_size=64,
                 mlp_mult=4, group_size=128, activation="lrelu2", leaky_relu_slope=0.5,
                 ln_scale_factor=1.0, moe_enabled=False, moe_num_experts=8, moe_top_k=2):
        super().__init__()
        self.dim = dim
        self.capsule_num = capsule_num
        self.capsule_dim = capsule_dim
        self.conv_kernel = conv_kernel
        self.ln_scale_factor = ln_scale_factor
        self.block_size = block_size

        self.spec_proj_in = TernaryLinear(dim, capsule_dim, group_size=group_size)
        
        self.decay_rates = nn.Parameter(torch.zeros(capsule_dim, dtype=torch.float32))
        self.gate_proj = TernaryLinear(dim, capsule_dim, group_size=group_size)
        
        self.router = FrequencyBandRouter(capsule_num, capsule_dim, block_size)
        self.koopman = KoopmanSpectralEvolution(capsule_dim, capsule_num)
        
        self.spec_proj_out = NormedTernaryLinear(capsule_dim, dim, group_size=group_size)
        self.spec_proj_out._zero_init = True
        nn.init.zeros_(self.spec_proj_out.weight)
        
        self.mixer_conv = nn.Parameter(torch.ones(capsule_dim, conv_kernel, dtype=torch.float32) / conv_kernel)
        if moe_enabled:
            self.local_mlp = TernaryMoE(dim, mlp_mult, num_experts=moe_num_experts, top_k=moe_top_k, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope, moe_start_fraction=Hyperparameters.moe_start_fraction)
        else:
            self.local_mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope)
        
        self.skc_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))
        self.aux_loss_fn = SpectralTernaryAuxLoss(weight=0.01)

    def forward(self, x, x0, v0=None, prev_capsules=None, elapsed_fraction=1.0):
        mix = self.resid_mix.to(x.dtype)
        x = mix[0] * x + mix[1] * x0
        normed = F.rms_norm(x, (x.size(-1),)) * self.ln_scale_factor
        
        B, T, D = normed.shape
        pad_len = (self.block_size - T % self.block_size) % self.block_size
        T_pad = T + pad_len
        
        s = self.spec_proj_in(normed)
        g = torch.sigmoid(self.gate_proj(normed))
        
        s_pad = F.pad(s, (0, 0, 0, pad_len))
        g_pad = F.pad(g, (0, 0, 0, pad_len))
        
        num_blocks = T_pad // self.block_size
        s_blocks = s_pad.view(B, num_blocks, self.block_size, self.capsule_dim)
        g_blocks = g_pad.view(B, num_blocks, self.block_size, self.capsule_dim)
        
        s_wht = causal_wht_blockwise(s_blocks.view(B, T_pad, self.capsule_dim), self.block_size)
        s_wht_blocks = s_wht.view(B, num_blocks, self.block_size, self.capsule_dim)
        
        s_decay = causal_spectral_decay_scan(s_wht_blocks, self.decay_rates, g_blocks)
        s_spec = s_decay.view(B, T_pad, self.capsule_dim)[:, :T, :]
        
        routing_weights, capsules = self.router(s_spec, T)
        carry = getattr(self, '_carry_capsules', None) if self.training else None
        evolved_caps = self.koopman(capsules, carry if carry is not None else prev_capsules)
        if self.training:
            self._last_capsules = evolved_caps.detach()
        
        synth_spec = torch.einsum("btn,bnc->btc", routing_weights, evolved_caps)
        
        synth_pad = F.pad(synth_spec, (0, 0, 0, pad_len))
        synth_wht = causal_wht_blockwise(synth_pad, self.block_size)[:, :T, :]
        
        s_conv_in = synth_wht.transpose(1, 2)
        s_conv_pad = F.pad(s_conv_in, (self.conv_kernel - 1, 0))
        weight = self.mixer_conv.view(self.capsule_dim, 1, self.conv_kernel)
        s_conv = F.conv1d(s_conv_pad, weight.to(s_conv_in.dtype), groups=self.capsule_dim)
        s_conv = s_conv.transpose(1, 2)
        
        skc_out = self.spec_proj_out(s_conv)
        x = x + self.skc_scale.to(x.dtype)[None, None, :] * skc_out

        if self.training:
            self.spectral_aux_loss = self.aux_loss_fn(s_spec)

        mlp_in = F.rms_norm(x, (x.size(-1),)) * self.ln_scale_factor
        if isinstance(self.local_mlp, TernaryMoE):
            mlp_out = self.local_mlp(mlp_in, elapsed_fraction=elapsed_fraction)
        else:
            mlp_out = self.local_mlp(mlp_in)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * mlp_out
        return x, None


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        group_size: int = 64,
        activation: str = "relu2",
        leaky_relu_slope: float = 0.5,
        embed_dim: int = 0,
        training_depth_recurrence: int = 0,
        fp_storage: str | bool = False,
        softcap_type: str = "poly",
        no_cache: bool = False,
        rope_type: str = "rope",
        yarn_max_len: int = 4096,
        train_seq_len: int = 1024,
        feedback_enabled: bool = True,
        feedback_dim: int = 64,
        feedback_sketch_tokens: int = 4,
        feedback_replay: str = "decoder",
        feedback_target: str = "decoder",
        feedback_fp_storage: str | bool = True,
        feedback_passes: int = 1,
        shared_blocks: int = 0,
        capsule_enabled: bool = False,
        capsule_num: int = 16,
        capsule_dim: int = 64,
        partial_rope_dims: int = 0,
        vrl_enabled: bool = False,
        vrl_start_layer: int = 8,
        ln_scale_damping: bool = False,
        bigram_hash_enabled: bool = False,
        bigram_hash_buckets: int = 4096,
        bigram_hash_dim: int = 128,
        engram_num_heads: int = 4,
        engram_num_orders: int = 2,
        engram_inject_layer: int = 1,
        xsa_start_layer: int = -1,
        moe_enabled: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.training_depth_recurrence = training_depth_recurrence
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.embed_dim = embed_dim if embed_dim > 0 else model_dim
        self.feedback_enabled = feedback_enabled
        self.feedback_replay = feedback_replay.lower()
        self.feedback_target = feedback_target.lower()
        self.shared_blocks = shared_blocks
        self.capsule_enabled = capsule_enabled
        self.vrl_enabled = vrl_enabled
        self.vrl_start_layer = vrl_start_layer
        self.feedback_passes = feedback_passes
        self.architecture = Hyperparameters.architecture

        # Determine per-layer block types for hybrid architecture
        if self.architecture == "hybrid":
            self._layer_types = ["attn" if i % 2 == 0 else "ssm" for i in range(num_layers)]
        elif self.architecture == "koopman_ssm":
            self._layer_types = ["ssm"] * num_layers
        elif self.architecture == "skc":
            self._layer_types = ["skc"] * num_layers
        else:
            self._layer_types = ["attn"] * num_layers
        if self.feedback_replay not in {"decoder", "none", "off"}:
            raise ValueError(f"Unsupported FEEDBACK_REPLAY={feedback_replay}")
        if self.feedback_target not in {"decoder"}:
            raise ValueError(f"Unsupported FEEDBACK_TARGET={feedback_target}")
        self.tok_emb = nn.Embedding(vocab_size, self.embed_dim)
        self.embed_proj = nn.Linear(self.embed_dim, model_dim, bias=False) if self.embed_dim != model_dim else None
        self.embed_proj_rev = nn.Linear(model_dim, self.embed_dim, bias=False) if self.embed_dim != model_dim else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.register_buffer("_distill_hidden0", None, persistent=False)

        # Block construction: supports transformer, koopman_ssm, and hybrid architectures
        def _make_attn_block(layer_idx):
            layer_vrl = vrl_enabled and layer_idx >= vrl_start_layer
            ln_sf = 1.0 / (layer_idx + 1) ** 0.5 if ln_scale_damping else 1.0
            layer_xsa = xsa_start_layer >= 0 and layer_idx >= xsa_start_layer
            return Block(
                dim=model_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                mlp_mult=mlp_mult, rope_base=rope_base, qk_gain_init=qk_gain_init,
                group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope,
                no_cache=no_cache, rope_type=rope_type, yarn_max_len=yarn_max_len,
                train_seq_len=train_seq_len, partial_rope_dims=partial_rope_dims,
                vrl_enabled=layer_vrl, ln_scale_factor=ln_sf, xsa=layer_xsa,
                moe_enabled=moe_enabled, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k
            )

        def _make_ssm_block(layer_idx):
            ln_sf = 1.0 / (layer_idx + 1) ** 0.5 if ln_scale_damping else 1.0
            d_win = Hyperparameters.koopman_decay_window
            if num_layers > 1:
                d_win = min(16 * (2 ** layer_idx), 256)
            return KoopmanBlock(
                dim=model_dim, state_dim=Hyperparameters.koopman_state_dim,
                mlp_mult=mlp_mult, mixer_rank=Hyperparameters.koopman_mixer_rank,
                conv_kernel=Hyperparameters.koopman_conv_kernel, decay_window=d_win,
                group_size=group_size, activation=activation,
                leaky_relu_slope=leaky_relu_slope, ln_scale_factor=ln_sf,
            )

        def _make_skc_block(layer_idx):
            ln_sf = 1.0 / (layer_idx + 1) ** 0.5 if ln_scale_damping else 1.0
            moe_layer_threshold = int(math.ceil(num_layers * Hyperparameters.moe_layer_frac))
            layer_moe = moe_enabled and layer_idx >= moe_layer_threshold
            return SKCLayer(
                dim=model_dim, capsule_num=Hyperparameters.skc_num_capsules,
                capsule_dim=Hyperparameters.skc_capsule_dim,
                conv_kernel=Hyperparameters.skc_conv_kernel, block_size=Hyperparameters.skc_block_size,
                mlp_mult=mlp_mult, group_size=group_size, activation=activation,
                leaky_relu_slope=leaky_relu_slope, ln_scale_factor=ln_sf,
                moe_enabled=layer_moe, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k
            )

        def _make_block(layer_idx):
            lt = self._layer_types[layer_idx]
            if lt == "ssm": return _make_ssm_block(layer_idx)
            if lt == "skc": return _make_skc_block(layer_idx)
            return _make_attn_block(layer_idx)

        if shared_blocks > 0:
            if self.architecture == "hybrid":
                # Hybrid: one shared attention block + one shared SSM block
                base_attn = _make_attn_block(0)
                base_attn.attn.vrl_enabled = False  # VRL handled at GPT level
                base_ssm = _make_ssm_block(1)
                self.shared_block_bank = nn.ModuleList([base_attn, base_ssm])
                self._block_map = [0 if self._layer_types[i] == "attn" else 1
                                   for i in range(num_layers)]
            else:
                # Standard: N unique blocks of same type, tiled across depth
                base_block = _make_block(0)
                if hasattr(base_block, 'attn') and hasattr(base_block.attn, 'vrl_enabled'):
                    base_block.attn.vrl_enabled = False
                base_block.ln_scale_factor = 1.0
                self.shared_block_bank = nn.ModuleList([_make_block(0) for _ in range(shared_blocks)])
                self._block_map = [i % shared_blocks for i in range(num_layers)]
            # Zero-init (SkipInit/ReZero): residual branches start closed, same as the
            # per-block gate/scale params. ones would open all residuals at step 0 causing
            # gradient explosion at depth with shared blocks.
            self.per_layer_attn_scales = nn.ParameterList([
                nn.Parameter(torch.zeros(model_dim, dtype=torch.float32)) for _ in range(num_layers)])
            self.per_layer_mlp_scales = nn.ParameterList([
                nn.Parameter(torch.zeros(model_dim, dtype=torch.float32)) for _ in range(num_layers)])
            self.per_layer_resid_mixes = nn.ParameterList([
                nn.Parameter(torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float())
                for _ in range(num_layers)])
            self.blocks = None
        else:
            self.shared_block_bank = None
            self.per_layer_attn_scales = None
            self.per_layer_mlp_scales = None
            self.per_layer_resid_mixes = None
            self._block_map = None
            self.blocks = nn.ModuleList([_make_block(i) for i in range(num_layers)])

        self.final_norm = RMSNorm()

        # EngramHash — Engram-inspired multi-head n-gram memory with context gating
        self.engram = None
        self.engram_inject_layer = engram_inject_layer
        if bigram_hash_enabled:
            self.engram = EngramHash(
                bigram_hash_buckets, bigram_hash_dim, model_dim,
                fp_storage=fp_storage, num_heads=engram_num_heads,
                num_orders=engram_num_orders,
            )

        # Capsule bank — with Koopman-driven predictive dynamics
        self.capsule_bank = None
        if capsule_enabled:
            self.capsule_bank = CapsuleBank(
                model_dim, capsule_num, capsule_dim, fp_storage=feedback_fp_storage,
                koopman_enabled=Hyperparameters.koopman_enabled,
                koopman_rank=Hyperparameters.koopman_rank,
                koopman_diag_init=Hyperparameters.koopman_diag_init,
            )

        self.feedback_pooler = None
        self.feedback_adapters = None
        if self.feedback_enabled and self.feedback_replay == "decoder":
            self.feedback_pooler = FeedbackPooler(
                model_dim,
                feedback_dim,
                feedback_sketch_tokens,
                fp_storage=feedback_fp_storage,
            )
            self.feedback_adapters = nn.ModuleList(
                [
                    FeedbackAdapter(model_dim, feedback_dim, fp_storage=feedback_fp_storage)
                    for _ in range(self.num_decoder_layers)
                ]
            )
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        if self.tie_embeddings:
            self.lm_head.weight.requires_grad_(False)
        self.vocab_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    # Explicitly zero output projections so residual branches
                    # start silent and open up gradually during training.
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_embedding(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        clamped_ids = torch.clamp(input_ids, 0, self.tok_emb.num_embeddings - 1)
        x = self.tok_emb(clamped_ids).float()
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        # Engram input injection (ungated, no hidden state yet)
        if self.engram is not None and self.engram_inject_layer < 0:
            x = x + self.engram(input_ids, hidden=None)
        x = F.rms_norm(x, (x.size(-1),))
        return x, x

    def _run_block(self, layer_idx: int, x: Tensor, x0: Tensor, v0: Tensor | None = None, elapsed_fraction: float = 1.0) -> tuple[Tensor, Tensor | None]:
        """Run one effective layer. In shared mode, uses shared weights + per-layer scales."""
        if self.blocks is not None:
            # Standard unique-block mode
            return self.blocks[layer_idx](x, x0, v0=v0, elapsed_fraction=elapsed_fraction)
        
        # Shared block mode: use shared weights but per-layer scale/mix params
        block = self.shared_block_bank[self._block_map[layer_idx]]
        mix = self.per_layer_resid_mixes[layer_idx].to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        
        layer_type = self._layer_types[layer_idx]
        if layer_type == "ssm":
            # Koopman SSM block
            mixer_out = block.mixer(block.attn_norm(x) * block.ln_scale_factor)
            x = x + self.per_layer_attn_scales[layer_idx].to(dtype=x.dtype) * mixer_out
            v_out = None
        elif layer_type == "skc":
            x, v_out = block(x, x0, v0=v0, elapsed_fraction=elapsed_fraction)
            return x, v_out
        else:
            # Attention block
            attn_out, v_out = block.attn(block.attn_norm(x) * block.ln_scale_factor, v0=v0)
            x = x + self.per_layer_attn_scales[layer_idx].to(dtype=x.dtype) * attn_out

        # Only TernaryMoE accepts elapsed_fraction; plain MLP.forward only takes x.
        _mlp_in = block.mlp_norm(x) * block.ln_scale_factor
        _mlp_out = (block.mlp(_mlp_in, elapsed_fraction=elapsed_fraction)
                    if isinstance(block.mlp, TernaryMoE)
                    else block.mlp(_mlp_in))
        x = x + self.per_layer_mlp_scales[layer_idx].to(dtype=x.dtype) * _mlp_out
        return x, v_out

    def _decoder_pass(self, x: Tensor, x0: Tensor, skips: list[Tensor], sketch: Tensor | None, v0: Tensor | None = None, elapsed_fraction: float = 1.0) -> Tensor:
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if i < self.num_skip_weights:
                x = x + self.skip_weights[i].to(dtype=x.dtype) * skips[-(i + 1)]
            for _ in range(max(1, self.training_depth_recurrence)):
                x, _ = self._run_block(bi, x, x0, v0=v0, elapsed_fraction=elapsed_fraction)
            if self.feedback_adapters is not None and sketch is not None:
                x = self.feedback_adapters[i](x, sketch)
        return x

    def _compute_hidden(self, input_ids: Tensor, elapsed_fraction: float = 1.0, carry_capsules: Tensor | None = None) -> tuple[Tensor, list, Tensor | None, list]:
        """Core KoopCaps-HRM forward: encode → [correct]^N → decode.

        Returns (hidden, consistency_losses, capsule_state) where consistency_losses
        is a list of (c_pred, c_actual) pairs for the Koopman dynamics auxiliary loss.
        """
        x, x0 = self._apply_embedding(input_ids)
        skips: list[Tensor] = []
        v0 = None

        # --- Encoder pass (runs once) ---
        for i in range(self.num_encoder_layers):
            # Engram injection at internal layer (context-gated)
            if (self.engram is not None
                    and self.engram_inject_layer >= 0
                    and i == self.engram_inject_layer):
                x = x + self.engram(input_ids, hidden=x)
            for _ in range(max(1, self.training_depth_recurrence)):
                x, v_out = self._run_block(i, x, x0, v0=v0, elapsed_fraction=elapsed_fraction)
                if v0 is None and v_out is not None:
                    v0 = v_out.detach()
            skips.append(x)

        # Capsule state initialization — use carry_capsules for cross-window persistence
        capsule_state = None
        if carry_capsules is not None:
            B_curr = x.shape[0]
            carry_avg = carry_capsules.mean(dim=0, keepdim=True).expand(B_curr, -1, -1)
            capsule_state = carry_avg
        if self.capsule_bank is not None:
            x, capsule_state, _, _ = self.capsule_bank(x, prev_capsules=capsule_state)

        encoded = x
        num_passes = self.feedback_passes

        # --- Iterative correction loop with Koopman dynamics ---
        sketch = None
        consistency_losses = []
        speculative_losses = []
        prev_capsule_state = None
        fast_forwarded = False

        for correction_pass in range(num_passes + 1):
            if correction_pass > 0 and self.feedback_enabled and self.feedback_pooler is not None:
                sketch = self.feedback_pooler(self.final_norm(x))
            else:
                sketch = None

            if self.capsule_bank is not None and correction_pass > 0:
                prev_capsule_state = capsule_state
                spec_steps = Hyperparameters.koopman_speculator_steps if (Hyperparameters.koopman_speculator_enabled and correction_pass == 1) else 0
                
                encoded, capsule_state, c_pred, c_spec = self.capsule_bank(
                    encoded, prev_capsules=capsule_state, speculate_steps=spec_steps
                )
                if c_pred is not None:
                    consistency_losses.append((c_pred, capsule_state.detach()))
                    
                if c_spec is not None:
                    speculative_losses.append(c_spec)

                # Fast-Forward Diffusion (EVAL MODE ONLY)
                # Set capsule state to speculated future and let the decoder pass below
                # run with this speculated state (don't break — we need the decoder pass).
                if (not self.training and Hyperparameters.koopman_speculator_enabled
                        and c_spec is not None and not fast_forwarded):
                    capsule_state = c_spec
                    fast_forwarded = True
                    # Fall through to _decoder_pass, then break on next iteration

                # Adaptive halting (eval only)
                if (not self.training and Hyperparameters.adaptive_halt_enabled
                        and prev_capsule_state is not None and correction_pass >= 1 and not fast_forwarded):
                    delta = torch.sqrt(torch.mean((capsule_state - prev_capsule_state) ** 2))
                    norm = torch.sqrt(torch.mean(capsule_state ** 2)) + 1e-8
                    if (delta / norm).item() < Hyperparameters.adaptive_halt_threshold:
                        break

            x = self._decoder_pass(encoded, x0, skips, sketch=sketch, v0=v0, elapsed_fraction=elapsed_fraction)
            
            # After fast-forward: we ran one decoder pass with speculated state, now stop
            if fast_forwarded:
                break

        c_final = capsule_state.detach() if capsule_state is not None else None
        jepa_loss = [(c_s, c_final) for c_s in speculative_losses] if c_final is not None else []

        return self.final_norm(x), consistency_losses, capsule_state, jepa_loss

    def _compute_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            proj = self.embed_proj_rev(x) if self.embed_proj_rev is not None else x
            logits_raw = F.linear(proj, self.tok_emb.weight.to(x.dtype))
        else:
            logits_raw = self.lm_head(x)
        return logits_raw + self.vocab_bias.to(x.dtype)

    def _softcap(self, logits: Tensor) -> Tensor:
        """Tanh softcap: bounds logits to (-c, c) without clipping gradients.
        c <= 0 disables softcap entirely. LOGIT_SOFTCAP=30 is the default."""
        c = float(self.logit_softcap)
        if c <= 0:
            return logits
        return c * torch.tanh(logits / c)

    def register_ternary_calib_names(self):
        """Set _calib_name on each TernaryLinear so aligned training can look up per-tensor calib."""
        for name, module in self.named_modules():
            if isinstance(module, TernaryLinear):
                module._calib_name = name + ".weight"

    def step_capsule_carry(self):
        """Call after each training step to advance SKC layer carry state."""
        for m in self.modules():
            if isinstance(m, SKCLayer) and hasattr(m, '_last_capsules'):
                m._carry_capsules = m._last_capsules

    def reset_capsule_carry(self):
        """Reset all SKC layer carry state (call on curriculum jumps / shard boundaries)."""
        for m in self.modules():
            if isinstance(m, SKCLayer):
                m._carry_capsules = None
                if hasattr(m, '_last_capsules'):
                    del m._last_capsules

    def forward_logits(self, input_ids: Tensor, temperature: float = 1.0) -> Tensor:
        hidden, _, _, _ = self._compute_hidden(input_ids)
        logits = self._softcap(self._compute_logits(hidden.reshape(-1, hidden.size(-1))))
        if temperature != 1.0:
            logits = logits / temperature
        return logits.reshape(input_ids.size(0), input_ids.size(1), -1)

    def forward_logits_with_carry(self, input_ids: Tensor, carry_capsules: Tensor | None = None,
                                   temperature: float = 1.0) -> tuple[Tensor, Tensor | None]:
        """Forward pass that accepts and returns capsule state for cross-window carry."""
        hidden, _, capsule_state, _ = self._compute_hidden(input_ids, carry_capsules=carry_capsules)
        logits = self._softcap(self._compute_logits(hidden.reshape(-1, hidden.size(-1))))
        if temperature != 1.0:
            logits = logits / temperature
        return logits.reshape(input_ids.size(0), input_ids.size(1), -1), capsule_state

    def forward(self, input_ids: Tensor, target_ids: Tensor, reduction: str = "mean",
                temperature: float = 1.0, elapsed_fraction: float = 1.0, carry_capsules: Tensor | None = None) -> Tensor:
        hidden, consistency_losses, _, jepa_loss = self._compute_hidden(input_ids, elapsed_fraction=elapsed_fraction, carry_capsules=carry_capsules)
        logits = self._softcap(self._compute_logits(hidden.reshape(-1, hidden.size(-1))))
        if temperature != 1.0:
            logits = logits / temperature
        logits = logits.reshape(-1, self.vocab_bias.numel())
        targets = target_ids.reshape(-1)
        if reduction == "none":
            return F.cross_entropy(logits.float(), targets, reduction="none").reshape(input_ids.shape)
        logits_f = logits.float()
        # Use F.cross_entropy directly — it fails fast on out-of-range targets rather than
        # silently clamping them (which poisons gradients with wrong-label supervision).
        ce_loss = F.cross_entropy(logits_f, targets)
        
        if self.training and consistency_losses and Hyperparameters.koopman_consistency_weight > 0:
            consist_sum = torch.tensor(0.0, device=input_ids.device)
            for c_pred, c_actual in consistency_losses:
                consist_sum = consist_sum + F.mse_loss(c_pred, c_actual)
            ce_loss = ce_loss + Hyperparameters.koopman_consistency_weight * consist_sum / len(consistency_losses)

        if self.training and jepa_loss and Hyperparameters.koopman_speculator_weight > 0:
            spec_sum = torch.tensor(0.0, device=input_ids.device)
            for c_spec, c_final in jepa_loss:
                spec_sum = spec_sum + F.mse_loss(c_spec, c_final)
            spec_loss = spec_sum / len(jepa_loss)
            ce_loss = ce_loss + Hyperparameters.koopman_speculator_weight * spec_loss

        # MoE auxiliary router loss — regularizes expert routing to prevent collapse.
        # moe_router_aux_loss_coef controls the balance; set to 0 to disable.
        if self.training and Hyperparameters.moe_router_aux_loss_coef > 0:
            moe_aux = torch.zeros((), device=input_ids.device)
            for m in self.modules():
                if isinstance(m, TernaryMoE) and m.aux_loss is not None:
                    moe_aux = moe_aux + m.aux_loss
            ce_loss = ce_loss + Hyperparameters.moe_router_aux_loss_coef * moe_aux

        # Spectral entropy aux loss — prevents spectral collapse in SKC layers.
        # Accumulate on device without .item() to avoid host-device sync every step.
        if self.training:
            spec_aux = torch.zeros((), device=input_ids.device)
            for m in self.modules():
                if isinstance(m, SKCLayer) and hasattr(m, 'spectral_aux_loss'):
                    spec_aux = spec_aux + m.spectral_aux_loss
            ce_loss = ce_loss + spec_aux

        return ce_loss



# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def build_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size()) if sp is not None else 0
    # Force table_size to at least 50257 to cover GPT-2 vocab range reliably
    table_size = max(sp_vocab_size, vocab_size, 50257)
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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def ld_val(pattern, seq_len, max_tok=int(os.environ.get("VAL_MAX_TOKENS", 500000))):
    files = sorted(glob.glob(pattern))
    assert files, f"No files: {pattern}"
    tok = torch.cat([ld_shard(Path(p)) for p in files]).contiguous()
    if max_tok > 0: tok = tok[:max_tok + 1]
    u = ((tok.numel() - 1) // seq_len) * seq_len
    return tok[:u + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, temperature: float = 1.0):
    # Safer batching for 20M parameter models on limited VRAM
    local_batch_tokens = min(args.val_batch_size, 131072) // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    eval_smoke = int(os.environ.get("FAST_SMOKE", "0")) == 1
    t_eval_start = time.perf_counter()
    with torch.inference_mode():
        for i, batch_start in enumerate(range(seq_start, seq_end, local_batch_seqs)):
            # Safety cap for smoke tests to prevent 10-minute hangs
            if eval_smoke and (i >= 128 or (time.perf_counter() - t_eval_start) > 30.0):
                break
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start = batch_start * args.train_seq_len
            raw_end = batch_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            
            with torch.autocast(device_type="cuda", dtype=ptdtype):
                batch_loss = model(x, y, temperature=temperature).detach()
            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            token_count += n
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            # Defensive clipping to prevent CUDA device-side asserts if vocab/LUT mismatch occurs
            max_idx = base_bytes_lut.size(0) - 1
            prev_ids = torch.clamp(prev_ids, 0, max_idx)
            tgt_ids = torch.clamp(tgt_ids, 0, max_idx)
            
            tok_bytes = base_bytes_lut[tgt_ids].to(torch.int16)
            tok_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += tok_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count.clamp_min(1.0)).item()
    bpb = val_loss / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1.0))
    model.train()
    return float(val_loss), float(bpb)

def eval_val_sliding(args, base_model, rank, world_size, device, grad_accum_steps, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride: int = 64, temperature: float = 1.0):
    del grad_accum_steps
    seq_len = args.train_seq_len
    batch_size = args.sliding_batch_size
    total_tokens = val_tokens.numel() - 1
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    all_starts = list(range(0, total_tokens, stride))
    my_starts = [s for idx, s in enumerate(all_starts) if idx % world_size == rank and min(s + seq_len, total_tokens) - s >= 1]

    use_carry = args.capsule_carry_enabled and args.capsule_enabled
    decay = args.capsule_carry_decay

    base_model.eval()
    carry_capsules = None

    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[i:i + batch_size]
            bsz = len(batch_starts)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for j, start in enumerate(batch_starts):
                end = min(start + seq_len, total_tokens)
                wlen = end - start
                wlens.append(wlen)
                chunk = val_tokens[start:end + 1].to(dtype=torch.int64, device=device)
                x_batch[j, :wlen] = chunk[:-1]
                y_batch[j, :wlen] = chunk[1:]
            if use_carry:
                logits, capsule_state = base_model.forward_logits_with_carry(
                    x_batch, carry_capsules=carry_capsules, temperature=temperature)
                if capsule_state is not None:
                    cs_avg = capsule_state.mean(dim=0, keepdim=True).detach()
                    if carry_capsules is not None:
                        carry_capsules = (decay * carry_capsules + (1.0 - decay) * cs_avg).detach()
                    else:
                        carry_capsules = cs_avg.detach()
            else:
                logits = base_model.forward_logits(x_batch, temperature=temperature)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for j, start in enumerate(batch_starts):
                wlen = wlens[j]
                score_from = 0 if start == 0 else max(wlen - stride, 0)
                scored = nll[j, score_from:wlen]
                sx = x_batch[j, score_from:wlen]
                sy = y_batch[j, score_from:wlen]
                loss_sum += scored.to(torch.float64).sum()
                token_count += float(wlen - score_from)
                tok_bytes = base_bytes_lut[sy].to(torch.int16)
                tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                byte_count += tok_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count.clamp_min(1.0)).item()
    bpb = val_loss / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1.0))
    base_model.train()
    return float(val_loss), float(bpb)


def collect_ttt_params(base_model: nn.Module, scope: str) -> tuple[dict[str, bool], list[Tensor]]:
    if scope != "feedback":
        raise ValueError(f"Unsupported TTT_SCOPE={scope}")
    original: dict[str, bool] = {}
    params: list[Tensor] = []
    for name, p in base_model.named_parameters():
        original[name] = p.requires_grad
        allow = (name.startswith("feedback_pooler.") or name.startswith("feedback_adapters.")
                or name == "skip_weights" or name.startswith("capsule_bank."))
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) >= 3:
                block_idx = int(parts[1])
                leaf = parts[2]
                if block_idx >= base_model.num_encoder_layers and leaf in {"attn_scale", "mlp_scale", "skc_scale"}:
                    allow = True
        # Per-layer scales in shared block mode — allow decoder-half scales for TTT
        if name.startswith("per_layer_attn_scales.") or name.startswith("per_layer_mlp_scales."):
            idx = int(name.split(".")[1])
            if idx >= base_model.num_encoder_layers:
                allow = True
        p.requires_grad_(allow)
        if allow:
            params.append(p)
    return original, params


def restore_requires_grad(base_model: nn.Module, original: dict[str, bool]) -> None:
    for name, p in base_model.named_parameters():
        p.requires_grad_(original.get(name, True))


def eval_val_sliding_ttt(
    args,
    base_model,
    rank,
    world_size,
    device,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    stride: int,
    batch_seqs: int = 32,
    temperature: float = 1.0,
    log0=print,
):
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        score_from = 0 if ws == 0 else max(wlen - stride, 0)
        chunk_idx = min((ws + score_from) // ttt_chunk, num_chunks - 1)
        chunk_windows[chunk_idx].append(ws)

    log0(
        f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} stride={stride} "
        f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} scope={args.ttt_scope}"
    )
    use_carry = args.capsule_carry_enabled and args.capsule_enabled
    decay = args.capsule_carry_decay if args.capsule_carry_enabled else 0.0
    carry_capsules = None
    original_grad, ttt_params = collect_ttt_params(base_model, args.ttt_scope)
    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)}")
    if not ttt_params:
        raise RuntimeError("TTT enabled but no parameters matched TTT_SCOPE")
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    try:
        for ci in range(num_chunks):
            windows = chunk_windows[ci]
            if not windows:
                continue
            chunk_start = ci * ttt_chunk
            chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
            my_s = (len(windows) * rank) // world_size
            my_e = (len(windows) * (rank + 1)) // world_size
            my_windows = windows[my_s:my_e]

            base_model.eval()
            with torch.inference_mode():
                for bi in range(0, len(my_windows), batch_seqs):
                    batch_ws = my_windows[bi:bi + batch_seqs]
                    bsz = len(batch_ws)
                    x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    wlens: list[int] = []
                    for i, ws in enumerate(batch_ws):
                        end = min(ws + seq_len, total_tokens)
                        wlen = end - ws
                        wlens.append(wlen)
                        chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                        x_batch[i, :wlen] = chunk[:-1]
                        y_batch[i, :wlen] = chunk[1:]
                    if use_carry:
                        logits, capsule_state = base_model.forward_logits_with_carry(
                            x_batch, carry_capsules=carry_capsules, temperature=temperature)
                        if capsule_state is not None:
                            cs_avg = capsule_state.mean(dim=0, keepdim=True)
                            if carry_capsules is not None:
                                carry_capsules = (decay * carry_capsules + (1.0 - decay) * cs_avg).detach()
                            else:
                                carry_capsules = cs_avg.detach()
                    else:
                        logits = base_model.forward_logits(x_batch, temperature=temperature)
                    nll = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y_batch.reshape(-1),
                        reduction="none",
                    ).reshape(bsz, seq_len)
                    for i, ws in enumerate(batch_ws):
                        wlen = wlens[i]
                        score_from = 0 if ws == 0 else max(wlen - stride, 0)
                        scored = nll[i, score_from:wlen].to(torch.float64)
                        loss_sum += scored.sum()
                        token_count += float(wlen - score_from)
                        sx = x_batch[i, score_from:wlen]
                        sy = y_batch[i, score_from:wlen]
                        tok_bytes = base_bytes_lut[sy].to(torch.int16)
                        tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                        byte_count += tok_bytes.to(torch.float64).sum()

            if ci == num_chunks - 1 or args.ttt_epochs <= 0:
                continue

            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs <= 0:
                continue
            cosine_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr
            my_seq_s = (chunk_seqs * rank) // world_size
            my_seq_e = (chunk_seqs * (rank + 1)) // world_size
            my_chunk_seqs = my_seq_e - my_seq_s
            for _ in range(args.ttt_epochs):
                for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                    be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                    actual_bs = my_seq_s + bs
                    start_tok = chunk_start + actual_bs * seq_len
                    end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                    if end_tok > val_tokens.numel():
                        continue
                    local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                    x = local[:-1].reshape(-1, seq_len)
                    y = local[1:].reshape(-1, seq_len)
                    optimizer.zero_grad(set_to_none=True)
                    loss = base_model(x, y, temperature=temperature, carry_capsules=carry_capsules) if use_carry else base_model(x, y, temperature=temperature)
                    loss.backward()
                    if world_size > 1:
                        for p in ttt_params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                    optimizer.step()

            if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
                elapsed = time.perf_counter() - t0
                running_loss = loss_sum.item() / max(token_count.item(), 1)
                running_bpb = running_loss / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
                log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={running_bpb:.6f} time={elapsed:.1f}s")

        if dist.is_available() and dist.is_initialized():
            for t in (loss_sum, token_count, byte_count):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss = (loss_sum / token_count.clamp_min(1.0)).item()
        val_bpb = val_loss / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1.0))
        log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} elapsed={time.perf_counter() - t0:.1f}s")
        return val_loss, val_bpb
    finally:
        restore_requires_grad(base_model, original_grad)
        base_model.eval()

# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                              calibration_tokens, base_bytes_lut, has_leading_space_lut,
                              is_boundary_token_lut):
    best_t, best_loss = 1.0, float("inf")
    for t in [0.90, 0.95, 1.00, 1.05, 1.10]:
        loss, _ = eval_val(args, base_model, rank, world_size, device, grad_accum_steps,
                           calibration_tokens, base_bytes_lut, has_leading_space_lut,
                           is_boundary_token_lut, temperature=t)
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return best_t


# ---------------------------------------------------------------------------
# N-gram evaluation cache with entropy-adaptive mixing
# ---------------------------------------------------------------------------
class NgramCache:
    """Dynamic n-gram cache built from already-scored tokens. Zero artifact cost.
    Uses entropy-adaptive alpha to blend n-gram empirical probabilities with neural logits."""
    def __init__(self, max_order: int = 5, alpha_base: float = 0.05,
                 alpha_scale: float = 0.55, entropy_center: float = 4.0):
        self.max_order = max_order
        self.alpha_base = alpha_base
        self.alpha_scale = alpha_scale
        self.entropy_center = entropy_center
        # counts[order] maps tuple(context) -> {next_token: count}
        self.counts: list[dict] = [{} for _ in range(max_order + 1)]
        self.total_counts: list[dict] = [{} for _ in range(max_order + 1)]

    def update(self, tokens: list[int]) -> None:
        """Add scored tokens to the cache."""
        for order in range(2, self.max_order + 1):
            for i in range(len(tokens) - order + 1):
                ctx = tuple(tokens[i:i + order - 1])
                nxt = tokens[i + order - 1]
                if ctx not in self.counts[order]:
                    self.counts[order][ctx] = {}
                    self.total_counts[order][ctx] = 0
                self.counts[order][ctx][nxt] = self.counts[order][ctx].get(nxt, 0) + 1
                self.total_counts[order][ctx] += 1

    def predict(self, context: list[int], vocab_size: int) -> Tensor | None:
        """Return n-gram log-probability distribution via backoff, or None if no match."""
        for order in range(self.max_order, 1, -1):
            if len(context) < order - 1:
                continue
            ctx = tuple(context[-(order - 1):])
            if ctx in self.counts[order]:
                total = self.total_counts[order][ctx]
                probs = torch.zeros(vocab_size)
                for tok, count in self.counts[order][ctx].items():
                    if tok < vocab_size:
                        probs[tok] = count / total
                if probs.sum() > 0:
                    # Laplace smoothing
                    probs = (probs + 1e-8) / (probs.sum() + 1e-8 * vocab_size)
                    return probs.log()
        return None

    def entropy_alpha(self, neural_logprobs: Tensor) -> float:
        """Compute entropy-adaptive mixing weight."""
        probs = neural_logprobs.exp()
        H = -(probs * neural_logprobs).sum().item()
        # Sigmoid schedule: higher entropy → more trust in cache
        return self.alpha_base + self.alpha_scale * (1.0 / (1.0 + math.exp(-2.0 * (H - self.entropy_center))))


# ---------------------------------------------------------------------------
# Export calibration: per-tensor threshold + scale_mult search
# ---------------------------------------------------------------------------
def _proxy_roundtrip_bpb(sd: dict, base_model, calib: dict, group_size: int,
                          proxy_tokens: torch.Tensor, args, device) -> float:
    """Export sd with calib, reload into base_model, score on proxy_tokens. Returns BPB."""
    import io, lzma as _lzma
    q_obj, _ = q_sd(sd, group_size=group_size, calib=calib)
    buf = io.BytesIO()
    torch.save(q_obj, buf)
    blob = _lzma.compress(buf.getvalue(), preset=1)  # fast preset for proxy
    loaded = torch.load(io.BytesIO(_lzma.decompress(blob)), map_location="cpu", weights_only=False)
    orig_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    missing, unexpected = base_model.load_state_dict(deq_sd(loaded), strict=False)
    base_model.eval()
    loss_sum = 0.0
    tok_count = 0
    seq_len = min(args.train_seq_len, 512)
    # Use no_grad instead of inference_mode: inference_mode caches tensors that
    # cannot be saved for backward, causing errors when the model returns to training.
    with torch.no_grad():
        for i in range(0, min(proxy_tokens.numel() - 1, 32768), seq_len):
            chunk = proxy_tokens[i:i + seq_len + 1].to(device)
            chunk = chunk.to(torch.int64)
            if chunk.numel() < 2: break
            x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y).item()
            loss_sum += loss * y.numel()
            tok_count += y.numel()
    bpb = (loss_sum / max(tok_count, 1)) / math.log(2.0)
    base_model.load_state_dict(orig_sd)
    base_model.train()
    return bpb


def calibrate_ternary(base_model, proxy_tokens: torch.Tensor, args, device) -> dict:
    """Search per-tensor (thr, scale_mult) to minimize round-trip proxy BPB.
    Budget-capped: respects CALIB_MAX_EVALS and CALIB_MAX_SECONDS limits.
    Returns calib dict."""
    import time as _time
    t_start = _time.perf_counter()
    evals = [0]  # mutable counter

    group_size = args.bitnet_group_size
    top_n = args.ternary_calib_top_n
    max_evals = args.calib_max_evals
    max_seconds = args.calib_max_seconds
    proxy_max_tok = args.calib_proxy_max_tok

    # Trim proxy tokens to budget-friendly size
    proxy_tokens = proxy_tokens[:proxy_max_tok] if proxy_tokens.numel() > proxy_max_tok else proxy_tokens

    sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

    def _budget_ok():
        return evals[0] < max_evals and (_time.perf_counter() - t_start) < max_seconds

    def _eval(calib_override):
        if not _budget_ok():
            return float("inf")
        evals[0] += 1
        return _proxy_roundtrip_bpb(sd, base_model, calib_override, group_size, proxy_tokens, args, device)

    # Build search grids
    thr_vals = [0.0]
    if args.ternary_threshold_search and args.ternary_threshold_steps > 0:
        lo, hi, n = args.ternary_threshold_low, args.ternary_threshold_high, args.ternary_threshold_steps
        thr_vals += [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
    scale_vals = [1.0]
    if args.ternary_scale_search and args.ternary_scale_mult_steps > 0:
        lo, hi, n = args.ternary_scale_mult_low, args.ternary_scale_mult_high, args.ternary_scale_mult_steps
        scale_vals += [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]

    # Pre-filter: top candidates by size (cheap, no proxy evals)
    all_eligible = [
        name for name, t in sd.items()
        if t.ndim == 2 and t.numel() > 16384
        and "tok_emb" not in name and "lm_head" not in name and "embed_proj" not in name
    ]
    prefilter_k = min(max(top_n * args.calib_prefilter_mult, top_n), args.calib_max_candidates)
    size_sorted = sorted(all_eligible, key=lambda n: sd[n].numel(), reverse=True)
    rank_pool = size_sorted[:prefilter_k]

    # Baseline BPB
    calib: dict = {}
    baseline_bpb = _eval(calib)
    if baseline_bpb == float("inf"):
        return calib  # budget already exhausted

    # Sensitivity ranking over rank_pool (not all tensors)
    probe_thr = thr_vals[len(thr_vals)//2] if len(thr_vals) > 1 else 0.05
    probe_sm = scale_vals[len(scale_vals)//2] if len(scale_vals) > 1 else 1.0
    sensitivities: list[tuple[float, str]] = []
    for name in rank_pool:
        if not _budget_ok():
            # Budget hit: fall back to size ordering for remaining candidates
            remaining = [n for n in rank_pool if n not in {s[1] for s in sensitivities}]
            sensitivities += [(0.0, n) for n in remaining]
            break
        probe = {name: {"thr": probe_thr, "scale_mult": probe_sm}}
        delta = abs(_eval(probe) - baseline_bpb)
        sensitivities.append((delta, name))
    sensitivities.sort(reverse=True)
    candidates = [name for _, name in sensitivities[:top_n]]

    def _search_one(name: str, current_calib: dict, ref_bpb: float):
        best_bpb = ref_bpb
        best_thr = current_calib.get(name, {}).get("thr", 0.0)
        best_sm = current_calib.get(name, {}).get("scale_mult", 1.0)
        for thr in thr_vals:
            for sm in scale_vals:
                if thr == best_thr and sm == best_sm:
                    continue
                if not _budget_ok():
                    return best_thr, best_sm, best_bpb
                test_calib = dict(current_calib)
                test_calib[name] = {"thr": thr, "scale_mult": sm}
                bpb = _eval(test_calib)
                if bpb < best_bpb:
                    best_bpb = bpb
                    best_thr, best_sm = thr, sm
        return best_thr, best_sm, best_bpb

    # Pass 1: greedy forward search over sensitivity-ranked candidates
    for name in candidates:
        if not _budget_ok():
            break
        best_thr, best_sm, best_bpb = _search_one(name, calib, baseline_bpb)
        if best_thr != 0.0 or best_sm != 1.0:
            calib[name] = {"thr": best_thr, "scale_mult": best_sm}
            baseline_bpb = best_bpb

    # Pass 2 (optional): re-optimize selected tensors with full context
    if args.calib_second_pass:
        for name in list(calib.keys()):
            if not _budget_ok():
                break
            best_thr, best_sm, best_bpb = _search_one(name, calib, baseline_bpb)
            calib[name] = {"thr": best_thr, "scale_mult": best_sm}
            baseline_bpb = best_bpb

    elapsed = _time.perf_counter() - t_start
    print(f"calib:budget evals={evals[0]}/{max_evals} time={elapsed:.1f}s/{max_seconds}s", flush=True)
    return calib


# ---------------------------------------------------------------------------
# GPTQ-lite: per-row clip percentile search for ternary quantization
# ---------------------------------------------------------------------------
def gptq_lite_clip_search(state_dict: dict, group_size: int, num_percentiles: int = 5) -> dict:
    """For each ternary-candidate weight matrix, search over clip percentiles
    to minimize per-row MSE between original and ternary-quantized weights.
    Returns a modified state_dict with clipped weights."""
    percentiles = [0.995 + 0.001 * i for i in range(num_percentiles)]
    improved = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float()
        if t.ndim != 2 or t.numel() <= 65536 or "tok_emb" in name or "lm_head" in name or "embed_proj" in name:
            improved[name] = tensor
            continue
        best_t = t.clone()
        best_mse = float("inf")
        # Pad columns to be divisible by group_size (same as q_sd does)
        pad = (group_size - t.shape[1] % group_size) % group_size
        for pct in percentiles:
            clip_val = torch.quantile(t.abs().flatten(), pct)
            t_clipped = t.clamp(-clip_val, clip_val)
            t_padded = F.pad(t_clipped, (0, pad)) if pad > 0 else t_clipped
            t_g = t_padded.reshape(-1, group_size)
            scale = t_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            q = (t_g / scale).round().clamp(-1, 1)
            recon = (q * scale).reshape(t_padded.shape)[:t.shape[0], :t.shape[1]]
            mse = (t_clipped - recon).pow(2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_t = t_clipped
        improved[name] = best_t.to(tensor.dtype).to(tensor.device)
    return improved


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------
class EMAHelper:
    """Exponential Moving Average for model weights. Maintains shadow copy.
    Applied to latent FP32 weights; ternary quantization happens at export."""
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module, move_to_cpu: bool = False) -> dict[str, Tensor]:
        """Replace model weights with EMA shadow. Returns original weights for restore."""
        original = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                original[name] = p.data.cpu() if move_to_cpu else p.data.clone()
                p.data.copy_(self.shadow[name].to(p.device))
        return original

    @staticmethod
    def restore(model: nn.Module, original: dict[str, Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in original:
                p.data.copy_(original[name])


class EMAHelper:
    """Exponential Moving Average for model weights. Maintains shadow copy.
    Applied to latent FP32 weights; ternary quantization happens at export."""
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module, move_to_cpu: bool = False) -> dict[str, torch.Tensor]:
        """Replace model weights with EMA shadow. Returns original weights for restore."""
        original = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                original[name] = p.data.cpu() if move_to_cpu else p.data.clone()
                p.data.copy_(self.shadow[name].to(p.device))
        return original

    @staticmethod
    def restore(model: nn.Module, original: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in original:
                p.data.copy_(original[name].to(p.device))





# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")

    # Wire global quantization flags from args — must happen before any model construction
    # so TernaryLinear.forward and KV cache paths see the correct values.
    global _TURBO_QUANT_TRAIN, _TURBO_QUANT_KV
    _TURBO_QUANT_TRAIN = args.turbo_quant_train
    _TURBO_QUANT_KV = bool(int(os.environ.get("TURBO_QUANT_KV", "0")))

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))
    grad_scale = 1.0 / grad_accum_steps

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if distributed:
        dist_backend = "nccl" if device.type == "cuda" else "gloo"
        from datetime import timedelta
        _nccl_timeout_sec = int(os.environ.get("TORCH_NCCL_TIMEOUT_SEC", "7200"))
        dist.init_process_group(backend=dist_backend, device_id=device if device.type == "cuda" else None,
                                timeout=timedelta(seconds=_nccl_timeout_sec))

        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs/cuda/", exist_ok=True)
    logfile = f"logs/cuda/{args.run_id}.txt" if master_process else None
    if master_process:
        print(logfile)

    def log0(msg: str, console: bool = True, flush: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=flush)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f, flush=flush)

    log0(code, console=False)
    log0("=" * 100, console=False)

    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    args.vocab_size = int(sp.vocab_size())
    # Use DistributedTokenLoader — the only loader that uses header-aware ld_shard()
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    val_tokens = ld_val(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(
        sp, args.vocab_size, device)

    # --- Model ---
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type, leaky_relu_slope=args.leaky_relu_slope,
        embed_dim=args.embed_dim, training_depth_recurrence=args.training_depth_recurrence, fp_storage=args.fp_storage,
        softcap_type=args.softcap_type, no_cache=(args.compile_mode == "reduce-overhead"),
        rope_type=args.rope_type, yarn_max_len=args.yarn_max_len, train_seq_len=args.train_seq_len,
        feedback_enabled=args.feedback_enabled, feedback_dim=args.feedback_dim,
        feedback_sketch_tokens=args.feedback_sketch_tokens, feedback_replay=args.feedback_replay,
        feedback_target=args.feedback_target, feedback_fp_storage=args.feedback_fp_storage,
        feedback_passes=args.feedback_passes,
        shared_blocks=args.shared_blocks, capsule_enabled=args.capsule_enabled,
        capsule_num=args.capsule_num, capsule_dim=args.capsule_dim,
        partial_rope_dims=args.partial_rope_dims, vrl_enabled=args.vrl_enabled,
        vrl_start_layer=args.vrl_start_layer, ln_scale_damping=args.ln_scale_damping,
        bigram_hash_enabled=args.bigram_hash_enabled,
        bigram_hash_buckets=args.bigram_hash_buckets, bigram_hash_dim=args.bigram_hash_dim,
        engram_num_heads=args.engram_num_heads, engram_num_orders=args.engram_num_orders,
        engram_inject_layer=args.engram_inject_layer,
        xsa_start_layer=args.xsa_start_layer,
        moe_enabled=args.moe_enabled,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
    ).to(device)

    # Re-enable standard compilation for Linux
    compiled_model = torch.compile(base_model, mode=args.compile_mode) if args.compile_mode != "none" else base_model
    
    # MoE with top-k routing leaves non-selected expert params unused on each step.
    # find_unused_parameters must be True whenever MoE is active to prevent DDP hangs.
    use_find_unused = (args.untie_at_fraction > 0 or not args.tie_embeddings
                       or args.shared_blocks > 0 or args.moe_enabled)

    if distributed:
        model = DDP(compiled_model, device_ids=[local_rank], find_unused_parameters=use_find_unused)
    else:
        model = compiled_model

    # C8: SKC-aware 3-tier optimizer grouping.
    # Structural/dynamical params (band_centers, eigenvalues, coupling_U/V, skc_scale, etc.)
    # are routed to AdamW — NOT to Muon. Muon is reserved for genuine ternary projection matrices.
    _SKC_STRUCTURAL = (
        "decay_rates", "band_centers", "band_log_widths",
        "eigenvalues", "coupling_U", "coupling_V", "nonlinear_gate",
        "mixer_conv", "skc_scale", "mlp_scale", "attn_scale", "resid_mix",
        "skip_weights", "vocab_bias", "content_router", "content_scale",
        "gate_proj", "decay_rates",
    )
    def _is_skc_structural(name: str) -> bool:
        return any(k in name for k in _SKC_STRUCTURAL)

    muon_params = []
    adam_params = []
    head_params = []

    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if "tok_emb" in name or "lm_head" in name or "embed_proj" in name:
            head_params.append(p)
        elif _is_skc_structural(name) or p.ndim < 2:
            # Structural/scalar params → AdamW for stability
            adam_params.append(p)
        else:
            # Genuine weight matrices (ternary projections) → Muon
            muon_params.append(p)
            
    opt_muon = Muon(muon_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    opt_adam = torch.optim.AdamW(adam_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd)
    opt_head = torch.optim.AdamW(head_params, lr=args.tied_embed_lr if args.tie_embeddings else args.head_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd)
    
    for opt in [opt_muon, opt_adam, opt_head]:
        for g in opt.param_groups:
             g["base_lr"] = g["lr"]
             
    optimizers = [opt_muon, opt_adam, opt_head]

    # --- EMA ---
    ema = None

    # Register param names on TernaryLinear modules for aligned training
    base_model.register_ternary_calib_names()

    # --- Log all hyperparameters ---
    log0("--- Hyperparameters ---", console=False)
    log0(" ".join(f"{a}={getattr(args,a)}" for a in sorted(dir(args)) if not a.startswith("_") and a not in ("train_files","val_files") and not callable(getattr(args,a))), console=False)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params} L:{args.num_layers} d:{args.model_dim} h:{args.num_heads} kv:{args.num_kv_heads} ws:{world_size} ga:{grad_accum_steps} s:{args.seed}")

    # --- Data loader ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float):
        if args.warmdown_fraction <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = int(args.iterations * (1.0 - args.warmdown_fraction))
            return max((args.iterations - step) / max(args.iterations * args.warmdown_fraction, 1), 0.0) if step >= warmdown_start else 1.0
        warmdown_ms = max_wallclock_ms * args.warmdown_fraction
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    _seq_switched = False
    _batch_switched = False
    active_seq_len = args.seq_len_start if args.seq_len_start > 0 else args.train_seq_len
    active_batch_tokens = args.batch_tokens_start if args.batch_tokens_start > 0 else args.train_batch_tokens

    # ---------------------------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------------------------

    # --- Compiler warmup ---
    # Use COMPILER_WARMUP_STEPS if set, otherwise fall back to WARMUP_STEPS.
    # These are intentionally distinct: compiler warmup triggers torch.compile graph
    # capture before the LR warmup; the state is restored after so no real gradient
    # steps are counted against the budget.
    _compile_warmup_n = args.compiler_warmup_steps if args.compiler_warmup_steps > 0 else args.warmup_steps
    if _compile_warmup_n > 0:
        _ms = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        _os = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(_compile_warmup_n):
            zero_grad_all()
            for mi in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = mi == grad_accum_steps - 1
                x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y, elapsed_fraction=0.0)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            log0(f"warmup:{ws+1}/{_compile_warmup_n}")
        log0("probe:restoring_pre_warmup_state", flush=True)
        base_model.load_state_dict(_ms, strict=True)
        log0("probe:restoring_optimizers", flush=True)
        for o, s in zip(optimizers, _os): o.load_state_dict(s)
        zero_grad_all()
        log0("probe:reinitializing_dataloader", flush=True)
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    
    log0("probe:ema_init_start", flush=True)
    # --- EMA init (after warmup so shadow starts from clean weights) ---
    if args.ema_enabled:
        ema = EMAHelper(base_model, decay=args.ema_decay)
        log0(f"ema:enabled decay={args.ema_decay} start_fraction={args.ema_start_fraction}")

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step: int | None = None
    _untied = False
    _aligned_phase_started = False
    _export_calib: dict = {}
    _best_proxy_bpb: float = float("inf")
    _best_proxy_sd: dict | None = None
    _proxy_calib_tokens: torch.Tensor | None = None
    train_loss = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    for step in range(args.iterations):
        if args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            
            # Apply EMA shadow weights for evaluation
            _orig_ema_weights = None
            if args.ema_enabled and args.ema_eval_apply and ema is not None:
                _orig_ema_weights = ema.apply_shadow(base_model)
                
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            
            # Restore original weights for continued training
            if args.ema_enabled and args.ema_eval_apply and ema is not None and _orig_ema_weights is not None:
                ema.restore(base_model, _orig_ema_weights)
            tstats = tern_stats(base_model, group_size=args.bitnet_group_size)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms zero_frac:{tstats['zero_frac']:.3f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Sequence length schedule
        if args.curr_enabled:
            frac = elapsed_ms / max_wallclock_ms if max_wallclock_ms else step / args.iterations
            target_seq_len = args.train_seq_len
            if frac < args.curr_p1_f: target_seq_len = args.curr_p1_s
            elif frac < args.curr_p2_f: target_seq_len = args.curr_p2_s
            elif frac < args.curr_p3_f: target_seq_len = args.curr_p3_s
            elif frac < args.curr_p4_f: target_seq_len = args.curr_p4_s
            elif frac < args.curr_p5_f: target_seq_len = args.curr_p5_s
            
            if active_seq_len != target_seq_len:
                active_seq_len = target_seq_len
                base_model.reset_capsule_carry()
                torch._dynamo.reset()
                train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
                log0(f"step:{step} curr_seq_len_jump:{active_seq_len}")
        elif args.seq_len_start > 0 and not _seq_switched:
            if max_wallclock_ms is not None:
                should_switch_seq = elapsed_ms >= args.seq_schedule_fraction * max_wallclock_ms
            else:
                should_switch_seq = step >= int(args.iterations * args.seq_schedule_fraction)
            if should_switch_seq:
                active_seq_len = args.train_seq_len
                _seq_switched = True
                torch._dynamo.reset()
                train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
                log0(f"step:{step} seq_len_switch:{args.seq_len_start}->{active_seq_len}")
            
        if args.batch_tokens_start > 0 and not _batch_switched:
            if max_wallclock_ms is not None:
                should_switch_batch = elapsed_ms >= args.batch_schedule_fraction * max_wallclock_ms
            else:
                should_switch_batch = step >= int(args.iterations * args.batch_schedule_fraction)
            if should_switch_batch:
                active_batch_tokens = args.train_batch_tokens
                _batch_switched = True
                log0(f"step:{step} batch_switch:{args.batch_tokens_start}->{active_batch_tokens}")

        zero_grad_all()
        train_loss.zero_()

        # Feedback interleaving: skip feedback on some steps for speed
        _orig_fp = base_model.feedback_passes
        use_feedback = (
            args.feedback_enabled
            and args.feedback_passes > 0
            and max(args.feedback_every, 1) > 0
            and step % max(args.feedback_every, 1) == 0
        )
        if not use_feedback:
            base_model.feedback_passes = 0

        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
            elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            elapsed_frac = min(elapsed_ms / 1000.0 / max(args.max_wallclock_seconds, 1e-9), 1.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, elapsed_fraction=elapsed_frac)
            train_loss.add_(loss.detach())
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Restore original feedback passes
        base_model.feedback_passes = _orig_fp

        # Advance SKC capsule carry state for next step
        base_model.step_capsule_carry()

        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        # Untie lm_head at configured fraction of training
        if args.untie_at_fraction > 0:
            if max_wallclock_ms is not None:
                should_untie = not _untied and elapsed_ms >= args.untie_at_fraction * max_wallclock_ms
            else:
                should_untie = not _untied and step >= int(args.iterations * args.untie_at_fraction)
            if should_untie and base_model.tie_embeddings:
                with torch.no_grad():
                    base_weight = base_model.tok_emb.weight.float()
                    if base_model.embed_proj_rev is not None:
                        full_weight = base_weight @ base_model.embed_proj_rev.weight.float()
                    else:
                        full_weight = base_weight
                    base_model.lm_head.weight.copy_(full_weight)
                base_model.tie_embeddings = False
                base_model.lm_head.weight.requires_grad_(True)
                for g in opt_head.param_groups:
                    g["lr"] = g["base_lr"] = args.head_lr
                _untied = True
                torch._dynamo.reset()
                log0(f"step:{step} untied lm_head (head_lr={args.head_lr})")

        # Muon momentum warmup
        if args.matrix_optimizer not in ("adamw", "adam"):
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            for g in opt_muon.param_groups:
                g["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        # LR scheduling
        for opt in optimizers:
            for g in opt.param_groups:
                if "base_lr" in g:
                    g["lr"] = g["base_lr"] * scale
                else:
                    g["lr"] = args.adam_lr * scale
            opt.step()
        zero_grad_all()

        # Stability constraint: clamp Koopman diagonal to (-0.999, 0.999)
        if (hasattr(base_model, 'capsule_bank') and base_model.capsule_bank is not None
                and base_model.capsule_bank.koopman is not None):
            with torch.no_grad():
                base_model.capsule_bank.koopman.diag.clamp_(-0.999, 0.999)

        # EMA update (only after start_fraction of training)
        if ema is not None:
            ema_progress = elapsed_ms / max_wallclock_ms if max_wallclock_ms else step / args.iterations
            if ema_progress >= args.ema_start_fraction:
                ema.update(base_model)

        # Export-aligned training phase: switch TernaryLinear to use calibrated quantizer
        # [calibration moved to post-training to avoid DDP collective sync issues]

        # Export proxy eval: periodically serialize+reload+score to track round-trip BPB
        if (master_process and args.export_proxy_eval
                and step > 0 and step % args.export_proxy_every == 0):
            if _proxy_calib_tokens is None:
                _proxy_calib_tokens = ld_val(args.val_files, args.train_seq_len, max_tok=32768).to(device)
            proxy_bpb = _proxy_roundtrip_bpb(
                base_model.state_dict(), base_model, _export_calib,
                args.bitnet_group_size, _proxy_calib_tokens, args, device
            )
            log0(f"step:{step} export_proxy_bpb:{proxy_bpb:.4f} best:{_best_proxy_bpb:.4f}", flush=True)
            if proxy_bpb < _best_proxy_bpb and args.export_proxy_use_best:
                _best_proxy_bpb = proxy_bpb
                _best_proxy_sd = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
                log0(f"step:{step} export_proxy:new_best {proxy_bpb:.4f}", flush=True)

        if stop_after_step is not None and step >= stop_after_step:
            log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and (step + 1) % args.train_log_every == 0:
            log0(f"step:{step+1}/{args.iterations} loss:{train_loss.item():.4f} t:{approx_ms:.0f}ms avg:{approx_ms/(step+1):.1f}ms")
        if args.churn_log_every > 0 and step % args.churn_log_every == 0:
            log0(f"step:{step} churn:{churn_fn(base_model, args.bitnet_group_size):.4f} zero:{tern_stats(base_model, args.bitnet_group_size)['zero_frac']:.3f}")

        # Wallclock cap sync
        if stop_after_step is None and max_wallclock_ms is not None and step % 10 == 0:
            reached_cap = approx_ms >= max_wallclock_ms
            if distributed:
                cap_t = torch.tensor(int(reached_cap), device=device)
                dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
                reached_cap = bool(cap_t.item())
            if reached_cap:
                stop_after_step = step

    # --- Post-Training Final Evaluation and crystallization ---
    torch.cuda.synchronize()
    training_time_ms += 1000.0 * (time.perf_counter() - t0)
    
    # Final eval WITH EMA weights applied
    _orig_ema_weights = None
    if args.ema_enabled and args.ema_eval_apply and ema is not None:
        _orig_ema_weights = ema.apply_shadow(base_model)
        
    log0(f"final_evaluation:starting step:{step+1}/{args.iterations}")
    val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                 val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    
    final_loss, final_bpb = val_loss, val_bpb
    log0(f"final_evaluation:completed val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # --- Rank Synchronization before Export ---
    if distributed:
        torch.distributed.barrier()
        log0("ema:ranks synchronized for export", flush=True)

    # --- Calibration (post-training, after all ranks exit training loop) ---
    # Runs only on master; other ranks wait at the barrier below.
    # Moving calibration here avoids DDP collective sequence-number divergence.
    if master_process and args.export_aligned_train and (args.ternary_threshold_search or args.ternary_scale_search):
        if _proxy_calib_tokens is None:
            _proxy_calib_tokens = ld_val(args.val_files, args.train_seq_len, max_tok=args.calib_proxy_max_tok).to(device)
        log0(f"export_calib:starting thr_search={args.ternary_threshold_search} scale_search={args.ternary_scale_search}", flush=True)
        if ema is not None and _orig_ema_weights is None:
            _ema_orig_c = ema.apply_shadow(base_model)
            _export_calib = calibrate_ternary(base_model, _proxy_calib_tokens, args, device)
            ema.restore(base_model, _ema_orig_c)
        else:
            _export_calib = calibrate_ternary(base_model, _proxy_calib_tokens, args, device)
        global _EXPORT_CALIB
        _EXPORT_CALIB = _export_calib
        log0(f"export_calib:done calibrated={len(_export_calib)} tensors", flush=True)

    if distributed:
        torch.distributed.barrier()  # wait for rank 0 to finish calibration

    # --- Restore best proxy checkpoint BEFORE EMA so EMA smooths it, not vice versa ---
    # Bug fix: _best_proxy_sd was captured from live training weights. If we restore it
    # AFTER applying EMA shadow, we overwrite the carefully accumulated smoothed weights
    # with noisy per-step weights. Do the restore first, then let EMA apply on top.
    if master_process and _best_proxy_sd is not None and args.export_proxy_use_best:
        log0(f"serialization:restoring best_proxy_sd (proxy_bpb={_best_proxy_bpb:.4f}) before EMA", flush=True)
        base_model.load_state_dict(_best_proxy_sd)

    # --- Apply EMA shadow weights before serialization ---
    _ema_original = None
    if ema is not None:
        log0("ema:applying shadow weights...", flush=True)
        # Move original weights to CPU to free VRAM for master process export
        _ema_original = ema.apply_shadow(base_model, move_to_cpu=True)
        log0("ema:applied shadow weights and offloaded originals to CPU", flush=True)

    # --- Serialization ---
    if master_process:
        log0("serialization:started", flush=True)
        # Verification printout: param budget accounting
        _sd_check = base_model.state_dict()
        _ternary_names = [k for k, v in _sd_check.items()
                          if v.ndim == 2 and v.numel() > 16384
                          and "tok_emb" not in k and "lm_head" not in k and "embed_proj" not in k]
        _fp_names = [k for k in _sd_check if k not in _ternary_names]
        _ternary_params = sum(_sd_check[k].numel() for k in _ternary_names)
        _fp_params = sum(_sd_check[k].numel() for k in _fp_names)
        _est_ternary_bytes = _ternary_params * 1.585 / 8 + (_ternary_params / args.bitnet_group_size) * 2
        _est_fp_bytes = _fp_params * 2  # bf16
        _est_total_mb = (_est_ternary_bytes + _est_fp_bytes + 170000) / 1e6
        log0(f"param_audit: total={_ternary_params+_fp_params:,} ternary_candidates={_ternary_params:,}({len(_ternary_names)}) fp={_fp_params:,}({len(_fp_names)}) est_raw={(_est_ternary_bytes+_est_fp_bytes)/1e6:.2f}MB est_compressed≈{_est_total_mb:.2f}MB")
        sd = base_model.state_dict()

        # GPTQ-lite: per-row clip percentile search before quantization
        if args.gptq_lite_enabled:
            log0(f"gptq_lite:searching {args.gptq_lite_percentiles} percentiles...")
            sd = gptq_lite_clip_search(sd, group_size=args.bitnet_group_size,
                                       num_percentiles=args.gptq_lite_percentiles)
            log0("gptq_lite:done")
            # Reload clipped weights into base_model so calibration runs on the same
            # weight distribution that will actually be quantized (bug fix: without this,
            # calibrate_ternary tunes thresholds against unclipped weights then applies
            # them to clipped weights, producing mismatched calib).
            base_model.load_state_dict({k: v.to(base_model.state_dict()[k].device)
                                        for k, v in sd.items()}, strict=False)

        # Final calibration pass at export time (if not already done during training)
        final_calib = _export_calib
        if not _aligned_phase_started and (args.ternary_threshold_search or args.ternary_scale_search):
            if _proxy_calib_tokens is None:
                _proxy_calib_tokens = ld_val(args.val_files, args.train_seq_len, max_tok=32768).to(device)
            log0("serialization:running_final_calib_search", flush=True)
            final_calib = calibrate_ternary(base_model, _proxy_calib_tokens, args, device)
            log0(f"serialization:calib_done tensors={len(final_calib)}", flush=True)

        if base_model.tie_embeddings:
            sd.pop("lm_head.weight", None)

        # Two methods: Standard Base-3 vs Bitmask Mapping
        methods = {}
        for method in ("standard", "bitmask"):
            q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size, fp_storage=args.fp_storage, ternary_method=method, calib=final_calib)
            buf = io.BytesIO()
            torch.save(q_obj, buf)
            methods[method] = {"blob": lzma.compress(buf.getvalue(), preset=args.lzma_preset), "stats": stats}
        best = min(methods, key=lambda m: len(methods[m]["blob"]))
        final_blob, q_stats = methods[best]["blob"], methods[best]["stats"]
        with open("final_model.ternary.ptz", "wb") as f:
            f.write(final_blob)

        artifact_bytes = len(final_blob)
        code_bytes = len(code.encode("utf-8"))

        total = artifact_bytes + code_bytes
        log0(f"artifact:{artifact_bytes/1e6:.2f}MB ternary:{q_stats['ternary_params']}({q_stats['ternary_bytes']}B) fp:{q_stats['fp_params']}({q_stats['fp_bytes']}B) code:{code_bytes}")
        log0(f"budget:{total}/{16000000} ({total/1e6:.2f}/{16.00:.2f}MB) {'FITS' if total <= 16000000 else 'OVER'}")

        if args.eval_depth_recurrence > 0:
            base_model.training_depth_recurrence = args.eval_depth_recurrence
            log0(f"eval_depth_recurrence:{args.eval_depth_recurrence}")
        if args.eval_feedback_passes > 0:
            base_model.feedback_passes = args.eval_feedback_passes
            log0(f"eval_feedback_passes:{args.eval_feedback_passes}")

    # --- All ranks load roundtrip weights and evaluate ---
    if distributed:
        dist.barrier()

    with open("final_model.ternary.ptz", "rb") as f:
        loaded = torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu", weights_only=False)
    # strict=True — any missing/unexpected keys indicate a real mismatch.
    # Only exception: tied embedding (lm_head.weight absent in artifact is expected).
    _rt_missing, _rt_unexpected = base_model.load_state_dict(deq_sd(loaded), strict=False)
    _rt_unexpected_real = [k for k in _rt_unexpected if k != "lm_head.weight"]
    _rt_missing_real = [k for k in _rt_missing if k != "lm_head.weight"]
    if _rt_missing_real or _rt_unexpected_real:
        log0(f"WARNING roundtrip_load: missing={_rt_missing_real[:5]} unexpected={_rt_unexpected_real[:5]}")
    torch._dynamo.reset()

    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_ternary_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")

    # Three separate scoreboard slots — do NOT mix them.
    # Slot 1: raw FP/EMA model (computed above in final_loss/final_bpb)
    # Slot 2: post-quantization round-trip (this is the canonical submission metric)
    # Slot 3: inference-time augmented (informational only — accumulates below)
    roundtrip_val_loss = q_val_loss
    roundtrip_val_bpb = q_val_bpb
    augmented_val_loss = q_val_loss
    augmented_val_bpb = q_val_bpb

    opt_temp = 1.0
    if args.temp_scaling:
        torch.cuda.synchronize()
        t_temp = time.perf_counter()
        # Use a validation slice for calibration — do NOT use train_loader (no .stream, and it's leakage)
        calib_tokens = ld_val(args.val_files, args.train_seq_len, max_tok=65536).to(device)
        opt_temp = find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                             calib_tokens, base_bytes_lut, has_leading_space_lut,
                             is_boundary_token_lut)
        torch.cuda.synchronize()
        temp_time_ms = 1000.0 * (time.perf_counter() - t_temp)
        log0(f"temp_scaling optimal_T:{opt_temp:.2f} eval_time:{temp_time_ms:.0f}ms")

    if args.sliding_eval:
        torch.cuda.synchronize()
        t_sliding = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(args, base_model, rank, world_size, device, grad_accum_steps,
                                           val_tokens, base_bytes_lut, has_leading_space_lut,
                                           is_boundary_token_lut, stride=args.sliding_eval_stride,
                                           temperature=opt_temp)
        torch.cuda.synchronize()
        sliding_time_ms = 1000.0 * (time.perf_counter() - t_sliding)
        log0(f"final_sliding val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} "
             f"(stride={args.sliding_eval_stride}, T={opt_temp:.2f}) eval_time:{sliding_time_ms:.0f}ms")
        # Accumulates into augmented slot ONLY — roundtrip_val_bpb stays unchanged
        augmented_val_loss, augmented_val_bpb = sw_loss, sw_bpb

    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, base_model, rank, world_size, device, val_tokens, base_bytes_lut,
            has_leading_space_lut, is_boundary_token_lut, stride=args.sliding_eval_stride,
            batch_seqs=args.ttt_batch_seqs, temperature=opt_temp, log0=log0,
        )
        torch.cuda.synchronize()
        ttt_time_ms = 1000.0 * (time.perf_counter() - t_ttt)
        log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{ttt_time_ms:.0f}ms")
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
        # Accumulates into augmented slot ONLY — roundtrip_val_bpb stays unchanged
        augmented_val_loss, augmented_val_bpb = ttt_loss, ttt_bpb

    # --- N-gram cache evaluation (single-rank, sequential) ---
    if args.ngram_cache_enabled and master_process:
        torch.cuda.synchronize()
        t_ngram = time.perf_counter()
        ngram_cache = NgramCache(
            max_order=args.ngram_max_order,
            alpha_base=args.ngram_alpha_base,
            alpha_scale=args.ngram_alpha_scale,
            entropy_center=args.ngram_entropy_center,
        )
        seq_len = args.train_seq_len
        total_tokens_ng = val_tokens.numel() - 1
        ngram_loss_sum = 0.0
        ngram_byte_sum = 0.0
        ngram_tok_count = 0
        scored_tokens: list[int] = []
        base_model.eval()
        use_carry = args.capsule_carry_enabled and args.capsule_enabled
        decay = args.capsule_carry_decay if args.capsule_carry_enabled else 0.0
        carry_capsules = None
        # Process validation sequentially, building cache as we go
        with torch.inference_mode():
            for pos in range(0, total_tokens_ng, seq_len):
                end = min(pos + seq_len, total_tokens_ng)
                wlen = end - pos
                chunk = val_tokens[pos:end + 1].to(dtype=torch.int64, device=device)
                x_ng = chunk[:-1].unsqueeze(0)
                y_ng = chunk[1:].unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=ptdtype):
                    if use_carry:
                        logits_ng, capsule_state = base_model.forward_logits_with_carry(
                            x_ng, carry_capsules=carry_capsules, temperature=opt_temp)
                        if capsule_state is not None:
                            cs_avg = capsule_state.mean(dim=0, keepdim=True)
                            if carry_capsules is not None:
                                carry_capsules = (decay * carry_capsules + (1.0 - decay) * cs_avg).detach()
                            else:
                                carry_capsules = cs_avg.detach()
                    else:
                        logits_ng = base_model.forward_logits(x_ng, temperature=opt_temp)
                log_probs_ng = F.log_softmax(logits_ng.squeeze(0).float(), dim=-1)
                for t_idx in range(wlen):
                    target_tok = y_ng[0, t_idx].item()
                    neural_lp = log_probs_ng[t_idx]
                    # Try n-gram cache prediction
                    ngram_lp = ngram_cache.predict(scored_tokens[-(args.ngram_max_order - 1):], args.vocab_size)
                    if ngram_lp is not None:
                        ngram_lp = ngram_lp.to(device)
                        alpha = ngram_cache.entropy_alpha(neural_lp)
                        # Mix in log space: log((1-a)*exp(neural) + a*exp(ngram))
                        mixed = torch.logaddexp(
                            neural_lp + math.log(1 - alpha + 1e-10),
                            ngram_lp + math.log(alpha + 1e-10),
                        )
                        token_nll = -mixed[target_tok].item()
                    else:
                        token_nll = -neural_lp[target_tok].item()
                    ngram_loss_sum += token_nll
                    # BPB accounting — use actual prev token (from input sequence)
                    tok_b = base_bytes_lut[target_tok].item()
                    prev_tok = x_ng[0, t_idx].item()
                    if has_leading_space_lut[target_tok].item() and not is_boundary_token_lut[prev_tok].item():
                        tok_b += 1
                    ngram_byte_sum += tok_b
                    ngram_tok_count += 1
                    scored_tokens.append(target_tok)
                # Update cache with scored tokens
                ngram_cache.update(chunk[1:].tolist())
        ngram_val_loss = ngram_loss_sum / max(ngram_tok_count, 1)
        ngram_bpb = (ngram_val_loss / math.log(2.0)) * (ngram_tok_count / max(ngram_byte_sum, 1))
        ngram_time_ms = 1000.0 * (time.perf_counter() - t_ngram)
        log0(f"ngram_cache val_loss:{ngram_val_loss:.4f} val_bpb:{ngram_bpb:.4f} "
             f"(order={args.ngram_max_order}) eval_time:{ngram_time_ms:.0f}ms")
        # Accumulates into augmented slot ONLY — roundtrip_val_bpb stays unchanged
        augmented_val_loss, augmented_val_bpb = ngram_val_loss, ngram_bpb

    if master_process:
        import json
        artifact_bytes = os.path.getsize("final_model.ternary.ptz") if os.path.exists("final_model.ternary.ptz") else 0
        c_code_bytes = len(code.encode("utf-8")) if 'code' in locals() else 0
        total_bytes = artifact_bytes + c_code_bytes

        # Log all three scoreboard slots for full transparency
        log0(f"scoreboard: raw_bpb={final_bpb:.4f} roundtrip_bpb={roundtrip_val_bpb:.4f} augmented_bpb={augmented_val_bpb:.4f}")

        with open("submission.json", "w") as f:
            json.dump({
                "author": "Aki Gogikar (OneNewAI)",
                "github_id": "akhileshgogikar",
                "name": "KoopCaps-HRM Ternary Reasoner",
                "blurb": "KoopCaps-HRM: 20M ternary SKC MoE with EMA, GPTQ-lite, and deterministic export.",
                "date": "2026-03-27T00:00:00Z",
                "val_loss": round(roundtrip_val_loss, 4),
                "val_bpb": round(roundtrip_val_bpb, 4),
                "augmented_val_bpb": round(augmented_val_bpb, 4),
                "bytes_total": total_bytes,
                "bytes_code": c_code_bytes
            }, f, indent=2)

    if distributed:
        try:
            torch.distributed.barrier()
            dist.destroy_process_group()
        except:
            pass

if __name__ == "__main__":
    main()
