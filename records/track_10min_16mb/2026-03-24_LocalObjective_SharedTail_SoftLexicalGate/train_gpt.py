"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func

    HAS_FLASH_ATTN_3 = True
except Exception:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_3_func

        HAS_FLASH_ATTN_3 = True
    except Exception:
        flash_attn_3_func = None
        HAS_FLASH_ATTN_3 = False

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None
    sdpa_kernel = None

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    zstandard = None
    HAS_ZSTD = False

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default winner-thin run:
# - 10 transformer blocks at width 448
# - 8 attention heads with 4 KV heads (GQA), moderate hashed 4-gram lane, partial RoPE, XSA
# - tiny shared-loop tail for recurrence-like bias without broad runtime tax
# - true 2048 train/eval, EMA, late QAT, mixed export, and FA3 enabled
DEFAULT_DATA_PATH = "./data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
DEFAULT_VOCAB_SIZE = 1024
COUNTERFACTUAL_RAW_OFF = -20.0

BOS_ID = -1
LEXICAL_CONFIDENCE_MARGIN = float(os.environ.get("LEXICAL_CONFIDENCE_MARGIN", "0.05"))
DEFAULT_PRIOR_GATE_INIT = float(os.environ.get("PRIOR_GATE_INIT", "-1.15"))

# Keep local proxy specialist-lane knobs in one place so the frontier and
# promotion recipes do not drift when we tweak early-training behavior.
LOCAL_PROXY_RECIPE_COMMON: dict[str, object] = {
    "train_seq_len": 2048,
    "eval_seq_len": 2048,
    "num_layers": 10,
    "num_kv_heads": 4,
    "mlp_hidden": 448,
    "use_smear_gate": True,
    "bigram_vocab_size": 16384,
    "bigram_dim": 64,
    "hashed_ngram_order": 3,
    "bigram_hidden_scale_init": 0.08,
    "bigram_scale_init": 0.15,
    "bigram_hidden_min_mult": 0.30,
    "lexical_shortcut_scale_init": 0.0045,
    "lexical_confidence_bias_init": 0.04,
    "use_pairwise_logit_prior": True,
    "pairwise_logit_rank": 32,
    "pairwise_logit_scale_init": 0.75,
    "pairwise_logit_gate_init": -1.1,
    "pairwise_logit_confidence_gain_init": 0.35,
    "use_residual_logit_adapter": True,
    "residual_logit_rank": 16,
    "residual_logit_scale_init": 0.20,
    "residual_logit_gate_init": -2.0,
    "use_online_doc_bias": True,
    "doc_bias_rank": 8,
    "doc_bias_scale_init": 0.10,
    "doc_bias_gate_init": -2.0,
    "use_confidence_branch_skip": True,
    "confidence_skip_margin": 1.25,
    "confidence_skip_sharpness": 6.0,
    "confidence_skip_lexical_min_keep": 0.05,
    "confidence_skip_pairwise_min_keep": 0.0,
    "confidence_skip_doc_bias_min_keep": 0.05,
    "orthogonal_init": True,
    "mup_proj_init": True,
    "overtone_embed_init": True,
    "resid_mix_phase_init": True,
    "smear_gate_init": -3.0,
    "smear_lexical_scale_init": 0.5,
    "smear_confidence_gain_init": 0.0,
    "smear_lexical_proj_init_std": 0.0,
    "lexical_enable_step": 0,
    "lexical_ramp_steps": 180,
    "lexical_calibration_enable_step": 0,
    "lexical_calibration_ramp_steps": 100,
    "lexical_hidden_enable_step": 64,
    "lexical_hidden_ramp_steps": 192,
    "lexical_shortcut_enable_step": 48,
    "lexical_shortcut_ramp_steps": 128,
    "lexical_batch_diversity_throttle": 0.35,
    "lexical_batch_repeat_throttle": 0.75,
    "lexical_batch_min_mult": 0.5,
    "matrix_lr": 0.0623,
    "scalar_lr": 0.035,
    "tied_embed_lr": 0.004,
    "force_fp16_tied_embed_export": True,
    "shared_tail_output_gate": False,
    "shared_tail_enable_step": 0,
    "shared_tail_ramp_steps": 400,
    "shared_tail_max_mult": 1.0,
    "use_alibi": True,
    "helper_path_full_steps": 1000,
    "helper_path_decay_end_step": 6000,
    "use_adaptive_rmsnorm": True,
    "adaptive_rmsnorm_gate_init": -2.0,
    "num_shared_layers": 0,
    "shared_layer_repeats": 0,
    "mid_aux_loss_coeff": 0.025,
    "mid_aux_enable_step": 0,
    "mid_aux_ramp_steps": 120,
    "mid_aux_decay_start_step": 280,
    "mid_aux_decay_end_step": 960,
    "mlp_matrix_lr_mult": 0.575,
    "token_weight_decay": 0.005,
    "head_weight_decay": 0.0,
}

    # Keep local proxy launch presets aligned with the shared recipe defaults and
    # enable a cheap early validation window so we can inspect the real trajectory
    # in steps 0-10 without mutating the configured full validation budget.
LOCAL_PROXY_PRESET_COMMON: dict[str, object] = {
    "curriculum_policy": "winner_int6",
    "optimizer_policy": "local_proxy_faststart",
    "eval_policy": "sliding64_2048",
    "runtime_policy": "compiled",
    "lr_warmup_steps": 100,
    "lr_warmup_init_scale": 0.08,
    "lr_warmup_power": 1.75,
    "tied_embed_init_std": 0.0075,
    "tied_embed_warmup_mult": 0.2,
    "tied_embed_warmup_steps": 80,
    "hashed_ngram_init_std": 0.0025,
    "early_val_max_seqs": 256,
    "val_loss_every": 100,
    "objective_counterfactual_bootstrap_only": False,
    "objective_counterfactual_interval": 100,
}

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH)
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))
    lexical_shortcut_attn_gain_share_min = float(os.environ.get("LEXICAL_SHORTCUT_ATTN_GAIN_SHARE_MIN", "0.375"))
    lexical_shortcut_attn_gain_share_full = float(os.environ.get("LEXICAL_SHORTCUT_ATTN_GAIN_SHARE_FULL", "0.475"))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    lr_warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", "0"))
    lr_warmup_init_scale = float(os.environ.get("LR_WARMUP_INIT_SCALE", "0.2"))
    lr_warmup_power = float(os.environ.get("LR_WARMUP_POWER", "1.5"))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", "2048"))
    training_preset = os.environ.get("TRAINING_PRESET", "").strip().lower()
    recipe_family = os.environ.get("RECIPE_FAMILY", "").strip().lower()
    curriculum_policy = os.environ.get("CURRICULUM_POLICY", "").strip().lower()
    data_policy = os.environ.get("DATA_POLICY", "").strip().lower()
    optimizer_policy = os.environ.get("OPTIMIZER_POLICY", "").strip().lower()
    precision_policy = os.environ.get("PRECISION_POLICY", "").strip().lower()
    eval_policy = os.environ.get("EVAL_POLICY", "").strip().lower()
    runtime_policy = os.environ.get("RUNTIME_POLICY", "").strip().lower()
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    rope_dims = int(os.environ.get("ROPE_DIMS", "16"))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    enable_torch_compile = bool(int(os.environ.get("ENABLE_TORCH_COMPILE", "1")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 256))
    val_max_seqs = int(os.environ.get("VAL_MAX_SEQS", "0"))
    early_val_max_seqs = int(os.environ.get("EARLY_VAL_MAX_SEQS", "256"))
    milestone_val_max_seqs = int(os.environ.get("MILESTONE_VAL_MAX_SEQS", "256"))
    final_eval_max_seqs = int(os.environ.get("FINAL_EVAL_MAX_SEQS", "0"))
    submission_size_limit_bytes = int(os.environ.get("SUBMISSION_SIZE_LIMIT_BYTES", 16_000_000))
    label_smoothing = float(os.environ.get("LABEL_SMOOTHING", 0.0))
    z_loss_coeff = float(os.environ.get("Z_LOSS_COEFF", 0.00005))
    row_norm_loss_coeff = float(os.environ.get("ROW_NORM_LOSS_COEFF", "0.0"))
    outlier_loss_coeff = float(os.environ.get("OUTLIER_LOSS_COEFF", "0.0"))
    outlier_loss_threshold = float(os.environ.get("OUTLIER_LOSS_THRESHOLD", "3.0"))
    logit_var_loss_coeff = float(os.environ.get("LOGIT_VAR_LOSS_COEFF", "0.0005"))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9999))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 200))
    mid_aux_loss_coeff = float(
        os.environ.get("MID_AUX_LOSS_COEFF", str(float(LOCAL_PROXY_RECIPE_COMMON["mid_aux_loss_coeff"])))
    )
    force_fp16_tied_embed_export = bool(int(os.environ.get("FORCE_FP16_TIED_EMBED_EXPORT", "1")))
    int8_auto_keep_budget_bytes = int(os.environ.get("INT8_AUTO_KEEP_BUDGET_BYTES", "0"))
    int8_auto_keep_max_tensors = int(os.environ.get("INT8_AUTO_KEEP_MAX_TENSORS", "0"))
    int8_auto_keep_min_numel = int(os.environ.get("INT8_AUTO_KEEP_MIN_NUMEL", "65536"))
    int8_sensitivity_log_topk = int(os.environ.get("INT8_SENSITIVITY_LOG_TOPK", "8"))
    int8_auto_keep_row_budget_bytes = int(os.environ.get("INT8_AUTO_KEEP_ROW_BUDGET_BYTES", "0"))
    int8_auto_keep_row_group_size = int(os.environ.get("INT8_AUTO_KEEP_ROW_GROUP_SIZE", "32"))
    objective_health_train_shock_ratio = float(os.environ.get("OBJECTIVE_HEALTH_TRAIN_SHOCK_RATIO", "1.40"))
    objective_health_train_shock_window = int(os.environ.get("OBJECTIVE_HEALTH_TRAIN_SHOCK_WINDOW", "5"))
    aux_release_min_step = int(os.environ.get("AUX_RELEASE_MIN_STEP", "30"))
    aux_release_max_mid_top1_agreement = float(os.environ.get("AUX_RELEASE_MAX_MID_TOP1_AGREEMENT", "0.10"))
    aux_release_min_backbone_gain_share = float(os.environ.get("AUX_RELEASE_MIN_BACKBONE_GAIN_SHARE", "0.75"))
    aux_release_train_loss_tol = float(os.environ.get("AUX_RELEASE_TRAIN_LOSS_TOL", "1.002"))
    objective_counterfactual_eval = bool(int(os.environ.get("OBJECTIVE_COUNTERFACTUAL_EVAL", "1")))
    objective_counterfactual_bootstrap_only = bool(int(os.environ.get("OBJECTIVE_COUNTERFACTUAL_BOOTSTRAP_ONLY", "0")))
    objective_counterfactual_interval = int(os.environ.get("OBJECTIVE_COUNTERFACTUAL_INTERVAL", "100"))
    objective_counterfactual_max_seqs = int(os.environ.get("OBJECTIVE_COUNTERFACTUAL_MAX_SEQS", "16"))
    objective_counterfactual_groups = os.environ.get(
        "OBJECTIVE_COUNTERFACTUAL_GROUPS",
        "lexical",
    ).strip()
    objective_track_module_attribution = bool(int(os.environ.get("OBJECTIVE_TRACK_MODULE_ATTRIBUTION", "1")))
    objective_module_attribution_interval = int(os.environ.get("OBJECTIVE_MODULE_ATTRIBUTION_INTERVAL", "1"))
    objective_bootstrap_steps = int(os.environ.get("OBJECTIVE_BOOTSTRAP_STEPS", "200"))
    objective_bootstrap_tokens = int(os.environ.get("OBJECTIVE_BOOTSTRAP_TOKENS", "0"))
    objective_bootstrap_train_log_every = int(os.environ.get("OBJECTIVE_BOOTSTRAP_TRAIN_LOG_EVERY", "10"))
    save_best_val_bpb_checkpoint = bool(int(os.environ.get("SAVE_BEST_VAL_BPB_CHECKPOINT", "1")))
    best_val_bpb_checkpoint_path = os.environ.get("BEST_VAL_BPB_CHECKPOINT_PATH", "best_val_bpb_model.pt").strip()
    init_model_path = os.environ.get("INIT_MODEL_PATH", "").strip()
    init_model_strict = bool(int(os.environ.get("INIT_MODEL_STRICT", "1")))
    regression_stop_min_step = int(os.environ.get("REGRESSION_STOP_MIN_STEP", "0"))
    regression_stop_patience = int(os.environ.get("REGRESSION_STOP_PATIENCE", "0"))
    regression_stop_min_delta = float(os.environ.get("REGRESSION_STOP_MIN_DELTA", "0.0"))
    train_random_offset_tokens = int(os.environ.get("TRAIN_RANDOM_OFFSET_TOKENS", "0"))
    use_packed_eval_windows = bool(int(os.environ.get("USE_PACKED_EVAL_WINDOWS", "1")))
    use_cuda_graphs = bool(int(os.environ.get("USE_CUDA_GRAPHS", "1")))
    cuda_graph_warmup_steps = int(os.environ.get("CUDA_GRAPH_WARMUP_STEPS", "0"))
    debug_static_shapes = bool(int(os.environ.get("DEBUG_STATIC_SHAPES", "0")))
    enable_cudnn_sdp = bool(int(os.environ.get("ENABLE_CUDNN_SDP", "0")))
    enable_flash_sdp = bool(int(os.environ.get("ENABLE_FLASH_SDP", "1")))
    enable_mem_efficient_sdp = bool(int(os.environ.get("ENABLE_MEM_EFFICIENT_SDP", "0")))
    enable_math_sdp = bool(int(os.environ.get("ENABLE_MATH_SDP", "1")))
    overtone_embed_init = bool(int(os.environ.get("OVERTONE_EMBED_INIT", "1")))
    overtone_embed_power = float(os.environ.get("OVERTONE_EMBED_POWER", "0.5"))
    overtone_embed_scale = float(os.environ.get("OVERTONE_EMBED_SCALE", "1.0"))
    resid_mix_phase_init = bool(int(os.environ.get("RESID_MIX_PHASE_INIT", "1")))
    resid_mix_phase_sharpness = float(os.environ.get("RESID_MIX_PHASE_SHARPNESS", "8.0"))
    resid_mix_phase_center = float(os.environ.get("RESID_MIX_PHASE_CENTER", "0.5"))
    use_alibi = bool(
        int(os.environ.get("USE_ALIBI", str(int(bool(LOCAL_PROXY_RECIPE_COMMON.get("use_alibi", False))))))
    )
    use_residual_attention = bool(int(os.environ.get("USE_RESIDUAL_ATTENTION", "0")))
    residual_attention_gain_init = float(os.environ.get("RESIDUAL_ATTENTION_GAIN_INIT", "0.5"))
    residual_attention_state_bits = int(os.environ.get("RESIDUAL_ATTENTION_STATE_BITS", "8"))
    manual_attention_query_chunk_size = int(os.environ.get("MANUAL_ATTENTION_QUERY_CHUNK_SIZE", "128"))
    manual_attention_key_chunk_size = int(os.environ.get("MANUAL_ATTENTION_KEY_CHUNK_SIZE", "128"))
    helper_path_full_steps = int(
        os.environ.get("HELPER_PATH_FULL_STEPS", str(int(LOCAL_PROXY_RECIPE_COMMON["helper_path_full_steps"])))
    )
    helper_path_decay_end_step = int(
        os.environ.get(
            "HELPER_PATH_DECAY_END_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["helper_path_decay_end_step"])),
        )
    )
    use_adaptive_rmsnorm = bool(int(os.environ.get("USE_ADAPTIVE_RMSNORM", "0")))
    adaptive_rmsnorm_gate_init = float(os.environ.get("ADAPTIVE_RMSNORM_GATE_INIT", "-2.0"))
    fake_quant_tail_steps = int(os.environ.get("FAKE_QUANT_TAIL_STEPS", "0"))
    fake_quant_bits = int(os.environ.get("FAKE_QUANT_BITS", "6"))
    fake_quant_full_run = bool(int(os.environ.get("FAKE_QUANT_FULL_RUN", "0")))
    late_qat = bool(int(os.environ.get("LATE_QAT", "1")))
    qat_threshold = float(os.environ.get("QAT_THRESHOLD", "0.15"))
    use_smear_gate = bool(
        int(os.environ.get("USE_SMEAR_GATE", "1" if LOCAL_PROXY_RECIPE_COMMON["use_smear_gate"] else "0"))
    )
    smear_gate_init = float(
        os.environ.get("SMEAR_GATE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["smear_gate_init"])))
    )
    smear_lexical_scale_init = float(
        os.environ.get("SMEAR_LEXICAL_SCALE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["smear_lexical_scale_init"])))
    )
    smear_confidence_gain_init = float(
        os.environ.get(
            "SMEAR_CONFIDENCE_GAIN_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["smear_confidence_gain_init"])),
        )
    )
    smear_lexical_proj_init_std = float(
        os.environ.get(
            "SMEAR_LEXICAL_PROJ_INIT_STD",
            str(float(LOCAL_PROXY_RECIPE_COMMON["smear_lexical_proj_init_std"])),
        )
    )
    lexical_enable_step = int(
        os.environ.get("LEXICAL_ENABLE_STEP", str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_enable_step"])))
    )
    lexical_ramp_steps = int(
        os.environ.get("LEXICAL_RAMP_STEPS", str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_ramp_steps"])))
    )
    lexical_calibration_enable_step = int(
        os.environ.get(
            "LEXICAL_CALIBRATION_ENABLE_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_calibration_enable_step"])),
        )
    )
    lexical_calibration_ramp_steps = int(
        os.environ.get(
            "LEXICAL_CALIBRATION_RAMP_STEPS",
            str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_calibration_ramp_steps"])),
        )
    )
    lexical_hidden_enable_step = int(
        os.environ.get(
            "LEXICAL_HIDDEN_ENABLE_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_hidden_enable_step"])),
        )
    )
    lexical_hidden_ramp_steps = int(
        os.environ.get(
            "LEXICAL_HIDDEN_RAMP_STEPS",
            str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_hidden_ramp_steps"])),
        )
    )
    lexical_shortcut_enable_step = int(
        os.environ.get(
            "LEXICAL_SHORTCUT_ENABLE_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_shortcut_enable_step"])),
        )
    )
    lexical_shortcut_ramp_steps = int(
        os.environ.get(
            "LEXICAL_SHORTCUT_RAMP_STEPS",
            str(int(LOCAL_PROXY_RECIPE_COMMON["lexical_shortcut_ramp_steps"])),
        )
    )
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", str(int(LOCAL_PROXY_RECIPE_COMMON["bigram_vocab_size"]))))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", str(int(LOCAL_PROXY_RECIPE_COMMON["bigram_dim"]))))
    hashed_ngram_order = int(
        os.environ.get("HASHED_NGRAM_ORDER", str(int(LOCAL_PROXY_RECIPE_COMMON["hashed_ngram_order"])))
    )
    hashed_ngram_init_std = float(os.environ.get("HASHED_NGRAM_INIT_STD", "0.0"))
    bigram_hidden_scale_init = float(
        os.environ.get("BIGRAM_HIDDEN_SCALE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["bigram_hidden_scale_init"])))
    )
    bigram_scale_init = float(
        os.environ.get("BIGRAM_SCALE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["bigram_scale_init"])))
    )
    lexical_shortcut_scale_init = float(
        os.environ.get("LEXICAL_SHORTCUT_SCALE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["lexical_shortcut_scale_init"])))
    )
    prior_gate_init = float(os.environ.get("PRIOR_GATE_INIT", str(DEFAULT_PRIOR_GATE_INIT)))
    prior_health_min_improvement = float(os.environ.get("PRIOR_HEALTH_MIN_IMPROVEMENT", "0.004"))
    prior_health_regression_tol = float(os.environ.get("PRIOR_HEALTH_REGRESSION_TOL", "0.02"))
    prior_health_confirmations = int(os.environ.get("PRIOR_HEALTH_CONFIRMATIONS", "2"))
    prior_health_threshold = float(os.environ.get("PRIOR_HEALTH_THRESHOLD", "0.75"))
    prior_health_recovery = float(os.environ.get("PRIOR_HEALTH_RECOVERY", "0.10"))
    prior_health_decay = float(os.environ.get("PRIOR_HEALTH_DECAY", "0.25"))
    prior_bigram_cap_floor = float(os.environ.get("PRIOR_BIGRAM_CAP_FLOOR", "0.35"))
    prior_bigram_ramp_tokens = int(os.environ.get("PRIOR_BIGRAM_RAMP_TOKENS", "320000000"))
    bigram_hidden_min_mult = float(
        os.environ.get("BIGRAM_HIDDEN_MIN_MULT", str(float(LOCAL_PROXY_RECIPE_COMMON["bigram_hidden_min_mult"])))
    )
    prior_lexical_cap_floor = float(os.environ.get("PRIOR_LEXICAL_CAP_FLOOR", "0.18"))
    prior_lexical_ramp_tokens = int(os.environ.get("PRIOR_LEXICAL_RAMP_TOKENS", "24000000"))
    lexical_confidence_bias_init = float(
        os.environ.get(
            "LEXICAL_CONFIDENCE_BIAS_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["lexical_confidence_bias_init"])),
        )
    )
    warmup_prior_init = bool(int(os.environ.get("WARMUP_PRIOR_INIT", "1")))
    warmup_prior_init_mode = os.environ.get("WARMUP_PRIOR_INIT_MODE", "bounded_delta").strip().lower()
    warmup_prior_init_blend = float(os.environ.get("WARMUP_PRIOR_INIT_BLEND", "0.10"))
    warmup_prior_init_delta_rms_mult = float(os.environ.get("WARMUP_PRIOR_INIT_DELTA_RMS_MULT", "0.35"))
    warmup_prior_init_delta_rms_floor = float(os.environ.get("WARMUP_PRIOR_INIT_DELTA_RMS_FLOOR", "0.0005"))
    warmup_prior_init_groups = tuple(
        item.strip()
        for item in os.environ.get("WARMUP_PRIOR_INIT_GROUPS", "lexical").split(",")
        if item.strip()
    )
    lexical_batch_diversity_throttle = float(
        os.environ.get(
            "LEXICAL_BATCH_DIVERSITY_THROTTLE",
            str(float(LOCAL_PROXY_RECIPE_COMMON["lexical_batch_diversity_throttle"])),
        )
    )
    lexical_batch_repeat_throttle = float(
        os.environ.get(
            "LEXICAL_BATCH_REPEAT_THROTTLE",
            str(float(LOCAL_PROXY_RECIPE_COMMON["lexical_batch_repeat_throttle"])),
        )
    )
    lexical_batch_min_mult = float(
        os.environ.get("LEXICAL_BATCH_MIN_MULT", str(float(LOCAL_PROXY_RECIPE_COMMON["lexical_batch_min_mult"])))
    )
    shared_tail_output_gate = bool(
        int(os.environ.get("SHARED_TAIL_OUTPUT_GATE", "1" if LOCAL_PROXY_RECIPE_COMMON["shared_tail_output_gate"] else "0"))
    )
    shared_tail_output_init = float(os.environ.get("SHARED_TAIL_OUTPUT_INIT", "-1.5"))
    shared_tail_enable_step = int(
        os.environ.get("SHARED_TAIL_ENABLE_STEP", str(int(LOCAL_PROXY_RECIPE_COMMON["shared_tail_enable_step"])))
    )
    shared_tail_ramp_steps = int(
        os.environ.get("SHARED_TAIL_RAMP_STEPS", str(int(LOCAL_PROXY_RECIPE_COMMON["shared_tail_ramp_steps"])))
    )
    shared_tail_max_mult = float(
        os.environ.get("SHARED_TAIL_MAX_MULT", str(float(LOCAL_PROXY_RECIPE_COMMON["shared_tail_max_mult"])))
    )
    signed_skip_weights = bool(int(os.environ.get("SIGNED_SKIP_WEIGHTS", "1")))
    mid_aux_enable_step = int(
        os.environ.get("MID_AUX_ENABLE_STEP", str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_enable_step"])))
    )
    mid_aux_ramp_steps = int(
        os.environ.get("MID_AUX_RAMP_STEPS", str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_ramp_steps"])))
    )
    mid_aux_decay_start_step = int(
        os.environ.get(
            "MID_AUX_DECAY_START_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_decay_start_step"])),
        )
    )
    mid_aux_decay_end_step = int(
        os.environ.get(
            "MID_AUX_DECAY_END_STEP",
            str(int(LOCAL_PROXY_RECIPE_COMMON["mid_aux_decay_end_step"])),
        )
    )
    orthogonal_init = bool(int(os.environ.get("ORTHOGONAL_INIT", "1")))
    mup_proj_init = bool(int(os.environ.get("MUP_PROJ_INIT", "1")))
    export_quant_bits = int(os.environ.get("EXPORT_QUANT_BITS", "5"))
    export_codec = os.environ.get("EXPORT_CODEC", "zstd").strip().lower()
    export_zstd_level = int(os.environ.get("EXPORT_ZSTD_LEVEL", "22"))
    export_high_precision_bits = int(os.environ.get("EXPORT_HIGH_PRECISION_BITS", "8"))
    export_high_precision_budget_bytes = int(os.environ.get("EXPORT_HIGH_PRECISION_BUDGET_BYTES", "300000"))
    export_high_precision_max_tensors = int(os.environ.get("EXPORT_HIGH_PRECISION_MAX_TENSORS", "4"))
    export_high_precision_min_numel = int(os.environ.get("EXPORT_HIGH_PRECISION_MIN_NUMEL", "65536"))
    use_flash_attn_3 = bool(int(os.environ.get("USE_FLASH_ATTN_3", "1")))
    attention_kv_mode = os.environ.get("ATTENTION_KV_MODE", "").strip().lower()
    use_mqa = bool(int(os.environ.get("USE_MQA", "0")))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", DEFAULT_VOCAB_SIZE))
    num_layers = int(os.environ.get("NUM_LAYERS", 10))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 448))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", "896"))
    num_shared_layers = int(os.environ.get("NUM_SHARED_LAYERS", "2"))
    shared_layer_repeats = int(os.environ.get("SHARED_LAYER_REPEATS", "2"))
    shared_loop_gate_init = float(os.environ.get("SHARED_LOOP_GATE_INIT", "-1.5"))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    use_output_logit_bias = bool(int(os.environ.get("USE_OUTPUT_LOGIT_BIAS", "0")))
    use_pairwise_logit_prior = bool(
        int(os.environ.get("USE_PAIRWISE_LOGIT_PRIOR", "1" if LOCAL_PROXY_RECIPE_COMMON["use_pairwise_logit_prior"] else "0"))
    )
    pairwise_logit_rank = int(os.environ.get("PAIRWISE_LOGIT_RANK", str(int(LOCAL_PROXY_RECIPE_COMMON["pairwise_logit_rank"]))))
    pairwise_logit_scale_init = float(
        os.environ.get("PAIRWISE_LOGIT_SCALE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["pairwise_logit_scale_init"])))
    )
    pairwise_logit_gate_init = float(
        os.environ.get("PAIRWISE_LOGIT_GATE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["pairwise_logit_gate_init"])))
    )
    pairwise_logit_confidence_gain_init = float(
        os.environ.get(
            "PAIRWISE_LOGIT_CONFIDENCE_GAIN_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["pairwise_logit_confidence_gain_init"])),
        )
    )
    use_online_doc_bias = bool(
        int(os.environ.get("USE_ONLINE_DOC_BIAS", "1" if LOCAL_PROXY_RECIPE_COMMON["use_online_doc_bias"] else "0"))
    )
    doc_bias_rank = int(os.environ.get("DOC_BIAS_RANK", str(int(LOCAL_PROXY_RECIPE_COMMON["doc_bias_rank"]))))
    doc_bias_scale_init = float(
        os.environ.get("DOC_BIAS_SCALE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["doc_bias_scale_init"])))
    )
    doc_bias_gate_init = float(
        os.environ.get("DOC_BIAS_GATE_INIT", str(float(LOCAL_PROXY_RECIPE_COMMON["doc_bias_gate_init"])))
    )
    use_confidence_branch_skip = bool(
        int(
            os.environ.get(
                "USE_CONFIDENCE_BRANCH_SKIP",
                "1" if LOCAL_PROXY_RECIPE_COMMON["use_confidence_branch_skip"] else "0",
            )
        )
    )
    confidence_skip_margin = float(
        os.environ.get(
            "CONFIDENCE_SKIP_MARGIN",
            str(float(LOCAL_PROXY_RECIPE_COMMON["confidence_skip_margin"])),
        )
    )
    confidence_skip_sharpness = float(
        os.environ.get(
            "CONFIDENCE_SKIP_SHARPNESS",
            str(float(LOCAL_PROXY_RECIPE_COMMON["confidence_skip_sharpness"])),
        )
    )
    confidence_skip_lexical_min_keep = float(
        os.environ.get(
            "CONFIDENCE_SKIP_LEXICAL_MIN_KEEP",
            str(float(LOCAL_PROXY_RECIPE_COMMON["confidence_skip_lexical_min_keep"])),
        )
    )
    confidence_skip_pairwise_min_keep = float(
        os.environ.get(
            "CONFIDENCE_SKIP_PAIRWISE_MIN_KEEP",
            str(float(LOCAL_PROXY_RECIPE_COMMON["confidence_skip_pairwise_min_keep"])),
        )
    )
    confidence_skip_doc_bias_min_keep = float(
        os.environ.get(
            "CONFIDENCE_SKIP_DOC_BIAS_MIN_KEEP",
            str(float(LOCAL_PROXY_RECIPE_COMMON["confidence_skip_doc_bias_min_keep"])),
        )
    )
    use_residual_logit_adapter = bool(
        int(
            os.environ.get(
                "USE_RESIDUAL_LOGIT_ADAPTER",
                "1" if LOCAL_PROXY_RECIPE_COMMON["use_residual_logit_adapter"] else "0",
            )
        )
    )
    residual_logit_rank = int(
        os.environ.get("RESIDUAL_LOGIT_RANK", str(int(LOCAL_PROXY_RECIPE_COMMON["residual_logit_rank"])))
    )
    residual_logit_scale_init = float(
        os.environ.get(
            "RESIDUAL_LOGIT_SCALE_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["residual_logit_scale_init"])),
        )
    )
    residual_logit_gate_init = float(
        os.environ.get(
            "RESIDUAL_LOGIT_GATE_INIT",
            str(float(LOCAL_PROXY_RECIPE_COMMON["residual_logit_gate_init"])),
        )
    )
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", str(float(LOCAL_PROXY_RECIPE_COMMON["tied_embed_lr"]))))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    tied_embed_warmup_mult = float(os.environ.get("TIED_EMBED_WARMUP_MULT", "0.3"))
    tied_embed_warmup_steps = int(os.environ.get("TIED_EMBED_WARMUP_STEPS", "40"))
    token_weight_decay = float(os.environ.get("TOKEN_WEIGHT_DECAY", str(float(LOCAL_PROXY_RECIPE_COMMON["token_weight_decay"]))))
    matrix_lr = float(os.environ.get("MATRIX_LR", str(float(LOCAL_PROXY_RECIPE_COMMON["matrix_lr"]))))
    mlp_matrix_lr_mult = float(os.environ.get("MLP_MATRIX_LR_MULT", str(float(LOCAL_PROXY_RECIPE_COMMON["mlp_matrix_lr_mult"]))))
    scalar_lr = float(os.environ.get("SCALAR_LR", str(float(LOCAL_PROXY_RECIPE_COMMON["scalar_lr"]))))
    head_weight_decay = float(os.environ.get("HEAD_WEIGHT_DECAY", str(float(LOCAL_PROXY_RECIPE_COMMON["head_weight_decay"]))))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", "0.03"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_backend_steps_light = int(os.environ.get("MUON_BACKEND_STEPS_LIGHT", "0"))
    muon_backend_refresh_interval = int(os.environ.get("MUON_BACKEND_REFRESH_INTERVAL", "1"))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    early_phase_steps = int(os.environ.get("EARLY_PHASE_STEPS", "0"))
    early_muon_attn_lr_scale = float(os.environ.get("EARLY_MUON_ATTN_LR_SCALE", "1.0"))
    early_muon_mlp_lr_scale = float(os.environ.get("EARLY_MUON_MLP_LR_SCALE", "1.0"))
    early_muon_wd_scale = float(os.environ.get("EARLY_MUON_WD_SCALE", "1.0"))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.25))

    def __init__(self):
        self._apply_policy_schema()
        self._apply_training_preset()
        self._resolve_attention_config()

    def _env_override_map(self) -> dict[str, str]:
        return {
            "train_seq_len": "TRAIN_SEQ_LEN",
            "eval_seq_len": "EVAL_SEQ_LEN",
            "eval_stride": "EVAL_STRIDE",
            "eval_batch_seqs": "EVAL_BATCH_SEQS",
            "train_batch_tokens": "TRAIN_BATCH_TOKENS",
            "early_val_max_seqs": "EARLY_VAL_MAX_SEQS",
            "lr_warmup_steps": "LR_WARMUP_STEPS",
            "lr_warmup_init_scale": "LR_WARMUP_INIT_SCALE",
            "num_layers": "NUM_LAYERS",
            "mlp_hidden": "MLP_HIDDEN",
            "rope_dims": "ROPE_DIMS",
            "ln_scale": "LN_SCALE",
            "tied_embed_lr": "TIED_EMBED_LR",
            "tied_embed_warmup_mult": "TIED_EMBED_WARMUP_MULT",
            "tied_embed_warmup_steps": "TIED_EMBED_WARMUP_STEPS",
            "matrix_lr": "MATRIX_LR",
            "scalar_lr": "SCALAR_LR",
            "muon_momentum": "MUON_MOMENTUM",
            "muon_momentum_warmup_start": "MUON_MOMENTUM_WARMUP_START",
            "muon_momentum_warmup_steps": "MUON_MOMENTUM_WARMUP_STEPS",
            "muon_backend_steps_light": "MUON_BACKEND_STEPS_LIGHT",
            "muon_backend_refresh_interval": "MUON_BACKEND_REFRESH_INTERVAL",
            "early_phase_steps": "EARLY_PHASE_STEPS",
            "early_muon_attn_lr_scale": "EARLY_MUON_ATTN_LR_SCALE",
            "early_muon_mlp_lr_scale": "EARLY_MUON_MLP_LR_SCALE",
            "early_muon_wd_scale": "EARLY_MUON_WD_SCALE",
            "warmdown_iters": "WARMDOWN_ITERS",
            "grad_clip_norm": "GRAD_CLIP_NORM",
            "muon_weight_decay": "MUON_WEIGHT_DECAY",
            "mid_aux_loss_coeff": "MID_AUX_LOSS_COEFF",
            "mid_aux_enable_step": "MID_AUX_ENABLE_STEP",
            "mid_aux_ramp_steps": "MID_AUX_RAMP_STEPS",
            "mid_aux_decay_start_step": "MID_AUX_DECAY_START_STEP",
            "mid_aux_decay_end_step": "MID_AUX_DECAY_END_STEP",
            "overtone_embed_init": "OVERTONE_EMBED_INIT",
            "resid_mix_phase_init": "RESID_MIX_PHASE_INIT",
            "resid_mix_phase_center": "RESID_MIX_PHASE_CENTER",
            "force_fp16_tied_embed_export": "FORCE_FP16_TIED_EMBED_EXPORT",
            "fake_quant_tail_steps": "FAKE_QUANT_TAIL_STEPS",
            "fake_quant_bits": "FAKE_QUANT_BITS",
            "fake_quant_full_run": "FAKE_QUANT_FULL_RUN",
            "late_qat": "LATE_QAT",
            "qat_threshold": "QAT_THRESHOLD",
            "int8_auto_keep_budget_bytes": "INT8_AUTO_KEEP_BUDGET_BYTES",
            "int8_auto_keep_max_tensors": "INT8_AUTO_KEEP_MAX_TENSORS",
            "int8_auto_keep_row_budget_bytes": "INT8_AUTO_KEEP_ROW_BUDGET_BYTES",
            "int8_auto_keep_row_group_size": "INT8_AUTO_KEEP_ROW_GROUP_SIZE",
            "comp_reference_train_tokens": "COMP_REFERENCE_TRAIN_TOKENS",
            "use_smear_gate": "USE_SMEAR_GATE",
            "use_packed_eval_windows": "USE_PACKED_EVAL_WINDOWS",
            "smear_gate_init": "SMEAR_GATE_INIT",
            "smear_lexical_scale_init": "SMEAR_LEXICAL_SCALE_INIT",
            "smear_confidence_gain_init": "SMEAR_CONFIDENCE_GAIN_INIT",
            "smear_lexical_proj_init_std": "SMEAR_LEXICAL_PROJ_INIT_STD",
            "lexical_enable_step": "LEXICAL_ENABLE_STEP",
            "lexical_ramp_steps": "LEXICAL_RAMP_STEPS",
            "lexical_calibration_enable_step": "LEXICAL_CALIBRATION_ENABLE_STEP",
            "lexical_calibration_ramp_steps": "LEXICAL_CALIBRATION_RAMP_STEPS",
            "lexical_hidden_enable_step": "LEXICAL_HIDDEN_ENABLE_STEP",
            "lexical_hidden_ramp_steps": "LEXICAL_HIDDEN_RAMP_STEPS",
            "lexical_shortcut_enable_step": "LEXICAL_SHORTCUT_ENABLE_STEP",
            "lexical_shortcut_ramp_steps": "LEXICAL_SHORTCUT_RAMP_STEPS",
            "bigram_vocab_size": "BIGRAM_VOCAB_SIZE",
            "bigram_dim": "BIGRAM_DIM",
            "hashed_ngram_order": "HASHED_NGRAM_ORDER",
            "hashed_ngram_init_std": "HASHED_NGRAM_INIT_STD",
            "bigram_hidden_scale_init": "BIGRAM_HIDDEN_SCALE_INIT",
            "bigram_scale_init": "BIGRAM_SCALE_INIT",
            "lexical_shortcut_scale_init": "LEXICAL_SHORTCUT_SCALE_INIT",
            "lexical_confidence_bias_init": "LEXICAL_CONFIDENCE_BIAS_INIT",
            "warmup_prior_init": "WARMUP_PRIOR_INIT",
            "warmup_prior_init_mode": "WARMUP_PRIOR_INIT_MODE",
            "warmup_prior_init_blend": "WARMUP_PRIOR_INIT_BLEND",
            "warmup_prior_init_delta_rms_mult": "WARMUP_PRIOR_INIT_DELTA_RMS_MULT",
            "warmup_prior_init_delta_rms_floor": "WARMUP_PRIOR_INIT_DELTA_RMS_FLOOR",
            "warmup_prior_init_groups": "WARMUP_PRIOR_INIT_GROUPS",
            "lexical_batch_diversity_throttle": "LEXICAL_BATCH_DIVERSITY_THROTTLE",
            "lexical_batch_repeat_throttle": "LEXICAL_BATCH_REPEAT_THROTTLE",
            "lexical_batch_min_mult": "LEXICAL_BATCH_MIN_MULT",
            "shared_tail_output_gate": "SHARED_TAIL_OUTPUT_GATE",
            "shared_tail_output_init": "SHARED_TAIL_OUTPUT_INIT",
            "shared_tail_enable_step": "SHARED_TAIL_ENABLE_STEP",
            "shared_tail_ramp_steps": "SHARED_TAIL_RAMP_STEPS",
            "shared_tail_max_mult": "SHARED_TAIL_MAX_MULT",
            "signed_skip_weights": "SIGNED_SKIP_WEIGHTS",
            "helper_path_full_steps": "HELPER_PATH_FULL_STEPS",
            "use_pairwise_logit_prior": "USE_PAIRWISE_LOGIT_PRIOR",
            "pairwise_logit_rank": "PAIRWISE_LOGIT_RANK",
            "pairwise_logit_scale_init": "PAIRWISE_LOGIT_SCALE_INIT",
            "pairwise_logit_gate_init": "PAIRWISE_LOGIT_GATE_INIT",
            "pairwise_logit_confidence_gain_init": "PAIRWISE_LOGIT_CONFIDENCE_GAIN_INIT",
            "use_online_doc_bias": "USE_ONLINE_DOC_BIAS",
            "doc_bias_rank": "DOC_BIAS_RANK",
            "doc_bias_scale_init": "DOC_BIAS_SCALE_INIT",
            "doc_bias_gate_init": "DOC_BIAS_GATE_INIT",
            "use_confidence_branch_skip": "USE_CONFIDENCE_BRANCH_SKIP",
            "confidence_skip_margin": "CONFIDENCE_SKIP_MARGIN",
            "confidence_skip_sharpness": "CONFIDENCE_SKIP_SHARPNESS",
            "confidence_skip_lexical_min_keep": "CONFIDENCE_SKIP_LEXICAL_MIN_KEEP",
            "confidence_skip_pairwise_min_keep": "CONFIDENCE_SKIP_PAIRWISE_MIN_KEEP",
            "confidence_skip_doc_bias_min_keep": "CONFIDENCE_SKIP_DOC_BIAS_MIN_KEEP",
            "use_residual_logit_adapter": "USE_RESIDUAL_LOGIT_ADAPTER",
            "residual_logit_rank": "RESIDUAL_LOGIT_RANK",
            "residual_logit_scale_init": "RESIDUAL_LOGIT_SCALE_INIT",
            "residual_logit_gate_init": "RESIDUAL_LOGIT_GATE_INIT",
            "helper_path_decay_end_step": "HELPER_PATH_DECAY_END_STEP",
            "use_adaptive_rmsnorm": "USE_ADAPTIVE_RMSNORM",
            "adaptive_rmsnorm_gate_init": "ADAPTIVE_RMSNORM_GATE_INIT",
            "orthogonal_init": "ORTHOGONAL_INIT",
            "mup_proj_init": "MUP_PROJ_INIT",
            "export_quant_bits": "EXPORT_QUANT_BITS",
            "export_codec": "EXPORT_CODEC",
            "export_high_precision_bits": "EXPORT_HIGH_PRECISION_BITS",
            "export_high_precision_budget_bytes": "EXPORT_HIGH_PRECISION_BUDGET_BYTES",
            "export_high_precision_max_tensors": "EXPORT_HIGH_PRECISION_MAX_TENSORS",
            "export_high_precision_min_numel": "EXPORT_HIGH_PRECISION_MIN_NUMEL",
            "train_random_offset_tokens": "TRAIN_RANDOM_OFFSET_TOKENS",
            "use_cuda_graphs": "USE_CUDA_GRAPHS",
            "cuda_graph_warmup_steps": "CUDA_GRAPH_WARMUP_STEPS",
            "debug_static_shapes": "DEBUG_STATIC_SHAPES",
            "attention_kv_mode": "ATTENTION_KV_MODE",
            "use_mqa": "USE_MQA",
            "init_model_path": "INIT_MODEL_PATH",
            "init_model_strict": "INIT_MODEL_STRICT",
        }

    def _apply_schema_values(self, values: dict[str, object]) -> None:
        env_overrides = self._env_override_map()
        for attr, value in values.items():
            env_name = env_overrides.get(attr)
            if env_name is not None and env_name in os.environ:
                continue
            setattr(self, attr, value)

    def _apply_policy_schema(self) -> None:
        recipe_families: dict[str, dict[str, object]] = {
            "local_proxy_frontier": {
                **LOCAL_PROXY_RECIPE_COMMON,
                "ema_decay": 0.0,
                "ema_start_step": 0,
            },
            "local_proxy_frontier_promotion": dict(LOCAL_PROXY_RECIPE_COMMON),
        }
        curriculum_policies: dict[str, dict[str, object]] = {
            "winner_int6": {"warmdown_iters": 3000, "grad_clip_norm": 0.3},
        }
        data_policies: dict[str, dict[str, object]] = {
            "sequential": {},
        }
        precision_policies: dict[str, dict[str, object]] = {
            "export_plus": {
                "export_quant_bits": 5,
                "fake_quant_tail_steps": 1024,
                "fake_quant_bits": 8,
                "int8_auto_keep_budget_bytes": 32768,
                "int8_auto_keep_max_tensors": 2,
                "int8_auto_keep_row_budget_bytes": 131072,
                "int8_auto_keep_row_group_size": 32,
            },
        }
        eval_policies: dict[str, dict[str, object]] = {
            "standard": {"eval_stride": 0},
            "sliding64_2048": {"eval_stride": 64, "eval_batch_seqs": 256},
        }
        optimizer_policies: dict[str, dict[str, object]] = {
            "local_proxy_faststart": {
                "muon_momentum": 0.96,
                "muon_momentum_warmup_start": 0.80,
                "muon_momentum_warmup_steps": 250,
                "muon_weight_decay": 0.03,
            },
        }
        runtime_policies: dict[str, dict[str, object]] = {
            "compiled": {"enable_torch_compile": True},
            "eager": {"enable_torch_compile": False},
        }

        for value, table, label in (
            (self.recipe_family, recipe_families, "RECIPE_FAMILY"),
            (self.curriculum_policy, curriculum_policies, "CURRICULUM_POLICY"),
            (self.data_policy, data_policies, "DATA_POLICY"),
            (self.optimizer_policy, optimizer_policies, "OPTIMIZER_POLICY"),
            (self.precision_policy, precision_policies, "PRECISION_POLICY"),
            (self.eval_policy, eval_policies, "EVAL_POLICY"),
            (self.runtime_policy, runtime_policies, "RUNTIME_POLICY"),
        ):
            if not value:
                continue
            if value not in table:
                valid = ", ".join(sorted(table))
                raise ValueError(f"Unknown {label}={value!r}; expected one of: {valid}")
            self._apply_schema_values(table[value])

    def _apply_training_preset(self) -> None:
        if not self.training_preset:
            return
        preset_aliases: dict[str, dict[str, str]] = {
            "local_proxy_frontier": {
                **LOCAL_PROXY_PRESET_COMMON,
                "recipe_family": "local_proxy_frontier",
                "fake_quant_tail_steps": 640,
                "fake_quant_bits": 6,
                "fake_quant_full_run": False,
            },
            "local_proxy_promotion": {
                **LOCAL_PROXY_PRESET_COMMON,
                "recipe_family": "local_proxy_frontier_promotion",
                "data_policy": "sequential",
                "precision_policy": "export_plus",
            },
        }
        if self.training_preset not in preset_aliases:
            valid = ", ".join(sorted(preset_aliases))
            raise ValueError(f"Unknown TRAINING_PRESET={self.training_preset!r}; expected one of: {valid}")
        # Explicit schema env vars win over legacy preset aliases.
        for attr, value in preset_aliases[self.training_preset].items():
            env_name = attr.upper()
            if env_name not in os.environ and not getattr(self, attr):
                setattr(self, attr, value)
        self._apply_policy_schema()

    def _resolve_attention_config(self) -> None:
        kv_mode = str(self.attention_kv_mode or "").strip().lower()
        if self.use_mqa:
            kv_mode = "mqa"
        if not kv_mode:
            if int(self.num_kv_heads) <= 1:
                kv_mode = "mqa"
            elif int(self.num_kv_heads) >= int(self.num_heads):
                kv_mode = "mha"
            else:
                kv_mode = "gqa"
        if kv_mode == "mqa":
            self.num_kv_heads = 1
        elif kv_mode == "mha":
            self.num_kv_heads = int(self.num_heads)
        elif kv_mode == "gqa":
            self.num_kv_heads = max(min(int(self.num_kv_heads), int(self.num_heads)), 1)
        else:
            raise ValueError(f"ATTENTION_KV_MODE={kv_mode!r} must be one of: mha, gqa, mqa")
        if int(self.num_heads) % int(self.num_kv_heads) != 0:
            raise ValueError(
                f"Resolved KV mode {kv_mode!r} is incompatible with NUM_HEADS={self.num_heads} "
                f"and NUM_KV_HEADS={self.num_kv_heads}"
            )
        self.attention_kv_mode = kv_mode


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
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
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        backend_steps_light: int = 0,
        backend_refresh_interval: int = 1,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                backend_steps_light=backend_steps_light,
                backend_refresh_interval=backend_refresh_interval,
                weight_decay=weight_decay,
                nesterov=nesterov,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            backend_steps_light = group["backend_steps_light"]
            backend_refresh_interval = max(int(group["backend_refresh_interval"]), 1)
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            step_index = int(group.get("step_index", 0)) + 1
            group["step_index"] = step_index
            use_full_backend = backend_refresh_interval <= 1 or step_index % backend_refresh_interval == 0
            current_backend_steps = backend_steps if use_full_backend else backend_steps_light
            for p in params:
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=current_backend_steps)
                # DDP already synchronizes gradients, so applying Muon locally on
                # every rank avoids an extra all-reduce over flattened updates.
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                g = g.to(dtype=p.dtype)
                state["last_applied_update"] = g.detach().clone()
                p.add_(g, alpha=-lr)

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
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


def build_tokenizers_json_luts(tokenizer_path: str, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    tok_vocab_size = int(tok.get_vocab_size())
    table_size = max(tok_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    special_tokens = {"[PAD]", "[BOS]", "[EOS]", "[UNK]"}
    for token_id in range(tok_vocab_size):
        piece = tok.id_to_token(token_id)
        if piece is None or piece in special_tokens:
            continue
        decoded = tok.decode([token_id], skip_special_tokens=False)
        if not decoded:
            continue
        base_bytes_np[token_id] = len(decoded.encode("utf-8"))
        is_boundary_token_np[token_id] = False
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def get_eval_seq_len(args: Hyperparameters) -> int:
    return args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len


def trace_shape(x: Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in x.shape)


def ensure_contiguous_lastdim(x: Tensor) -> Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def enforce_static_shape(x: Tensor, spec: tuple[int, ...], name: str = "tensor") -> Tensor:
    shape = trace_shape(x)
    if len(shape) != len(spec):
        raise ValueError(f"{name} rank mismatch: actual={shape} expected={spec}")
    for actual_dim, expected_dim in zip(shape, spec, strict=True):
        if expected_dim > 0 and actual_dim != expected_dim:
            raise ValueError(f"{name} shape mismatch: actual={shape} expected={spec}")
    return x


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def count_total_train_tokens(pattern: str) -> int:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return sum(_read_shard_num_tokens(file) for file in files)


def pad_token_pair_batch(
    x_seqs: list[Tensor], y_seqs: list[Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    if not x_seqs or len(x_seqs) != len(y_seqs):
        raise ValueError("Expected non-empty matching x/y sequence lists")
    lengths = [int(x.numel()) for x in x_seqs]
    max_len = max(lengths)
    batch_size = len(x_seqs)
    x_batch = torch.zeros((batch_size, max_len), dtype=torch.int64, device=device)
    y_batch = torch.zeros((batch_size, max_len), dtype=torch.int64, device=device)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    for i, (x_seq, y_seq) in enumerate(zip(x_seqs, y_seqs, strict=True)):
        seq_len = int(x_seq.numel())
        x_batch[i, :seq_len] = x_seq.to(device=device, dtype=torch.int64, non_blocking=True)
        y_batch[i, :seq_len] = y_seq.to(device=device, dtype=torch.int64, non_blocking=True)
        loss_mask[i, :seq_len] = True
    return x_batch, y_batch, loss_mask, lengths


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len_override: int = 0,
    max_seqs: int = 0,
) -> tuple[float, float, float, float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, EVAL_SEQ_LEN={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    if max_seqs > 0:
        total_seqs = min(total_seqs, max_seqs)
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    # Rotary caches persist across eval and training; inference_mode would cache
    # inference tensors here and then autograd would reject them on the next step.
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            if seq_len == args.train_seq_len:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y).detach()
                val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            else:
                logits_model = model.module if hasattr(model, "module") else model
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = logits_model.forward_logits(x)
                batch_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                    reduction="sum",
                )
                val_loss_sum += batch_loss.to(torch.float64)
            batch_token_count = float(y.numel())
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    bytes_per_token = val_byte_count.item() / val_token_count.item()
    model.train()
    return (
        float(val_loss.item()),
        float(bits_per_token * tokens_per_byte),
        float(bits_per_token),
        float(tokens_per_byte),
        float(bytes_per_token),
    )


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int,
    seq_len_override: int = 0,
    max_seqs: int = 0,
) -> tuple[float, float, float, float, float]:
    if batch_seqs <= 0:
        raise ValueError(f"EVAL_BATCH_SEQS must be positive for sliding eval, got {batch_seqs}")
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    if max_seqs > 0:
        total_tokens = min(total_tokens, max_seqs * seq_len)
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride]
    total_windows = len(window_starts)
    my_start = (total_windows * rank) // world_size
    my_end = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_start:my_end]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    # Keep cached RoPE tensors compatible with the subsequent training phase.
    with torch.no_grad():
        for batch_idx in range(0, len(my_windows), batch_seqs):
            batch_windows = my_windows[batch_idx : batch_idx + batch_seqs]
            x_seqs: list[Tensor] = []
            y_seqs: list[Tensor] = []
            window_lengths: list[int] = []

            for ws in batch_windows:
                end = min(ws + seq_len, total_tokens)
                window_len = end - ws
                window_lengths.append(window_len)
                chunk = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
                x_seqs.append(chunk[:-1])
                y_seqs.append(chunk[1:])

            if args.use_packed_eval_windows:
                x_batch, y_batch, _, _ = pad_token_pair_batch(x_seqs, y_seqs, device)
            else:
                batch_size = len(batch_windows)
                x_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
                y_batch = torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device)
                for i, (x_seq, y_seq) in enumerate(zip(x_seqs, y_seqs, strict=True)):
                    window_len = int(x_seq.numel())
                    x_batch[i, :window_len] = x_seq
                    y_batch[i, :window_len] = y_seq

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(x_batch)

            batch_size, batch_width = y_batch.shape
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(batch_size, batch_width)

            for i, ws in enumerate(batch_windows):
                window_len = window_lengths[i]
                score_start = 0 if ws == 0 else window_len - stride
                scored_nll = nll[i, score_start:window_len].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(window_len - score_start)
                tgt_ids = y_batch[i, score_start:window_len]
                prev_ids = x_batch[i, score_start:window_len]
                token_bytes = base_bytes_lut[tgt_ids].to(torch.float64)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                byte_count += token_bytes.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    bytes_per_token = byte_count.item() / token_count.item()
    base_model.train()
    return (
        float(val_loss.item()),
        float(bits_per_token * tokens_per_byte),
        float(bits_per_token),
        float(tokens_per_byte),
        float(bytes_per_token),
    )


def uses_sliding_eval(args: Hyperparameters) -> bool:
    return 0 < args.eval_stride < get_eval_seq_len(args)


def run_validation(
    args: Hyperparameters,
    model: nn.Module,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len_override: int = 0,
    max_seqs: int = 0,
) -> tuple[float, float, str, float, float, float]:
    eval_seq_len = seq_len_override if seq_len_override > 0 else get_eval_seq_len(args)
    if uses_sliding_eval(args):
        val_loss, val_bpb, bits_per_token, tokens_per_byte, bytes_per_token = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            seq_len_override=eval_seq_len,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
            max_seqs=max_seqs,
        )
        return val_loss, val_bpb, "sliding", bits_per_token, tokens_per_byte, bytes_per_token

    val_loss, val_bpb, bits_per_token, tokens_per_byte, bytes_per_token = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        seq_len_override=eval_seq_len,
        max_seqs=max_seqs,
    )
    return val_loss, val_bpb, "standard", bits_per_token, tokens_per_byte, bytes_per_token


def build_validation_diagnostic_batch(
    args: Hyperparameters,
    val_tokens: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    seq_len = get_eval_seq_len(args)
    total_tokens = max(int(val_tokens.numel()) - 1, 0)
    if total_tokens <= 0:
        empty = torch.zeros((1, 1), dtype=torch.int64, device=device)
        return empty, empty, torch.zeros_like(empty, dtype=torch.bool)
    if uses_sliding_eval(args):
        max_windows = max(min(int(args.eval_batch_seqs), 4), 1)
        window_starts = [ws for ws in range(0, total_tokens, args.eval_stride) if min(ws + seq_len, total_tokens) - ws >= args.eval_stride]
        if not window_starts:
            window_starts = [0]
        x_seqs: list[Tensor] = []
        y_seqs: list[Tensor] = []
        score_starts: list[int] = []
        for ws in window_starts[:max_windows]:
            end = min(ws + seq_len, total_tokens)
            chunk = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            if chunk.numel() >= 2:
                x_seqs.append(chunk[:-1])
                y_seqs.append(chunk[1:])
                window_len = int(chunk.numel() - 1)
                score_starts.append(0 if ws == 0 else max(window_len - args.eval_stride, 0))
        if x_seqs:
            x_batch, y_batch, loss_mask, _ = pad_token_pair_batch(x_seqs, y_seqs, device)
            for i, score_start in enumerate(score_starts):
                loss_mask[i, :score_start] = False
            return x_batch, y_batch, loss_mask
    max_seqs = 4
    x_seqs = []
    y_seqs = []
    for start in range(0, min(total_tokens, max_seqs * seq_len), seq_len):
        end = min(start + seq_len, total_tokens)
        chunk = val_tokens[start : end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
        if chunk.numel() >= 2:
            x_seqs.append(chunk[:-1])
            y_seqs.append(chunk[1:])
    if not x_seqs:
        chunk = val_tokens[: min(total_tokens, seq_len) + 1].to(device=device, dtype=torch.int64, non_blocking=True)
        x_seqs = [chunk[:-1]]
        y_seqs = [chunk[1:]]
    x_batch, y_batch, loss_mask, _ = pad_token_pair_batch(x_seqs, y_seqs, device)
    return x_batch, y_batch, loss_mask


@torch.no_grad()
def specialist_path_metrics(model: nn.Module) -> dict[str, float]:
    helper_path_schedule_mult = 1.0
    if getattr(model, "helper_path_schedule_mult_buffer", None) is not None:
        helper_path_schedule_mult = float(model.helper_path_schedule_mult_buffer.detach().float().item())
    lexical_schedule_mult = 1.0
    if getattr(model, "lexical_schedule_mult_buffer", None) is not None:
        lexical_schedule_mult = float(model.lexical_schedule_mult_buffer.detach().float().item())
    lexical_calibration_mult = lexical_schedule_mult
    if getattr(model, "lexical_calibration_mult_buffer", None) is not None:
        lexical_calibration_mult = float(model.lexical_calibration_mult_buffer.detach().float().item())
    lexical_hidden_mult = lexical_schedule_mult
    if getattr(model, "lexical_hidden_mult_buffer", None) is not None:
        lexical_hidden_mult = float(model.lexical_hidden_mult_buffer.detach().float().item())
    lexical_shortcut_mult = lexical_schedule_mult
    if getattr(model, "lexical_shortcut_mult_buffer", None) is not None:
        lexical_shortcut_mult = float(model.lexical_shortcut_mult_buffer.detach().float().item())
    lexical_shortcut_health_gate = 1.0
    if getattr(model, "lexical_shortcut_health_gate_buffer", None) is not None:
        lexical_shortcut_health_gate = float(model.lexical_shortcut_health_gate_buffer.detach().float().item())
    lexical_effective_mult = helper_path_schedule_mult * lexical_shortcut_mult
    bigram_hidden_schedule_mult = lexical_hidden_mult
    bigram_hidden_effective_mult = helper_path_schedule_mult * bigram_hidden_schedule_mult
    smear_gate = 0.0
    if getattr(model, "smear", None) is not None:
        smear_gate = float(torch.sigmoid(model.smear.gate.detach().float()).item())
    smear_gate_logit = float(getattr(model, "smear", None).gate.detach().float().item()) if getattr(model, "smear", None) is not None else 0.0
    lexical_batch_mult = float(getattr(model, "last_lexical_batch_mult", 1.0))
    lexical_batch_unique_frac = float(getattr(model, "last_lexical_batch_unique_frac", 1.0))
    lexical_batch_max_token_frac = float(getattr(model, "last_lexical_batch_max_token_frac", 0.0))
    lexical_shortcut_scale = 0.0
    if getattr(model, "lexical_shortcut_scale", None) is not None:
        lexical_shortcut_scale = float(F.softplus(model.lexical_shortcut_scale.detach().float()).item()) * lexical_effective_mult
    lexical_shortcut_scale_raw = 0.0
    if getattr(model, "lexical_shortcut_scale", None) is not None:
        lexical_shortcut_scale_raw = float(model.lexical_shortcut_scale.detach().float().item())
    lexical_confidence_scale = 0.0
    if getattr(model, "lexical_confidence_scale", None) is not None:
        lexical_confidence_scale = float(F.softplus(model.lexical_confidence_scale.detach().float()).item())
    lexical_confidence_scale_raw = 0.0
    if getattr(model, "lexical_confidence_scale", None) is not None:
        lexical_confidence_scale_raw = float(model.lexical_confidence_scale.detach().float().item())
    smear_lexical_scale = 0.0
    if getattr(model, "smear", None) is not None and getattr(model.smear, "lexical_scale", None) is not None:
        smear_lexical_scale = float(F.softplus(model.smear.lexical_scale.detach().float()).item()) * (
            helper_path_schedule_mult * lexical_hidden_mult
        )
    smear_lexical_scale_raw = 0.0
    if getattr(model, "smear", None) is not None and getattr(model.smear, "lexical_scale", None) is not None:
        smear_lexical_scale_raw = float(model.smear.lexical_scale.detach().float().item())
    bigram_scale = 0.0
    bigram_gate = 0.0
    bigram_cap = 0.0
    if getattr(model, "bigram", None) is not None:
        bigram_scale = float(F.softplus(model.bigram.scale.detach().float()).item()) * lexical_hidden_mult
        bigram_gate = float(torch.sigmoid(model.bigram.gate_logit.detach().float()).item())
        bigram_cap = float(model.bigram.influence_cap.detach().float().item())
    bigram_scale_raw = 0.0
    if getattr(model, "bigram", None) is not None:
        bigram_scale_raw = float(model.bigram.scale.detach().float().item())
    bigram_hidden_scale = 0.0
    if getattr(model, "bigram_hidden_scale", None) is not None:
        bigram_hidden_scale = (
            float(F.softplus(model.bigram_hidden_scale.detach().float()).item()) * bigram_hidden_effective_mult
        )
    bigram_hidden_scale_raw = 0.0
    if getattr(model, "bigram_hidden_scale", None) is not None:
        bigram_hidden_scale_raw = float(model.bigram_hidden_scale.detach().float().item())
    pairwise_logit_scale = 0.0
    if getattr(model, "pairwise_logit_scale", None) is not None:
        pairwise_logit_scale = float(F.softplus(model.pairwise_logit_scale.detach().float()).item())
    pairwise_logit_scale_raw = 0.0
    if getattr(model, "pairwise_logit_scale", None) is not None:
        pairwise_logit_scale_raw = float(model.pairwise_logit_scale.detach().float().item())
    pairwise_logit_gate = 0.0
    if getattr(model, "pairwise_logit_gate", None) is not None:
        pairwise_logit_gate = float(torch.sigmoid(model.pairwise_logit_gate.detach().float()).item())
    pairwise_logit_confidence_gain = 0.0
    if getattr(model, "pairwise_logit_confidence_gain", None) is not None:
        pairwise_logit_confidence_gain = float(model.pairwise_logit_confidence_gain.detach().float().item())
    pairwise_logit_gate_mean = float(getattr(model, "last_pairwise_logit_gate_mean", 0.0))
    pairwise_logit_rms = float(getattr(model, "last_pairwise_logit_rms", 0.0))
    shared_tail_gate = 0.0
    if getattr(model, "shared_tail_gate", None) is not None:
        shared_tail_gate = float(torch.sigmoid(model.shared_tail_gate.detach().float()).item())
    shared_tail_schedule_mult = 1.0
    if getattr(model, "shared_tail_schedule_mult_buffer", None) is not None:
        shared_tail_schedule_mult = float(model.shared_tail_schedule_mult_buffer.detach().float().item())
    shared_loop_gate_mean = 0.0
    shared_loop_gate_min = 0.0
    shared_loop_gate_max = 0.0
    shared_loop_depth = 0.0
    if getattr(model, "shared_loop_gates", None) is not None:
        shared_loop_gates = torch.sigmoid(model.shared_loop_gates.detach().float())
        shared_loop_gate_mean = float(shared_loop_gates.mean().item())
        shared_loop_gate_min = float(shared_loop_gates.amin().item())
        shared_loop_gate_max = float(shared_loop_gates.amax().item())
        shared_loop_depth = float(shared_loop_gates.numel())
    lexical_gate = 0.0
    lexical_cap = 0.0
    if getattr(model, "lexical_prior_gate_logit", None) is not None:
        lexical_gate = float(torch.sigmoid(model.lexical_prior_gate_logit.detach().float()).item())
        lexical_cap = float(model.lexical_cap.detach().float().item())
    return {
        "smear_gate": smear_gate,
        "smear_gate_logit": smear_gate_logit,
        "lexical_batch_mult": lexical_batch_mult,
        "lexical_batch_unique_frac": lexical_batch_unique_frac,
        "lexical_batch_max_token_frac": lexical_batch_max_token_frac,
        "lexical_shortcut_scale": lexical_shortcut_scale,
        "lexical_shortcut_scale_raw": lexical_shortcut_scale_raw,
        "lexical_confidence_scale": lexical_confidence_scale,
        "lexical_confidence_scale_raw": lexical_confidence_scale_raw,
        "smear_lexical_scale": smear_lexical_scale,
        "smear_lexical_scale_raw": smear_lexical_scale_raw,
        "bigram_scale": bigram_scale,
        "bigram_gate": bigram_gate,
        "bigram_cap": bigram_cap,
        "bigram_scale_raw": bigram_scale_raw,
        "bigram_hidden_scale": bigram_hidden_scale,
        "bigram_hidden_scale_raw": bigram_hidden_scale_raw,
        "pairwise_logit_scale": pairwise_logit_scale,
        "pairwise_logit_scale_raw": pairwise_logit_scale_raw,
        "pairwise_logit_gate": pairwise_logit_gate,
        "pairwise_logit_confidence_gain": pairwise_logit_confidence_gain,
        "pairwise_logit_gate_mean": pairwise_logit_gate_mean,
        "pairwise_logit_rms": pairwise_logit_rms,
        "helper_path_schedule_mult": helper_path_schedule_mult,
        "lexical_schedule_mult": lexical_schedule_mult,
        "lexical_calibration_mult": lexical_calibration_mult,
        "lexical_hidden_mult": lexical_hidden_mult,
        "lexical_shortcut_mult": lexical_shortcut_mult,
        "lexical_shortcut_health_gate": lexical_shortcut_health_gate,
        "lexical_gate": lexical_gate,
        "lexical_cap": lexical_cap,
        "shared_tail_gate": shared_tail_gate,
        "shared_tail_schedule_mult": shared_tail_schedule_mult,
        "shared_loop_gate_mean": shared_loop_gate_mean,
        "shared_loop_gate_min": shared_loop_gate_min,
        "shared_loop_gate_max": shared_loop_gate_max,
        "shared_loop_depth": shared_loop_depth,
    }


def objective_module_group(name: str) -> str:
    if name == "tok_emb.weight":
        return "embed_head_shared"
    if any(token in name for token in ("bigram", "lexical_", "smear", "lexical_residual_ngram")):
        return "lexical"
    if "lm_head" in name:
        return "head"
    if ".attn." in name:
        return "attn"
    if ".mlp." in name or "mlp_" in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def warmup_prior_init_param_names(model: nn.Module, groups: tuple[str, ...]) -> list[str]:
    return warmup_prior_init_param_names_for_mode(
        model,
        groups,
        getattr(Hyperparameters, "warmup_prior_init_mode", "safe"),
    )


def warmup_prior_init_param_names_for_mode(
    model: nn.Module,
    groups: tuple[str, ...],
    mode: str,
) -> list[str]:
    allowed_groups = {item.strip() for item in groups if item.strip()}
    if not allowed_groups:
        return []
    mode = str(mode or "safe").strip().lower()
    # Safe mode only restores calibration-style controls. Avoid direct influence
    # amplitudes here so warmup teaches branch centering and uncertainty handling
    # without preloading behavior into the true training trajectory.
    safe_name_patterns = (
        "smear.gate",
        "lexical_confidence_bias",
        "lexical_confidence_scale",
        "lexical_residual_confidence_bias",
        "lexical_residual_confidence_scale",
        "smear.confidence_gain",
        "cond_gate",
    )
    selected: list[str] = []
    for name, param in model.named_parameters():
        if objective_module_group(name) not in allowed_groups:
            continue
        is_control = param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        if param.dtype != torch.float32:
            continue
        if mode == "safe" and not any(pattern in name for pattern in safe_name_patterns):
            continue
        if mode in {"bounded_delta", "delta"} and not (is_control or param.ndim >= 2):
            continue
        selected.append(name)
    return selected


def blend_state_tensor(initial: Tensor, warmed: Tensor, blend: float) -> Tensor:
    blend = min(max(float(blend), 0.0), 1.0)
    if blend <= 0.0:
        return initial.clone()
    warmed = warmed.to(device=initial.device, dtype=initial.dtype)
    if blend >= 1.0:
        return warmed.clone()
    mixed = initial.float().lerp(warmed.float(), blend)
    return mixed.to(dtype=initial.dtype)


def transfer_warmup_state_tensor(
    initial: Tensor,
    warmed: Tensor,
    blend: float,
    mode: str,
    delta_rms_mult: float,
    delta_rms_floor: float,
) -> Tensor:
    mode = str(mode or "safe").strip().lower()
    if mode not in {"bounded_delta", "delta"}:
        return blend_state_tensor(initial, warmed, blend)
    blend = min(max(float(blend), 0.0), 1.0)
    if blend <= 0.0:
        return initial.clone()
    initial_f = initial.float()
    warmed_f = warmed.to(device=initial.device, dtype=torch.float32)
    delta = (warmed_f - initial_f) * blend
    delta_rms = delta.pow(2).mean().sqrt()
    initial_rms = initial_f.pow(2).mean().sqrt()
    max_delta_rms = max(float(initial_rms.item()) * max(float(delta_rms_mult), 0.0), float(delta_rms_floor))
    if float(delta_rms.item()) > max_delta_rms > 0.0:
        delta.mul_(max_delta_rms / max(float(delta_rms.item()), 1e-12))
    return (initial_f + delta).to(dtype=initial.dtype)


def init_module_objective_tracking(model: nn.Module) -> tuple[dict[str, Tensor], dict[str, dict[str, float]]]:
    snapshots = {name: param.detach().cpu().float().clone() for name, param in model.named_parameters()}
    accumulators: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        group = objective_module_group(name)
        stats = accumulators.setdefault(
            group,
            {
                "gain": 0.0,
                "grad_sq": 0.0,
                "update_sq": 0.0,
                "param_count": 0.0,
                "tracked_steps": 0.0,
            },
        )
        stats["param_count"] += float(param.numel())
    return snapshots, accumulators


def capture_module_objective_grads(model: nn.Module) -> dict[str, Tensor]:
    grad_snapshots: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_snapshots[name] = param.grad.detach().cpu().float().clone()
    return grad_snapshots


def capture_module_objective_optimizer_updates(
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
) -> dict[str, Tensor]:
    param_name_by_id = {id(param): name for name, param in model.named_parameters()}
    update_snapshots: dict[str, Tensor] = {}
    for opt in optimizers:
        if not isinstance(opt, Muon):
            continue
        for group in opt.param_groups:
            for param in group["params"]:
                state = opt.state.get(param, {})
                applied_update = state.get("last_applied_update")
                if not isinstance(applied_update, Tensor):
                    continue
                name = param_name_by_id.get(id(param))
                if name is None:
                    continue
                update_snapshots[name] = applied_update.detach().cpu().float().clone()
    return update_snapshots


def accumulate_module_objective_tracking(
    model: nn.Module,
    snapshots: dict[str, Tensor],
    accumulators: dict[str, dict[str, float]],
    grad_snapshots: dict[str, Tensor] | None = None,
    update_snapshots: dict[str, Tensor] | None = None,
) -> None:
    updated_groups: set[str] = set()
    for name, param in model.named_parameters():
        grad_cpu = update_snapshots.get(name) if update_snapshots is not None else None
        if grad_cpu is None:
            grad_cpu = grad_snapshots.get(name) if grad_snapshots is not None else None
        if grad_cpu is None:
            grad = param.grad
            if grad is None:
                continue
            grad_cpu = grad.detach().cpu().float()
        if grad_cpu is None:
            continue
        group = objective_module_group(name)
        stats = accumulators.setdefault(
            group,
            {"gain": 0.0, "grad_sq": 0.0, "update_sq": 0.0, "param_count": 0.0, "tracked_steps": 0.0},
        )
        prev_cpu = snapshots.get(name)
        curr_cpu = param.detach().cpu().float()
        if prev_cpu is None or prev_cpu.shape != curr_cpu.shape:
            snapshots[name] = curr_cpu.clone()
            continue
        delta_cpu = curr_cpu - prev_cpu
        stats["gain"] += float((-(grad_cpu * delta_cpu)).sum().item())
        stats["grad_sq"] += float(grad_cpu.square().sum().item())
        stats["update_sq"] += float(delta_cpu.square().sum().item())
        snapshots[name] = curr_cpu.clone()
        updated_groups.add(group)
    for group in updated_groups:
        accumulators[group]["tracked_steps"] += 1.0


@torch.no_grad()
def summarize_module_objective_tracking(
    model: nn.Module,
    accumulators: dict[str, dict[str, float]],
    export_quant_bits: int,
    force_fp16_tied_embed_export: bool,
) -> dict[str, dict[str, float]]:
    param_norm_sq_by_group: dict[str, float] = {}
    export_bytes_by_group: dict[str, float] = {}
    for name, param in model.named_parameters():
        group = objective_module_group(name)
        param_norm_sq_by_group[group] = param_norm_sq_by_group.get(group, 0.0) + float(
            param.detach().float().square().sum().item()
        )
        target_class = _classify_quant_target(name)
        bytes_per_elem = max(int(export_quant_bits), 1) / 8.0
        if target_class == "embed" and force_fp16_tied_embed_export:
            bytes_per_elem = 2.0
        export_bytes_by_group[group] = export_bytes_by_group.get(group, 0.0) + float(param.numel()) * bytes_per_elem
    summary: dict[str, dict[str, float]] = {}
    for group, stats in accumulators.items():
        gain = float(stats.get("gain", 0.0))
        grad_sq = float(stats.get("grad_sq", 0.0))
        update_sq = float(stats.get("update_sq", 0.0))
        param_count = max(float(stats.get("param_count", 0.0)), 1.0)
        export_bytes = max(float(export_bytes_by_group.get(group, 0.0)), 1e-9)
        param_norm = math.sqrt(max(float(param_norm_sq_by_group.get(group, 0.0)), 0.0))
        update_norm = math.sqrt(max(update_sq, 0.0))
        summary[group] = {
            "gain": gain,
            "gain_per_param": gain / param_count,
            "gain_per_byte": gain / export_bytes,
            "gain_per_update_norm": gain / max(update_norm, 1e-12),
            "grad_rms": math.sqrt(max(grad_sq, 0.0) / param_count),
            "update_rms": math.sqrt(max(update_sq, 0.0) / param_count),
            "param_rms": param_norm / math.sqrt(param_count),
            "param_count": param_count,
            "export_bytes": export_bytes,
            "tracked_steps": float(stats.get("tracked_steps", 0.0)),
        }
    return summary


def parse_counterfactual_groups(raw: str) -> list[str]:
    groups: list[str] = []
    seen: set[str] = set()
    for item in (piece.strip().lower() for piece in raw.split(",")):
        if not item or item in seen:
            continue
        if item != "lexical":
            continue
        groups.append(item)
        seen.add(item)
    return groups


def _backup_and_fill_(backups: list[tuple[Tensor, Tensor]], target: Tensor, value: float) -> None:
    backups.append((target, target.detach().clone()))
    target.fill_(float(value))


def prepare_counterfactual_group_ablation(model: nn.Module, group: str) -> callable:
    backups: list[tuple[Tensor, Tensor]] = []

    with torch.no_grad():
        if group == "lexical":
            if getattr(model, "lexical_shortcut_scale", None) is not None:
                _backup_and_fill_(backups, model.lexical_shortcut_scale, COUNTERFACTUAL_RAW_OFF)
            if getattr(model, "lexical_residual_scale", None) is not None:
                _backup_and_fill_(backups, model.lexical_residual_scale, COUNTERFACTUAL_RAW_OFF)
            if getattr(model, "bigram_hidden_scale", None) is not None:
                _backup_and_fill_(backups, model.bigram_hidden_scale, COUNTERFACTUAL_RAW_OFF)
            bigram = getattr(model, "bigram", None)
            if bigram is not None:
                _backup_and_fill_(backups, bigram.scale, COUNTERFACTUAL_RAW_OFF)
                _backup_and_fill_(backups, bigram.gate_logit, COUNTERFACTUAL_RAW_OFF)
            lexical_residual_ngram = getattr(model, "lexical_residual_ngram", None)
            if lexical_residual_ngram is not None:
                _backup_and_fill_(backups, lexical_residual_ngram.scale, COUNTERFACTUAL_RAW_OFF)
                _backup_and_fill_(backups, lexical_residual_ngram.gate_logit, COUNTERFACTUAL_RAW_OFF)
            smear = getattr(model, "smear", None)
            if smear is not None:
                _backup_and_fill_(backups, smear.gate, COUNTERFACTUAL_RAW_OFF)
                _backup_and_fill_(backups, smear.lexical_scale, COUNTERFACTUAL_RAW_OFF)
                _backup_and_fill_(backups, smear.confidence_gain, 0.0)

    def restore() -> None:
        with torch.no_grad():
            for target, original in reversed(backups):
                target.copy_(original)

    return restore


def run_counterfactual_group_eval(
    *,
    args: Hyperparameters,
    model: nn.Module,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len_override: int,
    max_seqs: int,
    groups: list[str],
    module_objective_summary: dict[str, dict[str, float]],
) -> tuple[tuple[float, float, str], list[dict[str, float | str]]]:
    base_metrics = run_validation(
        args,
        model,
        base_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        seq_len_override=seq_len_override,
        max_seqs=max_seqs,
    )
    base_loss, base_bpb, base_mode, _base_bits_per_token, _base_tokens_per_byte, _base_bytes_per_token = base_metrics
    results: list[dict[str, float | str]] = []
    for group in groups:
        restore = prepare_counterfactual_group_ablation(base_model, group)
        try:
            (
                ablated_loss,
                ablated_bpb,
                _ablated_mode,
                _ablated_bits_per_token,
                _ablated_tokens_per_byte,
                _ablated_bytes_per_token,
            ) = run_validation(
                args,
                model,
                base_model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                seq_len_override=seq_len_override,
                max_seqs=max_seqs,
            )
        finally:
            restore()
        export_bytes = float(module_objective_summary.get(group, {}).get("export_bytes", 0.0))
        delta_val_bpb = float(ablated_bpb - base_bpb)
        delta_val_loss = float(ablated_loss - base_loss)
        value_per_byte = delta_val_bpb / max(export_bytes, 1e-12)
        results.append(
            {
                "group": group,
                "base_val_loss": float(base_loss),
                "base_val_bpb": float(base_bpb),
                "ablated_val_loss": float(ablated_loss),
                "ablated_val_bpb": float(ablated_bpb),
                "delta_val_loss": delta_val_loss,
                "delta_val_bpb": delta_val_bpb,
                "export_bytes": export_bytes,
                "delta_val_bpb_per_byte": value_per_byte,
            }
        )
    return base_metrics, results

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get("INT8_KEEP_FLOAT_NAME_PATTERNS", "").split(",") if pattern
)
MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS",
        "tok_emb.weight,blocks.7.attn.c_k.weight,blocks.8.attn.c_k.weight",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
EXPORT_CALIBRATION_METHOD = os.environ.get("EXPORT_CALIBRATION_METHOD", "percentile_mse").strip().lower()
EXPORT_CALIBRATION_PERCENTILES_RAW = os.environ.get(
    "EXPORT_CALIBRATION_PERCENTILES",
    f"{INT8_CLIP_Q:.8f},0.999,0.9995,0.9999,0.99999,1.0",
)
MIXED_PRECISION_ROW_CLIP_SEARCH = bool(int(os.environ.get("MIXED_PRECISION_ROW_CLIP_SEARCH", "1")))
MIXED_PRECISION_ROW_CLIP_CANDIDATES = (0.999, 0.9995, 0.9999, 0.99999, 1.0)
FORCE_FP16_TIED_EMBED_EXPORT = bool(int(os.environ.get("FORCE_FP16_TIED_EMBED_EXPORT", "1")))
LATE_K_FP16_LAYERS = int(os.environ.get("LATE_K_FP16_LAYERS", "2"))
LARGE_MATRIX_QUANT_BITS = int(os.environ.get("LARGE_MATRIX_QUANT_BITS", "6"))
MIXED_PRECISION_EXPORT_MLP_BITS = int(os.environ.get("EXPORT_MLP_BITS", "0"))
MIXED_PRECISION_EXPORT_ATTN_BITS = int(os.environ.get("EXPORT_ATTN_BITS", "0"))
EXPORT_TENSOR_CANDIDATE_BITS_RAW = os.environ.get("EXPORT_TENSOR_CANDIDATE_BITS", "4,5,6,8")
EXPORT_TENSOR_SWEEP_OBJECTIVE = os.environ.get("EXPORT_TENSOR_SWEEP_OBJECTIVE", "mse_per_compressed_byte").strip().lower()
EXPORT_TENSOR_SWEEP_MIN_NUMEL = int((os.environ.get("EXPORT_TENSOR_SWEEP_MIN_NUMEL", "65536") or "65536").strip())
EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE = float(
    (os.environ.get("EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE", "0.0005") or "0.0005").strip()
)
EXPORT_TENSOR_SWEEP_LOG_TOPK = int((os.environ.get("EXPORT_TENSOR_SWEEP_LOG_TOPK", "8") or "8").strip())
EXPORT_BLOCK_PRUNE_ENABLED = bool(int(os.environ.get("EXPORT_BLOCK_PRUNE", "0")))
EXPORT_BLOCK_PRUNE_DENSITIES_RAW = os.environ.get("EXPORT_BLOCK_PRUNE_DENSITIES", "0.98,0.95,0.90")
EXPORT_BLOCK_PRUNE_BLOCK_RAW = os.environ.get("EXPORT_BLOCK_PRUNE_BLOCK", "16,16")
EXPORT_BLOCK_PRUNE_MIN_NUMEL = int(os.environ.get("EXPORT_BLOCK_PRUNE_MIN_NUMEL", "65536"))
EXPORT_BLOCK_PRUNE_MAX_EXTRA_REL_MSE = float(os.environ.get("EXPORT_BLOCK_PRUNE_MAX_EXTRA_REL_MSE", "0.0005"))
EXPORT_BLOCK_PRUNE_LOG_TOPK = int(os.environ.get("EXPORT_BLOCK_PRUNE_LOG_TOPK", "8"))
EXPORT_CHANNEL_EQUALIZATION = bool(int(os.environ.get("EXPORT_CHANNEL_EQUALIZATION", "1")))
EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN = max(float(os.environ.get("EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN", "4.0")), 1.0)

ACTIVE_EXPORT_CONFIG: dict[str, object] = {
    "num_layers": Hyperparameters.num_layers,
    "force_fp16_tied_embed_export": FORCE_FP16_TIED_EMBED_EXPORT,
    "late_k_fp16_layers": LATE_K_FP16_LAYERS,
    "int8_auto_keep_budget_bytes": Hyperparameters.int8_auto_keep_budget_bytes,
    "int8_auto_keep_max_tensors": Hyperparameters.int8_auto_keep_max_tensors,
    "int8_auto_keep_min_numel": Hyperparameters.int8_auto_keep_min_numel,
    "int8_sensitivity_log_topk": Hyperparameters.int8_sensitivity_log_topk,
    "int8_auto_keep_row_budget_bytes": Hyperparameters.int8_auto_keep_row_budget_bytes,
    "int8_auto_keep_row_group_size": Hyperparameters.int8_auto_keep_row_group_size,
    "export_codec": Hyperparameters.export_codec,
    "export_zstd_level": Hyperparameters.export_zstd_level,
    "export_high_precision_bits": Hyperparameters.export_high_precision_bits,
    "export_high_precision_budget_bytes": Hyperparameters.export_high_precision_budget_bytes,
    "export_high_precision_max_tensors": Hyperparameters.export_high_precision_max_tensors,
    "export_high_precision_min_numel": Hyperparameters.export_high_precision_min_numel,
}


def configure_runtime_export(args: Hyperparameters) -> None:
    ACTIVE_EXPORT_CONFIG.update(
        {
            "num_layers": int(args.num_layers),
            "force_fp16_tied_embed_export": bool(args.force_fp16_tied_embed_export),
            "late_k_fp16_layers": int(LATE_K_FP16_LAYERS),
            "int8_auto_keep_budget_bytes": int(args.int8_auto_keep_budget_bytes),
            "int8_auto_keep_max_tensors": int(args.int8_auto_keep_max_tensors),
            "int8_auto_keep_min_numel": int(args.int8_auto_keep_min_numel),
            "int8_sensitivity_log_topk": int(args.int8_sensitivity_log_topk),
            "int8_auto_keep_row_budget_bytes": int(args.int8_auto_keep_row_budget_bytes),
            "int8_auto_keep_row_group_size": int(args.int8_auto_keep_row_group_size),
            "export_codec": str(args.export_codec),
            "export_zstd_level": int(args.export_zstd_level),
            "export_high_precision_bits": int(args.export_high_precision_bits),
            "export_high_precision_budget_bytes": int(args.export_high_precision_budget_bytes),
            "export_high_precision_max_tensors": int(args.export_high_precision_max_tensors),
            "export_high_precision_min_numel": int(args.export_high_precision_min_numel),
        }
    )

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def should_keep_float_tensor(name: str, t: Tensor) -> bool:
    if bool(ACTIVE_EXPORT_CONFIG["force_fp16_tied_embed_export"]) and name == "tok_emb.weight":
        return True
    if (
        int(ACTIVE_EXPORT_CONFIG["late_k_fp16_layers"]) > 0
        and name.endswith("c_k.weight")
        and name.startswith("blocks.")
        and t.ndim == 2
    ):
        try:
            block_idx = int(name.split(".")[1])
            num_layers = int(ACTIVE_EXPORT_CONFIG["num_layers"])
            late_k_fp16_layers = int(ACTIVE_EXPORT_CONFIG["late_k_fp16_layers"])
            if block_idx >= max(num_layers - late_k_fp16_layers, 0):
                return True
        except (IndexError, ValueError):
            pass
    return t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or any(pattern in name for pattern in INT8_KEEP_FLOAT_NAME_PATTERNS)

def parse_export_calibration_percentiles(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if value > 1.0:
            value /= 100.0
        values.append(min(max(value, 1e-6), 1.0))
    if not values:
        values = [INT8_CLIP_Q, 1.0]
    values.append(1.0)
    return tuple(sorted(set(values)))


EXPORT_CALIBRATION_PERCENTILES = parse_export_calibration_percentiles(EXPORT_CALIBRATION_PERCENTILES_RAW)


def parse_export_block_prune_densities(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            continue
        if 0.0 < value <= 1.0:
            values.append(value)
    if not values:
        values = [0.98, 0.95, 0.90]
    return tuple(sorted(set(values), reverse=True))


def parse_export_block_prune_block(raw: str) -> tuple[int, int]:
    pieces = [piece.strip().lower() for piece in raw.replace("x", ",").split(",") if piece.strip()]
    if len(pieces) != 2:
        return (16, 16)
    try:
        block_h = max(int(pieces[0]), 1)
        block_w = max(int(pieces[1]), 1)
    except ValueError:
        return (16, 16)
    return block_h, block_w


def parse_export_tensor_candidate_bits(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for piece in raw.replace("/", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            bits = int(piece)
        except ValueError:
            continue
        if 2 <= bits <= 8:
            values.append(bits)
    if not values:
        values = [4, 5, 6, 8]
    return tuple(sorted(set(values)))


def parse_step_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for piece in raw.replace("/", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            step = int(piece)
        except ValueError:
            continue
        if step > 0:
            values.append(step)
    return tuple(sorted(set(values)))


EXPORT_BLOCK_PRUNE_DENSITIES = parse_export_block_prune_densities(EXPORT_BLOCK_PRUNE_DENSITIES_RAW)
EXPORT_BLOCK_PRUNE_BLOCK = parse_export_block_prune_block(EXPORT_BLOCK_PRUNE_BLOCK_RAW)
EXPORT_TENSOR_CANDIDATE_BITS = parse_export_tensor_candidate_bits(EXPORT_TENSOR_CANDIDATE_BITS_RAW)


def block_magnitude_prune_2d(t: Tensor, density: float, block: tuple[int, int]) -> Tensor:
    if t.ndim != 2:
        raise ValueError("block_magnitude_prune_2d expects a 2D tensor")
    keep_density = min(max(float(density), 0.0), 1.0)
    if keep_density >= 1.0:
        return t.contiguous()
    block_h, block_w = max(int(block[0]), 1), max(int(block[1]), 1)
    height, width = int(t.shape[0]), int(t.shape[1])
    tiles_h = (height + block_h - 1) // block_h
    tiles_w = (width + block_w - 1) // block_w
    pad_h = tiles_h * block_h - height
    pad_w = tiles_w * block_w - width
    padded = F.pad(t, (0, pad_w, 0, pad_h))
    blocks = padded.view(tiles_h, block_h, tiles_w, block_w).permute(0, 2, 1, 3)
    norms = blocks.reshape(tiles_h, tiles_w, -1).abs().sum(dim=-1)
    keep_blocks = max(1, int(round(keep_density * norms.numel())))
    threshold = norms.reshape(-1).kthvalue(norms.numel() - keep_blocks + 1).values
    keep = norms >= threshold
    keep_mask = keep.repeat_interleave(block_h, dim=0).repeat_interleave(block_w, dim=1)
    pruned = padded * keep_mask.to(dtype=padded.dtype)
    return pruned[:height, :width].contiguous()


def _reshape_tensor_into_blocks_2d(t: Tensor, block: tuple[int, int]) -> tuple[Tensor, Tensor, dict[str, int]]:
    if t.ndim != 2:
        raise ValueError("_reshape_tensor_into_blocks_2d expects a 2D tensor")
    block_h, block_w = max(int(block[0]), 1), max(int(block[1]), 1)
    height, width = int(t.shape[0]), int(t.shape[1])
    tiles_h = (height + block_h - 1) // block_h
    tiles_w = (width + block_w - 1) // block_w
    pad_h = tiles_h * block_h - height
    pad_w = tiles_w * block_w - width
    padded = F.pad(t, (0, pad_w, 0, pad_h))
    blocks = padded.view(tiles_h, block_h, tiles_w, block_w).permute(0, 2, 1, 3).contiguous()
    meta = {
        "height": height,
        "width": width,
        "block_h": block_h,
        "block_w": block_w,
        "tiles_h": tiles_h,
        "tiles_w": tiles_w,
        "padded_h": tiles_h * block_h,
        "padded_w": tiles_w * block_w,
    }
    return padded.contiguous(), blocks, meta


def _merge_blocks_into_tensor_2d(blocks: Tensor, meta: dict[str, int], dtype: torch.dtype) -> Tensor:
    padded = (
        blocks.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(meta["padded_h"]), int(meta["padded_w"]))
    )
    return padded[: int(meta["height"]), : int(meta["width"])].to(dtype=dtype).contiguous()


def _quantize_blocks_with_clip(blocks: Tensor, clip_abs: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    qmax = float(127 if bits == 8 else (2 ** (bits - 1)) - 1)
    qmin = -float(qmax if bits == 8 else (2 ** (bits - 1)))
    clip_abs = clip_abs.clamp_min(1e-12)
    clipped = torch.maximum(torch.minimum(blocks, clip_abs[..., None, None]), -clip_abs[..., None, None])
    scale = (clip_abs / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(clipped / scale[..., None, None]), qmin, qmax).to(torch.int8).contiguous()
    return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()


def quantize_mixed_precision_blockwise_2d(t: Tensor, bits: int, block: tuple[int, int]) -> tuple[Tensor, Tensor]:
    t32 = t.float().contiguous()
    _, blocks, meta = _reshape_tensor_into_blocks_2d(t32, block)
    block_abs = blocks.abs().reshape(int(meta["tiles_h"]), int(meta["tiles_w"]), -1)
    if EXPORT_CALIBRATION_METHOD == "absmax":
        q_blocks, scales = _quantize_blocks_with_clip(blocks, block_abs.amax(dim=-1), bits)
        return _merge_blocks_into_tensor_2d(q_blocks, meta, torch.int8), scales
    if EXPORT_CALIBRATION_METHOD == "percentile":
        q_blocks, scales = _quantize_blocks_with_clip(blocks, torch.quantile(block_abs, INT8_CLIP_Q, dim=-1), bits)
        return _merge_blocks_into_tensor_2d(q_blocks, meta, torch.int8), scales

    best_q, best_s = _quantize_blocks_with_clip(blocks, block_abs.amax(dim=-1), bits)
    best_mse = (best_q.float() * best_s[..., None, None].float() - blocks).square().mean(dim=(-1, -2))
    for clip_q in EXPORT_CALIBRATION_PERCENTILES:
        if clip_q >= 1.0:
            continue
        q_blocks, scales = _quantize_blocks_with_clip(blocks, torch.quantile(block_abs, clip_q, dim=-1), bits)
        mse = (q_blocks.float() * scales[..., None, None].float() - blocks).square().mean(dim=(-1, -2))
        better = mse < best_mse
        if bool(better.any().item()):
            best_mse = torch.where(better, mse, best_mse)
            best_s = torch.where(better, scales, best_s)
            best_q = torch.where(better[..., None, None], q_blocks, best_q)
    return _merge_blocks_into_tensor_2d(best_q, meta, torch.int8), best_s.contiguous()


def extract_blockwise_outliers_2d(
    t: Tensor, keep_density: float, block: tuple[int, int]
) -> tuple[Tensor, dict[str, Tensor] | None, float]:
    t32 = t.float().contiguous()
    keep_density = min(max(float(keep_density), 0.0), 1.0)
    if keep_density >= 1.0 or t32.numel() == 0:
        return t32, None, 1.0
    _, blocks, meta = _reshape_tensor_into_blocks_2d(t32, block)
    block_elems = max(int(meta["block_h"]) * int(meta["block_w"]), 1)
    outliers_per_block = int(round((1.0 - keep_density) * block_elems))
    outliers_per_block = min(max(outliers_per_block, 0), max(block_elems - 1, 0))
    if outliers_per_block <= 0:
        return t32, None, 1.0
    block_abs = blocks.abs().reshape(int(meta["tiles_h"]), int(meta["tiles_w"]), -1)
    topk = torch.topk(block_abs, k=outliers_per_block, dim=-1, largest=True, sorted=False).indices
    mask_flat = torch.zeros_like(block_abs, dtype=torch.bool)
    mask_flat.scatter_(-1, topk, True)
    mask_blocks = mask_flat.view(
        int(meta["tiles_h"]), int(meta["tiles_w"]), int(meta["block_h"]), int(meta["block_w"])
    )
    mask = _merge_blocks_into_tensor_2d(mask_blocks.to(dtype=torch.float32), meta, torch.bool)
    indices = mask.reshape(-1).nonzero(as_tuple=True)[0].to(dtype=torch.int32).contiguous()
    if indices.numel() == 0:
        return t32, None, 1.0
    values = t32.reshape(-1)[indices.to(dtype=torch.int64)].to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    masked = t32.masked_fill(mask, 0.0).contiguous()
    actual_density = 1.0 - (float(indices.numel()) / max(float(t32.numel()), 1.0))
    return masked, {"indices": indices, "values": values}, actual_density


def _quantize_per_row_with_clip(t32: Tensor, clip_abs: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    qmax = float(127 if bits == 8 else (2 ** (bits - 1)) - 1)
    qmin = -float(qmax if bits == 8 else (2 ** (bits - 1)))
    clip_abs = clip_abs.clamp_min(1e-12)
    clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
    scale = (clip_abs / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(clipped / scale[:, None]), qmin, qmax).to(torch.int8).contiguous()
    return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()


def _quantize_per_tensor_with_clip(t32: Tensor, clip_abs: float, bits: int) -> tuple[Tensor, Tensor]:
    qmax = float(127 if bits == 8 else (2 ** (bits - 1)) - 1)
    qmin = -float(qmax if bits == 8 else (2 ** (bits - 1)))
    clip_abs = max(float(clip_abs), 1e-12)
    scale = torch.tensor(clip_abs / qmax, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), qmin, qmax).to(torch.int8).contiguous()
    return q, scale


def quantize_tensor_calibrated(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.numel() == 0:
        return _quantize_per_tensor_with_clip(t32, 1.0, bits)

    if t32.ndim == 2:
        row_abs = t32.abs()
        if EXPORT_CALIBRATION_METHOD == "absmax":
            return _quantize_per_row_with_clip(t32, row_abs.amax(dim=1), bits)
        if EXPORT_CALIBRATION_METHOD == "percentile":
            return _quantize_per_row_with_clip(t32, torch.quantile(row_abs, INT8_CLIP_Q, dim=1), bits)

        best_q, best_s = _quantize_per_row_with_clip(t32, row_abs.amax(dim=1), bits)
        best_mse = (best_q.float() * best_s[:, None].float() - t32).square().mean(dim=1)
        for clip_q in EXPORT_CALIBRATION_PERCENTILES:
            if clip_q >= 1.0:
                continue
            q, s = _quantize_per_row_with_clip(t32, torch.quantile(row_abs, clip_q, dim=1), bits)
            mse = (q.float() * s[:, None].float() - t32).square().mean(dim=1)
            better = mse < best_mse
            if bool(better.any().item()):
                best_mse = torch.where(better, mse, best_mse)
                best_s = torch.where(better, s, best_s)
                best_q[better] = q[better]
        return best_q.contiguous(), best_s.contiguous()

    flat_abs = t32.abs().flatten()
    if EXPORT_CALIBRATION_METHOD == "absmax":
        return _quantize_per_tensor_with_clip(t32, float(flat_abs.amax().item()), bits)
    if EXPORT_CALIBRATION_METHOD == "percentile":
        return _quantize_per_tensor_with_clip(t32, float(torch.quantile(flat_abs, INT8_CLIP_Q).item()), bits)

    best_q, best_s = _quantize_per_tensor_with_clip(t32, float(flat_abs.amax().item()), bits)
    best_mse = float((best_q.float() * float(best_s.item()) - t32).square().mean().item())
    for clip_q in EXPORT_CALIBRATION_PERCENTILES:
        if clip_q >= 1.0:
            continue
        q, s = _quantize_per_tensor_with_clip(t32, float(torch.quantile(flat_abs, clip_q).item()), bits)
        mse = float((q.float() * float(s.item()) - t32).square().mean().item())
        if mse < best_mse:
            best_q, best_s, best_mse = q, s, mse
    return best_q, best_s


def quantize_float_tensor(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    return quantize_tensor_calibrated(t, bits=bits)


def quantize_tensor_row_groups(
    name: str,
    t: Tensor,
    row_group_size: int,
    passthrough_orig_dtypes: dict[str, str],
) -> list[dict[str, object]]:
    if t.ndim != 2 or row_group_size <= 0:
        return []
    groups: list[dict[str, object]] = []
    for row_start in range(0, t.shape[0], row_group_size):
        row_end = min(row_start + row_group_size, t.shape[0])
        chunk = t[row_start:row_end].contiguous()
        q, s = quantize_float_tensor(chunk)
        deq = dequantize_int8_tensor(q, s, t.dtype)
        diff = deq.float() - chunk.float()
        mse = float(diff.square().mean().item()) if diff.numel() else 0.0
        ref = float(chunk.float().square().mean().item()) if chunk.numel() else 0.0
        rel_mse = mse / max(ref, 1e-12)
        quant_bytes = tensor_nbytes(q) + tensor_nbytes(s)
        keep_bytes = tensor_nbytes(keep_float_tensor(name, chunk, passthrough_orig_dtypes))
        extra_bytes = max(keep_bytes - quant_bytes, 0)
        benefit = rel_mse * float(chunk.numel())
        groups.append(
            {
                "name": name,
                "row_start": row_start,
                "row_end": row_end,
                "extra_bytes": int(extra_bytes),
                "rel_mse": rel_mse,
                "benefit_per_extra_byte": benefit / max(extra_bytes, 1),
            }
        )
    return groups


def dequantize_int8_tensor(q: Tensor, s: Tensor, dtype: torch.dtype) -> Tensor:
    if s.ndim > 0:
        return (q.float() * s.to(dtype=torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
    return (q.float() * float(s.item())).to(dtype=dtype).contiguous()


def restore_nested_scale_tensor(s: Tensor, qmeta_entry: dict[str, object] | None) -> Tensor:
    if not isinstance(qmeta_entry, dict):
        return s.to(dtype=torch.float32).contiguous()
    nested_scale_scales = qmeta_entry.get("nested_scale_scales")
    if isinstance(nested_scale_scales, Tensor):
        return dequantize_int8_tensor(s, nested_scale_scales, torch.float32)
    return s.to(dtype=torch.float32).contiguous()


def dequantize_blockwise_int8_tensor(
    q: Tensor, s: Tensor, dtype: torch.dtype, block: tuple[int, int], qmeta_entry: dict[str, object] | None = None
) -> Tensor:
    _, q_blocks, meta = _reshape_tensor_into_blocks_2d(q.float(), block)
    scales = restore_nested_scale_tensor(s, qmeta_entry).view(int(meta["tiles_h"]), int(meta["tiles_w"]))
    deq_blocks = q_blocks.float() * scales[..., None, None].to(dtype=torch.float32)
    return _merge_blocks_into_tensor_2d(deq_blocks, meta, dtype)


def apply_outlier_passthrough(t: Tensor, outlier_entry: dict[str, Tensor] | None) -> Tensor:
    if not isinstance(outlier_entry, dict):
        return t
    indices = outlier_entry.get("indices")
    values = outlier_entry.get("values")
    if not isinstance(indices, Tensor) or not isinstance(values, Tensor) or indices.numel() == 0:
        return t
    out = t.contiguous()
    flat = out.reshape(-1)
    flat[indices.to(dtype=torch.int64)] = values.to(dtype=out.dtype)
    return flat.view_as(out).contiguous()


def dequantize_quantized_tensor(q: Tensor, s: Tensor, dtype: torch.dtype, qmeta_entry: dict[str, object] | None = None) -> Tensor:
    scheme = str((qmeta_entry or {}).get("scheme", "")).strip().lower()
    if scheme == "blockwise":
        block_raw = (qmeta_entry or {}).get("block", (16, 16))
        if isinstance(block_raw, (list, tuple)) and len(block_raw) == 2:
            block = (max(int(block_raw[0]), 1), max(int(block_raw[1]), 1))
        else:
            block = (16, 16)
        return dequantize_blockwise_int8_tensor(q, s, dtype, block, qmeta_entry)
    return dequantize_int8_tensor(q, s, dtype)


def quantize_mixed_precision_matrix_rows(t32: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    return quantize_tensor_calibrated(t32, bits=bits)


def quantize_mixed_precision_tensor(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    if t.ndim == 2:
        return quantize_mixed_precision_matrix_rows(t.float(), bits)
    return quantize_tensor_calibrated(t, bits=bits)


def resolve_mixed_precision_target_bits(name: str, bits: int, high_bits: int, high_precision_names: set[str]) -> int:
    target_class = _classify_quant_target(name)
    if target_class == "mlp":
        target_bits = MIXED_PRECISION_EXPORT_MLP_BITS if MIXED_PRECISION_EXPORT_MLP_BITS > 0 else bits
    elif target_class == "attn":
        target_bits = MIXED_PRECISION_EXPORT_ATTN_BITS if MIXED_PRECISION_EXPORT_ATTN_BITS > 0 else bits
    elif target_class == "embed":
        target_bits = 8
    else:
        target_bits = 8
    if name in high_precision_names:
        target_bits = high_bits
    return int(target_bits)


def _classify_quant_target(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if "bigram.embed" in name:
        return "attn"
    if "lexical_shortcut_head" in name or "lexical_residual_head" in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or ".proj." in name:
        return "attn"
    return "other"


def compressed_quantized_tensor_nbytes(
    q: Tensor, s: Tensor, codec: str, zstd_level: int, extra_payload: dict[str, object] | None = None
) -> int:
    buf = io.BytesIO()
    payload: dict[str, object] = {"q": q, "s": s}
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, buf)
    blob, _ = compress_blob(buf.getvalue(), codec, zstd_level)
    return int(len(blob))


def compressed_serialized_payload_nbytes(payload: dict[str, object], codec: str, zstd_level: int) -> int:
    buf = io.BytesIO()
    torch.save(payload, buf)
    blob, _ = compress_blob(buf.getvalue(), codec, zstd_level)
    return int(len(blob))


def relative_mse(reference: Tensor, approx: Tensor) -> float:
    diff = approx.float() - reference.float()
    mse = float(diff.square().mean().item()) if diff.numel() else 0.0
    ref = float(reference.float().square().mean().item()) if reference.numel() else 0.0
    return mse / max(ref, 1e-12)


def export_candidate_proxy_gain(name: str, rel_mse_gain: float, numel: int) -> float:
    base_gain = max(float(rel_mse_gain), 0.0) * max(int(numel), 1)
    if EXPORT_TENSOR_SWEEP_OBJECTIVE != "proxy_score_per_compressed_byte":
        return base_gain
    target_class = _classify_quant_target(name)
    class_weight = {"embed": 1.15, "attn": 1.0, "mlp": 0.9}.get(target_class, 0.8)
    return base_gain * class_weight


def summarize_export_candidate(candidate: dict[str, object]) -> dict[str, object]:
    summary = {
        "name": str(candidate["name"]),
        "numel": int(candidate["numel"]),
        "export_kind": str(candidate.get("export_kind", "quantized")),
        "scheme": str(candidate.get("scheme", "per_tensor")),
        "bits": int(candidate.get("bits", 0)),
        "rel_mse": float(candidate.get("rel_mse", 0.0)),
        "compressed_nbytes": int(candidate.get("compressed_nbytes", 0)),
        "payload_bytes": int(candidate.get("payload_bytes", 0)),
    }
    if "scale_scheme" in candidate:
        summary["scale_scheme"] = str(candidate["scale_scheme"])
    if "requested_density" in candidate:
        summary["requested_density"] = float(candidate["requested_density"])
    if "kept_density" in candidate:
        summary["kept_density"] = float(candidate["kept_density"])
    if "outlier_count" in candidate:
        summary["outlier_count"] = int(candidate["outlier_count"])
    return summary


def candidate_delta_metrics(name: str, baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, float]:
    baseline_rel_mse = float(baseline.get("rel_mse", 0.0))
    candidate_rel_mse = float(candidate.get("rel_mse", 0.0))
    delta_rel_mse = baseline_rel_mse - candidate_rel_mse
    compressed_delta = int(candidate.get("compressed_nbytes", 0)) - int(baseline.get("compressed_nbytes", 0))
    extra_bytes = max(compressed_delta, 0)
    bytes_saved = max(-compressed_delta, 0)
    proxy_gain = export_candidate_proxy_gain(name, delta_rel_mse, int(candidate.get("numel", baseline.get("numel", 0))))
    return {
        "delta_rel_mse": float(delta_rel_mse),
        "extra_rel_mse": float(max(candidate_rel_mse - baseline_rel_mse, 0.0)),
        "compressed_byte_delta": int(compressed_delta),
        "extra_bytes": int(extra_bytes),
        "bytes_saved": int(bytes_saved),
        "proxy_gain": float(proxy_gain),
        "benefit_per_extra_byte": float(proxy_gain / max(extra_bytes, 1)) if extra_bytes > 0 else 0.0,
        "loss_per_byte_saved": float(max(-proxy_gain, 0.0) / max(bytes_saved, 1)) if bytes_saved > 0 else 0.0,
    }


def build_quantized_export_candidate(
    name: str,
    t: Tensor,
    bits: int,
    codec: str,
    zstd_level: int,
    *,
    block: tuple[int, int] | None = None,
    keep_density: float = 1.0,
) -> dict[str, object]:
    t_cpu = t.detach().to("cpu").contiguous()
    if block is None:
        q, s = quantize_mixed_precision_tensor(t_cpu, bits)
        qmeta = {"scheme": "per_row" if s.ndim > 0 else "per_tensor", "axis": 0, "bits": int(bits)}
        deq = dequantize_quantized_tensor(q, s, t_cpu.dtype, qmeta)
        return {
            "name": str(name),
            "numel": int(t_cpu.numel()),
            "export_kind": "quantized",
            "scheme": str(qmeta["scheme"]),
            "bits": int(bits),
            "rel_mse": float(relative_mse(t_cpu, deq)),
            "compressed_nbytes": int(compressed_quantized_tensor_nbytes(q, s, codec, zstd_level)),
            "payload_bytes": int(tensor_nbytes(q) + tensor_nbytes(s)),
            "q": q,
            "s": s,
            "qmeta": qmeta,
            "outliers": None,
        }

    quant_input, outlier_passthrough, actual_density = extract_blockwise_outliers_2d(t_cpu, keep_density, block)
    q_blockwise, s_blockwise_plain = quantize_mixed_precision_blockwise_2d(quant_input, bits, block)
    scale_payloads: list[tuple[Tensor, dict[str, object]]] = [(s_blockwise_plain, {})]
    if s_blockwise_plain.ndim == 2 and s_blockwise_plain.numel() > 0:
        s_nested_q, s_nested_s = quantize_tensor_calibrated(s_blockwise_plain.float(), bits=8)
        scale_payloads.append(
            (
                s_nested_q.contiguous(),
                {
                    "scale_scheme": "nested_per_row",
                    "nested_scale_bits": 8,
                    "nested_scale_scales": s_nested_s.contiguous(),
                },
            )
        )
    best_candidate: dict[str, object] | None = None
    for scale_payload, scale_meta in scale_payloads:
        qmeta_entry = {
            "scheme": "blockwise",
            "axis": 0,
            "bits": int(bits),
            "block": [int(block[0]), int(block[1])],
            **scale_meta,
        }
        extra_payload: dict[str, object] = {}
        payload_bytes = tensor_nbytes(q_blockwise) + tensor_nbytes(scale_payload)
        if "nested_scale_scales" in scale_meta:
            extra_payload["nested_scale_scales"] = scale_meta["nested_scale_scales"]
            payload_bytes += tensor_nbytes(scale_meta["nested_scale_scales"])
        if outlier_passthrough is not None:
            extra_payload["outliers"] = outlier_passthrough
            qmeta_entry["outlier_count"] = int(outlier_passthrough["indices"].numel())
            payload_bytes += tensor_nbytes(outlier_passthrough["indices"]) + tensor_nbytes(outlier_passthrough["values"])
        compressed_candidate = compressed_quantized_tensor_nbytes(
            q_blockwise, scale_payload, codec, zstd_level, extra_payload=extra_payload
        )
        deq_candidate = dequantize_quantized_tensor(q_blockwise, scale_payload, t_cpu.dtype, qmeta_entry)
        deq_candidate = apply_outlier_passthrough(deq_candidate, outlier_passthrough)
        candidate = {
            "name": str(name),
            "numel": int(t_cpu.numel()),
            "export_kind": "quantized",
            "scheme": "blockwise",
            "bits": int(bits),
            "scale_scheme": str(scale_meta.get("scale_scheme", "fp16_block")),
            "requested_density": float(keep_density),
            "kept_density": float(actual_density),
            "outlier_count": int(0 if outlier_passthrough is None else outlier_passthrough["indices"].numel()),
            "rel_mse": float(relative_mse(t_cpu, deq_candidate)),
            "compressed_nbytes": int(compressed_candidate),
            "payload_bytes": int(payload_bytes),
            "q": q_blockwise,
            "s": scale_payload,
            "qmeta": qmeta_entry,
            "outliers": outlier_passthrough,
        }
        if best_candidate is None or (
            int(candidate["compressed_nbytes"]),
            float(candidate["rel_mse"]),
        ) < (
            int(best_candidate["compressed_nbytes"]),
            float(best_candidate["rel_mse"]),
        ):
            best_candidate = candidate
    if best_candidate is None:
        raise RuntimeError(f"failed to build blockwise candidate for {name}")
    return best_candidate


def build_float_export_candidate(name: str, t: Tensor, codec: str, zstd_level: int) -> dict[str, object]:
    t_cpu = t.detach().to("cpu").contiguous()
    passthrough_orig_dtypes: dict[str, str] = {}
    kept = keep_float_tensor(name, t_cpu, passthrough_orig_dtypes)
    restored = kept.detach().to("cpu").contiguous()
    orig_dtype = passthrough_orig_dtypes.get(name)
    if isinstance(orig_dtype, str):
        restored = restored.to(dtype=getattr(torch, orig_dtype)).contiguous()
    payload: dict[str, object] = {"value": kept}
    if isinstance(orig_dtype, str):
        payload["orig_dtype"] = orig_dtype
    return {
        "name": str(name),
        "numel": int(t_cpu.numel()),
        "export_kind": "float",
        "scheme": "float",
        "bits": int(16 if kept.dtype == torch.float16 else 32),
        "rel_mse": float(relative_mse(t_cpu, restored)),
        "compressed_nbytes": int(compressed_serialized_payload_nbytes(payload, codec, zstd_level)),
        "payload_bytes": int(tensor_nbytes(kept)),
        "kept": kept,
        "passthrough_orig_dtype": orig_dtype,
    }


def build_tensor_export_candidates(
    name: str,
    t: Tensor,
    default_bits: int,
    codec: str,
    zstd_level: int,
    *,
    extra_bits: tuple[int, ...] = (),
    include_float: bool = False,
    include_blockwise: bool = True,
) -> list[dict[str, object]]:
    candidate_bits = tuple(sorted(set((int(default_bits), *EXPORT_TENSOR_CANDIDATE_BITS, *extra_bits))))
    candidates: list[dict[str, object]] = []
    for bits in candidate_bits:
        candidates.append(build_quantized_export_candidate(name, t, bits, codec, zstd_level))
        blockwise_allowed = (
            include_blockwise
            and EXPORT_BLOCK_PRUNE_ENABLED
            and bool(EXPORT_BLOCK_PRUNE_DENSITIES)
            and t.ndim == 2
            and t.numel() >= max(EXPORT_BLOCK_PRUNE_MIN_NUMEL, EXPORT_TENSOR_SWEEP_MIN_NUMEL)
            and _classify_quant_target(name) in {"mlp", "attn"}
        )
        if blockwise_allowed:
            for keep_density in tuple(sorted(set((1.0, *EXPORT_BLOCK_PRUNE_DENSITIES)), reverse=True)):
                candidates.append(
                    build_quantized_export_candidate(
                        name,
                        t,
                        bits,
                        codec,
                        zstd_level,
                        block=EXPORT_BLOCK_PRUNE_BLOCK,
                        keep_density=keep_density,
                    )
                )
    if include_float:
        candidates.append(build_float_export_candidate(name, t, codec, zstd_level))
    deduped: dict[tuple[object, ...], dict[str, object]] = {}
    for candidate in candidates:
        key = (
            candidate.get("export_kind"),
            candidate.get("scheme"),
            int(candidate.get("bits", 0)),
            float(candidate.get("requested_density", 1.0)),
            str(candidate.get("scale_scheme", "")),
        )
        existing = deduped.get(key)
        if existing is None or (
            int(candidate["compressed_nbytes"]),
            float(candidate["rel_mse"]),
        ) < (
            int(existing["compressed_nbytes"]),
            float(existing["rel_mse"]),
        ):
            deduped[key] = candidate
    return list(deduped.values())


def select_default_export_candidate(
    name: str, baseline: dict[str, object], candidates: list[dict[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    ranked: list[dict[str, object]] = []
    best = baseline
    for candidate in candidates:
        if candidate is baseline or candidate.get("export_kind") != "quantized":
            continue
        delta = candidate_delta_metrics(name, baseline, candidate)
        if int(delta["bytes_saved"]) <= 0 or float(delta["extra_rel_mse"]) > EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE:
            continue
        entry = {**summarize_export_candidate(candidate), **delta}
        ranked.append(entry)
        if best is baseline or (
            int(entry["bytes_saved"]),
            -float(entry["extra_rel_mse"]),
            -float(entry["rel_mse"]),
        ) > (
            int(candidate_delta_metrics(name, baseline, best)["bytes_saved"]),
            -float(candidate_delta_metrics(name, baseline, best)["extra_rel_mse"]),
            -float(best["rel_mse"]),
        ):
            best = candidate
    ranked.sort(
        key=lambda item: (
            int(item["bytes_saved"]),
            -float(item["extra_rel_mse"]),
            -float(item["rel_mse"]),
            int(item["numel"]),
        ),
        reverse=True,
    )
    return best, ranked


def compress_blob(payload: bytes, codec: str, zstd_level: int) -> tuple[bytes, str]:
    normalized = codec.strip().lower()
    if normalized == "zstd" and HAS_ZSTD:
        return zstandard.ZstdCompressor(level=zstd_level).compress(payload), "zstd"
    return zlib.compress(payload, level=9), "zlib"


def decompress_blob(payload: bytes, codec: str) -> bytes:
    if codec == "zstd":
        if not HAS_ZSTD:
            raise RuntimeError("EXPORT_CODEC=zstd requested but zstandard is not installed")
        return zstandard.ZstdDecompressor().decompress(payload)
    return zlib.decompress(payload)


def select_auto_keep_float_tensors(state_dict: dict[str, Tensor]) -> tuple[set[str], dict[str, object]]:
    budget_bytes = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_budget_bytes"])
    max_tensors = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_max_tensors"])
    min_numel = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_min_numel"])
    log_topk = int(ACTIVE_EXPORT_CONFIG["int8_sensitivity_log_topk"])
    enabled = budget_bytes > 0 or max_tensors > 0
    info: dict[str, object] = {
        "enabled": enabled,
        "budget_bytes": int(budget_bytes),
        "max_tensors": int(max_tensors),
        "min_numel": int(min_numel),
        "selected_names": [],
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "top_candidates": [],
    }
    if not enabled:
        return set(), info

    candidates: list[dict[str, object]] = []
    codec = str(ACTIVE_EXPORT_CONFIG["export_codec"])
    zstd_level = int(ACTIVE_EXPORT_CONFIG["export_zstd_level"])
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point() or should_keep_float_tensor(name, t) or t.numel() < min_numel:
            continue
        q_bits = LARGE_MATRIX_QUANT_BITS if t.ndim == 2 else 8
        baseline = build_quantized_export_candidate(name, t, q_bits, codec, zstd_level)
        float_candidate = build_float_export_candidate(name, t, codec, zstd_level)
        delta = candidate_delta_metrics(name, baseline, float_candidate)
        if int(delta["extra_bytes"]) <= 0 or float(delta["proxy_gain"]) <= 0.0:
            continue
        max_abs_diff = float((float_candidate["kept"].float() - t.float()).abs().max().item()) if t.numel() else 0.0
        candidates.append(
            {
                "name": str(name),
                "numel": int(t.numel()),
                "keep_bytes": int(float_candidate["payload_bytes"]),
                "quant_bytes": int(baseline["payload_bytes"]),
                "extra_bytes": int(delta["extra_bytes"]),
                "compressed_byte_delta": int(delta["compressed_byte_delta"]),
                "rel_mse": float(baseline["rel_mse"]),
                "rel_mse_gain": float(delta["delta_rel_mse"]),
                "max_abs_diff": max_abs_diff,
                "benefit": float(delta["proxy_gain"]),
                "benefit_per_extra_byte": float(delta["benefit_per_extra_byte"]),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item["benefit_per_extra_byte"]),
            float(item["benefit"]),
            float(item["rel_mse"]),
            int(item["numel"]),
        ),
        reverse=True,
    )
    selected_names: list[str] = []
    used_extra_bytes = 0
    for item in ranked:
        if max_tensors > 0 and len(selected_names) >= max_tensors:
            break
        extra_bytes = int(item["extra_bytes"])
        if budget_bytes > 0 and used_extra_bytes + extra_bytes > budget_bytes:
            continue
        selected_names.append(str(item["name"]))
        used_extra_bytes += extra_bytes

    info["selected_names"] = selected_names
    info["selected_count"] = len(selected_names)
    info["selected_extra_bytes"] = int(used_extra_bytes)
    info["top_candidates"] = [
        {
            "name": str(item["name"]),
            "numel": int(item["numel"]),
            "extra_bytes": int(item["extra_bytes"]),
            "rel_mse": float(item["rel_mse"]),
            "max_abs_diff": float(item["max_abs_diff"]),
            "benefit_per_extra_byte": float(item["benefit_per_extra_byte"]),
        }
        for item in ranked[: max(log_topk, 0)]
    ]
    return set(selected_names), info


def select_auto_keep_float_row_groups(
    state_dict: dict[str, Tensor], passthrough_orig_dtypes: dict[str, str]
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, object]]:
    budget_bytes = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_row_budget_bytes"])
    group_size = int(ACTIVE_EXPORT_CONFIG["int8_auto_keep_row_group_size"])
    enabled = budget_bytes > 0 and group_size > 0
    info: dict[str, object] = {
        "enabled": enabled,
        "budget_bytes": int(budget_bytes),
        "row_group_size": int(group_size),
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "selected_groups": [],
    }
    if not enabled:
        return {}, info

    candidates: list[dict[str, object]] = []
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if (
            not t.is_floating_point()
            or should_keep_float_tensor(name, t)
            or t.ndim != 2
            or t.shape[0] < group_size
        ):
            continue
        candidates.extend(quantize_tensor_row_groups(name, t, group_size, passthrough_orig_dtypes))

    ranked = sorted(
        candidates,
        key=lambda item: (float(item["benefit_per_extra_byte"]), float(item["rel_mse"])),
        reverse=True,
    )
    selected: dict[str, list[tuple[int, int]]] = {}
    used_bytes = 0
    seen: set[tuple[str, int]] = set()
    for item in ranked:
        key = (str(item["name"]), int(item["row_start"]))
        if key in seen:
            continue
        extra = int(item["extra_bytes"])
        if used_bytes + extra > budget_bytes:
            continue
        seen.add(key)
        used_bytes += extra
        selected.setdefault(str(item["name"]), []).append((int(item["row_start"]), int(item["row_end"])))
    for name in selected:
        selected[name].sort()
    info["selected_count"] = sum(len(v) for v in selected.values())
    info["selected_extra_bytes"] = int(used_bytes)
    info["selected_groups"] = [
        {"name": name, "row_start": row_start, "row_end": row_end}
        for name, groups in selected.items()
        for row_start, row_end in groups
    ]
    return selected, info


def _equalization_channel_metric(t: Tensor, dim: int) -> Tensor:
    tf = t.detach().to(dtype=torch.float32)
    rms = tf.square().mean(dim=dim).add_(1e-12).sqrt()
    peak = tf.abs().amax(dim=dim).clamp_min_(1e-12)
    return (rms * peak).sqrt()


def _scaled_rows(weight: Tensor, scales: Tensor) -> Tensor:
    scaled = weight.detach().to(dtype=torch.float32, device="cpu") * scales[:, None].to(dtype=torch.float32, device="cpu")
    return scaled.to(dtype=weight.dtype).contiguous()


def _scaled_cols(weight: Tensor, scales: Tensor) -> Tensor:
    scaled = weight.detach().to(dtype=torch.float32, device="cpu") * scales[None, :].to(dtype=torch.float32, device="cpu")
    return scaled.to(dtype=weight.dtype).contiguous()


def _attention_equalization_layout(
    prefix: str, state_dict: dict[str, Tensor], v_channels: int, proj_in_dim: int
) -> tuple[int, int, int, int] | None:
    q_gain = state_dict.get(f"{prefix}.q_gain")
    if isinstance(q_gain, Tensor) and q_gain.ndim == 1:
        num_heads = int(q_gain.numel())
        if num_heads > 0 and proj_in_dim % num_heads == 0:
            head_dim = proj_in_dim // num_heads
            if head_dim > 0 and v_channels % head_dim == 0:
                num_kv_heads = v_channels // head_dim
                if num_kv_heads > 0 and num_heads % num_kv_heads == 0:
                    return num_heads, num_kv_heads, head_dim, num_heads // num_kv_heads
    if v_channels == proj_in_dim:
        return v_channels, v_channels, 1, 1
    return None


def _attention_equalization_scales(prefix: str, state_dict: dict[str, Tensor], v_weight: Tensor, proj_weight: Tensor) -> Tensor | None:
    if v_weight.ndim != 2 or proj_weight.ndim != 2:
        return None
    v_channels = int(v_weight.shape[0])
    proj_in_dim = int(proj_weight.shape[1])
    layout = _attention_equalization_layout(prefix, state_dict, v_channels, proj_in_dim)
    if layout is None:
        return None
    num_heads, num_kv_heads, head_dim, repeat = layout
    if v_channels != num_kv_heads * head_dim or proj_in_dim != num_heads * head_dim:
        return None
    producer = _equalization_channel_metric(v_weight, dim=1).view(num_kv_heads, head_dim)
    proj_metric = _equalization_channel_metric(proj_weight, dim=0).view(num_heads, head_dim)
    if repeat > 1:
        consumer = proj_metric.view(num_kv_heads, repeat, head_dim).square().mean(dim=1).sqrt()
    else:
        consumer = proj_metric.view(num_kv_heads, head_dim)
    gain = (consumer / producer.clamp_min(1e-12)).clamp_min(1e-12).sqrt()
    min_gain = 1.0 / EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN
    return gain.clamp(min=min_gain, max=EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN).reshape(-1)


def _expand_attention_proj_scales(
    prefix: str, state_dict: dict[str, Tensor], base_scales: Tensor, proj_in_dim: int
) -> Tensor | None:
    layout = _attention_equalization_layout(prefix, state_dict, int(base_scales.numel()), proj_in_dim)
    if layout is None:
        return None
    num_heads, num_kv_heads, head_dim, repeat = layout
    if int(base_scales.numel()) != num_kv_heads * head_dim or proj_in_dim != num_heads * head_dim:
        return None
    return base_scales.view(num_kv_heads, head_dim).repeat_interleave(repeat, dim=0).reshape(-1)


def canonicalize_state_dict_for_export(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if not EXPORT_CHANNEL_EQUALIZATION:
        return state_dict
    canonicalized = dict(state_dict)
    for name, tensor in state_dict.items():
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point() or tensor.ndim != 2:
            continue
        if name.endswith(".c_v.weight"):
            prefix = name[: -len(".c_v.weight")]
            proj_name = f"{prefix}.proj.weight"
            if (
                proj_name not in state_dict
                or f"{prefix}.inner_attn_norm.weight" in state_dict
                or f"{prefix}.inner_attn_norm.bias" in state_dict
            ):
                continue
            scales = _attention_equalization_scales(prefix, state_dict, tensor, state_dict[proj_name])
            proj_scales = None if scales is None else _expand_attention_proj_scales(prefix, state_dict, scales, int(state_dict[proj_name].shape[1]))
            if scales is None or proj_scales is None:
                continue
            canonicalized[name] = _scaled_rows(tensor, scales)
            canonicalized[proj_name] = _scaled_cols(state_dict[proj_name], proj_scales.reciprocal())
        elif name.endswith(".fc.weight"):
            prefix = name[: -len(".fc.weight")]
            proj_name = f"{prefix}.proj.weight"
            if (
                proj_name not in state_dict
                or f"{prefix}.inner_norm.weight" in state_dict
                or f"{prefix}.inner_norm.bias" in state_dict
            ):
                continue
            proj_weight = state_dict[proj_name]
            if proj_weight.ndim != 2 or int(proj_weight.shape[1]) != int(tensor.shape[0]):
                continue
            producer = _equalization_channel_metric(tensor, dim=1)
            consumer = _equalization_channel_metric(proj_weight, dim=0)
            gain = (consumer / producer.clamp_min(1e-12)).clamp_min(1e-12).sqrt()
            min_gain = 1.0 / EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN
            gain = gain.clamp(min=min_gain, max=EXPORT_CHANNEL_EQUALIZATION_MAX_GAIN)
            canonicalized[name] = _scaled_rows(tensor, gain)
            canonicalized[proj_name] = _scaled_cols(proj_weight, gain.square().reciprocal())
    return canonicalized


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    auto_keep_names, auto_keep_info = select_auto_keep_float_tensors(state_dict)
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    row_group_passthrough: dict[str, list[dict[str, object]]] = {}
    qmeta: dict[str, dict[str, object]] = {}
    row_keep_groups, row_keep_info = select_auto_keep_float_row_groups(state_dict, passthrough_orig_dtypes)
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
            "auto_keep_count",
            "auto_keep_extra_bytes",
            "auto_keep_row_group_count",
            "auto_keep_row_group_extra_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if name in auto_keep_names:
            stats["auto_keep_count"] += 1
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if should_keep_float_tensor(name, t):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if name in row_keep_groups:
            entries: list[dict[str, object]] = []
            for row_start, row_end in row_keep_groups[name]:
                kept = keep_float_tensor(name, t[row_start:row_end].contiguous(), passthrough_orig_dtypes)
                entries.append({"row_start": row_start, "row_end": row_end, "value": kept})
                stats["auto_keep_row_group_count"] += 1
                stats["int8_payload_bytes"] += tensor_nbytes(kept)
            row_group_passthrough[name] = entries
            qmeta[name] = {
                "scheme": "per_row",
                "axis": 0,
                "row_passthrough": [(entry["row_start"], entry["row_end"]) for entry in entries],
            }
        if s.ndim > 0:
            qmeta.setdefault(name, {"scheme": "per_row", "axis": 0})
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if row_group_passthrough:
        obj["row_group_passthrough"] = row_group_passthrough
    stats["auto_keep_extra_bytes"] = int(auto_keep_info["selected_extra_bytes"])
    stats["auto_keep_row_group_extra_bytes"] = int(row_keep_info["selected_extra_bytes"])
    return obj, stats, {**auto_keep_info, "row_groups": row_keep_info}

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    row_group_passthrough = obj.get("row_group_passthrough", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        out[name] = dequantize_quantized_tensor(q, s, dtype, qmeta.get(name, {}))
        if name in row_group_passthrough:
            for entry in row_group_passthrough[name]:
                row_start = int(entry["row_start"])
                row_end = int(entry["row_end"])
                patch = entry["value"].detach().to("cpu").contiguous()
                orig_dtype = passthrough_orig_dtypes.get(name)
                if isinstance(orig_dtype, str):
                    patch = patch.to(dtype=getattr(torch, orig_dtype)).contiguous()
                out[name][row_start:row_end] = patch.to(dtype=out[name].dtype)
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def quantize_state_dict_mixed_precision(state_dict: dict[str, Tensor], bits: int):
    high_bits = min(max(int(ACTIVE_EXPORT_CONFIG["export_high_precision_bits"]), bits), 8)
    codec = str(ACTIVE_EXPORT_CONFIG["export_codec"])
    zstd_level = int(ACTIVE_EXPORT_CONFIG["export_zstd_level"])
    high_precision_budget_bytes = int(ACTIVE_EXPORT_CONFIG["export_high_precision_budget_bytes"])
    high_precision_max_tensors = int(ACTIVE_EXPORT_CONFIG["export_high_precision_max_tensors"])
    high_precision_min_numel = int(ACTIVE_EXPORT_CONFIG["export_high_precision_min_numel"])
    target_bits_by_name: dict[str, int] = {}
    selected_base_candidates: dict[str, dict[str, object]] = {}
    selected_upgrade_candidates: dict[str, dict[str, object]] = {}
    tensor_sweep_selected: list[dict[str, object]] = []
    upgrade_ranked: list[tuple[dict[str, object], dict[str, object]]] = []
    tensor_sweep_info: dict[str, object] = {
        "enabled": bool(EXPORT_TENSOR_CANDIDATE_BITS),
        "objective": EXPORT_TENSOR_SWEEP_OBJECTIVE,
        "candidate_bits": [int(candidate_bits) for candidate_bits in EXPORT_TENSOR_CANDIDATE_BITS],
        "min_numel": int(EXPORT_TENSOR_SWEEP_MIN_NUMEL),
        "max_extra_rel_mse": float(EXPORT_TENSOR_SWEEP_MAX_EXTRA_REL_MSE),
        "selected_names": [],
        "selected_count": 0,
        "compressed_bytes_saved": 0,
        "top_candidates": [],
    }
    high_precision_info: dict[str, object] = {
        "enabled": high_bits > 0 and (high_precision_budget_bytes > 0 or high_precision_max_tensors > 0),
        "low_bits": int(bits),
        "high_bits": int(high_bits),
        "objective": EXPORT_TENSOR_SWEEP_OBJECTIVE,
        "candidate_bits": [int(candidate_bits) for candidate_bits in sorted(set((*EXPORT_TENSOR_CANDIDATE_BITS, high_bits)))],
        "budget_bytes": int(high_precision_budget_bytes),
        "max_tensors": int(high_precision_max_tensors),
        "min_numel": int(high_precision_min_numel),
        "selected_names": [],
        "selected_count": 0,
        "selected_extra_bytes": 0,
        "top_candidates": [],
    }
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point():
            continue
        if (
            should_keep_float_tensor(name, t)
            or any(pattern in name for pattern in MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS)
            or (_classify_quant_target(name) == "embed" and FORCE_FP16_TIED_EMBED_EXPORT)
        ):
            continue
        target_bits = resolve_mixed_precision_target_bits(name, bits, high_bits, set())
        target_bits_by_name[name] = target_bits
        if (
            _classify_quant_target(name) not in {"mlp", "attn"}
            or t.numel() < EXPORT_TENSOR_SWEEP_MIN_NUMEL
        ):
            continue
        baseline = build_quantized_export_candidate(name, t, target_bits, codec, zstd_level)
        candidates = build_tensor_export_candidates(
            name,
            t,
            target_bits,
            codec,
            zstd_level,
            extra_bits=(high_bits,),
            include_float=False,
            include_blockwise=True,
        )
        selected_base, ranked_defaults = select_default_export_candidate(name, baseline, candidates)
        if selected_base is not baseline:
            selected_base_candidates[name] = selected_base
            target_bits_by_name[name] = int(selected_base["bits"])
            selected_entry = {**summarize_export_candidate(selected_base), **candidate_delta_metrics(name, baseline, selected_base)}
            tensor_sweep_selected.append(selected_entry)
        if high_precision_info["enabled"] and t.numel() >= high_precision_min_numel:
            for candidate in candidates:
                if candidate is selected_base:
                    continue
                delta = candidate_delta_metrics(name, selected_base, candidate)
                if int(delta["extra_bytes"]) <= 0 or float(delta["proxy_gain"]) <= 0.0:
                    continue
                entry = {**summarize_export_candidate(candidate), **delta}
                upgrade_ranked.append((entry, candidate))
        if ranked_defaults:
            tensor_sweep_info["top_candidates"].extend(ranked_defaults[:1])

    upgrade_ranked.sort(
        key=lambda item: (
            float(item[0]["benefit_per_extra_byte"]),
            float(item[0]["delta_rel_mse"]),
            -int(item[0]["extra_bytes"]),
            int(item[0]["numel"]),
        ),
        reverse=True,
    )
    selected_upgrade_names: list[str] = []
    used_extra_bytes = 0
    if high_precision_info["enabled"]:
        for entry, candidate in upgrade_ranked:
            name = str(entry["name"])
            if name in selected_upgrade_candidates:
                continue
            if high_precision_max_tensors > 0 and len(selected_upgrade_names) >= high_precision_max_tensors:
                break
            extra_bytes = int(entry["extra_bytes"])
            if high_precision_budget_bytes > 0 and used_extra_bytes + extra_bytes > high_precision_budget_bytes:
                continue
            selected_upgrade_candidates[name] = candidate
            selected_upgrade_names.append(name)
            used_extra_bytes += extra_bytes

    tensor_sweep_selected.sort(
        key=lambda item: (
            int(item["bytes_saved"]),
            -float(item["extra_rel_mse"]),
            -float(item["rel_mse"]),
            int(item["numel"]),
        ),
        reverse=True,
    )
    tensor_sweep_info["selected_names"] = [str(item["name"]) for item in tensor_sweep_selected]
    tensor_sweep_info["selected_count"] = len(tensor_sweep_selected)
    tensor_sweep_info["compressed_bytes_saved"] = int(sum(int(item["bytes_saved"]) for item in tensor_sweep_selected))
    tensor_sweep_info["top_candidates"] = tensor_sweep_selected[: max(EXPORT_TENSOR_SWEEP_LOG_TOPK, 0)]
    high_precision_info["selected_names"] = selected_upgrade_names
    high_precision_info["selected_count"] = len(selected_upgrade_names)
    high_precision_info["selected_extra_bytes"] = int(used_extra_bytes)
    high_precision_info["top_candidates"] = [entry for entry, _candidate in upgrade_ranked[: max(EXPORT_TENSOR_SWEEP_LOG_TOPK, 0)]]

    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    outlier_passthrough: dict[str, dict[str, Tensor]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "payload_bytes",
            "high_precision_tensor_count",
            "high_precision_extra_bytes",
            "block_pruned_tensor_count",
            "block_prune_estimated_bytes_saved",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if (
            should_keep_float_tensor(name, t)
            or any(pattern in name for pattern in MIXED_PRECISION_KEEP_FP16_NAME_PATTERNS)
            or (_classify_quant_target(name) == "embed" and FORCE_FP16_TIED_EMBED_EXPORT)
        ):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        target_bits = int(target_bits_by_name.get(name, resolve_mixed_precision_target_bits(name, bits, high_bits, set())))
        selected_candidate = selected_upgrade_candidates.get(name) or selected_base_candidates.get(name)
        if name in selected_upgrade_candidates:
            stats["high_precision_tensor_count"] += 1
        if selected_candidate is not None and str(selected_candidate.get("scheme", "")) == "blockwise":
            stats["block_pruned_tensor_count"] += 1
        if selected_candidate is not None and selected_candidate.get("export_kind") == "quantized":
            q = selected_candidate["q"]
            s = selected_candidate["s"]
            qmeta[name] = dict(selected_candidate["qmeta"])
            if selected_candidate.get("outliers") is not None:
                outlier_passthrough[name] = selected_candidate["outliers"]
        else:
            q, s = quantize_mixed_precision_tensor(t, target_bits)
            qmeta[name] = {"scheme": "per_row" if s.ndim > 0 else "per_tensor", "axis": 0, "bits": int(target_bits)}
        if selected_candidate is not None and str(selected_candidate.get("scheme", "")) == "blockwise":
            qmeta[name]["block_prune"] = {
                "scheme": str(selected_candidate.get("scheme", "blockwise")),
                "scale_scheme": str(selected_candidate.get("scale_scheme", "fp16_block")),
                "block": [int(EXPORT_BLOCK_PRUNE_BLOCK[0]), int(EXPORT_BLOCK_PRUNE_BLOCK[1])],
                "requested_density": float(selected_candidate.get("requested_density", 1.0)),
                "kept_density": float(selected_candidate.get("kept_density", 1.0)),
                "outlier_count": int(selected_candidate.get("outlier_count", 0)),
            }
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        nested_scale_scales = qmeta[name].get("nested_scale_scales")
        if isinstance(nested_scale_scales, Tensor):
            stats["payload_bytes"] += tensor_nbytes(nested_scale_scales)
        if name in outlier_passthrough:
            stats["payload_bytes"] += tensor_nbytes(outlier_passthrough[name]["indices"])
            stats["payload_bytes"] += tensor_nbytes(outlier_passthrough[name]["values"])

    stats["high_precision_extra_bytes"] = int(high_precision_info["selected_extra_bytes"])
    stats["block_prune_estimated_bytes_saved"] = int(tensor_sweep_info["compressed_bytes_saved"])
    obj = {
        "__quant_format__": f"mixed_precision_{bits}bit_v2",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    if outlier_passthrough:
        obj["outlier_passthrough"] = outlier_passthrough
    return obj, stats, {**high_precision_info, "tensor_sweep": tensor_sweep_info, "block_prune": tensor_sweep_info}


def dequantize_state_dict_mixed_precision(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    qmeta = obj.get("qmeta", {})
    outlier_passthrough = obj.get("outlier_passthrough", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        out[name] = dequantize_quantized_tensor(q, s, dtype, qmeta.get(name, {}))
        out[name] = apply_outlier_passthrough(out[name], outlier_passthrough.get(name))
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

SHARD_HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_BYTES = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
SHARD_TOKEN_BYTES = np.dtype("<u2").itemsize


def _read_shard_num_tokens(file: Path) -> int:
    header = np.fromfile(file, dtype="<i4", count=SHARD_HEADER_INTS)
    if header.size != SHARD_HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = SHARD_HEADER_BYTES + num_tokens * SHARD_TOKEN_BYTES
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    return num_tokens


def load_data_shard(file: Path) -> Tensor:
    num_tokens = _read_shard_num_tokens(file)
    tokens_np = np.memmap(file, dtype="<u2", mode="c", offset=SHARD_HEADER_BYTES, shape=(num_tokens,))
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np)


def _available_loader_workers() -> int:
    override = int(os.environ.get("TOKEN_LOADER_WORKERS", "0"))
    if override > 0:
        return override
    try:
        available = len(os.sched_getaffinity(0))
    except Exception:
        available = os.cpu_count() or 1
    return max(int(available), 1)


def _resolve_loader_worker_count(env_name: str) -> int:
    override = int(os.environ.get(env_name, "0"))
    if override > 0:
        return override
    return _available_loader_workers()


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str, random_offset_tokens: int = 0, seed: int = 1337):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.random_offset_tokens = max(int(random_offset_tokens), 0)
        self.rng = random.Random(seed)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
        preload_workers = _resolve_loader_worker_count("TOKEN_LOADER_PRELOAD_WORKERS")
        self._preload_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=preload_workers) if len(self.files) > 1 else None
        )
        self._next_tokens_future: Future[Tensor] | None = None
        self._next_file_idx: int | None = None
        self.last_take_info: dict[str, object] = {
            "mode": "init",
            "file_idx": 0,
            "file": str(self.files[0]),
            "start_offset": 0,
            "requested_tokens": 0,
        }
        self._schedule_next_shard_preload()

    def __del__(self) -> None:
        if self._preload_executor is not None:
            self._preload_executor.shutdown(wait=False, cancel_futures=True)

    def state_dict(self) -> dict[str, object]:
        return {
            "file_idx": int(self.file_idx),
            "pos": int(self.pos),
            "rng_state": self.rng.getstate(),
            "last_take_info": copy.deepcopy(self.last_take_info),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.file_idx = int(state["file_idx"])
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = int(state["pos"])
        self.rng.setstate(state["rng_state"])  # type: ignore[arg-type]
        self.last_take_info = copy.deepcopy(state.get("last_take_info", self.last_take_info))
        self._next_tokens_future = None
        self._next_file_idx = None
        self._schedule_next_shard_preload()

    def _schedule_next_shard_preload(self) -> None:
        if self._preload_executor is None:
            return
        next_file_idx = (self.file_idx + 1) % len(self.files)
        if self._next_tokens_future is not None and self._next_file_idx == next_file_idx:
            return
        self._next_file_idx = next_file_idx
        self._next_tokens_future = self._preload_executor.submit(load_data_shard, self.files[next_file_idx])

    def _advance_file(self) -> None:
        next_file_idx = (self.file_idx + 1) % len(self.files)
        if self._next_tokens_future is not None and self._next_file_idx == next_file_idx:
            self.tokens = self._next_tokens_future.result()
        else:
            self.tokens = load_data_shard(self.files[next_file_idx])
        self.file_idx = next_file_idx
        self.pos = 0
        self._next_tokens_future = None
        self._next_file_idx = None
        self._schedule_next_shard_preload()

    def _skip_sequential(self, n: int) -> int:
        remaining = n
        files_touched = 0
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            self.pos += k
            remaining -= k
            files_touched += 1
        return files_touched

    def _collect_sequential(self, n: int) -> tuple[Tensor, int]:
        chunks: list[Tensor] = []
        remaining = n
        files_touched = 0
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
            files_touched += 1
        return (chunks[0] if len(chunks) == 1 else torch.cat(chunks), files_touched)

    def _take_sequential(self, n: int) -> Tensor:
        start_file_idx = self.file_idx
        start_file = str(self.files[self.file_idx])
        start_offset = self.pos
        chunk, files_touched = self._collect_sequential(n)
        self.last_take_info = {
            "mode": "sequential",
            "file_idx": int(start_file_idx),
            "file": start_file,
            "start_offset": int(start_offset),
            "requested_tokens": int(n),
            "files_touched": int(files_touched),
        }
        return chunk

    def _take_sequential_distributed(self, local_n: int, rank: int, world_size: int) -> Tensor:
        total_n = local_n * world_size
        start_file_idx = self.file_idx
        start_file = str(self.files[self.file_idx])
        start_offset = self.pos
        rank_offset = rank * local_n
        files_touched = self._skip_sequential(rank_offset)
        chunk, collected_files = self._collect_sequential(local_n)
        files_touched += collected_files
        files_touched += self._skip_sequential(total_n - rank_offset - local_n)
        self.last_take_info = {
            "mode": "sequential_distributed",
            "file_idx": int(start_file_idx),
            "file": start_file,
            "start_offset": int(start_offset),
            "requested_tokens": int(total_n),
            "returned_tokens": int(local_n),
            "rank_offset": int(rank_offset),
            "world_size": int(world_size),
            "files_touched": int(files_touched),
        }
        return chunk

    def take(self, n: int) -> Tensor:
        if self.random_offset_tokens <= 0:
            return self._take_sequential(n)

        while True:
            max_start = self.tokens.numel() - n
            if max_start < 0:
                self._advance_file()
                continue
            start_low = min(self.pos, max_start)
            start_high = min(max_start, start_low + self.random_offset_tokens)
            start = self.rng.randint(start_low, start_high) if start_high > start_low else start_low
            self.pos = start + n
            self.last_take_info = {
                "mode": "random_offset",
                "file_idx": int(self.file_idx),
                "file": str(self.files[self.file_idx]),
                "start_offset": int(start),
                "requested_tokens": int(n),
                "start_low": int(start_low),
                "start_high": int(start_high),
            }
            return self.tokens[start : start + n]

    def take_distributed(self, local_n: int, rank: int, world_size: int) -> Tensor:
        if local_n <= 0:
            raise ValueError(f"local_n must be positive, got {local_n}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}")
        if self.random_offset_tokens <= 0:
            return self._take_sequential_distributed(local_n, rank, world_size)

        total_n = local_n * world_size
        rank_offset = rank * local_n
        while True:
            max_start = self.tokens.numel() - total_n
            if max_start < 0:
                self._advance_file()
                continue
            start_low = min(self.pos, max_start)
            start_high = min(max_start, start_low + self.random_offset_tokens)
            start = self.rng.randint(start_low, start_high) if start_high > start_low else start_low
            local_start = start + rank_offset
            self.pos = start + total_n
            self.last_take_info = {
                "mode": "random_offset_distributed",
                "file_idx": int(self.file_idx),
                "file": str(self.files[self.file_idx]),
                "start_offset": int(start),
                "local_start_offset": int(local_start),
                "requested_tokens": int(total_n),
                "returned_tokens": int(local_n),
                "rank_offset": int(rank_offset),
                "world_size": int(world_size),
                "start_low": int(start_low),
                "start_high": int(start_high),
            }
            return self.tokens[local_start : local_start + local_n]


class DistributedTokenLoader:
    # Each call advances the shared logical token stream by one global chunk, but only
    # materializes this rank's disjoint local span. The extra "+1" token lets us build
    # (x, y) by shifting without reading every other rank's tokens first.
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        bos_token_id: int = -1,
        random_offset_tokens: int = 0,
        seed: int = 1337,
        debug_static_shapes: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.bos_token_id = int(bos_token_id)
        self.debug_static_shapes = bool(debug_static_shapes)
        self.stream = TokenStream(pattern, random_offset_tokens=random_offset_tokens, seed=seed + rank)
        self.last_batch_info: dict[str, object] = {}
        self.prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._prefetched_batch: dict[str, object] | None = None
        self._prefetched_kind: str | None = None
        batch_prep_workers = _resolve_loader_worker_count("TOKEN_LOADER_BATCH_PREP_WORKERS")
        self._batch_prep_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=batch_prep_workers)
        self._prepared_batch_future: Future[dict[str, object]] | None = None
        self._prepared_kind: str | None = None
        self._prepared_request: tuple[int, int, int] | None = None

    def __del__(self) -> None:
        self._batch_prep_executor.shutdown(wait=False, cancel_futures=True)

    def state_dict(self) -> dict[str, object]:
        return {
            "stream": self.stream.state_dict(),
            "last_batch_info": copy.deepcopy(self.last_batch_info),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.stream.load_state_dict(state["stream"])  # type: ignore[arg-type]
        self.last_batch_info = copy.deepcopy(state.get("last_batch_info", {}))
        self._prefetched_batch = None
        self._prefetched_kind = None
        self._prepared_batch_future = None
        self._prepared_kind = None
        self._prepared_request = None

    def _prepare_batch(
        self,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
        packed: bool,
    ) -> dict[str, object]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        local = self.stream.take_distributed(per_rank_span, self.rank, self.world_size).to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        batch_info: dict[str, object] = {
            **self.stream.last_take_info,
            "rank": int(self.rank),
            "world_size": int(self.world_size),
            "local_span_start": int(self.rank * per_rank_span),
            "local_span_end": int((self.rank + 1) * per_rank_span),
            "local_tokens": int(local_tokens),
            "seq_len": int(seq_len),
            "grad_accum_steps": int(grad_accum_steps),
        }
        loss_mask = None
        if packed:
            # Keep packed training on compile-stable shapes by using the same fixed
            # grid as the standard loader and masking only cross-document targets.
            # This preserves BOS-delimited document boundaries without creating
            # variable numbers of packed rows from short documents.
            loss_mask = y != self.bos_token_id
            batch_info.update(
                {
                    "batch_mode": "packed",
                    "packed_sequences": int(x.size(0)),
                    "packed_max_seq_len": int(seq_len),
                    "packed_valid_tokens": int(loss_mask.sum().item()),
                    "packed_boundary_drop_tokens": int(max(local_tokens - int(loss_mask.sum().item()), 0)),
                    "packed_fixed_shape": True,
                }
            )
        if self.debug_static_shapes:
            expected_rows = local_tokens // seq_len
            enforce_static_shape(x, (expected_rows, seq_len), name="packed_x" if packed else "x")
            enforce_static_shape(y, (expected_rows, seq_len), name="packed_y" if packed else "y")
            if packed and loss_mask is not None:
                enforce_static_shape(loss_mask, (expected_rows, seq_len), name="packed_loss_mask")
                batch_info["packed_expected_shape"] = [int(expected_rows), int(seq_len)]
                batch_info["packed_shape_trace"] = {
                    "x": list(trace_shape(x)),
                    "y": list(trace_shape(y)),
                    "loss_mask": list(trace_shape(loss_mask)),
                }
        return {
            "x_cpu": x.contiguous(),
            "y_cpu": y.contiguous(),
            "loss_mask_cpu": loss_mask.contiguous() if loss_mask is not None else None,
            "batch_info": batch_info,
        }

    def _copy_prepared_batch_to_device(self, prepared: dict[str, object]) -> dict[str, object]:
        x_cpu = prepared["x_cpu"]
        y_cpu = prepared["y_cpu"]
        loss_mask_cpu = prepared["loss_mask_cpu"]
        assert isinstance(x_cpu, Tensor) and isinstance(y_cpu, Tensor)
        assert loss_mask_cpu is None or isinstance(loss_mask_cpu, Tensor)

        x_host = x_cpu.pin_memory()
        y_host = y_cpu.pin_memory()
        x_host.copy_(x_cpu)
        y_host.copy_(y_cpu)
        loss_mask_host = None
        if loss_mask_cpu is not None:
            loss_mask_host = loss_mask_cpu.pin_memory()
            loss_mask_host.copy_(loss_mask_cpu)

        if self.prefetch_stream is None:
            x_dev = x_host.to(self.device, non_blocking=False)
            y_dev = y_host.to(self.device, non_blocking=False)
            loss_mask_dev = loss_mask_host.to(self.device, non_blocking=False) if loss_mask_host is not None else None
        else:
            with torch.cuda.stream(self.prefetch_stream):
                x_dev = x_host.to(self.device, non_blocking=True)
                y_dev = y_host.to(self.device, non_blocking=True)
                loss_mask_dev = (
                    loss_mask_host.to(self.device, non_blocking=True) if loss_mask_host is not None else None
                )

        return {
            "x": x_dev,
            "y": y_dev,
            "loss_mask": loss_mask_dev,
            "batch_info": prepared["batch_info"],
            "x_host": x_host,
            "y_host": y_host,
            "loss_mask_host": loss_mask_host,
        }

    def _submit_prepared_batch(
        self,
        kind: str,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> None:
        request = (global_tokens, seq_len, grad_accum_steps)
        if (
            self._prepared_batch_future is not None
            and self._prepared_kind == kind
            and self._prepared_request == request
        ):
            return
        packed = kind == "packed"
        self._prepared_batch_future = self._batch_prep_executor.submit(
            self._prepare_batch, global_tokens, seq_len, grad_accum_steps, packed
        )
        self._prepared_kind = kind
        self._prepared_request = request

    def _materialize_prefetched_batch(
        self,
        kind: str,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> None:
        request = (global_tokens, seq_len, grad_accum_steps)
        if (
            self._prepared_batch_future is None
            or self._prepared_kind != kind
            or self._prepared_request != request
        ):
            self._submit_prepared_batch(kind, global_tokens, seq_len, grad_accum_steps)
        prepared = self._prepared_batch_future.result()
        self._prepared_batch_future = None
        self._prepared_kind = None
        self._prepared_request = None
        self._prefetched_batch = self._copy_prepared_batch_to_device(prepared)
        self._prefetched_kind = kind

    def _get_prefetched_batch(
        self,
        kind: str,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self._prefetched_batch is None or self._prefetched_kind != kind:
            self._materialize_prefetched_batch(kind, global_tokens, seq_len, grad_accum_steps)
        batch = self._prefetched_batch
        assert batch is not None
        self._prefetched_batch = None
        self._prefetched_kind = None
        if self.prefetch_stream is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.prefetch_stream)
            batch["x"].record_stream(current_stream)
            batch["y"].record_stream(current_stream)
            if batch["loss_mask"] is not None:
                batch["loss_mask"].record_stream(current_stream)
        self.last_batch_info = batch["batch_info"]  # type: ignore[assignment]
        self._submit_prepared_batch(kind, global_tokens, seq_len, grad_accum_steps)
        return batch["x"], batch["y"], batch["loss_mask"]  # type: ignore[return-value]

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        x, y, _ = self._get_prefetched_batch("standard", global_tokens, seq_len, grad_accum_steps)
        return x, y

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

def fake_quantize_tensor(t: Tensor, bits: int) -> Tensor:
    if bits <= 0:
        return t
    t32 = t.float()
    if t32.numel() == 0:
        return t
    qmax = float((1 << (bits - 1)) - 1)
    if qmax <= 0:
        return t
    if t32.ndim == 2:
        clip_abs = t32.abs().amax(dim=1, keepdim=True)
    else:
        clip_abs = t32.abs().amax()
    # Preserve genuinely small tensors (for example tied embeddings at init)
    # instead of forcing an absolute quant step of 1/qmax, which can zero the
    # entire tensor when its dynamic range is tiny.
    scale = torch.where(clip_abs > 0, clip_abs / qmax, torch.ones_like(clip_abs))
    clipped = torch.clamp(t32, -clip_abs, clip_abs)
    quantized = torch.clamp(torch.round(clipped / scale), -qmax, qmax) * scale
    return (t32 + (quantized - t32).detach()).to(dtype=t.dtype)


def row_norm_regularization(module: nn.Module) -> Tensor:
    penalty = None
    for param in module.parameters():
        if param.ndim != 2:
            continue
        rows = param.float()
        row_rms = rows.square().mean(dim=1).add_(1e-8).sqrt()
        log_row_rms = row_rms.log()
        value = (log_row_rms - log_row_rms.mean()).square().mean()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return penalty


def outlier_regularization(module: nn.Module, threshold: float) -> Tensor:
    penalty = None
    clamped_threshold = max(float(threshold), 1e-4)
    for param in module.parameters():
        if param.ndim < 2:
            continue
        w = param.float()
        scale = w.square().mean().add_(1e-8).sqrt()
        excess = (w.abs() / scale - clamped_threshold).clamp_min(0.0)
        value = excess.square().mean()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return penalty


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_quant_bits = 0

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        weight = self.weight
        if self.fake_quant_bits > 0:
            weight = fake_quantize_tensor(weight, self.fake_quant_bits)
        return F.linear(x, weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    # Supports NTK-aware scaling when evaluated beyond the training context.
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        train_seq_len: int = 1024,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self.base = base
        self.train_seq_len = max(int(train_seq_len), 1)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, None]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                adjusted_base = self.base * (scale ** (self.dim / max(self.dim - 2, 1)))
                inv_freq = 1.0 / (
                    adjusted_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
                )
            else:
                inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype), None


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, scale: Tensor | None = None, inverse_scale: bool = False) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    if scale is None:
        return rot
    full_scale = torch.cat((scale, scale), dim=-1)
    return rot / full_scale if inverse_scale else rot * full_scale


def apply_partial_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rope_dims: int,
    scale: Tensor | None = None,
    inverse_scale: bool = False,
) -> Tensor:
    rope_dims = max(int(rope_dims), 0)
    if rope_dims <= 0:
        return x
    if rope_dims >= x.size(-1):
        return apply_rotary_emb(x, cos, sin, scale=scale, inverse_scale=inverse_scale)
    rot_x = x[..., :rope_dims]
    tail_x = x[..., rope_dims:]
    rot = apply_rotary_emb(rot_x, cos[..., : rope_dims // 2], sin[..., : rope_dims // 2], scale=scale, inverse_scale=inverse_scale)
    return torch.cat((rot, tail_x), dim=-1)


def _get_alibi_slopes(num_heads: int) -> Tensor:
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads), dtype=torch.float32)
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra = _get_alibi_slopes(2 * closest_power_of_2).tolist()[0::2][: num_heads - closest_power_of_2]
    return torch.tensor(slopes + extra, dtype=torch.float32)


def init_q_gain_vector(num_heads: int, base_gain: float) -> Tensor:
    if num_heads <= 1 or base_gain <= 0.0:
        return torch.full((max(int(num_heads), 1),), float(base_gain), dtype=torch.float32)
    head_pos = torch.linspace(-1.0, 1.0, steps=int(num_heads), dtype=torch.float32)
    multipliers = torch.exp(0.12 * head_pos)
    return float(base_gain) * multipliers


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int = 0, eps: float = 1e-5, gate_init: float = -2.0):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.base_log_scale = nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))
        self.cond_proj = CastedLinear(condition_dim, self.dim, bias=False) if condition_dim > 0 else None
        if self.cond_proj is not None:
            self.cond_proj._zero_init = True
        self.cond_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32)) if self.cond_proj is not None else None

    def forward(self, x: Tensor, condition: Tensor | None = None) -> Tensor:
        x_norm = F.rms_norm(x, (x.size(-1),), eps=self.eps)
        log_scale = 0.5 * torch.tanh(self.base_log_scale.to(dtype=x.dtype))[None, None, :]
        if (
            condition is not None
            and self.cond_proj is not None
            and condition.ndim == x.ndim
            and condition.size(0) == x.size(0)
            and condition.size(1) == x.size(1)
        ):
            cond_delta = torch.tanh(self.cond_proj(condition).to(dtype=x.dtype))
            cond_gain = torch.sigmoid(self.cond_gate.to(dtype=x.dtype))
            log_scale = log_scale + cond_gain * cond_delta
        return x_norm * torch.exp(log_scale)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_alibi: bool,
        use_residual_attention: bool,
        residual_attention_gain_init: float,
        helper_path_full_steps: int,
        helper_path_decay_end_step: int,
        train_seq_len: int = 1024,
        use_flash_attn_3: bool = False,
        rope_dims: int = 0,
        debug_static_shapes: bool = False,
        residual_attention_state_bits: int = 8,
        manual_attention_query_chunk_size: int = 128,
        manual_attention_key_chunk_size: int = 128,
        prior_gate_init: float = DEFAULT_PRIOR_GATE_INIT,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_alibi = use_alibi
        self.use_residual_attention = use_residual_attention
        self.helper_path_full_steps = max(int(helper_path_full_steps), 0)
        self.helper_path_decay_end_step = max(int(helper_path_decay_end_step), self.helper_path_full_steps)
        self.use_flash_attn_3 = bool(use_flash_attn_3 and HAS_FLASH_ATTN_3)
        self.debug_static_shapes = bool(debug_static_shapes)
        self.residual_attention_state_bits = max(int(residual_attention_state_bits), 0)
        self.manual_attention_query_chunk_size = max(int(manual_attention_query_chunk_size), 0)
        self.manual_attention_key_chunk_size = max(int(manual_attention_key_chunk_size), 0)
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_dims = self.head_dim if rope_dims <= 0 else min(int(rope_dims), self.head_dim)
        if self.rope_dims % 2 != 0:
            raise ValueError("rope_dims must be even")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(init_q_gain_vector(num_heads, qk_gain_init))
        self.residual_attention_gain = (
            nn.Parameter(torch.full((num_heads,), residual_attention_gain_init, dtype=torch.float32))
            if use_residual_attention
            else None
        )
        self.rotary = Rotary(
            self.rope_dims,
            base=rope_base,
            train_seq_len=train_seq_len,
        )
        self.register_buffer("alibi_slopes", _get_alibi_slopes(num_heads), persistent=False)
        self.register_buffer("helper_path_schedule_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self._alibi_bias_cache: Tensor | None = None
        self._alibi_bias_seq_len = 0
        self._manual_causal_mask_cache: Tensor | None = None
        self._manual_causal_mask_shape: tuple[object, ...] = (0, 0)
        self.fast_path = not self.use_alibi and not self.use_residual_attention

    @torch._dynamo.disable
    def _flash_attn_3(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if flash_attn_3_func is None:
            raise RuntimeError("FA3 path requested but flash_attn_interface is unavailable")
        return flash_attn_3_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            causal=True,
        ).transpose(1, 2).contiguous()

    def _get_alibi_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if (
            self._alibi_bias_cache is None
            or self._alibi_bias_seq_len != seq_len
            or self._alibi_bias_cache.device != device
        ):
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            rel = pos[None, :] - pos[:, None]
            rel = rel.clamp_min(0.0)
            bias = -self.alibi_slopes.to(device=device)[:, None, None] * rel[None, :, :]
            self._alibi_bias_cache = bias.unsqueeze(0)
            self._alibi_bias_seq_len = seq_len
        return self._alibi_bias_cache.to(dtype=dtype)

    def set_helper_path_train_step(self, step: int) -> None:
        mult = linear_anneal_multiplier(step, self.helper_path_full_steps, self.helper_path_decay_end_step)
        self.helper_path_schedule_mult_buffer.fill_(float(min(max(mult, 0.0), 1.0)))

    def _helper_path_schedule_mult(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.helper_path_schedule_mult_buffer.to(device=device, dtype=dtype)

    def _get_manual_causal_mask(self, seq_len: int, total_k: int, device: torch.device) -> Tensor:
        cache_key = (
            seq_len,
            total_k,
        )
        if (
            self._manual_causal_mask_cache is None
            or self._manual_causal_mask_shape != cache_key
            or self._manual_causal_mask_cache.device != device
        ):
            mem = max(total_k - seq_len, 0)
            q_pos = torch.arange(seq_len, device=device)
            k_pos = torch.arange(seq_len, device=device)
            causal = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).triu(1)
            token_mask = causal
            if mem > 0:
                full_mask = torch.zeros((seq_len, total_k), device=device, dtype=torch.bool)
                full_mask[:, mem:] = token_mask
            else:
                full_mask = token_mask
            self._manual_causal_mask_cache = full_mask
            self._manual_causal_mask_shape = cache_key
        return self._manual_causal_mask_cache

    def _get_manual_attention_bias_chunk(
        self,
        seq_len: int,
        total_k: int,
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        attn_bias = None
        q_chunk = q_end - q_start
        k_chunk = k_end - k_start
        mem = max(total_k - seq_len, 0)
        token_global_start = max(k_start, mem)
        token_global_end = max(min(k_end, total_k), mem)
        token_cols_start = token_global_start - k_start
        token_cols_end = token_global_end - k_start
        token_key_start = token_global_start - mem
        token_key_end = token_global_end - mem

        def add_component(component: Tensor) -> None:
            nonlocal attn_bias
            attn_bias = component if attn_bias is None else attn_bias + component

        def wrap_token_component(token_component: Tensor) -> Tensor:
            if token_cols_start == 0 and token_cols_end == k_chunk:
                return token_component
            component = torch.zeros((1, self.num_heads, q_chunk, k_chunk), device=device, dtype=dtype)
            component[:, :, :, token_cols_start:token_cols_end] = token_component
            return component

        if self.use_alibi and token_key_end > token_key_start:
            q_pos = torch.arange(q_start, q_end, device=device, dtype=torch.float32)[:, None]
            k_pos = torch.arange(token_key_start, token_key_end, device=device, dtype=torch.float32)[None, :]
            rel = (k_pos - q_pos).clamp_min(0.0)
            token_component = (
                -self.alibi_slopes.to(device=device)[:, None, None] * rel[None, :, :]
            ).unsqueeze(0).to(dtype=dtype)
            add_component(wrap_token_component(token_component))
        return attn_bias

    @torch._dynamo.disable
    def _sdpa_attention_with_bias(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        seqlen = q.size(-2)
        attn_bias = None
        if self.use_alibi:
            attn_bias = self._get_alibi_bias(seqlen, q.device, dtype)
        if attn_bias is None:
            return self._scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        causal_mask = torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool).triu(1)
        causal_bias = torch.zeros((1, 1, seqlen, seqlen), device=q.device, dtype=dtype)
        causal_bias = causal_bias.masked_fill(causal_mask[None, None, :, :], torch.finfo(dtype).min)
        attn_bias = attn_bias + causal_bias
        # Trainable additive bias paths should use math SDPA so gradients with
        # respect to the bias tensor are preserved reliably across backends.
        if sdpa_kernel is not None and SDPBackend is not None:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return self._scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
        return self._scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = True,
    ) -> Tensor:
        use_gqa = self.num_kv_heads != self.num_heads and k.size(1) != q.size(1)
        use_explicit_bias = attn_mask is not None
        out_dtype = q.dtype

        def run_sdpa() -> Tensor:
            q_sdpa = q
            k_sdpa = k
            v_sdpa = v
            attn_mask_sdpa = attn_mask
            if use_explicit_bias:
                # Preserve additive-bias gradients more reliably by running the
                # explicit-bias SDPA path in fp32 and avoiding enable_gqa.
                q_sdpa = q.float()
                k_sdpa = k.float()
                v_sdpa = v.float()
                attn_mask_sdpa = attn_mask.float() if attn_mask is not None else None
            if use_gqa:
                repeat = self.num_heads // self.num_kv_heads
                if use_explicit_bias:
                    k_sdpa = k_sdpa.repeat_interleave(repeat, dim=1)
                    v_sdpa = v_sdpa.repeat_interleave(repeat, dim=1)
                else:
                    try:
                        return F.scaled_dot_product_attention(
                            q_sdpa,
                            k_sdpa,
                            v_sdpa,
                            attn_mask=attn_mask_sdpa,
                            is_causal=is_causal,
                            enable_gqa=True,
                        ).to(dtype=out_dtype)
                    except TypeError:
                        k_sdpa = k_sdpa.repeat_interleave(repeat, dim=1)
                        v_sdpa = v_sdpa.repeat_interleave(repeat, dim=1)
            return F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask_sdpa,
                is_causal=is_causal,
            ).to(dtype=out_dtype)

        try:
            return run_sdpa()
        except RuntimeError as exc:
            if "Invalid backend" not in str(exc) or sdpa_kernel is None or SDPBackend is None:
                raise
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return run_sdpa()

    def _pack_residual_attention_state(self, logits: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        state = logits.detach()
        bits = self.residual_attention_state_bits
        if bits <= 0 or bits >= 16:
            return state
        max_q = (1 << (bits - 1)) - 1
        scale = state.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / float(max_q)
        quantized = torch.clamp(torch.round(state.float() / scale), -max_q, max_q).to(torch.int8)
        return quantized, scale.to(dtype=torch.float16)

    def _residual_attention_state_slice(
        self,
        state: Tensor | tuple[Tensor, Tensor] | None,
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor | None:
        if state is None:
            return None
        if isinstance(state, tuple):
            quantized, scale = state
            return (
                quantized[:, :, q_start:q_end, k_start:k_end].to(device=device, dtype=torch.float32)
                * scale[:, :, q_start:q_end, :].to(device=device, dtype=torch.float32)
            ).to(dtype=dtype)
        return state[:, :, q_start:q_end, k_start:k_end].to(device=device, dtype=dtype)

    @torch._dynamo.disable
    def _manual_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev_attn_logits: Tensor | tuple[Tensor, Tensor] | None,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor] | None]:
        bsz, _, seqlen, _ = q.shape
        if self.debug_static_shapes:
            q = ensure_contiguous_lastdim(q)
            k = ensure_contiguous_lastdim(k)
            v = ensure_contiguous_lastdim(v)
            enforce_static_shape(q, (bsz, self.num_heads, seqlen, self.head_dim), name="attn_q")
            enforce_static_shape(k, (bsz, self.num_heads, k.size(-2), self.head_dim), name="attn_k")
            enforce_static_shape(v, (bsz, self.num_heads, v.size(-2), self.head_dim), name="attn_v")
            if isinstance(prev_attn_logits, Tensor):
                enforce_static_shape(
                    prev_attn_logits,
                    (bsz, self.num_heads, seqlen, k.size(-2)),
                    name="prev_attn_logits",
                )
        total_k = k.size(-2)
        scale = self.head_dim**-0.5
        full_mask = self._get_manual_causal_mask(seqlen, total_k, q.device)
        residual_gain = None
        if prev_attn_logits is not None and self.use_residual_attention and self.residual_attention_gain is not None:
            residual_gain = torch.sigmoid(self.residual_attention_gain).to(dtype=dtype)[None, :, None, None]
            residual_gain = residual_gain * self._helper_path_schedule_mult(device=q.device, dtype=dtype)
        q_chunk_size = self.manual_attention_query_chunk_size if self.manual_attention_query_chunk_size > 0 else seqlen
        k_chunk_size = self.manual_attention_key_chunk_size if self.manual_attention_key_chunk_size > 0 else total_k
        use_chunked = q_chunk_size < seqlen or k_chunk_size < total_k
        if not use_chunked:
            attn_bias = self._get_manual_attention_bias_chunk(
                seqlen,
                total_k,
                0,
                seqlen,
                0,
                total_k,
                q.device,
                dtype,
            )
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_bias is not None:
                logits = logits + attn_bias
            if residual_gain is not None:
                prev_logits = self._residual_attention_state_slice(
                    prev_attn_logits, 0, seqlen, 0, total_k, dtype=dtype, device=q.device
                )
                if prev_logits is not None:
                    logits = logits + residual_gain * prev_logits
            logits = logits.masked_fill(full_mask[None, None, :, :], float("-inf"))
            probs = torch.softmax(logits.float(), dim=-1).to(dtype=dtype)
            y = torch.matmul(probs, v)
            next_attn_logits = self._pack_residual_attention_state(logits) if self.use_residual_attention else None
        else:
            y = torch.empty_like(q)
            quantized_state = 0 < self.residual_attention_state_bits < 16
            next_attn_logits = (
                torch.empty(
                    (bsz, self.num_heads, seqlen, total_k),
                    device=q.device,
                    dtype=torch.int8 if quantized_state else q.dtype,
                )
                if self.use_residual_attention
                else None
            )
            next_attn_scales = (
                torch.empty((bsz, self.num_heads, seqlen, 1), device=q.device, dtype=torch.float16)
                if self.use_residual_attention and quantized_state
                else None
            )
            for q_start in range(0, seqlen, q_chunk_size):
                q_end = min(q_start + q_chunk_size, seqlen)
                q_chunk = q[:, :, q_start:q_end, :]
                running_max = torch.full(
                    (bsz, self.num_heads, q_end - q_start, 1),
                    float("-inf"),
                    device=q.device,
                    dtype=torch.float32,
                )
                running_denom = torch.zeros_like(running_max)
                running_out = torch.zeros(
                    (bsz, self.num_heads, q_end - q_start, self.head_dim),
                    device=q.device,
                    dtype=torch.float32,
                )
                for k_start in range(0, total_k, k_chunk_size):
                    k_end = min(k_start + k_chunk_size, total_k)
                    mask_chunk = full_mask[q_start:q_end, k_start:k_end]
                    logits_chunk = torch.matmul(q_chunk, k[:, :, k_start:k_end, :].transpose(-2, -1)) * scale
                    attn_bias_chunk = self._get_manual_attention_bias_chunk(
                        seqlen,
                        total_k,
                        q_start,
                        q_end,
                        k_start,
                        k_end,
                        q.device,
                        dtype,
                    )
                    if attn_bias_chunk is not None:
                        logits_chunk = logits_chunk + attn_bias_chunk
                    if residual_gain is not None:
                        prev_logits_chunk = self._residual_attention_state_slice(
                            prev_attn_logits,
                            q_start,
                            q_end,
                            k_start,
                            k_end,
                            dtype=dtype,
                            device=q.device,
                        )
                        if prev_logits_chunk is not None:
                            logits_chunk = logits_chunk + residual_gain * prev_logits_chunk
                    logits_chunk = logits_chunk.masked_fill(mask_chunk[None, None, :, :], float("-inf"))
                    if next_attn_logits is not None:
                        if next_attn_scales is None:
                            next_attn_logits[:, :, q_start:q_end, k_start:k_end] = logits_chunk.detach().to(
                                dtype=next_attn_logits.dtype
                            )
                        else:
                            packed_chunk = self._pack_residual_attention_state(logits_chunk)
                            if isinstance(packed_chunk, tuple):
                                q_chunk_state, scale_chunk = packed_chunk
                                next_attn_logits[:, :, q_start:q_end, k_start:k_end] = q_chunk_state
                                if k_start == 0:
                                    next_attn_scales[:, :, q_start:q_end, :] = scale_chunk
                            else:
                                next_attn_logits[:, :, q_start:q_end, k_start:k_end] = packed_chunk.to(dtype=torch.int8)
                    logits_fp32 = logits_chunk.float()
                    block_max = logits_fp32.amax(dim=-1, keepdim=True)
                    new_max = torch.maximum(running_max, block_max)
                    carry_scale = torch.where(
                        torch.isfinite(running_max),
                        torch.exp(running_max - new_max),
                        torch.zeros_like(new_max),
                    )
                    exp_chunk = torch.where(
                        torch.isfinite(logits_fp32),
                        torch.exp(logits_fp32 - new_max),
                        torch.zeros_like(logits_fp32),
                    )
                    running_denom = running_denom * carry_scale + exp_chunk.sum(dim=-1, keepdim=True)
                    running_out = running_out * carry_scale.to(dtype=running_out.dtype) + torch.matmul(
                        exp_chunk,
                        v[:, :, k_start:k_end, :].to(dtype=torch.float32),
                    )
                    running_max = new_max
                y[:, :, q_start:q_end, :] = (running_out / running_denom.clamp_min(1e-20)).to(dtype=dtype)
            if next_attn_scales is not None:
                next_attn_logits = (next_attn_logits, next_attn_scales)
        return y, next_attn_logits

    def _project_qkv(self, x: Tensor, q_gain_delta: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seqlen, _dim = x.shape
        q_proj = self.c_q(x)
        q = q_proj.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_proj = self.c_v(x)
        v = v_proj.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if not self.use_alibi:
            cos, sin, scale = self.rotary(seqlen, x.device, q.dtype)
            q = apply_partial_rotary_emb(q, cos, sin, self.rope_dims, scale=scale)
            k = apply_partial_rotary_emb(k, cos, sin, self.rope_dims, scale=scale, inverse_scale=True)
        q_gain = self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if q_gain_delta is not None:
            q_gain_delta = q_gain_delta.to(dtype=q.dtype)
            if q_gain_delta.ndim == 2:
                q_gain = q_gain * (1.0 + q_gain_delta[:, :, None, None])
            else:
                q_gain = q_gain * (1.0 + q_gain_delta.permute(0, 2, 1)[:, :, :, None])
        q = q * q_gain
        return q, k, v

    def _repeat_gqa_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.num_kv_heads == self.num_heads:
            return k, v
        repeat = self.num_heads // self.num_kv_heads
        return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)

    def forward_dense(self, q: Tensor, k: Tensor, v: Tensor, dtype: torch.dtype) -> Tensor:
        if self.use_flash_attn_3 and not self.use_alibi:
            return self._flash_attn_3(q, k, v)
        return self._sdpa_attention_with_bias(q, k, v, dtype)

    @torch._dynamo.disable
    def forward_manual(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev_attn_logits: Tensor | tuple[Tensor, Tensor] | None,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor] | None]:
        return self._manual_attention(q, k, v, prev_attn_logits, dtype)

    def forward(
        self,
        x: Tensor,
        prev_attn_logits: Tensor | tuple[Tensor, Tensor] | None = None,
        q_gain_delta: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor] | None]:
        bsz, seqlen, dim = x.shape
        q, k, v = self._project_qkv(x, q_gain_delta=q_gain_delta)
        next_attn_logits = None
        if self.use_residual_attention:
            k, v = self._repeat_gqa_kv(k, v)
            y, next_attn_logits = self.forward_manual(q, k, v, prev_attn_logits, q.dtype)
        else:
            y = self.forward_dense(q, k, v, q.dtype)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), next_attn_logits

    def forward_simple(
        self,
        x: Tensor,
        q_gain_delta: Tensor | None = None,
    ) -> Tensor:
        if not self.fast_path:
            raise RuntimeError("forward_simple requested for a non-fast attention block")
        bsz, seqlen, dim = x.shape
        q, k, v = self._project_qkv(x, q_gain_delta=q_gain_delta)
        y = self.forward_dense(q, k, v, q.dtype)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int, hidden_dim: int = 0):
        super().__init__()
        inner_dim = hidden_dim if hidden_dim > 0 else mlp_mult * dim
        self.fc = CastedLinear(dim, inner_dim, bias=False)
        self.proj = CastedLinear(inner_dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SmearGate(nn.Module):
    def __init__(
        self,
        dim: int,
        gate_init: float = 0.0,
        lexical_scale_init: float = 1.0,
        confidence_gain_init: float = 0.0,
    ):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self.state_proj = CastedLinear(dim, 1, bias=False)
        self.lexical_proj = CastedLinear(dim, 1, bias=False)
        self.confidence_gain = nn.Parameter(torch.tensor(float(confidence_gain_init), dtype=torch.float32))
        self.lexical_scale = nn.Parameter(
            torch.tensor(inverse_softplus_scalar(float(lexical_scale_init)), dtype=torch.float32)
        )

    def forward(
        self,
        x: Tensor,
        lexical_features: Tensor | None = None,
        lexical_confidence: Tensor | None = None,
        lexical_scale_mult: Tensor | float = 1.0,
    ) -> Tensor:
        gate_logits = self.gate.to(dtype=x.dtype)
        gate_logits = gate_logits + self.state_proj(x).to(dtype=x.dtype)
        if lexical_features is not None and lexical_features.size(1) == x.size(1):
            lexical_scale = F.softplus(self.lexical_scale).to(dtype=x.dtype)
            lexical_scale = lexical_scale * torch.as_tensor(
                lexical_scale_mult,
                device=x.device,
                dtype=x.dtype,
            )
            gate_logits = gate_logits + lexical_scale * self.lexical_proj(lexical_features).to(dtype=x.dtype)
        if lexical_confidence is not None and lexical_confidence.size(1) == x.size(1):
            centered_confidence = 2.0 * lexical_confidence.to(dtype=x.dtype) - 1.0
            gate_logits = gate_logits + self.confidence_gain.to(dtype=x.dtype) * centered_confidence
        g = torch.sigmoid(gate_logits)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1.0 - g) * x + g * x_prev


class HashedNGramEmbedding(nn.Module):
    _mix_modulus = 2_147_483_629
    _mix_primes = (1_000_003, 1_000_033, 1_000_087, 1_000_103)
    _source_primes = (65_537, 65_579, 65_617, 65_659)
    _pad_constants = (911_382_323, 972_663_749, 1_034_039_771, 1_095_321_137)

    def __init__(
        self,
        vocab_size: int,
        bigram_dim: int,
        model_dim: int,
        order: int,
        scale_init: float = 0.2,
        prior_gate_init: float = DEFAULT_PRIOR_GATE_INIT,
    ):
        super().__init__()
        self.vocab_size = max(int(vocab_size), 1)
        self.order = max(int(order), 1)
        self.embed = nn.Embedding(self.vocab_size, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        # Keep lexical features large enough to influence early tied-embedding
        # logits before the shortcut branch has time to self-amplify.
        self.scale = nn.Parameter(torch.tensor(inverse_softplus_scalar(float(scale_init)), dtype=torch.float32))
        self.gate_logit = nn.Parameter(torch.tensor(float(prior_gate_init), dtype=torch.float32))
        self.register_buffer("influence_cap", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    def _hash(self, tokens: Tensor) -> Tensor:
        if self.vocab_size <= 1:
            return torch.zeros_like(tokens, dtype=torch.long)
        t = tokens.to(torch.int64)
        mix = torch.full_like(t, 1_461_597)
        for offset in range(self.order):
            shifted = torch.empty_like(t)
            if offset == 0:
                shifted.copy_(t)
            else:
                shifted[..., offset:] = t[..., :-offset]
                shifted[..., :offset] = self._pad_constants[(offset - 1) % len(self._pad_constants)]
            mix = torch.remainder(
                torch.bitwise_xor(
                    mix * self._mix_primes[offset % len(self._mix_primes)],
                    (shifted + 1) * self._source_primes[offset % len(self._source_primes)],
                ),
                self._mix_modulus,
            )
        return torch.remainder(mix, self.vocab_size).long()

    def forward(self, token_ids: Tensor) -> Tensor:
        hashed = self._hash(token_ids)
        h = self.embed(hashed)
        if self.proj is not None:
            h = self.proj(h)
        gate = torch.sigmoid(self.gate_logit).to(dtype=h.dtype) * self.influence_cap.to(device=h.device, dtype=h.dtype)
        return h * F.softplus(self.scale).to(dtype=h.dtype) * gate


def lexical_shortcut_confidence_from_logits(
    shortcut_logits: Tensor | None,
    confidence_scale: Tensor | None,
    confidence_bias: Tensor | None,
) -> Tensor | None:
    if shortcut_logits is None or confidence_scale is None or confidence_bias is None:
        return None
    # Let confidence-conditioned routing/smear pressure train the lexical
    # shortcut into producing larger top-2 margins instead of only learning a
    # more optimistic bias on top of weak logits.
    confidence_logits = shortcut_logits
    if confidence_logits.size(-1) <= 1:
        return torch.full(
            (*confidence_logits.shape[:-1], 1),
            0.5,
            dtype=confidence_logits.dtype,
            device=confidence_logits.device,
        )
    top2 = torch.topk(confidence_logits.float(), k=2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).unsqueeze(-1)
    confidence = bounded_sigmoid(
        confidence_scale.to(device=margin.device, dtype=margin.dtype) * margin
        + confidence_bias.to(device=margin.device, dtype=margin.dtype),
        margin=LEXICAL_CONFIDENCE_MARGIN,
    )
    return confidence.to(dtype=shortcut_logits.dtype)


class Block(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        rope_base: float,
        qk_gain_init: float,
        use_alibi: bool,
        use_residual_attention: bool,
        residual_attention_gain_init: float,
        helper_path_full_steps: int,
        helper_path_decay_end_step: int,
        residual_alpha: float,
        train_seq_len: int,
        use_flash_attn_3: bool,
        rope_dims: int,
        debug_static_shapes: bool,
        residual_attention_state_bits: int,
        manual_attention_query_chunk_size: int,
        manual_attention_key_chunk_size: int,
        ln_scale: bool,
        use_adaptive_rmsnorm: bool,
        adaptive_rmsnorm_gate_init: float,
        norm_condition_dim: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.residual_alpha = float(residual_alpha)
        norm_scale = (1.0 / math.sqrt(layer_idx + 1)) if ln_scale else 1.0
        self.register_buffer("norm_scale_buffer", torch.tensor(norm_scale, dtype=torch.float32), persistent=False)
        if use_adaptive_rmsnorm:
            self.attn_norm = AdaptiveRMSNorm(dim, condition_dim=norm_condition_dim, gate_init=adaptive_rmsnorm_gate_init)
            self.mlp_norm = AdaptiveRMSNorm(dim, condition_dim=norm_condition_dim, gate_init=adaptive_rmsnorm_gate_init)
        else:
            self.attn_norm = nn.RMSNorm(dim, elementwise_affine=False)
            self.mlp_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            use_alibi,
            use_residual_attention,
            residual_attention_gain_init,
            helper_path_full_steps,
            helper_path_decay_end_step,
            train_seq_len=train_seq_len,
            use_flash_attn_3=use_flash_attn_3,
            rope_dims=rope_dims,
            debug_static_shapes=debug_static_shapes,
            residual_attention_state_bits=residual_attention_state_bits,
            manual_attention_query_chunk_size=manual_attention_query_chunk_size,
            manual_attention_key_chunk_size=manual_attention_key_chunk_size,
        )
        self.helper_path_full_steps = max(int(helper_path_full_steps), 0)
        self.helper_path_decay_end_step = max(int(helper_path_decay_end_step), self.helper_path_full_steps)
        self.mlp = MLP(dim, mlp_mult, hidden_dim=mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.register_buffer("helper_path_schedule_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.fast_path = self.attn.fast_path

    def set_helper_path_train_step(self, step: int) -> None:
        mult = linear_anneal_multiplier(step, self.helper_path_full_steps, self.helper_path_decay_end_step)
        self.helper_path_schedule_mult_buffer.fill_(float(mult))

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        prev_attn_logits: Tensor | tuple[Tensor, Tensor] | None = None,
        q_gain_delta: Tensor | None = None,
        norm_condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor] | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        norm_scale = self.norm_scale_buffer.to(device=x.device, dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_normed = self.attn_norm(x, condition=norm_condition) if isinstance(self.attn_norm, AdaptiveRMSNorm) else self.attn_norm(x)
        attn_normed = attn_normed * norm_scale
        attn_out, next_attn_logits = self.attn(
            attn_normed,
            prev_attn_logits=prev_attn_logits,
            q_gain_delta=q_gain_delta,
        )
        x = x * self.residual_alpha + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x, condition=norm_condition) if isinstance(self.mlp_norm, AdaptiveRMSNorm) else self.mlp_norm(x)
        mlp_normed = mlp_normed * norm_scale
        x = x * self.residual_alpha + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        return x, next_attn_logits

    def forward_simple(
        self,
        x: Tensor,
        x0: Tensor,
        q_gain_delta: Tensor | None = None,
        norm_condition: Tensor | None = None,
    ) -> Tensor:
        if not self.fast_path:
            raise RuntimeError("forward_simple requested for a non-fast block")
        mix = self.resid_mix.to(dtype=x.dtype)
        norm_scale = self.norm_scale_buffer.to(device=x.device, dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        block_base = x
        attn_normed = self.attn_norm(x, condition=norm_condition) if isinstance(self.attn_norm, AdaptiveRMSNorm) else self.attn_norm(x)
        attn_normed = attn_normed * norm_scale
        attn_out = self.attn.forward_simple(attn_normed, q_gain_delta=q_gain_delta)
        x = x * self.residual_alpha + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_normed = self.mlp_norm(x, condition=norm_condition) if isinstance(self.mlp_norm, AdaptiveRMSNorm) else self.mlp_norm(x)
        mlp_normed = mlp_normed * norm_scale
        x = x * self.residual_alpha + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        tie_embeddings: bool,
        use_output_logit_bias: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        overtone_embed_init: bool,
        overtone_embed_power: float,
        overtone_embed_scale: float,
        resid_mix_phase_init: bool,
        resid_mix_phase_sharpness: float,
        resid_mix_phase_center: float,
        use_alibi: bool,
        num_shared_layers: int,
        shared_layer_repeats: int,
        shared_loop_gate_init: float,
        use_residual_attention: bool,
        residual_attention_gain_init: float,
        helper_path_full_steps: int,
        helper_path_decay_end_step: int,
        attention_kv_mode: str,
        use_smear_gate: bool,
        smear_gate_init: float,
        smear_lexical_scale_init: float,
        smear_confidence_gain_init: float,
        smear_lexical_proj_init_std: float,
        lexical_enable_step: int,
        lexical_ramp_steps: int,
        lexical_calibration_enable_step: int,
        lexical_calibration_ramp_steps: int,
        lexical_hidden_enable_step: int,
        lexical_hidden_ramp_steps: int,
        lexical_shortcut_enable_step: int,
        lexical_shortcut_ramp_steps: int,
        bigram_vocab_size: int,
        bigram_dim: int,
        hashed_ngram_order: int,
        hashed_ngram_init_std: float,
        bigram_hidden_scale_init: float,
        bigram_scale_init: float,
        bigram_hidden_min_mult: float,
        prior_gate_init: float,
        lexical_shortcut_scale_init: float,
        lexical_confidence_bias_init: float,
        use_pairwise_logit_prior: bool,
        pairwise_logit_rank: int,
        pairwise_logit_scale_init: float,
        pairwise_logit_gate_init: float,
        pairwise_logit_confidence_gain_init: float,
        use_online_doc_bias: bool,
        doc_bias_rank: int,
        doc_bias_scale_init: float,
        doc_bias_gate_init: float,
        use_confidence_branch_skip: bool,
        confidence_skip_margin: float,
        confidence_skip_sharpness: float,
        confidence_skip_lexical_min_keep: float,
        confidence_skip_pairwise_min_keep: float,
        confidence_skip_doc_bias_min_keep: float,
        use_residual_logit_adapter: bool,
        residual_logit_rank: int,
        residual_logit_scale_init: float,
        residual_logit_gate_init: float,
        lexical_batch_diversity_throttle: float,
        lexical_batch_repeat_throttle: float,
        lexical_batch_min_mult: float,
        shared_tail_output_gate: bool,
        shared_tail_output_init: float,
        shared_tail_enable_step: int,
        shared_tail_ramp_steps: int,
        shared_tail_max_mult: float,
        signed_skip_weights: bool,
        orthogonal_init: bool,
        mup_proj_init: bool,
        train_seq_len: int,
        use_flash_attn_3: bool,
        rope_dims: int,
        debug_static_shapes: bool,
        residual_attention_state_bits: int,
        manual_attention_query_chunk_size: int,
        manual_attention_key_chunk_size: int,
        ln_scale: bool,
        use_adaptive_rmsnorm: bool,
        adaptive_rmsnorm_gate_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.vocab_size = max(int(vocab_size), 1)
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.overtone_embed_init = overtone_embed_init
        self.overtone_embed_power = overtone_embed_power
        self.overtone_embed_scale = overtone_embed_scale
        self.resid_mix_phase_init = resid_mix_phase_init
        self.resid_mix_phase_sharpness = resid_mix_phase_sharpness
        self.resid_mix_phase_center = float(resid_mix_phase_center)
        self.use_alibi = use_alibi
        self.use_residual_attention = use_residual_attention
        self.residual_attention_gain_init = residual_attention_gain_init
        self.num_layers = max(int(num_layers), 0)
        self.helper_path_full_steps = max(int(helper_path_full_steps), 0)
        self.helper_path_decay_end_step = max(int(helper_path_decay_end_step), self.helper_path_full_steps)
        self.use_adaptive_rmsnorm = bool(use_adaptive_rmsnorm)
        self.adaptive_rmsnorm_gate_init = float(adaptive_rmsnorm_gate_init)
        self.num_heads = max(int(num_heads), 1)
        self.num_kv_heads = max(int(num_kv_heads), 1)
        self.attention_kv_mode = str(attention_kv_mode or "").strip().lower() or "gqa"
        self.use_smear_gate = use_smear_gate
        self.smear_gate_init = float(smear_gate_init)
        self.smear_lexical_scale_init = float(smear_lexical_scale_init)
        self.smear_confidence_gain_init = float(smear_confidence_gain_init)
        self.smear_lexical_proj_init_std = max(float(smear_lexical_proj_init_std), 0.0)
        self.lexical_enable_step = max(int(lexical_enable_step), 0)
        self.lexical_ramp_steps = max(int(lexical_ramp_steps), 0)
        self.lexical_calibration_enable_step = max(int(lexical_calibration_enable_step), 0)
        self.lexical_calibration_ramp_steps = max(int(lexical_calibration_ramp_steps), 0)
        self.lexical_hidden_enable_step = max(int(lexical_hidden_enable_step), 0)
        self.lexical_hidden_ramp_steps = max(int(lexical_hidden_ramp_steps), 0)
        self.lexical_shortcut_enable_step = max(int(lexical_shortcut_enable_step), 0)
        self.lexical_shortcut_ramp_steps = max(int(lexical_shortcut_ramp_steps), 0)
        self.bigram_hidden_scale_init = float(bigram_hidden_scale_init)
        self.bigram_scale_init = float(bigram_scale_init)
        self.bigram_hidden_min_mult = float(max(min(bigram_hidden_min_mult, 1.0), 0.0))
        self.prior_gate_init = float(prior_gate_init)
        self.use_pairwise_logit_prior = bool(use_pairwise_logit_prior)
        self.pairwise_logit_rank = max(int(pairwise_logit_rank), 0)
        self.pairwise_logit_scale_init = max(float(pairwise_logit_scale_init), 0.0)
        self.pairwise_logit_gate_init = float(pairwise_logit_gate_init)
        self.pairwise_logit_confidence_gain_init = float(pairwise_logit_confidence_gain_init)
        self.use_online_doc_bias = bool(use_online_doc_bias)
        self.doc_bias_rank = max(int(doc_bias_rank), 0)
        self.doc_bias_scale_init = max(float(doc_bias_scale_init), 0.0)
        self.doc_bias_gate_init = float(doc_bias_gate_init)
        self.use_confidence_branch_skip = bool(use_confidence_branch_skip)
        self.confidence_skip_margin = float(confidence_skip_margin)
        self.confidence_skip_sharpness = max(float(confidence_skip_sharpness), 0.0)
        self.confidence_skip_lexical_min_keep = min(max(float(confidence_skip_lexical_min_keep), 0.0), 1.0)
        self.confidence_skip_pairwise_min_keep = min(max(float(confidence_skip_pairwise_min_keep), 0.0), 1.0)
        self.confidence_skip_doc_bias_min_keep = min(max(float(confidence_skip_doc_bias_min_keep), 0.0), 1.0)
        self.use_residual_logit_adapter = bool(use_residual_logit_adapter)
        self.residual_logit_rank = max(int(residual_logit_rank), 0)
        self.residual_logit_scale_init = max(float(residual_logit_scale_init), 0.0)
        self.residual_logit_gate_init = float(residual_logit_gate_init)
        self.shared_tail_output_gate = bool(shared_tail_output_gate)
        self.shared_tail_output_init = float(shared_tail_output_init)
        self.shared_tail_enable_step = max(int(shared_tail_enable_step), 0)
        self.shared_tail_ramp_steps = max(int(shared_tail_ramp_steps), 0)
        self.shared_tail_max_mult = min(max(float(shared_tail_max_mult), 0.0), 1.0)
        self.signed_skip_weights = bool(signed_skip_weights)
        self.rope_dims = max(int(rope_dims), 0)
        self.ln_scale = bool(ln_scale)
        self.current_training_step = 0
        self.orthogonal_init = orthogonal_init
        self.mup_proj_init = mup_proj_init
        self.hashed_ngram_init_std = max(float(hashed_ngram_init_std), 0.0)
        self.lexical_shortcut_scale_init = float(lexical_shortcut_scale_init)
        self.lexical_confidence_bias_init = float(lexical_confidence_bias_init)
        self.lexical_batch_diversity_throttle = max(float(lexical_batch_diversity_throttle), 0.0)
        self.lexical_batch_repeat_throttle = max(float(lexical_batch_repeat_throttle), 0.0)
        self.lexical_batch_min_mult = min(max(float(lexical_batch_min_mult), 0.05), 1.0)
        self.norm_condition_dim = 0 if not self.use_adaptive_rmsnorm else 2
        self.fake_quant_bits = 0
        self.num_shared_layers = max(int(num_shared_layers), 0)
        self.shared_layer_repeats = max(int(shared_layer_repeats), 0)
        self.shared_loop_gate_init = shared_loop_gate_init
        residual_alpha = 1.0
        self.lexical_base_order = max(1, min(int(hashed_ngram_order), 2))
        self.use_output_logit_bias = bool(use_output_logit_bias)
        self.lexical_residual_order = 3 if int(hashed_ngram_order) >= 3 else 0
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = (
            HashedNGramEmbedding(
                bigram_vocab_size,
                bigram_dim,
                model_dim,
                order=self.lexical_base_order,
                scale_init=self.bigram_scale_init,
                prior_gate_init=self.prior_gate_init,
            )
            if bigram_vocab_size > 0
            else None
        )
        self.bigram_hidden_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(self.bigram_hidden_scale_init), dtype=torch.float32))
            if self.bigram is not None
            else None
        )
        self.lexical_residual_ngram = (
            HashedNGramEmbedding(
                bigram_vocab_size,
                bigram_dim,
                model_dim,
                order=self.lexical_residual_order,
                scale_init=self.bigram_scale_init,
                prior_gate_init=self.prior_gate_init,
            )
            if bigram_vocab_size > 0 and self.lexical_residual_order >= 3
            else None
        )
        self.smear = (
            SmearGate(
                model_dim,
                gate_init=self.smear_gate_init,
                lexical_scale_init=self.smear_lexical_scale_init,
                confidence_gain_init=self.smear_confidence_gain_init,
            )
            if use_smear_gate
            else None
        )
        self.lexical_shortcut_head = None if self.bigram is None else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lexical_shortcut_head is not None and not self.tie_embeddings:
            self.lexical_shortcut_head._zero_init = True
        self.lexical_residual_head = (
            None if self.lexical_residual_ngram is None else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lexical_residual_head is not None and not self.tie_embeddings:
            self.lexical_residual_head._zero_init = True
        self.lexical_shortcut_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(self.lexical_shortcut_scale_init), dtype=torch.float32))
            if self.bigram is not None
            else None
        )
        self.lexical_confidence_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(1.0), dtype=torch.float32))
            if self.bigram is not None
            else None
        )
        self.lexical_confidence_bias = (
            nn.Parameter(torch.tensor(self.lexical_confidence_bias_init, dtype=torch.float32))
            if self.bigram is not None
            else None
        )
        self.lexical_residual_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(self.lexical_shortcut_scale_init), dtype=torch.float32))
            if self.lexical_residual_ngram is not None
            else None
        )
        self.lexical_residual_confidence_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(1.0), dtype=torch.float32))
            if self.lexical_residual_ngram is not None
            else None
        )
        self.lexical_residual_confidence_bias = (
            nn.Parameter(torch.tensor(self.lexical_confidence_bias_init, dtype=torch.float32))
            if self.lexical_residual_ngram is not None
            else None
        )
        self.lexical_prior_gate_logit = (
            nn.Parameter(torch.tensor(self.prior_gate_init, dtype=torch.float32))
            if self.bigram is not None
            else None
        )
        self.register_buffer("lexical_cap", torch.ones((), dtype=torch.float32), persistent=False)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.shared_tail_gate = (
            nn.Parameter(torch.tensor(self.shared_tail_output_init, dtype=torch.float32))
            if self.shared_tail_output_gate
            else None
        )
        self.last_lexical_batch_mult = 1.0
        self.last_lexical_batch_unique_frac = 1.0
        self.last_lexical_batch_max_token_frac = 0.0
        self.register_buffer("lexical_batch_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("lexical_batch_unique_frac_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("lexical_batch_max_token_frac_buffer", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("helper_path_schedule_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("shared_tail_schedule_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("lexical_schedule_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("lexical_calibration_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("lexical_hidden_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("lexical_shortcut_mult_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.register_buffer("lexical_shortcut_health_gate_buffer", torch.ones((), dtype=torch.float32), persistent=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    i,
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    mlp_hidden,
                    rope_base,
                    qk_gain_init,
                    use_alibi,
                    use_residual_attention,
                    residual_attention_gain_init,
                    helper_path_full_steps=self.helper_path_full_steps,
                    helper_path_decay_end_step=self.helper_path_decay_end_step,
                    residual_alpha=residual_alpha,
                    train_seq_len=train_seq_len,
                    use_flash_attn_3=use_flash_attn_3,
                    rope_dims=self.rope_dims,
                    debug_static_shapes=debug_static_shapes,
                    residual_attention_state_bits=residual_attention_state_bits,
                    manual_attention_query_chunk_size=manual_attention_query_chunk_size,
                    manual_attention_key_chunk_size=manual_attention_key_chunk_size,
                    ln_scale=self.ln_scale,
                    use_adaptive_rmsnorm=self.use_adaptive_rmsnorm,
                    adaptive_rmsnorm_gate_init=self.adaptive_rmsnorm_gate_init,
                    norm_condition_dim=self.norm_condition_dim,
                )
                for i in range(num_layers)
            ]
        )
        self.shared_blocks = nn.ModuleList(
            [
                Block(
                    num_layers + i,
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    mlp_hidden,
                    rope_base,
                    qk_gain_init,
                    use_alibi,
                    use_residual_attention,
                    residual_attention_gain_init,
                    helper_path_full_steps=self.helper_path_full_steps,
                    helper_path_decay_end_step=self.helper_path_decay_end_step,
                    residual_alpha=residual_alpha,
                    train_seq_len=train_seq_len,
                    use_flash_attn_3=use_flash_attn_3,
                    rope_dims=self.rope_dims,
                    debug_static_shapes=debug_static_shapes,
                    residual_attention_state_bits=residual_attention_state_bits,
                    manual_attention_query_chunk_size=manual_attention_query_chunk_size,
                    manual_attention_key_chunk_size=manual_attention_key_chunk_size,
                    ln_scale=self.ln_scale,
                    use_adaptive_rmsnorm=self.use_adaptive_rmsnorm,
                    adaptive_rmsnorm_gate_init=self.adaptive_rmsnorm_gate_init,
                    norm_condition_dim=self.norm_condition_dim,
                )
                for i in range(self.num_shared_layers)
            ]
        )
        for block in list(self.blocks) + list(self.shared_blocks):
            if getattr(block, "attn", None) is not None:
                block.attn.set_helper_path_train_step(0)
        self.shared_loop_gates = None
        self.final_norm = nn.RMSNorm(model_dim, elementwise_affine=False)
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.output_logit_bias = (
            nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32)) if self.use_output_logit_bias else None
        )
        pairwise_prior_enabled = self.use_pairwise_logit_prior and self.pairwise_logit_rank > 0
        self.pairwise_prev_embed = (
            nn.Embedding(vocab_size, self.pairwise_logit_rank)
            if pairwise_prior_enabled
            else None
        )
        self.pairwise_logit_head = (
            CastedLinear(self.pairwise_logit_rank, vocab_size, bias=False)
            if pairwise_prior_enabled
            else None
        )
        self.pairwise_logit_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(self.pairwise_logit_scale_init), dtype=torch.float32))
            if pairwise_prior_enabled
            else None
        )
        self.pairwise_logit_gate = (
            nn.Parameter(torch.tensor(self.pairwise_logit_gate_init, dtype=torch.float32))
            if pairwise_prior_enabled
            else None
        )
        self.pairwise_logit_hidden_gate = (
            nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
            if pairwise_prior_enabled
            else None
        )
        self.pairwise_logit_confidence_gain = (
            nn.Parameter(torch.tensor(self.pairwise_logit_confidence_gain_init, dtype=torch.float32))
            if pairwise_prior_enabled
            else None
        )
        doc_bias_enabled = self.use_online_doc_bias and self.doc_bias_rank > 0
        self.doc_bias_embed = (
            nn.Embedding(vocab_size, self.doc_bias_rank)
            if doc_bias_enabled
            else None
        )
        self.doc_bias_head = (
            CastedLinear(self.doc_bias_rank, vocab_size, bias=False)
            if doc_bias_enabled
            else None
        )
        if self.doc_bias_head is not None:
            self.doc_bias_head._zero_init = True
        self.doc_bias_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(self.doc_bias_scale_init), dtype=torch.float32))
            if doc_bias_enabled
            else None
        )
        self.doc_bias_gate = (
            nn.Parameter(torch.tensor(self.doc_bias_gate_init, dtype=torch.float32))
            if doc_bias_enabled
            else None
        )
        residual_logit_enabled = self.use_residual_logit_adapter and self.residual_logit_rank > 0
        self.residual_logit_down = (
            CastedLinear(model_dim, self.residual_logit_rank, bias=False)
            if residual_logit_enabled
            else None
        )
        self.residual_logit_up = (
            CastedLinear(self.residual_logit_rank, vocab_size, bias=False)
            if residual_logit_enabled
            else None
        )
        if self.residual_logit_up is not None:
            self.residual_logit_up._zero_init = True
        self.residual_logit_scale = (
            nn.Parameter(torch.tensor(inverse_softplus_scalar(self.residual_logit_scale_init), dtype=torch.float32))
            if residual_logit_enabled
            else None
        )
        self.residual_logit_gate = (
            nn.Parameter(torch.tensor(self.residual_logit_gate_init, dtype=torch.float32))
            if residual_logit_enabled
            else None
        )
        self.mid_aux_head = CastedLinear(model_dim, vocab_size, bias=False)
        self.mid_aux_head._zero_init = True
        self.last_pairwise_logit_gate_mean = 0.0
        self.last_pairwise_logit_rms = 0.0
        self.fast_features_path = self.shared_loop_gates is None and all(block.fast_path for block in self.blocks)
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            if self.overtone_embed_init:
                self._init_overtone_embedding()
            else:
                nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.pairwise_prev_embed is not None:
            nn.init.normal_(self.pairwise_prev_embed.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.pairwise_logit_head is not None:
            nn.init.normal_(self.pairwise_logit_head.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.doc_bias_embed is not None:
            nn.init.normal_(self.doc_bias_embed.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.doc_bias_head is not None:
            nn.init.normal_(self.doc_bias_head.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        if self.tie_embeddings:
            with torch.no_grad():
                tied_weight = self._embedding_weight().detach()
                if self.lexical_shortcut_head is not None:
                    self.lexical_shortcut_head.weight.copy_(tied_weight)
                if self.lexical_residual_head is not None:
                    self.lexical_residual_head.weight.copy_(tied_weight)
        if self.bigram is not None:
            if self.hashed_ngram_init_std > 0.0:
                nn.init.normal_(self.bigram.embed.weight, mean=0.0, std=self.hashed_ngram_init_std)
            else:
                nn.init.zeros_(self.bigram.embed.weight)
            if self.bigram.proj is not None and self.hashed_ngram_init_std <= 0.0:
                nn.init.zeros_(self.bigram.proj.weight)
        if self.lexical_residual_ngram is not None:
            if self.hashed_ngram_init_std > 0.0:
                nn.init.normal_(self.lexical_residual_ngram.embed.weight, mean=0.0, std=self.hashed_ngram_init_std)
            else:
                nn.init.zeros_(self.lexical_residual_ngram.embed.weight)
            if self.lexical_residual_ngram.proj is not None and self.hashed_ngram_init_std <= 0.0:
                nn.init.zeros_(self.lexical_residual_ngram.proj.weight)
        if self.smear is not None:
            nn.init.zeros_(self.smear.state_proj.weight)
            if self.smear_lexical_proj_init_std > 0.0:
                nn.init.normal_(self.smear.lexical_proj.weight, mean=0.0, std=self.smear_lexical_proj_init_std)
            else:
                nn.init.zeros_(self.smear.lexical_proj.weight)
        if self.orthogonal_init:
            self._apply_orthogonal_init()
        if self.resid_mix_phase_init:
            self._init_resid_mix_phase_schedule()

    def _apply_orthogonal_init(self) -> None:
        proj_scale = 1.0 / math.sqrt(max(2 * len(self.blocks), 1)) if self.mup_proj_init else 1.0
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if getattr(module, "_zero_init", False):
                continue
            if module.weight.ndim != 2 or min(module.weight.shape) < 64:
                continue
            nn.init.orthogonal_(module.weight, gain=1.0)
            if self.mup_proj_init and (
                name.endswith(".proj")
                or name.endswith(".attn.proj")
                or name.endswith(".mlp.proj")
                or name.endswith("bigram.proj")
                or name.endswith("lexical_residual_ngram.proj")
            ):
                with torch.no_grad():
                    module.weight.mul_(proj_scale)

    def _init_overtone_embedding(self) -> None:
        rows, cols = self.tok_emb.weight.shape
        with torch.no_grad():
            basis = torch.randn((rows, cols), dtype=torch.float32, device=self.tok_emb.weight.device)
            q, _ = torch.linalg.qr(basis, mode="reduced")
            ranks = torch.arange(1, cols + 1, dtype=torch.float32, device=q.device)
            spectrum = ranks.pow(-self.overtone_embed_power)
            spectrum = spectrum / spectrum.norm() * (cols**0.5) * self.tied_embed_init_std * self.overtone_embed_scale
            self.tok_emb.weight.copy_(q * spectrum.unsqueeze(0))

    def _init_resid_mix_phase_schedule(self) -> None:
        total = max(len(self.blocks) - 1, 1)
        with torch.no_grad():
            for idx, block in enumerate(self.blocks):
                depth = idx / total
                alpha = torch.sigmoid(
                    torch.tensor((depth - self.resid_mix_phase_center) * self.resid_mix_phase_sharpness, dtype=torch.float32)
                ).item()
                block.resid_mix[0].fill_(alpha)
                block.resid_mix[1].fill_(1.0 - alpha)

    def _skip_scale(self, idx: int, dtype: torch.dtype) -> Tensor:
        scale = self.skip_weights[idx].to(dtype=dtype)
        return torch.tanh(scale) if self.signed_skip_weights else scale

    def set_fake_quant(self, bits: int) -> None:
        self.fake_quant_bits = max(int(bits), 0)
        for module in self.modules():
            if isinstance(module, CastedLinear):
                module.fake_quant_bits = self.fake_quant_bits

    def _helper_path_schedule_mult(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.helper_path_schedule_mult_buffer.to(device=device, dtype=dtype)

    def _shared_tail_schedule_mult(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.shared_tail_schedule_mult_buffer.to(device=device, dtype=dtype)

    def _lexical_hidden_mult(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.lexical_hidden_mult_buffer.to(device=device, dtype=dtype)

    def _lexical_shortcut_mult(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.lexical_shortcut_mult_buffer.to(device=device, dtype=dtype)

    def _embedding_weight(self) -> Tensor:
        weight = self.tok_emb.weight
        if self.fake_quant_bits > 0:
            weight = fake_quantize_tensor(weight, self.fake_quant_bits)
        return weight

    def _apply_output_head(self, head: nn.Module, x: Tensor) -> Tensor:
        if isinstance(head, CastedLinear):
            return head(x)
        weight = getattr(head, "weight", None)
        if isinstance(weight, Tensor):
            return head(x.to(dtype=weight.dtype))
        return head(x)

    def _mid_aux_logits(self, x: Tensor) -> Tensor:
        x = self.final_norm(x, condition=None) if isinstance(self.final_norm, AdaptiveRMSNorm) else self.final_norm(x)
        logits_proj = self._apply_output_head(self.mid_aux_head, x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _compute_pairwise_logit_prior(
        self,
        input_ids: Tensor,
        x: Tensor,
        lexical_confidence: Tensor | None = None,
        external_keep: Tensor | None = None,
    ) -> Tensor | None:
        if self.pairwise_prev_embed is None or self.pairwise_logit_head is None or self.pairwise_logit_scale is None:
            self.last_pairwise_logit_gate_mean = 0.0
            self.last_pairwise_logit_rms = 0.0
            return None
        pairwise_features = self.pairwise_prev_embed(input_ids)
        pairwise_logits = self._apply_output_head(self.pairwise_logit_head, pairwise_features)
        gate_logits = torch.zeros((*input_ids.shape, 1), device=x.device, dtype=x.dtype)
        if self.pairwise_logit_gate is not None:
            gate_logits = gate_logits + self.pairwise_logit_gate.to(device=x.device, dtype=x.dtype)
        if self.pairwise_logit_hidden_gate is not None:
            hidden_gate = self.pairwise_logit_hidden_gate.to(device=x.device, dtype=x.dtype)
            gate_logits = gate_logits + (x * hidden_gate).sum(dim=-1, keepdim=True) / math.sqrt(max(x.size(-1), 1))
        if lexical_confidence is not None and self.pairwise_logit_confidence_gain is not None:
            conf = 2.0 * lexical_confidence.to(device=x.device, dtype=x.dtype) - 1.0
            gate_logits = gate_logits + self.pairwise_logit_confidence_gain.to(device=x.device, dtype=x.dtype) * conf
        gate = torch.sigmoid(gate_logits)
        if external_keep is not None:
            gate = gate * external_keep.to(device=x.device, dtype=x.dtype)
        scale = F.softplus(self.pairwise_logit_scale).to(device=x.device, dtype=x.dtype)
        pairwise_logits = pairwise_logits.to(dtype=x.dtype)
        pairwise_logits = scale * gate * pairwise_logits
        if not torch._dynamo.is_compiling():
            self.last_pairwise_logit_gate_mean = float(gate.detach().float().mean().item())
            self.last_pairwise_logit_rms = float(pairwise_logits.detach().float().square().mean().sqrt().item())
        return pairwise_logits

    def _compute_online_doc_bias(self, input_ids: Tensor, dtype: torch.dtype) -> Tensor | None:
        if (
            self.doc_bias_embed is None
            or self.doc_bias_head is None
            or self.doc_bias_scale is None
            or self.doc_bias_gate is None
        ):
            return None
        token_features = self.doc_bias_embed(input_ids)
        prefix_cumsum = token_features.cumsum(dim=1)
        prefix_sum = torch.cat(
            [torch.zeros_like(prefix_cumsum[:, :1, :]), prefix_cumsum[:, :-1, :]],
            dim=1,
        )
        seq_len = int(input_ids.size(1))
        positions = torch.arange(seq_len, device=input_ids.device, dtype=token_features.dtype).view(1, seq_len, 1)
        denom = positions.clamp_min(1.0)
        prefix_mean = prefix_sum / denom
        doc_logits = self._apply_output_head(self.doc_bias_head, prefix_mean)
        scale = F.softplus(self.doc_bias_scale).to(device=input_ids.device, dtype=dtype)
        gate = torch.sigmoid(self.doc_bias_gate).to(device=input_ids.device, dtype=dtype)
        return scale * gate * doc_logits.to(dtype=dtype)

    def _confidence_skip_keep(self, base_logits: Tensor, min_keep: float) -> Tensor | None:
        if not self.use_confidence_branch_skip or base_logits.size(-1) <= 1:
            return None
        top2 = torch.topk(base_logits.float(), k=2, dim=-1).values
        margin = (top2[..., :1] - top2[..., 1:2]).to(dtype=base_logits.dtype)
        raw_keep = torch.sigmoid(
            (self.confidence_skip_margin - margin)
            * max(self.confidence_skip_sharpness, 0.0)
        )
        min_keep_t = torch.as_tensor(min_keep, device=base_logits.device, dtype=base_logits.dtype)
        return min_keep_t + (1.0 - min_keep_t) * raw_keep

    def _compute_residual_logit_adapter(self, x: Tensor) -> Tensor | None:
        if (
            self.residual_logit_down is None
            or self.residual_logit_up is None
            or self.residual_logit_scale is None
            or self.residual_logit_gate is None
        ):
            return None
        adapter_hidden = self._apply_output_head(self.residual_logit_down, x)
        adapter_logits = self._apply_output_head(self.residual_logit_up, adapter_hidden)
        scale = F.softplus(self.residual_logit_scale).to(device=x.device, dtype=x.dtype)
        gate = torch.sigmoid(self.residual_logit_gate).to(device=x.device, dtype=x.dtype)
        return scale * gate * adapter_logits.to(dtype=x.dtype)

    def _staged_mult(self, step: int, enable_step: int, ramp_steps: int) -> float:
        step = max(int(step), 0)
        enable_step = max(int(enable_step), 0)
        ramp_steps = max(int(ramp_steps), 0)
        if step < enable_step:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return min(max((step - enable_step) / max(ramp_steps, 1), 0.0), 1.0)

    def set_training_step(self, step: int) -> None:
        self.current_training_step = max(int(step), 0)
        helper_mult = linear_anneal_multiplier(step, self.helper_path_full_steps, self.helper_path_decay_end_step)
        self.helper_path_schedule_mult_buffer.fill_(float(helper_mult))
        step = self.current_training_step
        lexical_mult = self._staged_mult(step, self.lexical_enable_step, self.lexical_ramp_steps)
        lexical_calibration_mult = self._staged_mult(
            step,
            self.lexical_calibration_enable_step,
            self.lexical_calibration_ramp_steps,
        )
        lexical_hidden_mult = self._staged_mult(
            step,
            self.lexical_hidden_enable_step,
            self.lexical_hidden_ramp_steps,
        )
        lexical_shortcut_mult = self._staged_mult(
            step,
            self.lexical_shortcut_enable_step,
            self.lexical_shortcut_ramp_steps,
        )
        lexical_shortcut_mult *= float(self.lexical_shortcut_health_gate_buffer.detach().float().item())
        shared_tail_mult = self._staged_mult(
            step,
            self.shared_tail_enable_step,
            self.shared_tail_ramp_steps,
        ) * self.shared_tail_max_mult
        self.shared_tail_schedule_mult_buffer.fill_(float(shared_tail_mult))
        self.lexical_schedule_mult_buffer.fill_(float(max(lexical_hidden_mult, lexical_shortcut_mult)))
        self.lexical_calibration_mult_buffer.fill_(float(lexical_calibration_mult))
        self.lexical_hidden_mult_buffer.fill_(float(lexical_hidden_mult))
        self.lexical_shortcut_mult_buffer.fill_(float(lexical_shortcut_mult))
        for block in list(self.blocks) + list(self.shared_blocks):
            if getattr(block, "attn", None) is not None:
                if hasattr(block.attn, "set_helper_path_train_step"):
                    block.attn.set_helper_path_train_step(step)
            if hasattr(block, "set_helper_path_train_step"):
                block.set_helper_path_train_step(step)

    @torch.no_grad()
    def set_lexical_shortcut_external_gate(self, gate: float) -> None:
        self.lexical_shortcut_health_gate_buffer.fill_(float(min(max(gate, 0.0), 1.0)))

    def set_prior_caps(
        self,
        *,
        bigram: float | None = None,
        lexical: float | None = None,
    ) -> None:
        if bigram is not None:
            if self.bigram is not None:
                self.bigram.influence_cap.fill_(float(bigram))
            if self.lexical_residual_ngram is not None:
                self.lexical_residual_ngram.influence_cap.fill_(float(bigram))
        if lexical is not None:
            self.lexical_cap.fill_(float(lexical))

    def _loss_from_logits(
        self,
        logits: Tensor,
        target_ids: Tensor,
        loss_mask: Tensor | None = None,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        logit_var_loss_coeff: float = 0.0,
    ) -> Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)
        mask = None if loss_mask is None else loss_mask.reshape(-1).to(dtype=torch.float32)
        ce = F.cross_entropy(flat_logits.float(), targets, reduction="none", label_smoothing=label_smoothing)
        if mask is None:
            loss = ce.mean()
        else:
            loss = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        if z_loss_coeff > 0.0:
            z_term = torch.logsumexp(flat_logits.float(), dim=-1).square()
            z_loss = z_term.mean() if mask is None else (z_term * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + z_loss_coeff * z_loss
        if logit_var_loss_coeff > 0.0:
            # Keep per-token logit spread from exploding in short runs, which tends
            # to help the later quantized softcapped head behave more smoothly.
            logit_std = flat_logits.float().std(dim=-1)
            target_std = float(self.logit_softcap) / 4.0
            logit_var = (logit_std - target_std).square()
            if mask is None:
                loss = loss + logit_var_loss_coeff * logit_var.mean()
            else:
                loss = loss + logit_var_loss_coeff * (logit_var * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    def _apply_shared_tail(self, x: Tensor, x0: Tensor, norm_condition: Tensor | None = None) -> Tensor:
        if self.num_shared_layers <= 0 or self.shared_layer_repeats <= 0:
            return x
        tail_base = x
        prev_attn_logits = None
        tail_mult = self._helper_path_schedule_mult(device=x.device, dtype=x.dtype)
        tail_mult = tail_mult * self._shared_tail_schedule_mult(device=x.device, dtype=x.dtype)
        if self.shared_tail_gate is not None:
            tail_mult = tail_mult * torch.sigmoid(self.shared_tail_gate.to(dtype=x.dtype))
        for repeat_idx in range(self.shared_layer_repeats):
            for layer_idx, block in enumerate(self.shared_blocks):
                prev_x = x
                x, prev_attn_logits = self._run_block(
                    block,
                    x,
                    x0,
                    prev_attn_logits=prev_attn_logits,
                    norm_condition=norm_condition,
                )
                x = prev_x + tail_mult * (x - prev_x)
        return tail_base + tail_mult * (x - tail_base)

    def _embed_token_ids_unique(self, token_ids: Tensor) -> Tensor:
        embed_weight = self._embedding_weight()
        return F.embedding(token_ids, embed_weight)

    def _compute_lexical_batch_mult(self, input_ids: Tensor) -> Tensor:
        flat = input_ids.reshape(-1)
        if flat.numel() == 0:
            throttle = torch.ones((), device=input_ids.device, dtype=torch.float32)
            unique_frac = torch.ones((), device=input_ids.device, dtype=torch.float32)
            max_token_frac = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        else:
            vocab_size = int(self.vocab_size)
            counts = torch.zeros((vocab_size,), device=input_ids.device, dtype=torch.float32)
            counts.scatter_add_(0, flat.detach(), torch.ones_like(flat, dtype=torch.float32))
            unique_frac = counts.count_nonzero().to(dtype=torch.float32) / float(max(vocab_size, 1))
            max_token_frac = counts.max().to(dtype=torch.float32) / float(max(int(flat.numel()), 1))
            throttle = 1.0
            throttle -= self.lexical_batch_diversity_throttle * (1.0 - unique_frac)
            throttle -= self.lexical_batch_repeat_throttle * max_token_frac
            throttle = throttle.clamp(min=self.lexical_batch_min_mult, max=1.0)
        if not torch._dynamo.is_compiling():
            self.last_lexical_batch_mult = float(throttle.detach().cpu().item())
            self.last_lexical_batch_unique_frac = float(unique_frac.detach().cpu().item())
            self.last_lexical_batch_max_token_frac = float(max_token_frac.detach().cpu().item())
            self.lexical_batch_mult_buffer.copy_(
                throttle.detach().to(device=self.lexical_batch_mult_buffer.device, dtype=self.lexical_batch_mult_buffer.dtype)
            )
            self.lexical_batch_unique_frac_buffer.copy_(
                unique_frac.detach().to(
                    device=self.lexical_batch_unique_frac_buffer.device,
                    dtype=self.lexical_batch_unique_frac_buffer.dtype,
                )
            )
            self.lexical_batch_max_token_frac_buffer.copy_(
                max_token_frac.detach().to(
                    device=self.lexical_batch_max_token_frac_buffer.device,
                    dtype=self.lexical_batch_max_token_frac_buffer.dtype,
                )
            )
        return throttle

    def _build_norm_condition(
        self,
        lexical_confidence: Tensor | None,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if not self.use_adaptive_rmsnorm:
            return None
        if lexical_confidence is None:
            return None
        conf = 2.0 * lexical_confidence.to(device=device, dtype=dtype) - 1.0
        mask = torch.ones((batch_size, seq_len, 1), device=device, dtype=dtype)
        return torch.cat((conf, mask), dim=-1)

    def _project_lexical_features(self, lexical_features: Tensor | None, residual: bool = False) -> Tensor | None:
        if lexical_features is None:
            return None
        head = self.lexical_residual_head if residual else self.lexical_shortcut_head
        if head is not None:
            return self._apply_output_head(head, lexical_features)
        if self.tie_embeddings:
            embed_weight = self._embedding_weight()
            return F.linear(lexical_features.to(dtype=embed_weight.dtype), embed_weight)
        return None

    def _bigram_hidden_schedule_mult(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self._lexical_hidden_mult(device=device, dtype=dtype)

    def _compute_lexical_shortcut(
        self,
        lexical_features: Tensor | None,
        lexical_residual_features: Tensor | None = None,
        lexical_batch_mult: Tensor | float = 1.0,
    ) -> tuple[Tensor | None, Tensor | None]:
        if lexical_features is None or self.lexical_shortcut_scale is None:
            return None, None
        base_logits = self._project_lexical_features(lexical_features, residual=False)
        if base_logits is None:
            return None, None
        lexical_batch_gain = torch.as_tensor(lexical_batch_mult, device=base_logits.device, dtype=base_logits.dtype)
        helper_mult = self._helper_path_schedule_mult(device=base_logits.device, dtype=base_logits.dtype)
        lexical_shortcut_mult = self._lexical_shortcut_mult(device=base_logits.device, dtype=base_logits.dtype)
        lexical_effective_mult = helper_mult * lexical_shortcut_mult
        lexical_gate = 1.0
        if self.lexical_prior_gate_logit is not None:
            lexical_gate = torch.sigmoid(self.lexical_prior_gate_logit).to(dtype=base_logits.dtype)
            lexical_gate = lexical_gate * self.lexical_cap.to(device=base_logits.device, dtype=base_logits.dtype)
        shortcut_scale = F.softplus(self.lexical_shortcut_scale).to(dtype=base_logits.dtype)
        confidence_logits = lexical_gate * base_logits
        shortcut_logits = lexical_batch_gain * lexical_effective_mult * lexical_gate * shortcut_scale * base_logits
        if lexical_residual_features is not None and self.lexical_residual_scale is not None:
            residual_logits = self._project_lexical_features(lexical_residual_features, residual=True)
            if residual_logits is not None:
                residual_scale = F.softplus(self.lexical_residual_scale).to(dtype=residual_logits.dtype)
                confidence_logits = confidence_logits + lexical_gate.to(dtype=residual_logits.dtype) * residual_scale * residual_logits
                residual_logits = (
                    lexical_batch_gain
                    * lexical_effective_mult.to(dtype=residual_logits.dtype)
                    * lexical_gate
                    * residual_scale
                    * residual_logits
                )
                residual_confidence = lexical_shortcut_confidence_from_logits(
                    residual_logits,
                    F.softplus(self.lexical_residual_confidence_scale),
                    self.lexical_residual_confidence_bias,
                )
                if residual_confidence is not None:
                    residual_logits = residual_logits * residual_confidence.to(dtype=residual_logits.dtype)
                shortcut_logits = shortcut_logits + residual_logits
        confidence = lexical_shortcut_confidence_from_logits(
            confidence_logits,
            F.softplus(self.lexical_confidence_scale),
            self.lexical_confidence_bias,
        )
        return shortcut_logits, confidence

    def _run_block(
        self,
        block: Block,
        x: Tensor,
        x0: Tensor,
        prev_attn_logits: Tensor | tuple[Tensor, Tensor] | None,
        norm_condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor] | None]:
        x0_for_block = x0.clone() if x is x0 else x0
        if block.fast_path and prev_attn_logits is None:
            return block.forward_simple(x, x0_for_block, norm_condition=norm_condition), None
        return block(
            x,
            x0_for_block,
            prev_attn_logits=prev_attn_logits,
            norm_condition=norm_condition,
        )

    def _forward_features(
        self,
        input_ids: Tensor,
        return_pre_final: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None] | tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor]:
        if self.fast_features_path:
            return self._forward_features_fast(input_ids, return_pre_final=return_pre_final)
        lexical_batch_mult = self._compute_lexical_batch_mult(input_ids)
        x = self._embed_token_ids_unique(input_ids)
        lexical_features = None
        lexical_residual_features = None
        lexical_shortcut_logits = None
        lexical_confidence = None
        helper_mult = self._helper_path_schedule_mult(device=x.device, dtype=x.dtype)
        lexical_hidden_mult = self._lexical_hidden_mult(device=x.device, dtype=x.dtype)
        lexical_effective_mult = helper_mult * lexical_hidden_mult
        if self.bigram is not None:
            bigram = self.bigram(input_ids)
            lexical_features = bigram
            if self.lexical_residual_ngram is not None:
                lexical_residual_features = self.lexical_residual_ngram(input_ids)
            lexical_shortcut_logits, lexical_confidence = self._compute_lexical_shortcut(
                lexical_features,
                lexical_residual_features=lexical_residual_features,
                lexical_batch_mult=lexical_batch_mult,
            )
            bigram_hidden = bigram
            if self.bigram_hidden_scale is not None:
                bigram_hidden = (
                    F.softplus(self.bigram_hidden_scale).to(dtype=bigram.dtype)
                    * helper_mult.to(dtype=bigram.dtype)
                    * self._bigram_hidden_schedule_mult(device=bigram.device, dtype=bigram.dtype)
                    * bigram_hidden
                )
            if self.lexical_prior_gate_logit is not None:
                lexical_gate = torch.sigmoid(self.lexical_prior_gate_logit).to(dtype=bigram_hidden.dtype)
                lexical_gate = lexical_gate * self.lexical_cap.to(device=bigram_hidden.device, dtype=bigram_hidden.dtype)
                bigram_hidden = bigram_hidden * lexical_gate
            bigram_hidden = bigram_hidden * lexical_batch_mult
            x = x + bigram_hidden
        norm_condition = self._build_norm_condition(
            lexical_confidence,
            seq_len=int(input_ids.size(1)),
            batch_size=int(input_ids.size(0)),
            device=x.device,
            dtype=x.dtype,
        )
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            lexical_features_for_smear = (
                lexical_features * lexical_batch_mult if lexical_features is not None else None
            )
            smeared = self.smear(
                x,
                lexical_features=lexical_features_for_smear,
                lexical_confidence=lexical_confidence,
                lexical_scale_mult=lexical_effective_mult,
            )
            x = x + lexical_hidden_mult * (smeared - x)
        # Keep the block-level residual reference non-aliased so torch.compile
        # does not recompile `_run_block` when the first block sees x is x0.
        x0 = x.clone()
        skips: list[Tensor] = []
        prev_attn_logits = None
        h_mid: Tensor | None = None
        mid_idx = len(self.blocks) // 2

        for i in range(self.num_encoder_layers):
            x, prev_attn_logits = self._run_block(
                self.blocks[i],
                x,
                x0,
                prev_attn_logits=prev_attn_logits,
                norm_condition=norm_condition,
            )
            if i == mid_idx:
                h_mid = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self._skip_scale(i, x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            x, prev_attn_logits = self._run_block(
                self.blocks[bi],
                x,
                x0,
                prev_attn_logits=prev_attn_logits,
                norm_condition=norm_condition,
            )
            if bi == mid_idx:
                h_mid = x
        x = self._apply_shared_tail(x, x0, norm_condition=norm_condition)
        pre_final = x
        x = self.final_norm(x, condition=norm_condition) if isinstance(self.final_norm, AdaptiveRMSNorm) else self.final_norm(x)
        if return_pre_final:
            return x, h_mid, lexical_shortcut_logits, lexical_confidence, pre_final
        return x, h_mid, lexical_shortcut_logits, lexical_confidence

    def _forward_features_fast(
        self,
        input_ids: Tensor,
        return_pre_final: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None] | tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor]:
        lexical_batch_mult = self._compute_lexical_batch_mult(input_ids)
        x = self._embed_token_ids_unique(input_ids)
        lexical_features = None
        lexical_residual_features = None
        lexical_shortcut_logits = None
        lexical_confidence = None
        helper_mult = self._helper_path_schedule_mult(device=x.device, dtype=x.dtype)
        lexical_hidden_mult = self._lexical_hidden_mult(device=x.device, dtype=x.dtype)
        lexical_effective_mult = helper_mult * lexical_hidden_mult
        if self.bigram is not None:
            bigram = self.bigram(input_ids)
            lexical_features = bigram
            if self.lexical_residual_ngram is not None:
                lexical_residual_features = self.lexical_residual_ngram(input_ids)
            lexical_shortcut_logits, lexical_confidence = self._compute_lexical_shortcut(
                lexical_features,
                lexical_residual_features=lexical_residual_features,
                lexical_batch_mult=lexical_batch_mult,
            )
            bigram_hidden = bigram
            if self.bigram_hidden_scale is not None:
                bigram_hidden = (
                    F.softplus(self.bigram_hidden_scale).to(dtype=bigram.dtype)
                    * helper_mult.to(dtype=bigram.dtype)
                    * self._bigram_hidden_schedule_mult(device=bigram.device, dtype=bigram.dtype)
                    * bigram_hidden
                )
            if self.lexical_prior_gate_logit is not None:
                lexical_gate = torch.sigmoid(self.lexical_prior_gate_logit).to(dtype=bigram_hidden.dtype)
                lexical_gate = lexical_gate * self.lexical_cap.to(device=bigram_hidden.device, dtype=bigram_hidden.dtype)
                bigram_hidden = bigram_hidden * lexical_gate
            bigram_hidden = bigram_hidden * lexical_batch_mult
            x = x + bigram_hidden
        norm_condition = self._build_norm_condition(
            lexical_confidence,
            seq_len=int(input_ids.size(1)),
            batch_size=int(input_ids.size(0)),
            device=x.device,
            dtype=x.dtype,
        )
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            lexical_features_for_smear = (
                lexical_features * lexical_batch_mult if lexical_features is not None else None
            )
            smeared = self.smear(
                x,
                lexical_features=lexical_features_for_smear,
                lexical_confidence=lexical_confidence,
                lexical_scale_mult=lexical_effective_mult,
            )
            x = x + lexical_hidden_mult * (smeared - x)
        # Keep the block-level residual reference non-aliased so torch.compile
        # does not recompile `forward_simple` when the first block sees x is x0.
        x0 = x.clone()
        skips: list[Tensor] = []
        h_mid: Tensor | None = None
        mid_idx = len(self.blocks) // 2
        for i in range(self.num_encoder_layers):
            x = self.blocks[i].forward_simple(x, x0, norm_condition=norm_condition)
            if i == mid_idx:
                h_mid = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self._skip_scale(i, x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            x = self.blocks[bi].forward_simple(x, x0, norm_condition=norm_condition)
            if bi == mid_idx:
                h_mid = x
        x = self._apply_shared_tail(x, x0, norm_condition=norm_condition)
        pre_final = x
        x = self.final_norm(x, condition=norm_condition) if isinstance(self.final_norm, AdaptiveRMSNorm) else self.final_norm(x)
        if return_pre_final:
            return x, h_mid, lexical_shortcut_logits, lexical_confidence, pre_final
        return x, h_mid, lexical_shortcut_logits, lexical_confidence

    def _project_logits(
        self,
        x: Tensor,
        input_ids: Tensor | None = None,
        lexical_shortcut_logits: Tensor | None = None,
        lexical_confidence: Tensor | None = None,
    ) -> Tensor:
        if self.tie_embeddings:
            embed_weight = self._embedding_weight()
            logits_proj = F.linear(x.to(dtype=embed_weight.dtype), embed_weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self._apply_output_head(self.lm_head, x)
        if self.output_logit_bias is not None:
            logits_proj = logits_proj + self.output_logit_bias.to(device=logits_proj.device, dtype=logits_proj.dtype)
        lexical_keep = self._confidence_skip_keep(logits_proj, self.confidence_skip_lexical_min_keep)
        pairwise_keep = self._confidence_skip_keep(logits_proj, self.confidence_skip_pairwise_min_keep)
        doc_bias_keep = self._confidence_skip_keep(logits_proj, self.confidence_skip_doc_bias_min_keep)
        if lexical_shortcut_logits is not None:
            lexical_logits = lexical_shortcut_logits.to(dtype=logits_proj.dtype)
            if lexical_keep is not None:
                lexical_logits = lexical_logits * lexical_keep.to(dtype=lexical_logits.dtype)
            logits_proj = logits_proj + lexical_logits
        if input_ids is not None:
            pairwise_logits = self._compute_pairwise_logit_prior(
                input_ids,
                x,
                lexical_confidence=lexical_confidence,
                external_keep=pairwise_keep,
            )
            if pairwise_logits is not None:
                logits_proj = logits_proj + pairwise_logits.to(dtype=logits_proj.dtype)
            doc_bias_logits = self._compute_online_doc_bias(input_ids, dtype=logits_proj.dtype)
            if doc_bias_logits is not None:
                if doc_bias_keep is not None:
                    doc_bias_logits = doc_bias_logits * doc_bias_keep.to(dtype=doc_bias_logits.dtype)
                logits_proj = logits_proj + doc_bias_logits.to(dtype=logits_proj.dtype)
        residual_adapter_logits = self._compute_residual_logit_adapter(x)
        if residual_adapter_logits is not None:
            logits_proj = logits_proj + residual_adapter_logits.to(dtype=logits_proj.dtype)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        loss_mask: Tensor | None = None,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        logit_var_loss_coeff: float = 0.0,
        mid_aux_loss_coeff: float | Tensor = 0.01,
    ) -> Tensor:
        features, h_mid, lexical_shortcut_logits, lexical_confidence = self._forward_features(input_ids)
        logits = self._project_logits(
            features,
            input_ids=input_ids,
            lexical_shortcut_logits=lexical_shortcut_logits,
            lexical_confidence=lexical_confidence,
        )
        loss = self._loss_from_logits(
            logits,
            target_ids,
            loss_mask=loss_mask,
            label_smoothing=label_smoothing,
            z_loss_coeff=z_loss_coeff,
            logit_var_loss_coeff=logit_var_loss_coeff,
        )
        if h_mid is not None:
            if h_mid.size(1) == target_ids.size(1):
                mid_logits = self._mid_aux_logits(h_mid)
                mid_ce = F.cross_entropy(
                    mid_logits.reshape(-1, mid_logits.size(-1)).float(),
                    target_ids.reshape(-1),
                    reduction="none",
                    label_smoothing=label_smoothing,
                )
                if loss_mask is None:
                    mid_loss = mid_ce.mean()
                else:
                    mid_mask = loss_mask.reshape(-1).to(dtype=mid_ce.dtype)
                    mid_loss = (mid_ce * mid_mask).sum() / mid_mask.sum().clamp_min(1.0)
                if not torch.is_tensor(mid_aux_loss_coeff):
                    mid_aux_loss_coeff = torch.tensor(
                        float(mid_aux_loss_coeff),
                        device=mid_loss.device,
                        dtype=mid_loss.dtype,
                    )
                else:
                    mid_aux_loss_coeff = mid_aux_loss_coeff.to(device=mid_loss.device, dtype=mid_loss.dtype)
                loss = loss + mid_aux_loss_coeff * mid_loss
        return loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x, _, lexical_shortcut_logits, lexical_confidence = self._forward_features(input_ids)
        return self._project_logits(
            x,
            input_ids=input_ids,
            lexical_shortcut_logits=lexical_shortcut_logits,
            lexical_confidence=lexical_confidence,
        )

    @torch.no_grad()
    def convergence_diagnostics(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        loss_mask: Tensor | None = None,
    ) -> dict[str, object]:
        was_training = self.training
        if was_training:
            self.eval()
        try:
            features, h_mid, lexical_shortcut_logits, lexical_confidence, pre_final_features = self._forward_features(
                input_ids,
                return_pre_final=True,
            )
            if self.tie_embeddings:
                embed_weight = self._embedding_weight()
                logits_proj = F.linear(features.to(dtype=embed_weight.dtype), embed_weight)
            else:
                if self.lm_head is None:
                    raise RuntimeError("lm_head is required when tie_embeddings=False")
                logits_proj = self._apply_output_head(self.lm_head, features)
            if self.output_logit_bias is not None:
                logits_proj = logits_proj + self.output_logit_bias.to(device=logits_proj.device, dtype=logits_proj.dtype)
            lexical_keep = self._confidence_skip_keep(logits_proj, self.confidence_skip_lexical_min_keep)
            pairwise_keep = self._confidence_skip_keep(logits_proj, self.confidence_skip_pairwise_min_keep)
            doc_bias_keep = self._confidence_skip_keep(logits_proj, self.confidence_skip_doc_bias_min_keep)
            if lexical_shortcut_logits is not None:
                lexical_logits = lexical_shortcut_logits.to(dtype=logits_proj.dtype)
                if lexical_keep is not None:
                    lexical_logits = lexical_logits * lexical_keep.to(dtype=lexical_logits.dtype)
                logits_proj = logits_proj + lexical_logits
            pairwise_logits = self._compute_pairwise_logit_prior(
                input_ids,
                features,
                lexical_confidence=lexical_confidence,
                external_keep=pairwise_keep,
            )
            if pairwise_logits is not None:
                logits_proj = logits_proj + pairwise_logits.to(dtype=logits_proj.dtype)
            doc_bias_logits = self._compute_online_doc_bias(input_ids, dtype=logits_proj.dtype)
            if doc_bias_logits is not None:
                if doc_bias_keep is not None:
                    doc_bias_logits = doc_bias_logits * doc_bias_keep.to(dtype=doc_bias_logits.dtype)
                logits_proj = logits_proj + doc_bias_logits.to(dtype=logits_proj.dtype)
            residual_adapter_logits = self._compute_residual_logit_adapter(features)
            if residual_adapter_logits is not None:
                logits_proj = logits_proj + residual_adapter_logits.to(dtype=logits_proj.dtype)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            valid = loss_mask.to(dtype=torch.bool) if loss_mask is not None else torch.ones_like(target_ids, dtype=torch.bool)
            valid_count = int(valid.sum().item())
            ln2 = math.log(2.0)

            def _masked_feature_rms(value: Tensor) -> float:
                if valid_count <= 0:
                    return 0.0
                if value.ndim != 3 or value.size(0) != valid.size(0) or value.size(1) != valid.size(1):
                    return float(value.float().square().mean().sqrt().item()) if value.numel() > 0 else 0.0
                masked = value.float().masked_select(valid.unsqueeze(-1))
                return float(masked.square().mean().sqrt().item()) if masked.numel() > 0 else 0.0

            def _masked_logit_std(value: Tensor) -> float:
                if valid_count <= 1:
                    return 0.0
                if value.ndim != 3 or value.size(0) != valid.size(0) or value.size(1) != valid.size(1):
                    return float(value.float().std(unbiased=False).item()) if value.numel() > 1 else 0.0
                masked = value.float().masked_select(valid.unsqueeze(-1))
                return float(masked.std(unbiased=False).item()) if masked.numel() > 1 else 0.0

            def _masked_logit_max_abs(value: Tensor) -> float:
                if valid_count <= 0:
                    return 0.0
                if value.ndim != 3 or value.size(0) != valid.size(0) or value.size(1) != valid.size(1):
                    return float(value.float().abs().amax().item()) if value.numel() > 0 else 0.0
                masked = value.float().masked_select(valid.unsqueeze(-1))
                return float(masked.abs().amax().item()) if masked.numel() > 0 else 0.0

            def _mean(value: Tensor) -> float:
                if valid_count <= 0:
                    return 0.0
                masked = value.float().masked_select(valid)
                return float(masked.mean().item()) if masked.numel() > 0 else 0.0

            def _std(value: Tensor) -> float:
                if valid_count <= 1:
                    return 0.0
                masked = value.float().masked_select(valid)
                return float(masked.std(unbiased=False).item()) if masked.numel() > 0 else 0.0

            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            final_entropy_bits = -(probs * (log_probs / ln2)).sum(dim=-1)
            final_top1_conf = probs.amax(dim=-1)
            final_top1 = logits.argmax(dim=-1)
            final_top1_acc = final_top1.eq(target_ids).to(dtype=torch.float32)
            if logits.size(-1) > 1:
                final_top2 = torch.topk(logits.float(), k=2, dim=-1).values
                final_margin = final_top2[..., 0] - final_top2[..., 1]
            else:
                final_margin = torch.ones_like(final_top1_conf)
            token_loss_bits = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                target_ids.reshape(-1),
                reduction="none",
            ).view_as(target_ids) / ln2

            diag: dict[str, object] = {
                "diag_token_count": valid_count,
                "pre_final_feature_rms": _masked_feature_rms(pre_final_features),
                "final_feature_rms": _masked_feature_rms(features),
                "pre_softcap_logit_std": _masked_logit_std(logits_proj),
                "final_logit_std": _masked_logit_std(logits),
                "pre_softcap_logit_max_abs": _masked_logit_max_abs(logits_proj),
                "final_logit_max_abs": _masked_logit_max_abs(logits),
                "tied_embed_weight_rms": float(self._embedding_weight().float().square().mean().sqrt().item()),
                "final_entropy_bits": _mean(final_entropy_bits),
                "final_entropy_std_bits": _std(final_entropy_bits),
                "final_top1_conf": _mean(final_top1_conf),
                "final_margin": _mean(final_margin),
                "final_top1_acc": _mean(final_top1_acc),
                "token_loss_bits": _mean(token_loss_bits),
                "token_loss_std_bits": _std(token_loss_bits),
                "lexical_conf_mean": 0.0,
                "lexical_conf_entropy_bits": 0.0,
                "lexical_conf_high_rate": 0.0,
                "lexical_conf_low_rate": 0.0,
                "mid_feature_rms": 0.0,
                "mid_entropy_bits": 0.0,
                "entropy_gap_bits": 0.0,
                "mid_loss_bits": 0.0,
                "mid_top1_conf": 0.0,
                "mid_final_kl_bits": 0.0,
                "mid_top1_agreement": 0.0,
                "mid_final_top1_flip_rate": 0.0,
            }

            lexical_confidence = lexical_shortcut_confidence_from_logits(
                lexical_shortcut_logits,
                self.lexical_confidence_scale,
                self.lexical_confidence_bias,
            )
            if lexical_confidence is not None:
                lexical_conf = lexical_confidence.squeeze(-1).float().clamp(1e-6, 1.0 - 1e-6)
                lexical_entropy = -(lexical_conf * torch.log2(lexical_conf) + (1.0 - lexical_conf) * torch.log2(1.0 - lexical_conf))
                diag["lexical_conf_mean"] = _mean(lexical_conf)
                diag["lexical_conf_entropy_bits"] = _mean(lexical_entropy)
                diag["lexical_conf_high_rate"] = _mean((lexical_conf >= 0.9).to(dtype=torch.float32))
                diag["lexical_conf_low_rate"] = _mean((lexical_conf <= 0.1).to(dtype=torch.float32))

            if h_mid is not None:
                if h_mid.size(1) == target_ids.size(1):
                    diag["mid_feature_rms"] = _masked_feature_rms(h_mid)
                    mid_logits = self._mid_aux_logits(h_mid)
                    mid_log_probs = F.log_softmax(mid_logits.float(), dim=-1)
                    mid_probs = mid_log_probs.exp()
                    mid_entropy_bits = -(mid_probs * (mid_log_probs / ln2)).sum(dim=-1)
                    mid_top1_conf = mid_probs.amax(dim=-1)
                    mid_loss_bits = F.cross_entropy(
                        mid_logits.reshape(-1, mid_logits.size(-1)).float(),
                        target_ids.reshape(-1),
                        reduction="none",
                    ).view_as(target_ids) / ln2
                    mid_top1 = mid_logits.argmax(dim=-1)
                    mid_top1_agreement = mid_top1.eq(final_top1).to(dtype=torch.float32)
                    mid_final_top1_flip_rate = mid_top1.ne(final_top1).to(dtype=torch.float32)
                    mid_final_kl_bits = (probs * (log_probs - mid_log_probs)).sum(dim=-1) / ln2
                    diag["mid_entropy_bits"] = _mean(mid_entropy_bits)
                    diag["entropy_gap_bits"] = _mean(mid_entropy_bits - final_entropy_bits)
                    diag["mid_loss_bits"] = _mean(mid_loss_bits)
                    diag["mid_top1_conf"] = _mean(mid_top1_conf)
                    diag["mid_final_kl_bits"] = _mean(mid_final_kl_bits)
                    diag["mid_top1_agreement"] = _mean(mid_top1_agreement)
                    diag["mid_final_top1_flip_rate"] = _mean(mid_final_top1_flip_rate)

            return diag
        finally:
            if was_training:
                self.train()

def init_ema_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone().float() for name, tensor in module.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state: dict[str, Tensor], module: nn.Module, decay: float) -> None:
    one_minus_decay = 1.0 - decay
    for name, tensor in module.state_dict().items():
        ema_state[name].mul_(decay).add_(tensor.detach().float(), alpha=one_minus_decay)


def fake_quant_active(args: Hyperparameters, step: int, elapsed_ms: float, lr_scale: float) -> bool:
    if args.fake_quant_bits <= 0:
        return False
    if args.fake_quant_full_run:
        return True
    if args.late_qat and lr_scale <= args.qat_threshold:
        return True
    if args.fake_quant_tail_steps <= 0:
        return False
    if step >= max(args.iterations - args.fake_quant_tail_steps, 0):
        return True
    if args.max_wallclock_seconds > 0 and step > 0:
        step_ms = elapsed_ms / step
        remaining_ms = 1000.0 * args.max_wallclock_seconds - elapsed_ms
        return remaining_ms <= args.fake_quant_tail_steps * step_ms
    return False


def bounded_sigmoid(logits: Tensor, margin: float) -> Tensor:
    margin = min(max(float(margin), 0.0), 0.49)
    probs = torch.sigmoid(logits)
    if margin <= 0.0:
        return probs
    return probs * (1.0 - 2.0 * margin) + margin


def inverse_softplus_scalar(value: float) -> float:
    value = max(float(value), 1e-6)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def linear_anneal_multiplier(step: int, full_steps: int, decay_end_step: int) -> float:
    step = max(int(step), 0)
    full_steps = max(int(full_steps), 0)
    decay_end_step = max(int(decay_end_step), 0)
    if decay_end_step <= 0:
        return 1.0
    if decay_end_step <= full_steps:
        return 1.0 if step <= full_steps else 0.0
    if step <= full_steps:
        return 1.0
    if step >= decay_end_step:
        return 0.0
    return 1.0 - float(step - full_steps) / float(decay_end_step - full_steps)


def _clamp_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def compute_prior_effective_caps(
    train_tokens_seen: int,
    release_train_tokens: dict[str, int | None],
    base_health: float | dict[str, float],
    args: Hyperparameters,
) -> dict[str, float]:
    specs = {
        "bigram": (args.prior_bigram_cap_floor, args.prior_bigram_ramp_tokens),
        "lexical": (args.prior_lexical_cap_floor, args.prior_lexical_ramp_tokens),
    }
    caps: dict[str, float] = {}
    for name, (floor, ramp_tokens) in specs.items():
        floor = _clamp_unit(floor)
        if isinstance(base_health, dict):
            health = _clamp_unit(float(base_health.get(name, 0.0)))
        else:
            health = _clamp_unit(base_health)
        release_at = release_train_tokens.get(name)
        if release_at is None:
            caps[name] = floor
            continue
        progress = max(int(train_tokens_seen) - int(release_at), 0)
        ramp = _clamp_unit(progress / max(int(ramp_tokens), 1))
        caps[name] = floor + (1.0 - floor) * health * ramp
    return caps


def load_initial_model_state(model: nn.Module, path: str, strict: bool) -> tuple[list[str], list[str]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported INIT_MODEL_PATH payload type: {type(payload).__name__}")
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(getattr(incompatible, "missing_keys", [])), list(getattr(incompatible, "unexpected_keys", []))


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global BOS_ID, zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    configure_runtime_export(args)
    if args.enable_torch_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    launched_with_dist_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = launched_with_dist_env and world_size > 1
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps_override = os.environ.get("GRAD_ACCUM_STEPS")
    if grad_accum_steps_override is None:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
        grad_accum_steps = 8 // world_size
    else:
        grad_accum_steps = int(grad_accum_steps_override)
        if grad_accum_steps <= 0:
            raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {grad_accum_steps}")
    grad_scale = 1.0 / grad_accum_steps
    if args.train_batch_tokens % (world_size * grad_accum_steps * args.train_seq_len) != 0:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN; "
            f"got TRAIN_BATCH_TOKENS={args.train_batch_tokens}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(args.enable_cudnn_sdp)
    enable_flash_sdp(args.enable_flash_sdp)
    enable_mem_efficient_sdp(args.enable_mem_efficient_sdp)
    enable_math_sdp(args.enable_math_sdp)

    raw_model_path = "final_model.pt"
    quant_model_path = "final_model.int8.ptz"
    best_raw_model_path = args.best_val_bpb_checkpoint_path or "best_val_bpb_model.pt"
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = os.path.join("logs", f"{args.run_id}.txt")
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer_kind = ""
    if args.tokenizer_path.endswith(".model"):
        tokenizer_kind = "sentencepiece"
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        tokenizer_vocab_size = int(sp.vocab_size())
        bos_token_id = int(sp.bos_id())
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            sp, args.vocab_size, device
        )
    elif args.tokenizer_path.endswith(".json"):
        from tokenizers import Tokenizer

        tokenizer_kind = "tokenizers_json"
        tok = Tokenizer.from_file(args.tokenizer_path)
        tokenizer_vocab_size = int(tok.get_vocab_size())
        bos_token_id = tok.token_to_id("<s>")
        if bos_token_id is None:
            bos_token_id = tok.token_to_id("<bos>")
        if bos_token_id is None:
            bos_token_id = -1
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_tokenizers_json_luts(
            args.tokenizer_path, args.vocab_size, device
        )
    else:
        raise ValueError(f"Unsupported tokenizer format for TOKENIZER_PATH={args.tokenizer_path!r}; expected .model or .json")
    if tokenizer_vocab_size != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={tokenizer_vocab_size}"
        )
    BOS_ID = int(bos_token_id)
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    total_train_tokens = count_total_train_tokens(args.train_files)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    eval_seq_len = get_eval_seq_len(args)
    val_tokens_eval = val_tokens if eval_seq_len == args.train_seq_len else load_validation_tokens(args.val_files, eval_seq_len)
    log0(f"tokenizer:{tokenizer_kind} dataset:{dataset_name} train_tokens:{total_train_tokens}")
    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    budget_started_at = time.perf_counter()

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        tie_embeddings=args.tie_embeddings,
        use_output_logit_bias=args.use_output_logit_bias,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        overtone_embed_init=args.overtone_embed_init,
        overtone_embed_power=args.overtone_embed_power,
        overtone_embed_scale=args.overtone_embed_scale,
        resid_mix_phase_init=args.resid_mix_phase_init,
        resid_mix_phase_sharpness=args.resid_mix_phase_sharpness,
        resid_mix_phase_center=args.resid_mix_phase_center,
        use_alibi=args.use_alibi,
        num_shared_layers=args.num_shared_layers,
        shared_layer_repeats=args.shared_layer_repeats,
        shared_loop_gate_init=args.shared_loop_gate_init,
        use_residual_attention=args.use_residual_attention,
        residual_attention_gain_init=args.residual_attention_gain_init,
        helper_path_full_steps=args.helper_path_full_steps,
        helper_path_decay_end_step=args.helper_path_decay_end_step,
        attention_kv_mode=args.attention_kv_mode,
        use_smear_gate=args.use_smear_gate,
        smear_gate_init=args.smear_gate_init,
        smear_lexical_scale_init=args.smear_lexical_scale_init,
        smear_confidence_gain_init=args.smear_confidence_gain_init,
        smear_lexical_proj_init_std=args.smear_lexical_proj_init_std,
        lexical_enable_step=args.lexical_enable_step,
        lexical_ramp_steps=args.lexical_ramp_steps,
        lexical_calibration_enable_step=args.lexical_calibration_enable_step,
        lexical_calibration_ramp_steps=args.lexical_calibration_ramp_steps,
        lexical_hidden_enable_step=args.lexical_hidden_enable_step,
        lexical_hidden_ramp_steps=args.lexical_hidden_ramp_steps,
        lexical_shortcut_enable_step=args.lexical_shortcut_enable_step,
        lexical_shortcut_ramp_steps=args.lexical_shortcut_ramp_steps,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        hashed_ngram_order=args.hashed_ngram_order,
        hashed_ngram_init_std=args.hashed_ngram_init_std,
        bigram_hidden_scale_init=args.bigram_hidden_scale_init,
        bigram_scale_init=args.bigram_scale_init,
        bigram_hidden_min_mult=args.bigram_hidden_min_mult,
        prior_gate_init=args.prior_gate_init,
        lexical_shortcut_scale_init=args.lexical_shortcut_scale_init,
        lexical_confidence_bias_init=args.lexical_confidence_bias_init,
        use_pairwise_logit_prior=args.use_pairwise_logit_prior,
        pairwise_logit_rank=args.pairwise_logit_rank,
        pairwise_logit_scale_init=args.pairwise_logit_scale_init,
        pairwise_logit_gate_init=args.pairwise_logit_gate_init,
        pairwise_logit_confidence_gain_init=args.pairwise_logit_confidence_gain_init,
        use_online_doc_bias=args.use_online_doc_bias,
        doc_bias_rank=args.doc_bias_rank,
        doc_bias_scale_init=args.doc_bias_scale_init,
        doc_bias_gate_init=args.doc_bias_gate_init,
        use_confidence_branch_skip=args.use_confidence_branch_skip,
        confidence_skip_margin=args.confidence_skip_margin,
        confidence_skip_sharpness=args.confidence_skip_sharpness,
        confidence_skip_lexical_min_keep=args.confidence_skip_lexical_min_keep,
        confidence_skip_pairwise_min_keep=args.confidence_skip_pairwise_min_keep,
        confidence_skip_doc_bias_min_keep=args.confidence_skip_doc_bias_min_keep,
        use_residual_logit_adapter=args.use_residual_logit_adapter,
        residual_logit_rank=args.residual_logit_rank,
        residual_logit_scale_init=args.residual_logit_scale_init,
        residual_logit_gate_init=args.residual_logit_gate_init,
        lexical_batch_diversity_throttle=args.lexical_batch_diversity_throttle,
        lexical_batch_repeat_throttle=args.lexical_batch_repeat_throttle,
        lexical_batch_min_mult=args.lexical_batch_min_mult,
        shared_tail_output_gate=args.shared_tail_output_gate,
        shared_tail_output_init=args.shared_tail_output_init,
        shared_tail_enable_step=args.shared_tail_enable_step,
        shared_tail_ramp_steps=args.shared_tail_ramp_steps,
        shared_tail_max_mult=args.shared_tail_max_mult,
        signed_skip_weights=args.signed_skip_weights,
        orthogonal_init=args.orthogonal_init,
        mup_proj_init=args.mup_proj_init,
        train_seq_len=args.train_seq_len,
        use_flash_attn_3=args.use_flash_attn_3,
        rope_dims=args.rope_dims,
        debug_static_shapes=args.debug_static_shapes,
        residual_attention_state_bits=args.residual_attention_state_bits,
        manual_attention_query_chunk_size=args.manual_attention_query_chunk_size,
        manual_attention_key_chunk_size=args.manual_attention_key_chunk_size,
        ln_scale=args.ln_scale,
        use_adaptive_rmsnorm=args.use_adaptive_rmsnorm,
        adaptive_rmsnorm_gate_init=args.adaptive_rmsnorm_gate_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if args.init_model_path:
        missing_keys, unexpected_keys = load_initial_model_state(
            base_model,
            args.init_model_path,
            strict=args.init_model_strict,
        )
        log0(
            f"init_model:loaded path:{args.init_model_path} strict:{int(args.init_model_strict)} "
            f"missing:{len(missing_keys)} unexpected:{len(unexpected_keys)}"
        )
        if missing_keys:
            log0(f"init_model_missing:{','.join(missing_keys[:12])}")
        if unexpected_keys:
            log0(f"init_model_unexpected:{','.join(unexpected_keys[:12])}")
    enable_model_compile = args.enable_torch_compile
    if enable_model_compile:
        pass
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=False)
        if enable_model_compile
        else base_model
    )
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    ema_state = init_ema_state(base_model) if args.ema_decay > 0.0 else None
    ema_state_updated = False
    if not args.use_cuda_graphs:
        cuda_graph_disable_reason = "disabled"
    else:
        cuda_graph_disable_reason = "eligible"
    cuda_graph_eligible = cuda_graph_disable_reason == "eligible"

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    embed_params = [base_model.tok_emb.weight]
    if base_model.bigram is not None:
        embed_params.append(base_model.bigram.embed.weight)
    if getattr(base_model, "lexical_residual_ngram", None) is not None:
        embed_params.append(base_model.lexical_residual_ngram.embed.weight)
    if getattr(base_model, "pairwise_prev_embed", None) is not None:
        embed_params.append(base_model.pairwise_prev_embed.weight)
    if getattr(base_model, "doc_bias_embed", None) is not None:
        embed_params.append(base_model.doc_bias_embed.weight)
    block_named_params = (
        [(f"blocks.{name}", p) for name, p in base_model.blocks.named_parameters()]
        + [(f"shared_blocks.{name}", p) for name, p in base_model.shared_blocks.named_parameters()]
    )
    attn_matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and ".attn." in name
    ]
    mlp_matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and ".mlp." in name
    ]
    other_matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        and ".attn." not in name
        and ".mlp." not in name
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.shared_loop_gates is not None:
        scalar_params.append(base_model.shared_loop_gates)
    if base_model.shared_tail_gate is not None:
        scalar_params.append(base_model.shared_tail_gate)
    if base_model.smear is not None:
        scalar_params.append(base_model.smear.gate)
        scalar_params.append(base_model.smear.confidence_gain)
        scalar_params.append(base_model.smear.lexical_scale)
        other_matrix_params.append(base_model.smear.state_proj.weight)
        other_matrix_params.append(base_model.smear.lexical_proj.weight)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
        scalar_params.append(base_model.bigram.gate_logit)
        if base_model.bigram.proj is not None:
            other_matrix_params.append(base_model.bigram.proj.weight)
    if getattr(base_model, "lexical_residual_ngram", None) is not None:
        scalar_params.append(base_model.lexical_residual_ngram.scale)
        scalar_params.append(base_model.lexical_residual_ngram.gate_logit)
        if base_model.lexical_residual_ngram.proj is not None:
            other_matrix_params.append(base_model.lexical_residual_ngram.proj.weight)
    if base_model.lexical_shortcut_scale is not None:
        scalar_params.append(base_model.lexical_shortcut_scale)
    if base_model.lexical_confidence_scale is not None:
        scalar_params.append(base_model.lexical_confidence_scale)
    if base_model.lexical_confidence_bias is not None:
        scalar_params.append(base_model.lexical_confidence_bias)
    if base_model.lexical_shortcut_head is not None:
        other_matrix_params.append(base_model.lexical_shortcut_head.weight)
    if getattr(base_model, "lexical_residual_scale", None) is not None:
        scalar_params.append(base_model.lexical_residual_scale)
    if getattr(base_model, "lexical_residual_confidence_scale", None) is not None:
        scalar_params.append(base_model.lexical_residual_confidence_scale)
    if getattr(base_model, "lexical_residual_confidence_bias", None) is not None:
        scalar_params.append(base_model.lexical_residual_confidence_bias)
    if getattr(base_model, "lexical_residual_head", None) is not None:
        other_matrix_params.append(base_model.lexical_residual_head.weight)
    if getattr(base_model, "lexical_prior_gate_logit", None) is not None:
        scalar_params.append(base_model.lexical_prior_gate_logit)
    if getattr(base_model, "output_logit_bias", None) is not None:
        scalar_params.append(base_model.output_logit_bias)
    if getattr(base_model, "pairwise_logit_scale", None) is not None:
        scalar_params.append(base_model.pairwise_logit_scale)
    if getattr(base_model, "pairwise_logit_gate", None) is not None:
        scalar_params.append(base_model.pairwise_logit_gate)
    if getattr(base_model, "pairwise_logit_hidden_gate", None) is not None:
        scalar_params.append(base_model.pairwise_logit_hidden_gate)
    if getattr(base_model, "pairwise_logit_confidence_gain", None) is not None:
        scalar_params.append(base_model.pairwise_logit_confidence_gain)
    if getattr(base_model, "pairwise_logit_head", None) is not None:
        other_matrix_params.append(base_model.pairwise_logit_head.weight)
    if getattr(base_model, "doc_bias_scale", None) is not None:
        scalar_params.append(base_model.doc_bias_scale)
    if getattr(base_model, "doc_bias_gate", None) is not None:
        scalar_params.append(base_model.doc_bias_gate)
    if getattr(base_model, "doc_bias_head", None) is not None:
        other_matrix_params.append(base_model.doc_bias_head.weight)
    if getattr(base_model, "residual_logit_scale", None) is not None:
        scalar_params.append(base_model.residual_logit_scale)
    if getattr(base_model, "residual_logit_gate", None) is not None:
        scalar_params.append(base_model.residual_logit_gate)
    if getattr(base_model, "residual_logit_down", None) is not None:
        other_matrix_params.append(base_model.residual_logit_down.weight)
    if getattr(base_model, "residual_logit_up", None) is not None:
        other_matrix_params.append(base_model.residual_logit_up.weight)
    if isinstance(base_model.final_norm, AdaptiveRMSNorm):
        scalar_params.append(base_model.final_norm.base_log_scale)
        if base_model.final_norm.cond_gate is not None:
            scalar_params.append(base_model.final_norm.cond_gate)
        if base_model.final_norm.cond_proj is not None:
            other_matrix_params.append(base_model.final_norm.cond_proj.weight)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": embed_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.token_weight_decay,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok]
    muon_optimizers: list[torch.optim.Optimizer] = []
    if attn_matrix_params:
        optimizer_muon_attn = Muon(
            attn_matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            backend_steps_light=args.muon_backend_steps_light,
            backend_refresh_interval=args.muon_backend_refresh_interval,
            weight_decay=args.muon_weight_decay,
        )
        for group in optimizer_muon_attn.param_groups:
            group["base_lr"] = args.matrix_lr
        muon_optimizers.append(optimizer_muon_attn)
        optimizers.append(optimizer_muon_attn)
    if mlp_matrix_params:
        mlp_matrix_lr = args.matrix_lr * args.mlp_matrix_lr_mult
        optimizer_muon_mlp = Muon(
            mlp_matrix_params,
            lr=mlp_matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            backend_steps_light=args.muon_backend_steps_light,
            backend_refresh_interval=args.muon_backend_refresh_interval,
            weight_decay=args.muon_weight_decay,
        )
        for group in optimizer_muon_mlp.param_groups:
            group["base_lr"] = mlp_matrix_lr
        muon_optimizers.append(optimizer_muon_mlp)
        optimizers.append(optimizer_muon_mlp)
    if other_matrix_params:
        optimizer_muon_other = Muon(
            other_matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            backend_steps_light=args.muon_backend_steps_light,
            backend_refresh_interval=args.muon_backend_refresh_interval,
            weight_decay=args.muon_weight_decay,
        )
        for group in optimizer_muon_other.param_groups:
            group["base_lr"] = args.matrix_lr
        muon_optimizers.append(optimizer_muon_other)
        optimizers.append(optimizer_muon_other)
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers.append(optimizer_scalar)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.head_weight_decay,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    positional_mode = "alibi" if args.use_alibi else "rope_only"
    log0(f"positional_mode:{positional_mode} train_seq_len:{args.train_seq_len} iterations:{args.iterations}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        bos_token_id=bos_token_id,
        random_offset_tokens=args.train_random_offset_tokens,
        seed=args.seed,
        debug_static_shapes=args.debug_static_shapes,
    )

    def zero_grad_all(set_to_none: bool = True) -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        scale = 1.0
        if args.lr_warmup_steps > 0:
            warmup_progress = min(max(step, 0) / max(args.lr_warmup_steps, 1), 1.0)
            warmup_progress = warmup_progress ** max(float(args.lr_warmup_power), 1e-6)
            start_scale = min(max(args.lr_warmup_init_scale, 0.0), 1.0)
            scale *= start_scale + (1.0 - start_scale) * warmup_progress
        if args.warmdown_iters <= 0:
            return scale
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                scale *= max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return scale
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            scale *= remaining_ms / max(warmdown_ms, 1e-9)
        return scale

    def tied_embed_lr_mul(step: int) -> float:
        warmup_mult = min(max(args.tied_embed_warmup_mult, 0.0), 1.0)
        warmup_steps = max(int(args.tied_embed_warmup_steps), 0)
        if warmup_steps <= 0:
            return 1.0
        progress = min(max(step, 0) / max(warmup_steps, 1), 1.0)
        return warmup_mult + (1.0 - warmup_mult) * progress

    def early_phase_frac(step: int) -> float:
        if args.early_phase_steps <= 0:
            return 0.0
        return 1.0 - min(max(step, 0) / max(args.early_phase_steps, 1), 1.0)

    def next_train_microbatch() -> tuple[Tensor, Tensor, Tensor | None]:
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        loss_mask = None
        return x, y, loss_mask

    current_train_step = 0
    mid_aux_health_cap = 1.0
    lexical_shortcut_health_gate = 1.0 if args.lexical_shortcut_attn_gain_share_full <= 0.0 else 0.0
    active_mid_aux_loss_coeff = float(args.mid_aux_loss_coeff)
    active_lexical_shortcut_attn_gain_share_min = float(args.lexical_shortcut_attn_gain_share_min)
    active_lexical_shortcut_attn_gain_share_full = float(args.lexical_shortcut_attn_gain_share_full)
    def compute_training_loss(
        x: Tensor,
        y: Tensor,
        loss_mask: Tensor | None,
    ) -> Tensor:
        mid_aux_sched = 1.0
        if args.mid_aux_ramp_steps > 0:
            mid_aux_sched = min(
                max(float(current_train_step - args.mid_aux_enable_step) / float(args.mid_aux_ramp_steps), 0.0),
                1.0,
            )
        elif current_train_step < args.mid_aux_enable_step:
            mid_aux_sched = 0.0
        if args.mid_aux_decay_end_step > args.mid_aux_decay_start_step:
            if current_train_step >= args.mid_aux_decay_end_step:
                mid_aux_sched = 0.0
            elif current_train_step > args.mid_aux_decay_start_step:
                decay_progress = (
                    float(current_train_step - args.mid_aux_decay_start_step)
                    / float(args.mid_aux_decay_end_step - args.mid_aux_decay_start_step)
                )
                mid_aux_sched *= max(1.0 - decay_progress, 0.0)
        current_mid_aux_coeff = active_mid_aux_loss_coeff * mid_aux_sched * mid_aux_health_cap
        current_mid_aux_coeff_t = torch.tensor(current_mid_aux_coeff, device=x.device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            base_model.set_lexical_shortcut_external_gate(lexical_shortcut_health_gate)
            base_model.set_training_step(current_train_step)
            loss = model(
                x,
                y,
                loss_mask=loss_mask,
                label_smoothing=args.label_smoothing,
                z_loss_coeff=args.z_loss_coeff,
                logit_var_loss_coeff=args.logit_var_loss_coeff,
                mid_aux_loss_coeff=current_mid_aux_coeff_t,
            )
            if args.row_norm_loss_coeff > 0.0:
                loss = loss + args.row_norm_loss_coeff * row_norm_regularization(base_model)
            if args.outlier_loss_coeff > 0.0:
                loss = loss + args.outlier_loss_coeff * outlier_regularization(
                    base_model, args.outlier_loss_threshold
                )
        return loss

    def build_train_cuda_graph(
        sample_x: Tensor,
        sample_y: Tensor,
        sample_loss_mask: Tensor | None,
    ) -> dict[str, object]:
        # Keep long-lived buffers stable so replay can swap in new batch contents
        # without changing addresses captured into the graph.
        static_x = torch.empty_like(sample_x)
        static_y = torch.empty_like(sample_y)
        static_loss_mask = torch.empty_like(sample_loss_mask) if sample_loss_mask is not None else None
        static_loss = torch.zeros((), device=device, dtype=torch.float32)
        static_x.copy_(sample_x, non_blocking=True)
        static_y.copy_(sample_y, non_blocking=True)
        if static_loss_mask is not None and sample_loss_mask is not None:
            static_loss_mask.copy_(sample_loss_mask, non_blocking=True)
        zero_grad_all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_loss = compute_training_loss(static_x, static_y, static_loss_mask)
            static_loss.copy_(captured_loss.detach().to(dtype=static_loss.dtype))
            (captured_loss * grad_scale).backward()
        return {
            "graph": graph,
            "static_x": static_x,
            "static_y": static_y,
            "static_loss_mask": static_loss_mask,
            "static_loss": static_loss,
        }

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    warmup_prior_names = (
        warmup_prior_init_param_names(base_model, args.warmup_prior_init_groups)
        if args.warmup_prior_init and args.warmup_steps > 0
        else []
    )
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        warmed_prior_state: dict[str, Tensor] = {}
        model.train()
        base_model.set_fake_quant(0)
        base_model.set_training_step(0)
        for warmup_step in range(args.warmup_steps):
            current_train_step = warmup_step
            warmup_scale = lr_mul(warmup_step, 0.0)
            token_scale = warmup_scale * tied_embed_lr_mul(warmup_step)
            for opt in optimizers:
                scaled_lr = token_scale if opt is optimizer_tok else warmup_scale
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * scaled_lr
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y, loss_mask = next_train_microbatch()
                warmup_loss = compute_training_loss(x, y, loss_mask)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        if warmup_prior_names:
            current_state = base_model.state_dict()
            warmed_prior_state = {
                name: current_state[name].detach().cpu().clone()
                for name in warmup_prior_names
                if name in current_state
            }
        restored_model_state = {name: tensor.detach().clone() for name, tensor in initial_model_state.items()}
        if warmed_prior_state:
            blend = min(max(float(args.warmup_prior_init_blend), 0.0), 1.0)
            transferred_param_count = 0
            transferred_element_count = 0
            for name, warmed_tensor in warmed_prior_state.items():
                if name not in restored_model_state:
                    continue
                restored_model_state[name] = transfer_warmup_state_tensor(
                    restored_model_state[name],
                    warmed_tensor,
                    blend,
                    args.warmup_prior_init_mode,
                    args.warmup_prior_init_delta_rms_mult,
                    args.warmup_prior_init_delta_rms_floor,
                )
                transferred_param_count += 1
                transferred_element_count += int(restored_model_state[name].numel())
            preview = ",".join(warmup_prior_names[:6])
            if len(warmup_prior_names) > 6:
                preview = f"{preview},..."
            log0(
                f"warmup_prior_init:enabled groups:{','.join(args.warmup_prior_init_groups)} "
                f"mode:{args.warmup_prior_init_mode} blend:{blend:.3f} "
                f"params:{transferred_param_count} elements:{transferred_element_count} names:{preview}"
            )
        base_model.load_state_dict(restored_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        current_train_step = 0
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        base_model.set_fake_quant(0)
        train_loader = DistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            bos_token_id=bos_token_id,
            random_offset_tokens=args.train_random_offset_tokens,
            seed=args.seed,
            debug_static_shapes=args.debug_static_shapes,
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 1000.0 * (time.perf_counter() - budget_started_at)
    stop_after_step: int | None = None
    raw_final_val_loss = math.nan
    raw_final_val_bpb = math.nan
    best_val_bpb = math.inf
    best_val_loss = math.nan
    best_val_step = -1
    best_val_train_tokens = 0
    best_bootstrap_val_bpb = math.inf
    recent_train_losses: deque[float] = deque(maxlen=max(int(args.objective_health_train_shock_window), 1))
    regression_bad_validations = 0
    scale = lr_mul(0, training_time_ms)
    module_param_snapshots: dict[str, Tensor] = {}
    module_objective_accumulators: dict[str, dict[str, float]] = {}
    if args.objective_track_module_attribution:
        module_param_snapshots, module_objective_accumulators = init_module_objective_tracking(base_model)
    combined_lexical_release_train_token: int | None = None
    combined_lexical_positive_confirmations = 0
    combined_lexical_health_score = 0.0
    initial_prior_caps = compute_prior_effective_caps(
        train_tokens_seen=0,
        release_train_tokens={
            "bigram": combined_lexical_release_train_token,
            "lexical": combined_lexical_release_train_token,
        },
        base_health={
            "bigram": combined_lexical_health_score,
            "lexical": combined_lexical_health_score,
        },
        args=args,
    )
    base_model.set_prior_caps(
        bigram=initial_prior_caps["bigram"],
        lexical=initial_prior_caps["lexical"],
    )
    cuda_graph_runner: dict[str, object] | None = None
    cuda_graph_runner_fake_quant_bits: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def aux_release_ready(
        *,
        step: int,
        val_bpb: float,
        best_reference_bpb: float,
        diag: dict[str, float],
        attn_gain_share: float,
        mlp_gain_share: float,
    ) -> bool:
        if step < max(int(args.aux_release_min_step), 0):
            return False
        if math.isfinite(best_reference_bpb) and float(val_bpb) > float(best_reference_bpb) + args.prior_health_regression_tol:
            return False
        if float(diag.get("mid_top1_agreement", 0.0)) >= args.aux_release_max_mid_top1_agreement:
            return False
        if float(attn_gain_share + mlp_gain_share) < args.aux_release_min_backbone_gain_share:
            return False
        window = max(int(args.objective_health_train_shock_window), 3)
        if len(recent_train_losses) < window:
            return False
        losses = list(recent_train_losses)[-window:]
        baseline = min(losses)
        if baseline <= 0.0:
            return False
        if losses[-1] > losses[0] * max(float(args.aux_release_train_loss_tol), 1.0):
            return False
        if max(losses) / baseline > args.objective_health_train_shock_ratio:
            return False
        return True

    def maybe_update_prior_releases(
        *,
        step: int,
        train_tokens_seen: int,
        bootstrap_active: bool,
        aux_release_ready: bool,
        val_bpb: float,
        best_reference_bpb: float,
        counterfactual_results: list[dict[str, float | str]],
        lexical_module: dict[str, float],
    ) -> None:
        nonlocal combined_lexical_release_train_token
        nonlocal combined_lexical_positive_confirmations
        nonlocal combined_lexical_health_score
        regressing = math.isfinite(best_reference_bpb) and float(val_bpb) > float(best_reference_bpb) + args.prior_health_regression_tol
        evidence = False
        negative_evidence = False
        cf_by_group = {str(item["group"]): item for item in counterfactual_results}
        lexical_cf = cf_by_group.get("lexical")
        if lexical_cf is not None:
            lexical_delta = float(lexical_cf.get("delta_val_bpb", 0.0))
            evidence = lexical_delta >= args.prior_health_min_improvement and not regressing
            negative_evidence = lexical_delta <= -args.prior_health_min_improvement
        else:
            lexical_gain = float(lexical_module.get("gain_per_update_norm", 0.0))
            evidence = lexical_gain >= args.prior_health_threshold and not regressing

        if not aux_release_ready:
            evidence = False
            negative_evidence = False

        if evidence:
            combined_lexical_positive_confirmations += 1
            combined_lexical_health_score = min(
                1.0,
                combined_lexical_health_score + args.prior_health_recovery,
            )
        else:
            combined_lexical_positive_confirmations = 0
            if negative_evidence or regressing:
                combined_lexical_health_score = max(
                    0.0,
                    combined_lexical_health_score - args.prior_health_decay,
                )
        if (
            combined_lexical_release_train_token is None
            and combined_lexical_positive_confirmations >= max(int(args.prior_health_confirmations), 1)
        ):
            combined_lexical_release_train_token = int(train_tokens_seen)
        effective_caps = compute_prior_effective_caps(
            train_tokens_seen=train_tokens_seen,
            release_train_tokens={
                "bigram": combined_lexical_release_train_token,
                "lexical": combined_lexical_release_train_token,
            },
            base_health={
                "bigram": combined_lexical_health_score,
                "lexical": combined_lexical_health_score,
            },
            args=args,
        )
        base_model.set_prior_caps(
            bigram=effective_caps["bigram"],
            lexical=effective_caps["lexical"],
        )

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        train_tokens_seen = int(step) * int(args.train_batch_tokens)
        bootstrap_active = (
            step < max(int(args.objective_bootstrap_steps), 0)
            or train_tokens_seen < max(int(args.objective_bootstrap_tokens), 0)
        )
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            current_train_step = step
            base_model.set_training_step(current_train_step)
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            validation_max_seqs = args.val_max_seqs
            val_loss, val_bpb, _val_mode, _val_bits_per_token, _val_tokens_per_byte, _val_bytes_per_token = run_validation(
                args,
                model,
                base_model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                max_seqs=validation_max_seqs,
            )
            if bootstrap_active and math.isfinite(float(val_bpb)):
                best_bootstrap_val_bpb = min(best_bootstrap_val_bpb, float(val_bpb))
            diag_x, diag_y, diag_loss_mask = build_validation_diagnostic_batch(args, val_tokens_eval, device)
            diag = base_model.convergence_diagnostics(diag_x, diag_y, loss_mask=diag_loss_mask)
            specialist_diag = specialist_path_metrics(base_model)
            entropy_gap_bits = float(diag["entropy_gap_bits"])
            mid_final_kl_bits = float(diag["mid_final_kl_bits"])
            final_entropy_bits = float(diag["final_entropy_bits"])
            final_top1_conf = float(diag["final_top1_conf"])
            pre_final_feature_rms = float(diag["pre_final_feature_rms"])
            pre_softcap_logit_std = float(diag["pre_softcap_logit_std"])
            module_objective_summary = (
                summarize_module_objective_tracking(
                    base_model,
                    module_objective_accumulators,
                    args.export_quant_bits,
                    args.force_fp16_tied_embed_export,
                )
                if args.objective_track_module_attribution
                else {}
            )
            counterfactual_groups = parse_counterfactual_groups(args.objective_counterfactual_groups)
            should_run_counterfactual = (
                args.objective_counterfactual_eval
                and bool(counterfactual_groups)
                and int(args.objective_counterfactual_max_seqs) > 0
                and (not args.objective_counterfactual_bootstrap_only or bootstrap_active)
                and (
                    last_step
                    or (
                        max(int(args.objective_counterfactual_interval), 0) > 0
                        and step > 0
                        and step % max(int(args.objective_counterfactual_interval), 1) == 0
                    )
                )
            )
            attn_module = module_objective_summary.get("attn", {})
            mlp_module = module_objective_summary.get("mlp", {})
            lexical_module = module_objective_summary.get("lexical", {})
            bootstrap_total_gain = sum(float(stats.get("gain", 0.0)) for stats in module_objective_summary.values())
            attn_gain = float(attn_module.get("gain", 0.0))
            mlp_gain = float(mlp_module.get("gain", 0.0))
            attn_gain_share = attn_gain / max(bootstrap_total_gain, 1e-12)
            aux_release_ready_flag = aux_release_ready(
                step=step,
                val_bpb=float(val_bpb),
                best_reference_bpb=best_bootstrap_val_bpb if bootstrap_active else best_val_bpb,
                diag=diag,
                attn_gain_share=attn_gain_share,
                mlp_gain_share=float(mlp_gain / max(bootstrap_total_gain, 1e-12)),
            )
            counterfactual_results: list[dict[str, float | str]] = []
            if should_run_counterfactual:
                counterfactual_max_seqs = min(
                    validation_max_seqs if validation_max_seqs > 0 else args.objective_counterfactual_max_seqs,
                    args.objective_counterfactual_max_seqs,
                )
                _counterfactual_base_metrics, counterfactual_results = run_counterfactual_group_eval(
                    args=args,
                    model=model,
                    base_model=base_model,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    grad_accum_steps=grad_accum_steps,
                    val_tokens=val_tokens,
                    base_bytes_lut=base_bytes_lut,
                    has_leading_space_lut=has_leading_space_lut,
                    is_boundary_token_lut=is_boundary_token_lut,
                    seq_len_override=0,
                    max_seqs=counterfactual_max_seqs,
                    groups=counterfactual_groups,
                    module_objective_summary=module_objective_summary,
                )
            maybe_update_prior_releases(
                step=step,
                train_tokens_seen=train_tokens_seen,
                bootstrap_active=bootstrap_active,
                aux_release_ready=aux_release_ready_flag,
                val_bpb=float(val_bpb),
                best_reference_bpb=best_bootstrap_val_bpb if bootstrap_active else best_val_bpb,
                counterfactual_results=counterfactual_results,
                lexical_module=lexical_module,
            )
            log0(
                f"step:{step}/{args.iterations} "
                f"val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            raw_final_val_loss = val_loss
            raw_final_val_bpb = val_bpb
            improved_best = math.isfinite(float(val_bpb)) and (
                best_val_step < 0 or float(val_bpb) < best_val_bpb - args.regression_stop_min_delta
            )
            if improved_best:
                best_val_bpb = float(val_bpb)
                best_val_loss = float(val_loss)
                best_val_step = int(step)
                best_val_train_tokens = int(train_tokens_seen)
                regression_bad_validations = 0
                if master_process and args.save_best_val_bpb_checkpoint:
                    checkpoint_payload = {
                        "model_state_dict": base_model.state_dict(),
                        "step": best_val_step,
                        "train_tokens_seen": best_val_train_tokens,
                        "val_bpb": best_val_bpb,
                        "val_loss": best_val_loss,
                        "run_id": args.run_id,
                    }
                    torch.save(checkpoint_payload, best_raw_model_path)
            elif (
                args.regression_stop_patience > 0
                and step >= args.regression_stop_min_step
                and best_val_step >= 0
                and float(val_bpb) > best_val_bpb + args.regression_stop_min_delta
            ):
                regression_bad_validations += 1
                if regression_bad_validations >= args.regression_stop_patience and stop_after_step is None:
                    stop_after_step = int(step)
            else:
                regression_bad_validations = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        fake_quant_bits = args.fake_quant_bits if fake_quant_active(args, step, elapsed_ms, scale) else 0
        base_model.set_fake_quant(fake_quant_bits)
        graph_step_active = cuda_graph_eligible and step >= max(args.cuda_graph_warmup_steps, 0)
        if cuda_graph_runner is not None and cuda_graph_runner_fake_quant_bits != fake_quant_bits:
            cuda_graph_runner = None
            cuda_graph_runner_fake_quant_bits = None
        zero_grad_all(set_to_none=not (graph_step_active and cuda_graph_runner is not None))
        train_loss = torch.zeros((), device=device)
        last_x = None
        last_y = None
        last_loss_mask = None
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y, loss_mask = next_train_microbatch()
            last_x = x
            last_y = y
            last_loss_mask = loss_mask
            loss_value: Tensor | None = None
            used_cuda_graph = False
            if graph_step_active:
                if cuda_graph_runner is None:
                    try:
                        cuda_graph_runner = build_train_cuda_graph(x, y, loss_mask)
                        cuda_graph_runner_fake_quant_bits = fake_quant_bits
                        used_cuda_graph = True
                        loss_value = cuda_graph_runner["static_loss"]  # type: ignore[assignment]
                    except Exception as exc:
                        cuda_graph_eligible = False
                        cuda_graph_disable_reason = f"capture_failed:{type(exc).__name__}"
                        cuda_graph_runner = None
                        cuda_graph_runner_fake_quant_bits = None
                        zero_grad_all()
                else:
                    static_x = cuda_graph_runner["static_x"]
                    static_y = cuda_graph_runner["static_y"]
                    static_loss_mask = cuda_graph_runner["static_loss_mask"]
                    static_x.copy_(x, non_blocking=True)
                    static_y.copy_(y, non_blocking=True)
                    if static_loss_mask is not None and loss_mask is not None:
                        static_loss_mask.copy_(loss_mask, non_blocking=True)
                    cuda_graph_runner["graph"].replay()
                    used_cuda_graph = True
                    loss_value = cuda_graph_runner["static_loss"]  # type: ignore[assignment]
            if not used_cuda_graph:
                loss = compute_training_loss(x, y, loss_mask)
                loss_value = loss.detach()
                (loss * grad_scale).backward()
            train_loss += loss_value
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in muon_optimizers:
            for group in opt.param_groups:
                group["momentum"] = muon_momentum

        early_frac = early_phase_frac(step)
        attn_lr_scale = 1.0 + early_frac * (args.early_muon_attn_lr_scale - 1.0)
        mlp_lr_scale = 1.0 + early_frac * (args.early_muon_mlp_lr_scale - 1.0)
        muon_wd_scale = 1.0 + early_frac * (args.early_muon_wd_scale - 1.0)

        token_scale = scale * tied_embed_lr_mul(step)
        for opt in optimizers:
            scaled_lr = token_scale if opt is optimizer_tok else scale
            if opt is optimizer_muon_attn:
                scaled_lr *= attn_lr_scale
            elif opt is optimizer_muon_mlp:
                scaled_lr *= mlp_lr_scale
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scaled_lr
                if opt is optimizer_muon_attn or opt is optimizer_muon_mlp:
                    group["weight_decay"] = args.muon_weight_decay * muon_wd_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        module_grad_snapshots: dict[str, Tensor] | None = None
        module_update_snapshots: dict[str, Tensor] | None = None
        if args.objective_track_module_attribution and (step + 1) % max(int(args.objective_module_attribution_interval), 1) == 0:
            module_grad_snapshots = capture_module_objective_grads(base_model)
        for opt in optimizers:
            opt.step()
        if args.objective_track_module_attribution and (step + 1) % max(int(args.objective_module_attribution_interval), 1) == 0:
            module_update_snapshots = capture_module_objective_optimizer_updates(base_model, optimizers)
            accumulate_module_objective_tracking(
                base_model,
                module_param_snapshots,
                module_objective_accumulators,
                grad_snapshots=module_grad_snapshots,
                update_snapshots=module_update_snapshots,
            )
        zero_grad_all(set_to_none=not (graph_step_active and cuda_graph_runner is not None))
        if ema_state is not None and step + 1 >= args.ema_start_step:
            update_ema_state(ema_state, base_model, args.ema_decay)
            ema_state_updated = True

        train_loss_value = float(train_loss.item())
        recent_train_losses.append(train_loss_value)
        if args.objective_track_module_attribution:
            module_objective_summary_gate = summarize_module_objective_tracking(
                base_model,
                module_objective_accumulators,
                args.export_quant_bits,
                args.force_fp16_tied_embed_export,
            )
            gate_total_gain = sum(float(stats.get("gain", 0.0)) for stats in module_objective_summary_gate.values())
            gate_attn_gain = float(module_objective_summary_gate.get("attn", {}).get("gain", 0.0))
            gate_attn_gain_share = gate_attn_gain / max(gate_total_gain, 1e-12)
            gate_lo = float(active_lexical_shortcut_attn_gain_share_min)
            gate_hi = max(float(active_lexical_shortcut_attn_gain_share_full), gate_lo + 1e-6)
            lexical_shortcut_health_gate = min(max((gate_attn_gain_share - gate_lo) / (gate_hi - gate_lo), 0.0), 1.0)
        else:
            lexical_shortcut_health_gate = 1.0

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if ema_state is not None and ema_state_updated:
        base_model.load_state_dict(ema_state, strict=True)

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), raw_model_path)
        model_bytes = os.path.getsize(raw_model_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    export_state_dict = canonicalize_state_dict_for_export(base_model.state_dict())
    if args.export_quant_bits < 8:
        quant_obj, mixed_stats, mixed_precision_info = quantize_state_dict_mixed_precision(
            export_state_dict, bits=args.export_quant_bits
        )
        quant_stats = {
            "param_count": mixed_stats["param_count"],
            "num_tensors": mixed_stats["num_tensors"],
            "num_float_tensors": mixed_stats["num_float_tensors"],
            "num_nonfloat_tensors": mixed_stats["num_nonfloat_tensors"],
            "baseline_tensor_bytes": mixed_stats["baseline_tensor_bytes"],
            "int8_payload_bytes": mixed_stats["payload_bytes"],
            "auto_keep_count": mixed_stats["high_precision_tensor_count"],
            "auto_keep_extra_bytes": mixed_stats["high_precision_extra_bytes"],
            "auto_keep_row_group_count": 0,
            "auto_keep_row_group_extra_bytes": 0,
            "block_pruned_tensor_count": mixed_stats["block_pruned_tensor_count"],
            "block_prune_estimated_bytes_saved": mixed_stats["block_prune_estimated_bytes_saved"],
        }
        auto_keep_info = {**mixed_precision_info, "row_groups": {}}
    else:
        quant_obj, quant_stats, auto_keep_info = quantize_state_dict_int8(export_state_dict)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob, used_codec = compress_blob(quant_raw, args.export_codec, args.export_zstd_level)
    quant_raw_bytes = len(quant_raw)
    submission_over_limit = False
    quant_file_bytes = None
    total_submission_bytes = None
    if master_process:
        with open(quant_model_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(quant_model_path)
        code_bytes = len(code.encode("utf-8"))
        total_submission_bytes = quant_file_bytes + code_bytes
        submission_over_limit = total_submission_bytes > args.submission_size_limit_bytes

    if distributed:
        over_limit_tensor = torch.tensor(int(submission_over_limit), device=device)
        dist.all_reduce(over_limit_tensor, op=dist.ReduceOp.MAX)
        submission_over_limit = bool(over_limit_tensor.item())
    if submission_over_limit:
        raise RuntimeError(
            "Final int8+zlib artifact exceeds SUBMISSION_SIZE_LIMIT_BYTES; "
            "adjust model/code size or quantization settings."
        )

    if distributed:
        dist.barrier()
    with open(quant_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_blob(quant_blob_disk, used_codec)), map_location="cpu")
    if args.export_quant_bits < 8:
        base_model.load_state_dict(dequantize_state_dict_mixed_precision(quant_state), strict=True)
    else:
        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    final_eval_is_exact = True
    if uses_sliding_eval(args):
        final_eval_is_exact = args.final_eval_max_seqs <= 0
        q_val_loss, q_val_bpb, _, _, _ = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens_eval,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            seq_len_override=eval_seq_len,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
            max_seqs=args.final_eval_max_seqs,
        )
    else:
        final_eval_is_exact = args.final_eval_max_seqs <= 0
        q_val_loss, q_val_bpb = eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens_eval,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            seq_len_override=eval_seq_len,
            max_seqs=args.final_eval_max_seqs,
        )
    torch.cuda.synchronize()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
