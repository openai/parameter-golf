from __future__ import annotations

import os
import uuid

import torch


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    experiment_name = os.environ.get("EXPERIMENT_NAME", "")
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    use_mhc = bool(int(os.environ.get("USE_MHC", "0")))
    mhc_type = os.environ.get("MHC_TYPE", "mhc")
    mhc_num_streams = int(os.environ.get("MHC_NUM_STREAMS", 2))
    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
    use_compile_model = bool(int(os.environ.get("USE_COMPILE_MODEL", "1")))
    use_compile_muon = bool(int(os.environ.get("USE_COMPILE_MUON", "1")))
    allow_compile_with_mhc = bool(int(os.environ.get("ALLOW_COMPILE_WITH_MHC", "0")))
    compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "1")))
    compile_dynamic = bool(int(os.environ.get("COMPILE_DYNAMIC", "0")))

    flash_attn_version = int(os.environ.get("USE_FLASHATTENTION3", os.environ.get("USE_FLASHATTENTION2", "0")))
    if int(os.environ.get("USE_FLASHATTENTION3", "1")):
        flash_attn_version = 3
    elif int(os.environ.get("USE_FLASHATTENTION2", "0")):
        flash_attn_version = 2

    # LR scheduler (see train_gpt_lib/lr_schedulers.py for options).
    lr_schedule = os.environ.get("LR_SCHEDULE", "trapezoid")

    # Optimizer strategy (see train_gpt_lib/optim.py OPTIMIZER_REGISTRY for options).
    optimizer = os.environ.get("OPTIMIZER", "muon_adam")
    adamw_weight_decay = float(os.environ.get("ADAMW_WEIGHT_DECAY", "0.1"))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    comet_enable = bool(int(os.environ.get("COMET_ENABLE", "0")))
    comet_api_key = os.environ.get("COMET_API_KEY", "")
    comet_project_name = os.environ.get("COMET_PROJECT_NAME", "parameter-golf")
    comet_workspace = os.environ.get("COMET_WORKSPACE", "")
    comet_log_train_every = int(os.environ.get("COMET_LOG_TRAIN_EVERY", 100))

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for key in dir(self):
            if key.startswith("_"):
                continue
            value = getattr(self, key)
            if callable(value):
                continue
            if isinstance(value, (bool, int, float, str)):
                out[key] = value
        return out


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
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

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
