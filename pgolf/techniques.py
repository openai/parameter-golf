"""Technique definitions and test configurations."""

from __future__ import annotations

# Small model for fast validation smoke tests
VALIDATION_BASELINE = {
    "RUN_ID": "baseline",
    "NUM_LAYERS": 4,
    "MODEL_DIM": 256,
    "MLP_MULT": 2,
    "NUM_HEADS": 4,
    "NUM_KV_HEADS": 2,
}

# Full model for ablation / search
FULL_BASELINE = {
    "RUN_ID": "baseline",
    "NUM_LAYERS": 9,
    "MODEL_DIM": 512,
    "MLP_MULT": 2,
    "NUM_HEADS": 8,
    "NUM_KV_HEADS": 4,
}

# (label, env_var_overrides) — each tested independently against baseline
TECHNIQUE_TESTS = [
    # --- Baseline ---
    ("baseline", {}),
    # --- Activations (mutually exclusive) ---
    ("activation_leaky_relu_squared", {"ACTIVATION": "leaky_relu_squared"}),
    ("activation_star_relu", {"ACTIVATION": "star_relu"}),
    ("activation_polycom", {"ACTIVATION": "polycom"}),
    # --- Architecture toggles ---
    ("hybridnorm", {"ENABLE_HYBRIDNORM": 1}),
    ("smeargate", {"ENABLE_SMEARGATE": 1}),
    ("diff_attn", {"ENABLE_DIFF_ATTN": 1}),
    ("pope", {"ENABLE_POPE": 1}),
    ("wavelet", {"ENABLE_WAVELET": 1}),
    ("vga", {"ENABLE_VGA": 1}),
    ("xsa_3", {"XSA_LAST_N": 3}),
    ("xsa_6", {"XSA_LAST_N": 6}),
    ("mtp_1", {"MTP_NUM_HEADS": 1}),
    ("mtp_2", {"MTP_NUM_HEADS": 2}),
    # --- Training toggles ---
    ("ema_0.997", {"EMA_DECAY": 0.997}),
    ("ema_0.999", {"EMA_DECAY": 0.999}),
    ("swa", {"ENABLE_SWA": 1}),
    ("qat", {"ENABLE_QAT": 1}),
    ("ema_swa", {"EMA_DECAY": 0.997, "ENABLE_SWA": 1}),
    # --- Quantization ---
    ("quant_int6", {"QUANT_BITS": 6}),
    ("quant_int5", {"QUANT_BITS": 5}),
    ("optrot", {"ENABLE_OPTROT": 1}),
    ("gptq", {"ENABLE_GPTQ": 1}),
    ("pruning_2pct", {"ENABLE_PRUNING": 1, "PRUNE_FRACTION": 0.02}),
    ("entropy_coding", {"ENABLE_ENTROPY_CODING": 1}),
    ("optrot_gptq", {"ENABLE_OPTROT": 1, "ENABLE_GPTQ": 1}),
    (
        "optrot_gptq_pruning",
        {"ENABLE_OPTROT": 1, "ENABLE_GPTQ": 1, "ENABLE_PRUNING": 1},
    ),
    # --- Eval-time ---
    ("ttt", {"ENABLE_TTT": 1}),
    ("ngram", {"ENABLE_NGRAM": 1}),
    ("knn", {"ENABLE_KNN": 1}),
    ("ttt_tempcal", {"ENABLE_TTT": 1, "TTT_TEMP": 0.98}),
    ("ngram_knn", {"ENABLE_NGRAM": 1, "ENABLE_KNN": 1}),
    (
        "ttt_ngram_knn",
        {"ENABLE_TTT": 1, "TTT_TEMP": 0.98, "ENABLE_NGRAM": 1, "ENABLE_KNN": 1},
    ),
    # --- Model size variations ---
    ("int5_11L", {"QUANT_BITS": 5, "NUM_LAYERS": 11}),
    ("int5_11L_mlp3x", {"QUANT_BITS": 5, "NUM_LAYERS": 11, "MLP_MULT": 3}),
    ("int5_13L", {"QUANT_BITS": 5, "NUM_LAYERS": 13}),
    ("int6_11L", {"QUANT_BITS": 6, "NUM_LAYERS": 11}),
    ("int6_11L_mlp3x", {"QUANT_BITS": 6, "NUM_LAYERS": 11, "MLP_MULT": 3}),
]
