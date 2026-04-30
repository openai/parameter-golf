"""Self-contained training script for the parameter-golf 16 MB / 600 s track.

Architecture: 11-layer transformer, model dim 512, 8/4 GQA, DualMLP (4x expansion
split half-width), tied 8192-vocab SentencePiece BPE, partial RoPE (16/64 dims),
gated attention, EMA 0.9965, Muon (NS-3) + fused AdamW hybrid. 35,988,657 params.
Quantization: GPTQ int6 matrix + int5 embedding + 2:4 sparsity + freeze-dry + zstd-22.
"""

import os as _bootstrap_os
import sys as _bootstrap_sys
import types as _bootstrap_types

try:
    import torch as _bootstrap_torch

    _bootstrap_torch._inductor.config.triton.enable_persistent_tma_matmul = True
except Exception:
    pass
_bootstrap_os.environ.setdefault("USE_LLOYD_MAX", "1")
_bootstrap_os.environ.setdefault("USE_PARALLEL_GPTQ", "1")
_bootstrap_os.environ.setdefault("USE_FREEZE_DRY", "1")
_bootstrap_os.environ.setdefault("COMPRESSOR", "zstd")
_bootstrap_os.environ.setdefault("USE_DUAL_MLP", "1")
_bootstrap_os.environ.setdefault("DUAL_MLP_RATIO", "0.375")
_bootstrap_os.environ.setdefault("USE_ASYMMETRIC_SKIP_INIT", "1")
_bootstrap_os.environ.setdefault("USE_CROSS_WINDOW", "1")
_bootstrap_os.environ.setdefault("ITERATIONS", "20000")
_bootstrap_os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "600")
_bootstrap_os.environ.setdefault("TTT_ENABLED", "1")
_bootstrap_os.environ.setdefault("TTT_LR", "0.005")
_bootstrap_os.environ.setdefault("TTT_EPOCHS", "3")
_bootstrap_os.environ.setdefault("TTT_CHUNK_TOKENS", "32768")
_bootstrap_os.environ.setdefault("SLIDING_WINDOW_ENABLED", "1")
_bootstrap_os.environ.setdefault("USE_NGRAM_BIAS", "0")
_bootstrap_os.environ.setdefault("GPTQ_CALIB_USE_VAL", "0")
_bootstrap_os.environ.setdefault("PREQUANT_TTT_ENABLED", "0")
_bootstrap_os.environ.setdefault("MATRIX_BITS", "6")
_bootstrap_os.environ.setdefault("TRAIN_BATCH_TOKENS", "524288")
_bootstrap_os.environ.setdefault("VAL_BATCH_TOKENS", "262144")
_bootstrap_os.environ.setdefault("SEED", "42")
_bootstrap_os.environ.setdefault("USE_NGRAM_BF16", "1")
_bootstrap_os.environ.setdefault("TTT_BATCH_SEQS", "16")
_bootstrap_os.environ.setdefault("PREQUANT_TTT_BATCH_SEQS", "16")
_bootstrap_os.environ.setdefault("USE_SPARSITY_24", "1")
_bootstrap_os.environ.setdefault("USE_CMP_QUANT_VALUE_DEDUP", "1")
_bootstrap_os.environ.setdefault("EMBED_BITS", "5")
_bootstrap_os.environ.setdefault("QK_GAIN_INIT", "5.25")
_bootstrap_os.environ.setdefault("EMA_DECAY", "0.9965")
_bootstrap_os.environ.setdefault("ADAM_WD", "0.095")
_bootstrap_os.environ.setdefault("MATRIX_LR", "0.022")
_bootstrap_os.environ.setdefault("WARMDOWN_FRAC", "0.72")
_bootstrap_os.environ.setdefault("PARALLEL_RESIDUAL_START", "7")
_bootstrap_os.environ.setdefault("MUON_WD", "0.12")
_bootstrap_os.environ.setdefault("MUON_MOMENTUM", "0.98")
_bootstrap_os.environ.setdefault("USE_CURRICULUM_SHARD", "1")
_bootstrap_os.environ.setdefault("CURRICULUM_MANIFEST_PATH", "./data/curriculum_manifest.npz")
_bootstrap_os.environ.setdefault("CURRICULUM_BUCKET_FLOOR_WEIGHT", "0.02")
_bootstrap_os.environ.setdefault("MUON_BACKEND_STEPS", "3")
_bootstrap_os.environ.setdefault("USE_PREFETCH_LOADER", "1")
_bootstrap_os.environ.setdefault("PREFETCH_DEPTH", "4")
_bootstrap_os.environ.setdefault("PREFETCH_PIN_MEMORY", "1")
for _pkg in ("submission", "submission.ideas"):
    if _pkg not in _bootstrap_sys.modules:
        _bootstrap_sys.modules[_pkg] = _bootstrap_types.ModuleType(_pkg)
_idea_module_idea_phase6_sparsity_24 = _bootstrap_types.ModuleType("submission.ideas.idea_phase6_sparsity_24")
_idea_module_idea_phase6_sparsity_24.__file__ = "<inlined>"
_idea_source_idea_phase6_sparsity_24 = '"""PHASE6 - 2:4 structured sparsity compression.\n\nGoal: cut bytes by ~30% versus the int6 GPTQ baseline by exploiting the fact\nthat (a) weight matrices carry a lot of "below the noise floor" values and\n(b) we can encode which two-of-four positions we kept with only 2 bits per\nblock of 4 elements. Standard NVIDIA sparse-tensor format, but used here as\na pure storage-side compression trick - we never actually run in sparse mode,\njust round-trip the surviving values through serialize/deserialize.\n\nAlgorithm (per 2D weight matrix):\n  1. Reshape along axis 1 into contiguous blocks of 4 elements.\n     (If the last axis isn\'t a multiple of 4, pad with zeros - padding count\n     is stored in the packed dict so dequantize can trim.)\n  2. In each block, find the indices of the 2 largest-|value| elements.\n  3. Keep their values, zero the other two.\n  4. Store:\n       - positions: 2 bits per block saying which-of-6 pairs survived\n                    (C(4,2)=6 possible pairs; fits in 3 bits but we use 4 so\n                     the packing is trivial and brotli handles the rest)\n       - values:    the 2 surviving values per block, SPARSITY_24_BITS-bit\n                    quantized, per-row scaled.\n       - scale:     per-row float32 dequant factor.\n       - shape:     original (n_rows, n_cols).\n       - pad:       int32 count of zero columns added to reach multiple-of-4.\n\nExpected savings versus int6 baseline:\n  - int6 = 6 bits/weight\n  - ours = (2 values × SPARSITY_24_BITS bits + 4 bits position) / 4 weights\n        = (6 + 4) / 4 = 2.5 bits/weight at SPARSITY_24_BITS=3\n  - that\'s a ~58% byte reduction vs int6, though brotli eats into the gap\n    because int6 compresses well too. Real-world we expect ~30%.\n\nEnv vars:\n  USE_SPARSITY_24=0|1         (default 0)\n  SPARSITY_24_BITS=3          (bits used to store each surviving value)\n\nHook point: tournament/train.py serialize() - replaces .q/.scale with\n.__sparsity_24_packed.<field> for eligible tensors; marks\nquant_meta[name]=\'sparsity_24\'. The deserialize path reads those keys back\nand reconstructs the dense matrix (with the non-surviving positions as 0s).\n"""\nfrom __future__ import annotations\n\nimport os\nfrom typing import Dict\n\nimport numpy as np\n\n\n# ─── Env gates ──────────────────────────────────────────────────────────────\n\ndef is_enabled() -> bool:\n    return bool(int(os.environ.get("USE_SPARSITY_24", "0")))\n\n\ndef _value_bits() -> int:\n    # How many bits per surviving value. 3 bits = 8 levels (uint3),\n    # 4 bits = 16 levels, etc.  Clamped to [2, 8].\n    b = int(os.environ.get("SPARSITY_24_BITS", "3"))\n    return max(2, min(8, b))\n\n\n# ─── Pair-index tables ──────────────────────────────────────────────────────\n# The 6 unique (i,j) pairs over {0,1,2,3} with i<j.\n_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]\n# Reverse map: pair tuple -> index in _PAIRS\n_PAIR_LOOKUP = {p: i for i, p in enumerate(_PAIRS)}\n\n\ndef _select_top2_pair_idx(block: np.ndarray) -> int:\n    """Given a length-4 array, return the index in _PAIRS of the two positions\n    with largest magnitude."""\n    mags = np.abs(block)\n    # argpartition gives us the two largest without a full sort\n    top2 = np.argpartition(mags, -2)[-2:]\n    a, b = sorted(int(x) for x in top2)\n    return _PAIR_LOOKUP[(a, b)]\n\n\n# ─── Core pack ──────────────────────────────────────────────────────────────\n\ndef quantize_sparsity_24(W: np.ndarray, value_bits: int | None = None) -> Dict[str, np.ndarray]:\n    """2:4 structured-sparsity quantization of a 2D weight matrix.\n\n    Parameters\n    ----------\n    W : np.ndarray of shape (m, n), float\n    value_bits : int, bits per surviving value (default from env)\n\n    Returns\n    -------\n    packed : dict with keys \'values\', \'positions\', \'scale\', \'shape\', \'pad\', \'bits\'.\n    """\n    if value_bits is None:\n        value_bits = _value_bits()\n    if W.ndim != 2:\n        raise ValueError(f"sparsity_24 needs a 2D matrix, got shape {W.shape}")\n\n    m, n = W.shape\n    pad = (4 - n % 4) % 4\n    if pad:\n        W = np.concatenate([W, np.zeros((m, pad), dtype=W.dtype)], axis=1)\n    n_padded = W.shape[1]\n    assert n_padded % 4 == 0\n    n_blocks_per_row = n_padded // 4\n\n    # Reshape into (m, n_blocks, 4) so each last-axis block is a 2:4 group.\n    W3 = W.reshape(m, n_blocks_per_row, 4)\n\n    # Compute per-block pair index (one uint8 per block; actual entropy ≤3 bits).\n    mags = np.abs(W3)\n    # For each block, sort indices desc by magnitude; take the two largest.\n    # argsort along axis=2, keep the top 2 positions.\n    top2_idx = np.argpartition(mags, -2, axis=2)[:, :, -2:]\n    top2_idx_sorted = np.sort(top2_idx, axis=2)  # (m, nblk, 2), positions with i<j\n    # Map (i,j) -> pair index via lookup table.\n    pair_codes = np.zeros((m, n_blocks_per_row), dtype=np.uint8)\n    for pi, (i, j) in enumerate(_PAIRS):\n        pair_codes[(top2_idx_sorted[:, :, 0] == i) & (top2_idx_sorted[:, :, 1] == j)] = pi\n\n    # Gather the two surviving values per block.\n    rows_idx = np.arange(m).reshape(m, 1, 1)\n    blk_idx = np.arange(n_blocks_per_row).reshape(1, n_blocks_per_row, 1)\n    surviving_vals = W3[rows_idx, blk_idx, top2_idx_sorted]  # (m, nblk, 2)\n    # Flatten to (m, nblk*2) for per-row quantization.\n    vals_flat = surviving_vals.reshape(m, n_blocks_per_row * 2).astype(np.float32)\n\n    # Per-row symmetric quant with value_bits levels.\n    levels = 1 << value_bits\n    qmax = levels - 1\n    row_max = np.max(np.abs(vals_flat), axis=1, keepdims=True)\n    row_max = np.where(row_max == 0, 1.0, row_max)\n    scale = row_max / (qmax / 2)\n    q = np.round(vals_flat / scale + (qmax / 2)).astype(np.int32)\n    q = np.clip(q, 0, qmax).astype(np.uint8)\n\n    return {\n        "values":    q,                                # uint8, shape (m, 2*nblk)\n        "positions": pair_codes,                       # uint8, shape (m, nblk), 0..5\n        "scale":     scale.astype(np.float32),         # (m, 1)\n        "shape":     np.array([m, n], dtype=np.int32),\n        "pad":       np.int32(pad),\n        "bits":      np.int32(value_bits),\n    }\n\n\ndef dequantize(packed: Dict[str, np.ndarray]) -> np.ndarray:\n    """Inverse of quantize_sparsity_24. Pure numpy."""\n    q = packed["values"].astype(np.float32)              # (m, 2*nblk)\n    pair_codes = packed["positions"].astype(np.int32)    # (m, nblk)\n    scale = packed["scale"].astype(np.float32)           # (m, 1)\n    m_orig, n_orig = (int(x) for x in packed["shape"])\n    pad = int(packed["pad"])\n    bits = int(packed["bits"])\n    qmax = (1 << bits) - 1\n\n    m = q.shape[0]\n    n_vals = q.shape[1]\n    n_blocks_per_row = n_vals // 2\n    n_padded = n_orig + pad\n    assert n_blocks_per_row * 4 == n_padded\n\n    # Dequantize values.\n    vals_flat = (q - (qmax / 2)) * scale                 # (m, 2*nblk)\n    vals = vals_flat.reshape(m, n_blocks_per_row, 2)     # (m, nblk, 2)\n\n    # Scatter back into a (m, nblk, 4) dense block layout.\n    dense = np.zeros((m, n_blocks_per_row, 4), dtype=np.float32)\n    # For each pair code, scatter both survivors.\n    for pi, (i, j) in enumerate(_PAIRS):\n        mask = (pair_codes == pi)                        # (m, nblk)\n        # mask is (m, nblk); vals[...,0] is (m, nblk). Assigning into dense[mask, i]\n        # only writes to the selected (row, block) pairs, which matches vals[mask, 0].\n        dense[mask, i] = vals[mask, 0]\n        dense[mask, j] = vals[mask, 1]\n\n    W = dense.reshape(m, n_padded)\n    if pad:\n        W = W[:, :n_orig]\n    assert W.shape == (m_orig, n_orig)\n    return W.astype(np.float32)\n\n\n# ─── Size estimation ────────────────────────────────────────────────────────\n\ndef estimate_compressed_bytes(W: np.ndarray, bits: int) -> int:\n    m, n = W.shape\n    n_padded = n + (4 - n % 4) % 4\n    n_blocks = (m * n_padded) // 4\n    # positions: 1 byte per block (entropy ≤3 bits; brotli handles the rest).\n    pos_bytes = n_blocks\n    # values: 2 * bits per block, rounded up to bytes.\n    val_bytes = int(np.ceil(2 * n_blocks * bits / 8))\n    meta_bytes = m * 4\n    return pos_bytes + val_bytes + meta_bytes\n\n\nif __name__ == "__main__":\n    # Smoke test\n    rng = np.random.default_rng(42)\n    W = rng.standard_normal((128, 512)).astype(np.float32)\n\n    packed = quantize_sparsity_24(W, value_bits=3)\n    W_rec = dequantize(packed)\n    rmse = float(np.sqrt(np.mean((W - W_rec) ** 2)))\n\n    # Count zeros in the reconstruction - 2:4 means exactly half are zero.\n    zero_frac = float(np.mean(W_rec == 0))\n    n_correct_zero_frac = abs(zero_frac - 0.5) < 0.02\n\n    print(f"original:       {W.size * 4} bytes (fp32)")\n    print(f"int6 baseline:  {int(W.size * 6 / 8)} bytes")\n    print(f"sparsity_24 @3: {estimate_compressed_bytes(W, 3)} bytes")\n    print(f"shape:          {packed[\'shape\'].tolist()}")\n    print(f"RMSE:           {rmse:.5f}")\n    print(f"zero fraction:  {zero_frac:.3f} (expect ≈0.5; pass={n_correct_zero_frac})")\n    # Spot-check round-trip at extreme sparsity: the block with the biggest magnitude\n    # should survive in full (modulo quantization).\n    biggest_idx = np.unravel_index(np.argmax(np.abs(W)), W.shape)\n    surviving = abs(W_rec[biggest_idx]) > 1e-4\n    print(f"biggest-mag survived: {surviving}")\n\n    # Also sanity-check handling of non-multiple-of-4 widths.\n    W2 = rng.standard_normal((16, 37)).astype(np.float32)\n    p2 = quantize_sparsity_24(W2, value_bits=3)\n    W2_rec = dequantize(p2)\n    print(f"odd-width {W2.shape} round-trip RMSE: {float(np.sqrt(np.mean((W2 - W2_rec) ** 2))):.5f}")\n    assert W2_rec.shape == W2.shape\n    print("OK")\n'
exec(_idea_source_idea_phase6_sparsity_24, _idea_module_idea_phase6_sparsity_24.__dict__)
_bootstrap_sys.modules["submission.ideas.idea_phase6_sparsity_24"] = _idea_module_idea_phase6_sparsity_24
_idea_module_idea_051_freeze_dry = _bootstrap_types.ModuleType("submission.ideas.idea_051_freeze_dry")
_idea_module_idea_051_freeze_dry.__file__ = "<inlined>"
_idea_source_idea_051_freeze_dry = '"""IDEA-051 - Freeze-drying: detect & drop weights that are linear combos of neighbors.\n\nAfter training, for each weight w_{i,j}, fit a linear model predicting it from\nits row/column neighbors. If fit RMSE < threshold, mark as "reconstructable" and\ndrop it. Store only a bitmask + shared reconstruction coefficients.\n\nEnv vars:\n  USE_FREEZE_DRY=0|1                    (default 0)\n  FREEZE_DRY_RMSE_THRESH=0.005         (default 0.005 - max RMSE for reconstruction)\n  FREEZE_DRY_MIN_FRACTION=0.05         (default 0.05 - only apply if >5% reconstructable)\n\nHook point: submission/train.py serialize() - after GPTQ, analyze weight structure\nand drop linearly-reconstructable weights before brotli compression.\n"""\n\nimport os\nfrom typing import Dict, Tuple, Optional\n\nimport numpy as np\n\n\ndef is_enabled() -> bool:\n    return bool(int(os.environ.get("USE_FREEZE_DRY", "0")))\n\n\ndef get_rmse_thresh() -> float:\n    return float(os.environ.get("FREEZE_DRY_RMSE_THRESH", "0.005"))\n\n\ndef get_min_fraction() -> float:\n    return float(os.environ.get("FREEZE_DRY_MIN_FRACTION", "0.05"))\n\n\ndef analyze_linear_redundancy(\n    w: np.ndarray,\n    rmse_thresh: float = 0.005,\n) -> Tuple[np.ndarray, float]:\n    """Analyze a weight matrix for linear redundancy along rows.\n\n    For each element w[i,j], fit: w[i,j] ≈ a*w[i,j-1] + b*w[i,j+1]\n    (2-neighbor linear prediction). If RMSE < threshold, mark as reconstructable.\n\n    Args:\n        w: [out_dim, in_dim] float32 weight matrix\n        rmse_thresh: max RMSE for a weight to be considered reconstructable\n\n    Returns:\n        mask: [out_dim, in_dim] bool - True = keep, False = reconstructable (can drop)\n        fraction_reconstructable: float\n    """\n    out_dim, in_dim = w.shape\n    if in_dim < 3:\n        return np.ones_like(w, dtype=bool), 0.0\n\n    # For each interior column j (1..in_dim-2), predict from j-1 and j+1\n    # via least-squares: w[:,j] ≈ a * w[:,j-1] + b * w[:,j+1]\n    mask = np.ones_like(w, dtype=bool)\n    total_checked = 0\n    total_reconstructable = 0\n\n    for j in range(1, in_dim - 1):\n        # Stack neighbors as [out_dim, 2] design matrix\n        X = np.stack([w[:, j - 1], w[:, j + 1]], axis=1)  # [out_dim, 2]\n        y = w[:, j]  # [out_dim]\n\n        # Solve least squares for coefficients a, b\n        try:\n            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)\n        except np.linalg.LinAlgError:\n            continue\n\n        # Compute per-element RMSE\n        pred = X @ coeffs\n        errors = np.abs(y - pred)\n\n        # Mark elements with error < threshold as reconstructable\n        recon_mask = errors < rmse_thresh\n        mask[:, j] = ~recon_mask  # True = keep (NOT reconstructable)\n        total_checked += out_dim\n        total_reconstructable += recon_mask.sum()\n\n    fraction = total_reconstructable / max(total_checked, 1)\n    return mask, fraction\n\n\ndef freeze_dry_state_dict(\n    state_dict: Dict[str, "torch.Tensor"],\n    rmse_thresh: float = None,\n    min_fraction: float = None,\n) -> Dict[str, "torch.Tensor"]:\n    """Apply freeze-drying to all weight matrices in state_dict.\n\n    Weights identified as linearly reconstructable are set to zero.\n    Zeros compress efficiently under brotli.\n\n    Returns: modified state_dict (in-place)\n    """\n    import torch\n\n    if not is_enabled():\n        return state_dict\n\n    rmse_thresh = rmse_thresh or get_rmse_thresh()\n    min_fraction = min_fraction or get_min_fraction()\n\n    total_removed = 0\n    total_weights = 0\n\n    for name, tensor in state_dict.items():\n        if tensor.dim() != 2 or tensor.numel() < 65536:\n            continue\n        if not tensor.is_floating_point():\n            continue\n\n        w_np = tensor.detach().cpu().float().numpy()\n        mask, frac = analyze_linear_redundancy(w_np, rmse_thresh)\n\n        if frac < min_fraction:\n            continue\n\n        # Zero out reconstructable weights\n        removed = (~mask).sum()\n        state_dict[name] = tensor * torch.from_numpy(mask.astype(np.float32)).to(tensor.device).to(tensor.dtype)\n\n        total_removed += removed\n        total_weights += tensor.numel()\n\n        print(\n            f"[IDEA-051 freeze_dry] {name}: {removed}/{tensor.numel()} "\n            f"({100*frac:.1f}%) weights zeroed",\n            flush=True,\n        )\n\n    if total_weights > 0:\n        print(\n            f"[IDEA-051 freeze_dry] total: {total_removed}/{total_weights} "\n            f"({100*total_removed/total_weights:.2f}%) weights zeroed",\n            flush=True,\n        )\n\n    return state_dict\n'
exec(_idea_source_idea_051_freeze_dry, _idea_module_idea_051_freeze_dry.__dict__)
_bootstrap_sys.modules["submission.ideas.idea_051_freeze_dry"] = _idea_module_idea_051_freeze_dry
_idea_module_idea_064_parallel_gptq = _bootstrap_types.ModuleType("submission.ideas.idea_064_parallel_gptq")
_idea_module_idea_064_parallel_gptq.__file__ = "<inlined>"
_idea_source_idea_064_parallel_gptq = '"""IDEA-064 - Parallel-search GPTQ: try 50+ clip percentiles using all 208 CPUs.\n\nPR #414 tries 5 clip percentiles per row. We have 208 vCPUs. Run 50+ candidates\nin parallel, pick per-row optimum. Strictly better GPTQ at zero GPU cost.\n\nEnv vars:\n  USE_PARALLEL_GPTQ=0|1                 (default 0)\n  PARALLEL_GPTQ_N_CLIPS=50             (default 50 clip candidates)\n  PARALLEL_GPTQ_WORKERS=0              (default 0 - auto-detect CPU count)\n\nHook point: submission/train.py gptq_quantize_weight() - replace with\nparallel_gptq_quantize_weight when enabled.\n"""\n\nimport os\nimport math\nfrom concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor\nfrom typing import Tuple, List, Optional\n\nimport numpy as np\n\n\ndef is_enabled() -> bool:\n    return bool(int(os.environ.get("USE_PARALLEL_GPTQ", "0")))\n\n\ndef get_n_clips() -> int:\n    return int(os.environ.get("PARALLEL_GPTQ_N_CLIPS", "50"))\n\n\ndef get_workers() -> int:\n    n = int(os.environ.get("PARALLEL_GPTQ_WORKERS", "0"))\n    if n <= 0:\n        n = min(os.cpu_count() or 8, 64)\n    return n\n\n\ndef generate_clip_candidates(n: int = 50, sigma_min: float = 2.0, sigma_max: float = 20.0) -> List[float]:\n    """Generate n clip-sigma candidates log-spaced between sigma_min and sigma_max."""\n    log_min = math.log(sigma_min)\n    log_max = math.log(sigma_max)\n    return [math.exp(log_min + (log_max - log_min) * i / max(n - 1, 1)) for i in range(n)]\n\n\ndef _quantize_row_at_clip(args):\n    """Worker function: quantize a single row at a given clip_sigma.\n\n    Returns (row_idx, clip_sigma, reconstruction_error).\n    """\n    row_idx, w_row, h_diag_row, clip_sigma, clip_range = args\n    w = w_row.copy()\n    n = len(w)\n    # Clip outliers\n    std = np.std(w)\n    clip_val = clip_sigma * std\n    w = np.clip(w, -clip_val, clip_val)\n    # Scale to int range\n    w_max = np.abs(w).max()\n    if w_max < 1e-10:\n        return row_idx, clip_sigma, 0.0\n    scale = w_max / clip_range\n    q = np.round(w / scale).clip(-clip_range, clip_range)\n    recon = q * scale\n    # Hessian-weighted reconstruction error\n    err = (w_row - recon) ** 2\n    if h_diag_row is not None and len(h_diag_row) == len(err):\n        err *= (1.0 + np.abs(h_diag_row))\n    return row_idx, clip_sigma, float(err.sum())\n\n\ndef parallel_search_best_clips(\n    weight: "np.ndarray",\n    hessian_diag: "Optional[np.ndarray]",\n    clip_range: int = 31,\n    n_clips: int = None,\n    n_workers: int = None,\n) -> "np.ndarray":\n    """Find optimal clip_sigma per row using parallel search.\n\n    Args:\n        weight: [out_dim, in_dim] float32 numpy array\n        hessian_diag: [out_dim] or [out_dim, in_dim] Hessian diagonal\n        clip_range: max int value (31 for int6)\n        n_clips: number of clip candidates to try\n        n_workers: parallel workers\n\n    Returns:\n        best_clips: [out_dim] best clip_sigma per row\n    """\n    if not is_enabled():\n        return None\n\n    n_clips = n_clips or get_n_clips()\n    n_workers = n_workers or get_workers()\n    clips = generate_clip_candidates(n_clips)\n    out_dim = weight.shape[0]\n\n    # Prepare per-row Hessian\n    if hessian_diag is not None:\n        if hessian_diag.ndim == 1:\n            h_rows = [None] * out_dim  # per-output-dim scalar, not per-element\n        else:\n            h_rows = [hessian_diag[i] for i in range(out_dim)]\n    else:\n        h_rows = [None] * out_dim\n\n    # Build task list: (row_idx, w_row, h_row, clip_sigma, clip_range)\n    tasks = []\n    for row_idx in range(out_dim):\n        for clip_sigma in clips:\n            tasks.append((row_idx, weight[row_idx].copy(), h_rows[row_idx], clip_sigma, clip_range))\n\n    # Run in parallel (use threads not processes to avoid pickle overhead)\n    best_clips = np.full(out_dim, clips[len(clips) // 2])  # default to median\n    best_errors = np.full(out_dim, float("inf"))\n\n    with ThreadPoolExecutor(max_workers=n_workers) as pool:\n        for row_idx, clip_sigma, err in pool.map(_quantize_row_at_clip, tasks):\n            if err < best_errors[row_idx]:\n                best_errors[row_idx] = err\n                best_clips[row_idx] = clip_sigma\n\n    total_improvement = (best_errors.sum()) / max(out_dim, 1)\n    print(\n        f"[IDEA-064 parallel_gptq] searched {n_clips} clips × {out_dim} rows "\n        f"using {n_workers} workers, avg_best_err={total_improvement:.6f}",\n        flush=True,\n    )\n    return best_clips\n'
exec(_idea_source_idea_064_parallel_gptq, _idea_module_idea_064_parallel_gptq.__dict__)
_bootstrap_sys.modules["submission.ideas.idea_064_parallel_gptq"] = _idea_module_idea_064_parallel_gptq
_idea_module_tournament_quant_01_lloyd_max = _bootstrap_types.ModuleType(
    "submission.ideas.tournament_quant_01_lloyd_max"
)
_idea_module_tournament_quant_01_lloyd_max.__file__ = "<inlined>"
_idea_source_tournament_quant_01_lloyd_max = '"""Tournament Quant 01 -- Lloyd-Max codebook quantization for int6 weights.\n\nReplace standard uniform int6 quantization with optimal non-uniform\nquantization based on a pre-computed 64-level Lloyd-Max codebook.  The\ncodebook is trained offline to minimise MSE for the empirical weight\ndistribution (Gaussian-like, heavy tails).\n\nFor each weight value, find the nearest codebook centroid and store its\n6-bit index (0-63).  Dequantize by table lookup: weight_approx =\ncodebook[index].  Because the codebook places more centroids near\nzero (where most weights live), reconstruction error drops vs uniform\nspacing at the same 6 bits per weight.\n\nCodebook path: data/lloyd_max_codebook_64.npy  (64 float32 values,\nsorted ascending, pre-trained via the Lloyd-Max algorithm on a sample\nof trained model weights).\n\nEnv vars:\n  USE_LLOYD_MAX=0|1               (default 0)\n  LLOYD_MAX_CODEBOOK=<path>       (default data/lloyd_max_codebook_64.npy)\n\nHook point: submission/train.py quantize() -- replace uniform int6\nround-and-clip with ``lloyd_max_quantize()``; at inference call\n``lloyd_max_dequantize()`` to recover approximate float values.\n"""\n\nimport os\nfrom pathlib import Path\nfrom typing import Tuple\n\nimport numpy as np\n\n\n# ---------------------------------------------------------------------------\n# Env-var gate\n# ---------------------------------------------------------------------------\n\ndef is_enabled() -> bool:\n    return bool(int(os.environ.get("USE_LLOYD_MAX", "0")))\n\n\ndef _codebook_path() -> str:\n    default = str(Path(__file__).resolve().parents[2] / "data" / "lloyd_max_codebook_64.npy")\n    return os.environ.get("LLOYD_MAX_CODEBOOK", default)\n\n\n# ---------------------------------------------------------------------------\n# Codebook loading (cached)\n# ---------------------------------------------------------------------------\n\n_CODEBOOK_CACHE = None\n\n\ndef load_codebook() -> np.ndarray:\n    """Load and cache the 64-level Lloyd-Max codebook (sorted ascending).\n\n    Returns:\n        1-D float32 array of 64 centroid values.\n    """\n    global _CODEBOOK_CACHE\n    if _CODEBOOK_CACHE is None:\n        cb = np.load(_codebook_path()).astype(np.float32).ravel()\n        assert cb.shape[0] == 64, f"Expected 64 codebook entries, got {cb.shape[0]}"\n        cb.sort()\n        _CODEBOOK_CACHE = cb\n    return _CODEBOOK_CACHE\n\n\n# ---------------------------------------------------------------------------\n# Quantize / dequantize\n# ---------------------------------------------------------------------------\n\ndef lloyd_max_quantize(\n    tensor: np.ndarray,\n    codebook: np.ndarray = None,\n) -> Tuple[np.ndarray, np.ndarray]:\n    """Quantize a weight tensor using Lloyd-Max codebook lookup.\n\n    For every element, find the nearest codebook centroid and store its\n    6-bit index.\n\n    Args:\n        tensor: arbitrary-shape float32 weight array.\n        codebook: 1-D sorted array of 64 centroids (default: load from disk).\n\n    Returns:\n        (indices, codebook) where indices is uint8 array (0-63) with the\n        same shape as tensor, and codebook is the 64-entry centroid array.\n    """\n    if codebook is None:\n        codebook = load_codebook()\n\n    flat = tensor.astype(np.float32).ravel()\n    # Nearest-centroid assignment via binary search on sorted codebook\n    insert_idx = np.searchsorted(codebook, flat)\n    # Clamp to valid range\n    insert_idx = np.clip(insert_idx, 0, len(codebook) - 1)\n    # Check if the left neighbour is closer\n    left_idx = np.clip(insert_idx - 1, 0, len(codebook) - 1)\n    dist_right = np.abs(flat - codebook[insert_idx])\n    dist_left = np.abs(flat - codebook[left_idx])\n    indices = np.where(dist_left < dist_right, left_idx, insert_idx)\n    indices = indices.astype(np.uint8).reshape(tensor.shape)\n    return indices, codebook\n\n\ndef lloyd_max_dequantize(\n    indices: np.ndarray,\n    codebook: np.ndarray = None,\n) -> np.ndarray:\n    """Dequantize 6-bit indices back to float32 via codebook lookup.\n\n    Args:\n        indices: uint8 array of codebook indices (0-63), any shape.\n        codebook: 1-D sorted array of 64 centroids (default: load from disk).\n\n    Returns:\n        float32 array of the same shape with approximate weight values.\n    """\n    if codebook is None:\n        codebook = load_codebook()\n\n    return codebook[indices.ravel()].reshape(indices.shape)\n'
exec(_idea_source_tournament_quant_01_lloyd_max, _idea_module_tournament_quant_01_lloyd_max.__dict__)
_bootstrap_sys.modules["submission.ideas.tournament_quant_01_lloyd_max"] = _idea_module_tournament_quant_01_lloyd_max
_idea_module_tournament_mlp_01_dual_mlp = _bootstrap_types.ModuleType("submission.ideas.tournament_mlp_01_dual_mlp")
_idea_module_tournament_mlp_01_dual_mlp.__file__ = "<inlined>"
_idea_source_tournament_mlp_01_dual_mlp = '"""TOURNAMENT-MLP-01 -- Dual MLP: two parallel half-width MLPs per layer.\n\nReplace each layer\'s single MLP with two parallel MLPs, each with half the\nnormal hidden size, then average their outputs:\n\n    mlp_out = 0.5 * (mlp_a(x) + mlp_b(x))\n\nTotal parameter count is the same as a single full-width MLP (since each\nhalf-width MLP has half the parameters). The benefit is implicit ensemble\naveraging: the two MLPs can specialise on different patterns and their\naverage is a smoother, more robust transformation than a single MLP.\n\nThis is analogous to dropout-as-ensemble (Srivastava et al. 2014) but\nstructural rather than stochastic: two independent paths that are always\nboth active. It also relates to mixture-of-experts with uniform routing\n(every expert always active, equal weight).\n\nEnv vars:\n  USE_DUAL_MLP=0|1              (default 0)\n\nHook point: submission/train.py GPT.__init__() -- after blocks are created,\ncall apply_dual_mlp(model, h) to replace each block\'s MLP.\n\nIntegration:\n  from submission.ideas.tournament_mlp_01_dual_mlp import (\n      is_enabled, apply_dual_mlp\n  )\n  if is_enabled():\n      apply_dual_mlp(model, h)\n"""\n\nimport os\n\nimport torch\nimport torch.nn as nn\n\n\n# ---------------------------------------------------------------------------\n# Configuration\n# ---------------------------------------------------------------------------\n\ndef is_enabled() -> bool:\n    return bool(int(os.environ.get("USE_DUAL_MLP", "0")))\n\n\n# ---------------------------------------------------------------------------\n# Dual MLP module\n# ---------------------------------------------------------------------------\n\nclass DualMLP(nn.Module):\n    """Two parallel half-width MLPs whose outputs are averaged.\n\n    Each sub-MLP has hidden_dim = original_hidden_dim / 2, so total params\n    match a single full-width MLP.\n    """\n\n    def __init__(self, model_dim: int, mlp_mult: float):\n        super().__init__()\n        # Full hidden dim that a single MLP would use\n        full_hidden = int(model_dim * mlp_mult)\n        # Each sub-MLP gets half\n        half_hidden = full_hidden // 2\n\n        self.mlp_a = nn.Sequential(\n            nn.Linear(model_dim, half_hidden, bias=False),\n            nn.SiLU(),\n            nn.Linear(half_hidden, model_dim, bias=False),\n        )\n        self.mlp_b = nn.Sequential(\n            nn.Linear(model_dim, half_hidden, bias=False),\n            nn.SiLU(),\n            nn.Linear(half_hidden, model_dim, bias=False),\n        )\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return 0.5 * (self.mlp_a(x) + self.mlp_b(x))\n\n\n# ---------------------------------------------------------------------------\n# Integration\n# ---------------------------------------------------------------------------\n\ndef apply_dual_mlp(model, h):\n    """Replace each block\'s MLP with a DualMLP.\n\n    Args:\n        model: GPT model with .blocks ModuleList\n        h: hyperparameters namespace (needs model_dim, mlp_mult)\n    """\n    if not is_enabled():\n        return\n\n    model_dim = getattr(h, "model_dim", 512)\n    mlp_mult = getattr(h, "mlp_mult", 3.0)\n\n    replaced = 0\n    for block in model.blocks:\n        # Handle both direct blocks and wrapped blocks (e.g. HyperConnectedBlock)\n        target = block\n        if hasattr(block, "block"):\n            target = block.block\n\n        if hasattr(target, "mlp"):\n            target.mlp = DualMLP(model_dim, mlp_mult)\n            replaced += 1\n\n    print(\n        f"[TOURNAMENT-MLP-01 dual_mlp] replaced {replaced} MLPs with DualMLP "\n        f"(2x half-width={int(model_dim * mlp_mult) // 2}, averaged)",\n        flush=True,\n    )\n'
exec(_idea_source_tournament_mlp_01_dual_mlp, _idea_module_tournament_mlp_01_dual_mlp.__dict__)
_bootstrap_sys.modules["submission.ideas.tournament_mlp_01_dual_mlp"] = _idea_module_tournament_mlp_01_dual_mlp
_idea_module_tournament_embed_03_asymmetric_skip = _bootstrap_types.ModuleType(
    "submission.ideas.tournament_embed_03_asymmetric_skip"
)
_idea_module_tournament_embed_03_asymmetric_skip.__file__ = "<inlined>"
_idea_source_tournament_embed_03_asymmetric_skip = '"""TOURNAMENT-EMBED-03 -- Asymmetric skip-connection initialization.\n\nInitialize U-Net skip_weights at 0.5 instead of the default 1.0. This\ncreates an information bottleneck at the encoder-decoder boundary: the\ndecoder receives a halved skip signal, forcing it to learn its own\nrepresentations rather than simply copying encoder outputs.\n\nAt 1.0, skip connections pass encoder activations through unchanged.\nAt 0.5, the decoder must reconstruct half the signal from its own\ncomputation, encouraging more expressive decoder layers.\n\nEnv vars:\n  USE_ASYMMETRIC_SKIP_INIT=0|1   (default 0)\n\nHook point: model initialization -- after creating skip_weights, override\ntheir values with 0.5.\n"""\n\nimport os\nimport torch\n\n\ndef is_enabled() -> bool:\n    return bool(int(os.environ.get("USE_ASYMMETRIC_SKIP_INIT", "0")))\n\n\ndef apply_asymmetric_skip_init(skip_weights: torch.Tensor) -> torch.Tensor:\n    """Re-initialize skip weights to 0.5.\n\n    Args:\n        skip_weights: the model\'s skip connection weights, any shape.\n\n    Returns:\n        New tensor filled with 0.5, same shape and device.\n    """\n    return torch.full_like(skip_weights, 0.5)\n'
exec(_idea_source_tournament_embed_03_asymmetric_skip, _idea_module_tournament_embed_03_asymmetric_skip.__dict__)
_bootstrap_sys.modules["submission.ideas.tournament_embed_03_asymmetric_skip"] = (
    _idea_module_tournament_embed_03_asymmetric_skip
)
_idea_module_idea_curriculum_shard = _bootstrap_types.ModuleType("submission.ideas.idea_curriculum_shard")
_idea_module_idea_curriculum_shard.__file__ = "<inlined>"
_idea_source_idea_curriculum_shard = '"""Entropy-bucket curriculum shard loader - drop-in replacement for\nShuffledSequenceLoader\'s `next_batch` API.\n\nWhen USE_CURRICULUM_SHARD=1, training samples sequences from pre-computed\nentropy buckets with a time-varying weight schedule that crossfades from easy\n(low-entropy) to hard (high-entropy) as training progresses. A floor weight\nprevents any bucket from going to zero (avoids catastrophic forgetting of\neither tail).\n\nSchedule per sequence:\n  d[b]   = b / (N-1)                    # bucket difficulty, 0 easiest\n  w[b]   = (1 - d[b]) * (1 - p) + d[b] * p,   # p = training progress fraction\n  w[b] <- max(w[b], floor)\n  sample bucket ~ w / sum(w), then sample a sequence uniformly from that bucket.\n\nEnv vars:\n  USE_CURRICULUM_SHARD=0|1                  (default 0)\n  CURRICULUM_MANIFEST_PATH=./data/curriculum_manifest.npz\n  CURRICULUM_BUCKET_FLOOR_WEIGHT=0.02\n\nExpects a manifest built by submission/final/assign_buckets.py (output of\nsubmission/final/compute_entropy.py). This module only defines the loader -\nthe host script is responsible for substituting it at ShuffledSequenceLoader\ncall sites when is_enabled() returns True.\n"""\nfrom __future__ import annotations\n\nimport glob\nimport os\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\n\n\ndef is_enabled() -> bool:\n    return bool(int(os.environ.get("USE_CURRICULUM_SHARD", "0")))\n\n\ndef get_manifest_path() -> str:\n    return os.environ.get("CURRICULUM_MANIFEST_PATH", "./data/curriculum_manifest.npz")\n\n\ndef get_bucket_floor_weight() -> float:\n    return float(os.environ.get("CURRICULUM_BUCKET_FLOOR_WEIGHT", "0.02"))\n\n\n_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize\n\n\ndef _read_num_tokens(file: Path) -> int:\n    header = np.fromfile(file, dtype="<i4", count=3)\n    return int(header[2])\n\n\ndef _load_token_shard(file: Path) -> torch.Tensor:\n    num_tokens = _read_num_tokens(file)\n    tokens = np.fromfile(\n        file, dtype="<u2", count=num_tokens, offset=_SHARD_HEADER_BYTES,\n    )\n    return torch.from_numpy(tokens.astype(np.int32))\n\n\nclass CurriculumSequenceLoader:\n    """Shard loader with entropy-bucket-weighted sampling. Matches the external\n    API that tournament/train.py expects on a train loader:\n      .next_batch(global_tokens, grad_accum_steps) -> (x, y) torch.Tensor pair\n      .prefill(global_tokens, grad_accum_steps[, target_depth][, timeout_s])\n    """\n\n    def __init__(self, h, device: torch.device) -> None:\n        self.h = h\n        self.device = device\n        self.seq_len = h.train_seq_len\n        self.rank = h.rank\n        self.world_size = h.world_size\n        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]\n        if not all_files:\n            raise FileNotFoundError(f"curriculum: no files for {h.train_files!r}")\n\n        manifest_path = Path(get_manifest_path())\n        if not manifest_path.exists():\n            raise FileNotFoundError(\n                f"curriculum manifest missing: {manifest_path} "\n                "(run submission/final/compute_entropy.py + assign_buckets.py first)",\n            )\n        manifest = np.load(manifest_path, allow_pickle=True)\n        shard_paths = list(manifest["shard_paths"])\n        seq_starts = list(manifest["seq_starts"])\n        bucket_ids = list(manifest["bucket_ids"])\n        self.num_buckets = int(manifest["num_buckets"])\n        manifest_seq_len = int(manifest["seq_len"])\n        if manifest_seq_len != self.seq_len:\n            raise ValueError(\n                f"manifest seq_len={manifest_seq_len} != train_seq_len={self.seq_len}",\n            )\n\n        assigned_paths = all_files[self.rank :: self.world_size]\n        by_basename = {p.name: p for p in assigned_paths}\n\n        per_bucket: list[list[tuple[Path, int]]] = [[] for _ in range(self.num_buckets)]\n        for mpath, mstarts, mbuckets in zip(shard_paths, seq_starts, bucket_ids, strict=True):\n            mpath_str = str(mpath)\n            basename = Path(mpath_str).name\n            if basename not in by_basename:\n                continue\n            resolved = by_basename[basename]\n            starts_arr = np.asarray(mstarts, dtype=np.int64)\n            buckets_arr = np.asarray(mbuckets, dtype=np.int8)\n            for start, bucket in zip(starts_arr.tolist(), buckets_arr.tolist(), strict=True):\n                per_bucket[int(bucket)].append((resolved, int(start)))\n\n        if sum(len(b) for b in per_bucket) == 0:\n            raise RuntimeError(\n                f"curriculum: rank {self.rank}/{self.world_size} has no matching shards",\n            )\n\n        self._per_bucket = per_bucket\n        self._floor_weight = get_bucket_floor_weight()\n        self._rng = np.random.Generator(np.random.PCG64(self.rank))\n        self._shard_cache: dict[Path, torch.Tensor] = {}\n        self._start_time: float | None = None\n        self._max_wallclock_seconds = max(1.0, float(h.max_wallclock_seconds))\n        print(\n            f"[curriculum] rank={self.rank}/{self.world_size} "\n            f"buckets={self.num_buckets} total_seqs={sum(len(b) for b in per_bucket)} "\n            f"floor={self._floor_weight}",\n            flush=True,\n        )\n\n    def _progress_fraction(self) -> float:\n        if self._start_time is None:\n            self._start_time = time.monotonic()\n            return 0.0\n        elapsed = time.monotonic() - self._start_time\n        return min(1.0, elapsed / self._max_wallclock_seconds)\n\n    def _bucket_weights(self, progress: float) -> np.ndarray:\n        n = self.num_buckets\n        difficulty = np.arange(n, dtype=np.float64) / max(n - 1, 1)\n        weights = (1.0 - difficulty) * (1.0 - progress) + difficulty * progress\n        has_entries = np.array([len(b) > 0 for b in self._per_bucket], dtype=bool)\n        weights = np.where(has_entries, np.maximum(weights, self._floor_weight), 0.0)\n        total = float(weights.sum())\n        if total <= 0:\n            raise RuntimeError("curriculum: all buckets empty for this rank")\n        return weights / total\n\n    def _get_shard_tokens(self, shard_path: Path) -> torch.Tensor:\n        tokens = self._shard_cache.get(shard_path)\n        if tokens is None:\n            tokens = _load_token_shard(shard_path)\n            self._shard_cache[shard_path] = tokens\n        return tokens\n\n    def _take_sequence(self) -> torch.Tensor:\n        weights = self._bucket_weights(self._progress_fraction())\n        bucket = int(self._rng.choice(self.num_buckets, p=weights))\n        entries = self._per_bucket[bucket]\n        idx = int(self._rng.integers(len(entries)))\n        shard_path, start = entries[idx]\n        tokens = self._get_shard_tokens(shard_path)\n        end = start + self.seq_len + 1\n        if end > tokens.numel():\n            start = max(0, tokens.numel() - self.seq_len - 1)\n            end = start + self.seq_len + 1\n        return tokens[start:end]\n\n    def _build_batch_cpu(self, global_tokens: int, grad_accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:\n        device_tokens = global_tokens // (self.world_size * grad_accum_steps)\n        device_batch_size = device_tokens // self.seq_len\n        sequences = [self._take_sequence() for _ in range(device_batch_size)]\n        stacked = torch.stack(sequences, dim=0).to(dtype=torch.int64)\n        pinned = stacked.pin_memory() if self.device.type == "cuda" else stacked\n        x = pinned[:, :-1].contiguous()\n        y = pinned[:, 1:].contiguous()\n        return x, y\n\n    def next_batch(self, global_tokens: int, grad_accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:\n        x, y = self._build_batch_cpu(global_tokens, grad_accum_steps)\n        if self.device.type == "cuda":\n            x = x.to(self.device, non_blocking=True)\n            y = y.to(self.device, non_blocking=True)\n        return x, y\n\n    def prefill(self, *args, **kwargs) -> None:  # noqa: ARG002\n        # The curriculum loader does not use a prefetch thread. prefill is a no-op.\n        return None\n\n    def prefetch_queue_depth(self) -> int:\n        return 0\n'
exec(_idea_source_idea_curriculum_shard, _idea_module_idea_curriculum_shard.__dict__)
_bootstrap_sys.modules["submission.ideas.idea_curriculum_shard"] = _idea_module_idea_curriculum_shard
for _bootstrap_key in list(globals()):
    if _bootstrap_key.startswith(("_bootstrap_", "_idea_module_", "_idea_source_")):
        del globals()[_bootstrap_key]
del _bootstrap_key
import collections, copy, glob, io, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as _fa3_raw

    def flash_attn_3_func(q, k, v, causal=True):
        return _fa3_raw(q, k, v, causal=causal)
except ImportError:

    def flash_attn_3_func(q, k, v, causal=True):
        qt = q.transpose(1, 2)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        n_q = qt.size(1)
        n_kv = kt.size(1)
        if n_q != n_kv:
            n_rep = n_q // n_kv
            kt = kt.repeat_interleave(n_rep, dim=1)
            vt = vt.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2).contiguous()


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.667))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
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
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 2048))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 4.0))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 4))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.5))
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", "-1"))
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-08))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    muon_wd = float(os.environ.get("MUON_WD", 0.085))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    prequant_ttt_enabled = bool(int(os.environ.get("PREQUANT_TTT_ENABLED", "0")))
    prequant_ttt_lr = float(os.environ.get("PREQUANT_TTT_LR", 0.00045))
    prequant_ttt_epochs = int(os.environ.get("PREQUANT_TTT_EPOCHS", 8))
    prequant_ttt_freeze_blocks = int(os.environ.get("PREQUANT_TTT_FREEZE_BLOCKS", 1))
    prequant_ttt_batch_seqs = int(os.environ.get("PREQUANT_TTT_BATCH_SEQS", 32))
    prequant_ttt_grad_clip = float(os.environ.get("PREQUANT_TTT_GRAD_CLIP", 1.0))
    prequant_ttt_cosine_decay = bool(int(os.environ.get("PREQUANT_TTT_COSINE_DECAY", "1")))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 12.0))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(data_dir, "datasets", f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    tokenizer_path = os.path.join(data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model")
    logfile = f"logs/{run_id}.txt"
    model_path = os.environ.get("MODEL_PATH", "final_model.pt")
    quantized_model_path = os.environ.get("QUANTIZED_MODEL_PATH", "final_model.int6.ptz")


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
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(
            self.sp, h.vocab_size, device
        )
        self.boundary_mask = None
        self.pmi_matrix = None


def build_sentencepiece_luts(sp, vocab_size, device):
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
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
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


class ShuffledSequenceLoader:
    """Training data loader with optional background prefetch thread.

    Set USE_PREFETCH_LOADER=1 to enable CPU-side prefetch: a daemon thread
    builds batches into pinned-memory tensors ahead of the GPU, so the CPU
    data loader runs IN PARALLEL with GPU forward/backward. Queue depth is
    controlled by PREFETCH_DEPTH (default 4).

    Without prefetch (default), behaves identically to the original
    synchronous path: next_batch() builds the batch on the main thread then
    ships to GPU via .to(non_blocking=True).

    Thread safety: only the worker thread touches self.rng and self.start_inds
    once prefetch is active. The main thread only pops from the queue and
    does the H2D transfer. self.files / self.num_tokens are read-only after
    __init__ so memmap access is safe across threads.
    """

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
        self._use_prefetch = bool(int(os.environ.get("USE_PREFETCH_LOADER", "0")))
        self._prefetch_depth = int(os.environ.get("PREFETCH_DEPTH", "4"))
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._prefetch_args = None
        self._prefetch_use_pinned = bool(int(os.environ.get("PREFETCH_PIN_MEMORY", "1")))
        self._prefetch_stats = {"batches_served": 0, "queue_waits_empty": 0, "queue_waits_full": 0}

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def _build_batch_cpu(self, global_tokens, grad_accum_steps):
        """Build one (x, y) batch on CPU. Returns pinned tensors if
        PREFETCH_PIN_MEMORY=1 (default). Thread-safe for single-worker use."""
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        if self._prefetch_use_pinned:
            x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64, pin_memory=True)
            y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64, pin_memory=True)
        else:
            x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
            y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return (x, y)

    def _prefetch_worker(self):
        """Background daemon thread: loops forever, pushing batches into the
        queue. Any exception is surfaced to the main thread via a sentinel
        tuple ('__ERROR__', exc)."""
        try:
            while True:
                x, y = self._build_batch_cpu(*self._prefetch_args)
                self._prefetch_queue.put((x, y))
        except Exception as exc:
            try:
                self._prefetch_queue.put(("__ERROR__", exc))
            except Exception:
                pass

    def _ensure_prefetch_started(self, global_tokens, grad_accum_steps):
        if self._prefetch_queue is not None:
            return
        import queue as _queue
        import threading as _threading

        self._prefetch_queue = _queue.Queue(maxsize=self._prefetch_depth)
        self._prefetch_args = (global_tokens, grad_accum_steps)
        self._prefetch_thread = _threading.Thread(
            target=self._prefetch_worker, daemon=True, name="ShuffledSequenceLoader-prefetch"
        )
        self._prefetch_thread.start()
        print(f"[prefetch] daemon started: depth={self._prefetch_depth} pinned={self._prefetch_use_pinned}", flush=True)

    def next_batch(self, global_tokens, grad_accum_steps):
        if self._use_prefetch:
            self._ensure_prefetch_started(global_tokens, grad_accum_steps)
            if self._prefetch_queue.empty():
                self._prefetch_stats["queue_waits_empty"] += 1
            item = self._prefetch_queue.get()
            if isinstance(item, tuple) and len(item) >= 1 and (item[0] == "__ERROR__"):
                raise item[1]
            x, y = item
            self._prefetch_stats["batches_served"] += 1
            return (x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
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
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return (x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))

    def prefetch_queue_depth(self):
        """Current depth of the prefetch queue (for telemetry). Returns -1 if
        prefetch is disabled."""
        if self._prefetch_queue is None:
            return -1
        return self._prefetch_queue.qsize()

    def prefill(self, global_tokens, grad_accum_steps, target_depth=None, timeout_s=120.0):
        """Pre-fill the prefetch queue during pretime so training starts with
        a full queue. Front-loads CPU work into pretime (free) so the CPU is
        nearly idle during the 600s training budget (available for metric
        logging, optimizer offload, async checkpoint writes, etc.).

        Blocks until the queue has `target_depth` batches, or until timeout.
        Only runs if USE_PREFETCH_LOADER=1.

        Env var override: PREFETCH_PREFILL_BATCHES (default = PREFETCH_DEPTH).
        """
        if not self._use_prefetch:
            print("[prefetch] prefill: USE_PREFETCH_LOADER=0, skipping", flush=True)
            return
        if target_depth is None:
            target_depth = int(os.environ.get("PREFETCH_PREFILL_BATCHES", str(self._prefetch_depth)))
        target_depth = min(target_depth, self._prefetch_depth)
        self._ensure_prefetch_started(global_tokens, grad_accum_steps)
        import time as _time

        t0 = _time.perf_counter()
        last_log = t0
        print(
            f"[prefetch] prefill: target_depth={target_depth}, maxsize={self._prefetch_depth}, timeout={timeout_s}s",
            flush=True,
        )
        while True:
            current = self._prefetch_queue.qsize()
            if current >= target_depth:
                elapsed = _time.perf_counter() - t0
                print(f"[prefetch] prefill: reached depth {current}/{target_depth} in {elapsed:.2f}s", flush=True)
                return
            elapsed = _time.perf_counter() - t0
            if elapsed >= timeout_s:
                print(f"[prefetch] prefill: TIMEOUT at depth {current}/{target_depth} after {elapsed:.1f}s", flush=True)
                return
            if _time.perf_counter() - last_log > 5.0:
                print(f"[prefetch] prefill progress: {current}/{target_depth} at {elapsed:.1f}s", flush=True)
                last_log = _time.perf_counter()
            _time.sleep(0.1)


def _make_shard_loader(h, device):
    """Dispatch shard loading: curriculum if USE_CURRICULUM_SHARD=1, else
    the standard ShuffledSequenceLoader (rank-partitioned shuffle)."""
    try:
        from submission.ideas.idea_curriculum_shard import (
            CurriculumSequenceLoader as _CurriculumShard,
            is_enabled as _curriculum_enabled,
        )

        if _curriculum_enabled():
            return _CurriculumShard(h, device)
    except Exception:
        pass
    return ShuffledSequenceLoader(h, device)


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


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(self.train_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("_cos_pre", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("_sin_pre", freqs.sin()[None, :, None, :], persistent=False)
        self._max_pre_seq_len = self.train_seq_len
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if seq_len <= self._max_pre_seq_len:
            return (self._cos_pre[:, :seq_len].to(dtype=dtype), self._sin_pre[:, :seq_len].to(dtype=dtype))
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            scale = seq_len / self.train_seq_len
            new_base = self.base * scale ** (rd / (rd - 2))
            inv_freq = 1.0 / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return (self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype))


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = (x[..., :rope_dims], x[..., rope_dims:])
        half = rope_dims // 2
        x1, x2 = (x_rope[..., :half], x_rope[..., half:])
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = (x[..., :half], x[..., half:])
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
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
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False
        self.gate_proj = CastedLinear(dim, num_heads, bias=True)
        with torch.no_grad():
            self.gate_proj.weight.zero_()
            if self.gate_proj.bias is not None:
                self.gate_proj.bias.fill_(2.94)
        self.use_gated_attention = bool(int(os.environ.get("USE_GATED_ATTENTION", "0")))

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.use_gated_attention:
            gate = torch.sigmoid(self.gate_proj(x).float()).to(dtype=y.dtype)
            y = y * gate.unsqueeze(-1)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.use_norm_pct_dropout = bool(int(os.environ.get("USE_NORM_PCT_DROPOUT", "0")))
        self.norm_pct_thresh = float(os.environ.get("NORM_PCT_THRESH", "0.99"))

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
        if self.training and self.use_norm_pct_dropout:
            orig_shape = x.shape
            x_flat = x.reshape(-1, orig_shape[-1])
            row_norms = x_flat.float().norm(dim=-1)
            kth = torch.quantile(row_norms, self.norm_pct_thresh)
            keep = (row_norms < kth).to(dtype=x.dtype).unsqueeze(-1)
            x_flat = x_flat * keep
            x = x_flat.reshape(orig_shape)
        return self.proj(x)


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
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.layer_idx = layer_idx
        self._parallel_residuals = bool(int(os.environ.get("USE_PARALLEL_RESIDUALS", "0"))) or (
            int(os.environ.get("PARALLEL_RESIDUAL_START", "-1")) >= 0
            and layer_idx >= int(os.environ.get("PARALLEL_RESIDUAL_START", "-1"))
        )

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self._parallel_residuals:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return (
                x_in
                + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
                + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
            )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor
        )
        return x_out

    def forward_attn(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        return x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out

    def forward_mlp(self, x):
        return x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale_factor)


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
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        _per_layer_mlp = [h.mlp_mult] * h.num_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    _per_layer_mlp[i],
                    h.rope_base,
                    h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                )
                for i in range(h.num_layers)
            ]
        )
        self._huffman_remapper = None
        self._smeargate = None
        self._bigramhash_learned = None
        self._fused_ops = None
        self._fused_int6 = None
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims
                )
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = (
            nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32))
            if h.skip_gates_enabled
            else None
        )
        self.parallel_start_layer = int(os.environ.get("PARALLEL_START_LAYER", "7"))
        self.lane_merge = nn.Parameter(torch.tensor(0.5)) if self.parallel_start_layer > 0 else None
        self._ngram_enabled = bool(int(os.environ.get("USE_NGRAM_BIAS", "0")))
        self._ngram_w_bigram = float(os.environ.get("NGRAM_W_BIGRAM", "0.20"))
        self._ngram_w_trigram = float(os.environ.get("NGRAM_W_TRIGRAM", "0.15"))
        self._ngram_w_fourgram = float(os.environ.get("NGRAM_W_FOURGRAM", "0.10"))
        self._ngram_hash = int(os.environ.get("NGRAM_HASH_BUCKETS", "16384"))
        self._ngram_backoff = bool(int(os.environ.get("USE_NGRAM_BACKOFF", "0")))
        self._ngram_backoff_t4 = float(os.environ.get("NGRAM_BACKOFF_THRESH4", "1.0"))
        self._ngram_backoff_t3 = float(os.environ.get("NGRAM_BACKOFF_THRESH3", "1.0"))
        self._ngram_backoff_alpha = float(os.environ.get("NGRAM_BACKOFF_ALPHA", "0.4"))
        self._nlfi_enabled = bool(int(os.environ.get("USE_NGR_LOG_FREQ_INV", "0")))
        self._nlfi_applied = False
        self.register_buffer("_nlfi_bigram_mult", torch.ones(self._ngram_hash, dtype=torch.float32), persistent=False)
        self.register_buffer("_nlfi_trigram_mult", torch.ones(self._ngram_hash, dtype=torch.float32), persistent=False)
        self.register_buffer("_nlfi_fourgram_mult", torch.ones(self._ngram_hash, dtype=torch.float32), persistent=False)
        self.register_buffer("_nlfi_stored_flag", torch.zeros(1, dtype=torch.int64), persistent=False)
        self._ctx_part_tab_enabled = bool(int(os.environ.get("USE_CTX_PARTITIONED_TAB", "0")))
        self._ctx_part_slices = int(os.environ.get("CTX_PARTITION_SLICES", "16"))
        self.register_buffer("_bigram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_trigram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        self.register_buffer("_fourgram_tab", torch.zeros(1, dtype=torch.float32), persistent=False)
        if self._ngram_enabled:
            vs = h.vocab_size
            _ngram_bf16 = bool(int(os.environ.get("USE_NGRAM_BF16", "0")))
            _ngram_dtype = torch.bfloat16 if _ngram_bf16 else torch.float32
            _ngram_bigram_only = bool(int(os.environ.get("USE_NGRAM_BIGRAM_ONLY", "0")))
            for tab_attr, fname, label in [
                ("_bigram_tab", f"data/bigram_tab_{vs}v.npy", "bigram"),
                ("_trigram_tab", f"data/trigram_logprobs_{vs}v.npy", "trigram"),
                ("_fourgram_tab", f"data/fourgram_logprobs_{vs}v.npy", "fourgram"),
            ]:
                if _ngram_bigram_only and tab_attr != "_bigram_tab":
                    print(f"NGRAM_BIAS: {label} SKIPPED (USE_NGRAM_BIGRAM_ONLY=1)", flush=True)
                    continue
                try:
                    _arr = np.load(fname)
                    _tab = torch.from_numpy(_arr).to(dtype=_ngram_dtype)
                    setattr(self, tab_attr, _tab)
                    print(f"NGRAM_BIAS: loaded {label} {_arr.shape} from {fname} dtype={_ngram_dtype}", flush=True)
                except Exception as _e:
                    print(f"NGRAM_BIAS: {label} load failed ({fname}): {_e}", flush=True)
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and (module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        if getattr(self, "_smeargate", None) is not None:
            x = self._smeargate(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        )
        for i in enc_iter:
            x = self.blocks[i](x, x0)
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            if lane0 is None:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                if i >= psl and psl > 0:
                    lane0 = x
                    lane1 = x.clone()
                    lane0 = self.blocks[i].forward_attn(lane0, x0)
                    lane1 = self.blocks[i].forward_mlp(lane1)
                else:
                    x = self.blocks[i](x, x0)
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :] * skips.pop()
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(scaled_skip, lane0, g)
                    else:
                        lane0 = lane0 + scaled_skip
                lane0 = self.blocks[i].forward_attn(lane0, x0)
                lane1 = self.blocks[i].forward_mlp(lane1)
        if lane0 is not None:
            lm = self.lane_merge.to(dtype=lane0.dtype)
            x = lm * lane0 + (1.0 - lm) * lane1
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if getattr(self, "_bigramhash_learned", None) is not None:
            logits = logits + self._bigramhash_learned(input_ids).to(dtype=logits.dtype)
        if self._ngram_enabled and self._bigram_tab.numel() > 1:
            B, S = input_ids.shape
            _zeros1 = torch.zeros(B, 1, device=input_ids.device, dtype=input_ids.dtype)
            _zeros2 = torch.zeros(B, 2, device=input_ids.device, dtype=input_ids.dtype)
            _ids_flat = input_ids.reshape(-1).long()
            _prev2 = torch.cat([_zeros1, input_ids[:, :-1]], dim=1).reshape(-1).long()
            _prev3 = torch.cat([_zeros2, input_ids[:, :-2]], dim=1).reshape(-1).long()
            H = self._ngram_hash
            _h_bi = _ids_flat * 36313 % H
            if self._ctx_part_tab_enabled:
                _S_slices = self._ctx_part_slices
                _zone = _ids_flat % _S_slices * (H // _S_slices)
                _h_bi = (_h_bi + _zone) % H
            _h_tri = (_prev2 * 36313 + _ids_flat * 27191) % H
            _h_four = (_prev3 * 36313 + _prev2 * 27191 + _ids_flat * 51497) % H
            _bi = self._bigram_tab[_h_bi].reshape(B, S, -1)
            if self._ngram_backoff and self._trigram_tab.numel() > 1 and (self._fourgram_tab.numel() > 1):
                _tri = self._trigram_tab[_h_tri].reshape(B, S, -1)
                _four = self._fourgram_tab[_h_four].reshape(B, S, -1)
                _peak4 = _four.amax(dim=-1, keepdim=True)
                _peak3 = _tri.amax(dim=-1, keepdim=True)
                _use_4 = (_peak4 > self._ngram_backoff_t4).to(_four.dtype)
                _use_3 = (1 - _use_4) * (_peak3 > self._ngram_backoff_t3).to(_tri.dtype)
                _use_bi = 1 - _use_4 - _use_3
                _alpha = self._ngram_backoff_alpha
                _ng = _use_4 * _four + _use_3 * _tri * _alpha + _use_bi * _bi * (_alpha * _alpha)
                logits = logits + _ng.to(dtype=logits.dtype)
            else:
                _bias = self._ngram_w_bigram * _bi
                if self._trigram_tab.numel() > 1:
                    _bias = _bias + self._ngram_w_trigram * self._trigram_tab[_h_tri].reshape(B, S, -1)
                if self._fourgram_tab.numel() > 1:
                    _bias = _bias + self._ngram_w_fourgram * self._fourgram_tab[_h_four].reshape(B, S, -1)
                logits = logits + _bias.to(dtype=logits.dtype)
        return logits

    @torch.no_grad()
    def _apply_nlfi_once(self, input_ids):
        if self._nlfi_applied or not self._nlfi_enabled:
            return
        if not (self._ngram_enabled and self._bigram_tab.numel() > 1):
            return
        try:
            _ids_flat = input_ids.reshape(-1).long()
            H = self._ngram_hash
            if int(self._nlfi_stored_flag.item()) == 1:
                _bg_mult = self._nlfi_bigram_mult
                _tg_mult = self._nlfi_trigram_mult
                _fg_mult = self._nlfi_fourgram_mult
                print("NGR_LOG_FREQ_INV: restored multipliers from state_dict", flush=True)
            else:
                _bg_h_init = _ids_flat * 36313 % H
                _bg_counts = torch.zeros(H, dtype=torch.float32, device=_ids_flat.device)
                _bg_counts.scatter_add_(0, _bg_h_init, torch.ones_like(_bg_h_init, dtype=torch.float32))
                _bg_mult = 1.0 / torch.log(2.0 + _bg_counts)
                _tg_h_init = (_ids_flat * 36313 ^ _ids_flat * 39979 >> 1) % H
                _tg_counts = torch.zeros(H, dtype=torch.float32, device=_ids_flat.device)
                _tg_counts.scatter_add_(0, _tg_h_init, torch.ones_like(_tg_h_init, dtype=torch.float32))
                _tg_mult = 1.0 / torch.log(2.0 + _tg_counts)
                _fg_h_init = (_ids_flat * 36313 ^ _ids_flat * 39979 >> 1 ^ _ids_flat * 41077 >> 2) % H
                _fg_counts = torch.zeros(H, dtype=torch.float32, device=_ids_flat.device)
                _fg_counts.scatter_add_(0, _fg_h_init, torch.ones_like(_fg_h_init, dtype=torch.float32))
                _fg_mult = 1.0 / torch.log(2.0 + _fg_counts)
                self._nlfi_bigram_mult.data = _bg_mult.detach().to(self._nlfi_bigram_mult.dtype)
                self._nlfi_trigram_mult.data = _tg_mult.detach().to(self._nlfi_trigram_mult.dtype)
                self._nlfi_fourgram_mult.data = _fg_mult.detach().to(self._nlfi_fourgram_mult.dtype)
                self._nlfi_stored_flag.data = torch.ones(1, dtype=torch.int64, device=self._nlfi_stored_flag.device)
                print("NGR_LOG_FREQ_INV: computed + saved multipliers from current batch", flush=True)
            if self._bigram_tab.numel() > 1:
                if self._bigram_tab.dim() == 2:
                    self._bigram_tab.mul_(_bg_mult.to(self._bigram_tab.dtype).unsqueeze(1))
                else:
                    self._bigram_tab.mul_(_bg_mult.to(self._bigram_tab.dtype))
            if self._trigram_tab.numel() > 1:
                if self._trigram_tab.dim() == 2:
                    self._trigram_tab.mul_(_tg_mult.to(self._trigram_tab.dtype).unsqueeze(1))
                else:
                    self._trigram_tab.mul_(_tg_mult.to(self._trigram_tab.dtype))
            if self._fourgram_tab.numel() > 1:
                if self._fourgram_tab.dim() == 2:
                    self._fourgram_tab.mul_(_fg_mult.to(self._fourgram_tab.dtype).unsqueeze(1))
                else:
                    self._fourgram_tab.mul_(_fg_mult.to(self._fourgram_tab.dtype))
            print("NGR_LOG_FREQ_INV: applied mutation to n-gram tables (one-time per process)", flush=True)
        except Exception as _e:
            print(f"NGR_LOG_FREQ_INV: mutation failed ({_e})", flush=True)
        self._nlfi_applied = True

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")


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
    a, b, c = (3.4445, -4.775, 2.0315)
    X = G.bfloat16()
    if X.dim() == 2:
        X = X / (X.norm() + eps)
    else:
        X = X / (X.flatten(start_dim=-2).norm(dim=-1, keepdim=True).unsqueeze(-1) + eps)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.transpose(-2, -1) if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_normalize=False):
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

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        _use_parallel_muon = int(os.environ.get("USE_PARALLEL_MUON", "0"))
        _use_normuon = int(os.environ.get("USE_NORMUON", "0"))
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum((int(p.numel()) for p in params))
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            _offsets = [0]
            for p in params:
                _offsets.append(_offsets[-1] + p.numel())
            if _use_parallel_muon:
                _shape_groups = {}
                for i, p in enumerate(params):
                    if i % world_size != rank:
                        continue
                    if p.grad is None:
                        continue
                    sh = tuple(p.grad.shape)
                    _shape_groups.setdefault(sh, []).append((i, p))
                for sh, grp in _shape_groups.items():
                    _grads = []
                    for i, p in grp:
                        g = p.grad
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(g)
                        if nesterov:
                            g = g.add(buf, alpha=momentum)
                        if group.get("row_normalize", False):
                            _rn = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                            g = g / _rn.to(g.dtype)
                        _grads.append(g)
                    _stacked = torch.stack(_grads, dim=0)
                    _result = zeropower_via_newtonschulz5(_stacked, steps=backend_steps)
                    for _bi, (i, p) in enumerate(grp):
                        g = _result[_bi]
                        if _use_normuon:
                            _post_norm = g.float().norm(dim=-1, keepdim=True).clamp(min=1e-08)
                            g = g / _post_norm.to(g.dtype)
                        g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                        updates_flat[_offsets[i] : _offsets[i + 1]] = g.reshape(-1)
            else:
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
                        if group.get("row_normalize", False):
                            row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                            g = g / row_norms.to(g.dtype)
                        g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                        if _use_normuon:
                            _post_norm = g.float().norm(dim=-1, keepdim=True).clamp(min=1e-08)
                            g = g / _post_norm.to(g.dtype)
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                        updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                    curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    (
        pattern
        for pattern in os.environ.get(
            "CONTROL_TENSOR_NAME_PATTERNS",
            "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,lane_merge",
        ).split(",")
        if pattern
    )
)


class Optimizers:
    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [
            p
            for name, p in block_named_params
            if p.ndim == 2 and (not any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]
        scalar_params = [
            p
            for name, p in block_named_params
            if p.ndim < 2 or any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if base_model.lane_merge is not None:
            scalar_params.append(base_model.lane_merge)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
        self.optimizer_tok = torch.optim.AdamW(
            tok_params, betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True
        )
        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": h.head_lr, "base_lr": h.head_lr}],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (
            param.ndim < 2 or any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
        ) and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def make_hook(name):

        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm

        def make_output_hook(name):

            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)

            return hook_fn

        hooks.append(hook_module.register_forward_hook(make_output_hook("tok_emb.weight")))
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
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
    Q = Q[:, invperm]
    if int(os.environ.get("USE_CMP_QUANT_VALUE_DEDUP", "0")):
        _cqvd_step = int(os.environ.get("CMP_QUANT_DEDUP_STEP", "2"))
        if _cqvd_step > 1:
            Q = (Q.to(torch.int16) // _cqvd_step * _cqvd_step).to(torch.int8)
    return (Q, s)


def gptq_mixed_quantize(state_dict, hessians, h):
    _use_sigma_delta = False
    _use_vernier = False
    try:
        from submission.ideas.idea_064_parallel_gptq import is_enabled as pg_on

        if pg_on():
            log("[IDEA-064 parallel_gptq] enabled - multi-clip search active")
    except ImportError:
        pass
    _cascade_bits = None
    _mixed_int_bits = {}
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        cs = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        if name in _mixed_int_bits:
            bits = _mixed_int_bits[name]
        if _cascade_bits is not None and "tok_emb" not in name:
            import re as _re

            _m = _re.search("blocks\\.(\\d+)\\.", name)
            if _m:
                bits = _cascade_bits.get(int(_m.group(1)), bits)
        _use_par_gptq = False
        try:
            from submission.ideas.idea_064_parallel_gptq import is_enabled as pg2_on

            _use_par_gptq = pg2_on()
        except ImportError:
            pass
        if _use_par_gptq and "tok_emb" not in name:
            try:
                from submission.ideas.idea_064_parallel_gptq import parallel_search_best_clips
                import numpy as _np

                _best_clips = parallel_search_best_clips(
                    _np.array(t.numpy()),
                    _np.array(hessians[name].numpy()) if name in hessians else None,
                    clip_range=2 ** (bits - 1) - 1,
                )
                if _best_clips is not None:
                    cs = float(_best_clips.mean())
            except Exception:
                pass
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub("\\.\\d+$", "", re.sub("blocks\\.\\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return (result, meta)


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = (result[name + ".q"], result[name + ".scale"])
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))).to(orig_dtype)
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
    _bpe_codebook = None
    if compressor == "lzma":
        compressed = lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli

        compressed = brotli.compress(data, quality=11)
    elif compressor == "zstd":
        import zstandard

        compressed = zstandard.ZstdCompressor(level=22).compress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    if _bpe_codebook is not None:
        header = len(_bpe_codebook).to_bytes(4, "big") + _bpe_codebook
    else:
        header = b"\x00\x00\x00\x00"
    return header + compressed


def _decompress(data, compressor):
    _bpe_header_len = int.from_bytes(data[:4], "big")
    if _bpe_header_len > 0:
        _bpe_codebook_bytes = data[4 : 4 + _bpe_header_len]
        data = data[4 + _bpe_header_len :]
    else:
        _bpe_codebook_bytes = None
        data = data[4:]
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli

        raw = brotli.decompress(data)
    elif compressor == "zstd":
        import zstandard

        raw = zstandard.ZstdDecompressor().decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    if _bpe_codebook_bytes is not None:
        pass
    raw = _byte_unshuffle(raw)
    return raw


class _ValCalibLoader:
    def __init__(self, val_tokens, h, device):
        self.val_tokens = val_tokens
        self.h = h
        self.device = device
        self._offset = 0

    def next_batch(self, batch_tokens, grad_accum_steps):
        seq_len = self.h.train_seq_len
        batch_seqs = max(1, batch_tokens // (seq_len * max(1, grad_accum_steps)))
        needed = batch_seqs * seq_len + 1
        if self._offset + needed > self.val_tokens.numel():
            self._offset = 0
        chunk = self.val_tokens[self._offset : self._offset + needed].to(device=self.device, dtype=torch.int64)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        self._offset += needed - 1
        return (x, y)


def serialize(h, base_model, code, val_data=None):
    code_bytes = len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device("cuda", h.local_rank)
    if int(os.environ.get("GPTQ_CALIB_USE_VAL", "0")):
        log(
            "WARNING: GPTQ_CALIB_USE_VAL is disabled - calibrating on val tokens violates Rule 6 (data leakage). Ignoring."
        )
    log("GPTQ:collecting Hessians from calibration data...")
    calib_loader = ShuffledSequenceLoader(h, device)
    t0 = time.perf_counter()
    hessians = collect_hessians(base_model, calib_loader, h, device, n_calibration_batches=h.gptq_calibration_batches)
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    try:
        from submission.ideas.idea_051_freeze_dry import freeze_dry_state_dict

        freeze_dry_state_dict(quant_result)
    except ImportError:
        pass
    # IDEA-038 vernier and IDEA-023 sigma_delta are inactive in this build -
    # the dead-code remover stripped their try/except wrappers, so the failure
    # log lines that referenced the bound exception variable are now `pass`.
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
    return (bytes_total, quant_file_bytes)


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu")
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return (val_loss, val_bpb)


def eval_val(h, device, val_data, model):
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
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(
                dtype=torch.int16
            )
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    _eval_pool = None
    _cmix_mixer = None
    _rep_detector = None
    _state_machine = None
    _ctx_ngram = None
    _tilt_tables = None
    _ngram_cache_mixer = None
    _sa_mixer = None
    _ctw_mixer = None
    _dirichlet = None
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi : bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws : we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            _cpu_fut = None
            if _eval_pool is not None:
                import numpy as _np

                _cpu_fut = _eval_pool.submit(_np.array(x_batch[0, : wlens[0]].cpu()))
            _wse_active = False
            if _wse_active:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = logits_fn(x_batch)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = logits_fn(x_batch)
            if _cpu_fut is not None:
                _cpu_biases = _cpu_fut.result()
                _total_bias = _eval_pool.blend_biases(_cpu_biases)
                import torch as _torch

                _bias_t = _torch.from_numpy(_total_bias).to(logits.device).to(logits.dtype)
                for _bi in range(bsz):
                    logits[_bi, : wlens[_bi] if _bi < len(wlens) else seq_len] += _bias_t
            if _tilt_tables is not None:
                pass
            if getattr(val_data, "pmi_matrix", None) is not None:
                pass
            if _cmix_mixer is not None:
                pass
            if _rep_detector is not None:
                pass
            if _state_machine is not None:
                pass
            if _ctx_ngram:
                pass
            if _ngram_cache_mixer is not None:
                pass
            if _ctw_mixer is not None:
                pass
            if _sa_mixer is not None:
                pass
            if _dirichlet is not None:
                try:
                    import numpy as _dnp

                    _dbias = _dirichlet.predict(_dnp.array(x_batch[0, : wlens[0]].cpu()), h.vocab_size)
                    import torch as _dt

                    logits[:1, : wlens[0]] += _dt.from_numpy(_dbias).to(logits.device).to(logits.dtype)
                except ImportError:
                    pass
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction="none"
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
            _scored = y_batch[0, context_size : wlens[0]].tolist() if len(wlens) > 0 else []
            if _eval_pool is not None:
                _eval_pool.update_all(_scored)
            if _cmix_mixer is not None:
                _cmix_mixer.observe(_scored)
            if _rep_detector is not None:
                _rep_detector.observe(_scored)
            if _state_machine is not None:
                for _tok in _scored:
                    _state_machine.update(int(_tok))
            if _ngram_cache_mixer is not None:
                _ngram_cache_mixer.observe(_scored)
            if _ctw_mixer is not None:
                _ctw_mixer.observe(_scored)
            if _dirichlet is not None:
                _dirichlet.observe(_scored)
            if _sa_mixer is not None:
                _sa_mixer.observe_batch(_scored)
    if _eval_pool is not None:
        _eval_pool.shutdown()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def eval_val_sliding_ttt(h, base_model, rank, world_size, device, val_data, stride):
    seq_len = h.eval_seq_len
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log(
        f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} total_windows={len(window_starts)} stride={stride} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs} freeze_blocks={h.ttt_freeze_blocks}"
    )
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    frozen_block_ids = set(range(min(h.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"blocks.{bi}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)
    log(
        f"ttt_sliding:params unfrozen={sum((p.numel() for p in ttt_params))} frozen={sum((p.numel() for p in base_model.parameters() if not p.requires_grad))}"
    )
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    t0 = time.perf_counter()
    batch_seqs = h.ttt_batch_seqs
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi : bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction="none"
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(
                        torch.float64
                    )
                    byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, h.ttt_grad_clip)
                        optimizer.step()
        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
            log(f"  ttt_chunk [{ci + 1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} elapsed={time.perf_counter() - t0:.1f}s")
    return (val_loss, val_bpb)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    return (val_loss, val_bpb)


def _load_train_sample_for_nlfi(h, device):
    """RULE COMPLIANCE: NLFI bias mutation must use TRAIN data, not val (the comp
    rules forbid accessing val data during training). Loads the first eval_seq_len
    tokens from the first train shard. Deterministic, so train-side and eval-side
    NLFI setup compute matching multipliers."""
    try:
        _train_files = sorted(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin"))
        if not _train_files:
            return None
        _arr = np.fromfile(str(_train_files[0]), dtype=np.uint16, count=h.eval_seq_len)
        if _arr.size < h.eval_seq_len:
            return None
        return torch.from_numpy(_arr.astype(np.int64)).to(device).view(1, -1)
    except Exception as _e:
        print(f"NLFI: train sample load failed ({_e}), falling back to no setup", flush=True)
        return None


def train_model(h, device, val_data, contrastive_init=None):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    if contrastive_init is not None:
        try:
            _ci_loaded = 0
            for k, v in contrastive_init.items():
                if k in base_model.state_dict():
                    base_model.state_dict()[k].copy_(v)
                    _ci_loaded += 1
            log(f"[IDEA-024 contrastive] transferred {_ci_loaded}/{len(contrastive_init)} pretrained weight tensors")
        except Exception as e:
            log(f"[IDEA-024 contrastive] weight transfer failed: {e}")
    if getattr(base_model, "_nlfi_enabled", False) and (not getattr(base_model, "_nlfi_applied", False)):
        _sample = _load_train_sample_for_nlfi(h, device)
        if _sample is not None:
            base_model._apply_nlfi_once(_sample)
    _compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
    if _compile_mode == "default":
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    else:
        log(f"torch.compile mode={_compile_mode}")
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True, mode=_compile_mode)
    if h.distributed:
        # find_unused_parameters=True: per-layer params (skip_gates, lane_merge,
        # XSA projections) are conditionally unused on a given step depending on
        # the active forward path. Required to avoid DDP raising on first step.
        model = DDP(
            compiled_model,
            device_ids=[h.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = compiled_model
    log(f"model_params:{sum((p.numel() for p in base_model.parameters()))}")
    optimizers = Optimizers(h, base_model)
    train_loader = _make_shard_loader(h, device)
    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    train_loader.prefill(h.train_batch_tokens, h.grad_accum_steps)
    _curriculum_results = []
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1000.0
        log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
    _fuzzy_enabled = int(os.environ.get("USE_FUZZY_LR_BANDIT", "0"))
    _fuzzy_arms = [0.5, 1.0, 2.0]
    _fuzzy_means = [0.0, 0.0, 0.0]
    _fuzzy_counts = [1, 1, 1]
    _fuzzy_prev_loss = None
    _fuzzy_arm_idx = 1
    if _fuzzy_enabled:
        log(f"FUZZY_LR_BANDIT: enabled arms={_fuzzy_arms} (Shot 17)")
    _fermented_state = None
    _rare_weights = None
    _sharpen_state = None
    _bma_mgr = None
    _rnt_tables = None
    _maml_state = None
    try:
        from submission.ideas.idea_051_freeze_dry import is_enabled as fd_on

        if fd_on():
            log("[IDEA-051 freeze_dry] enabled - linear-combo pruning active")
    except ImportError:
        pass
    _distil_state = None

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

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            if _rnt_tables is not None:
                pass
            train_loss += loss.detach()
            if _rare_weights is not None:
                try:
                    _batch_toks = y.detach().cpu().numpy().flatten()
                    _rw_scale = float(_rare_weights[_batch_toks % len(_rare_weights)].mean())
                    loss = loss * _rw_scale
                except Exception:
                    pass
            if _fermented_state is not None:
                try:
                    _fp_scale = _fermented_state.step(loss.detach().unsqueeze(0).unsqueeze(0))
                    if _fp_scale is not None:
                        loss = loss * _fp_scale.mean().to(loss.device)
                except Exception:
                    pass
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = _make_shard_loader(h, device)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    if _distil_state is not None and _distil_state.variant == "v1":
        try:
            _distil_state.setup_v1_teacher(base_model.state_dict())
        except Exception:
            pass
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        if h.num_loops > 0 and (not base_model.looping_active) and (frac >= h.enable_looping_at):
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        if _fuzzy_enabled:
            _samples = [
                _fuzzy_means[i] + random.gauss(0, 1.0 / _fuzzy_counts[i] ** 0.5) for i in range(len(_fuzzy_arms))
            ]
            _fuzzy_arm_idx = _samples.index(max(_samples))
            scale = scale * _fuzzy_arms[_fuzzy_arm_idx]
        if _curriculum_results:
            pass
        train_loss = step_fn(step, scale)
        if _maml_state is not None and _maml_state.should_run():
            try:
                x_s, y_s = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
                x_q, y_q = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
                _ml = _maml_state.maml_loss(base_model, x_s, y_s, x_q, y_q)
                if _ml is not None:
                    _ml.backward()
            except Exception:
                pass
        if _distil_state is not None and _distil_state.teacher_state is not None:
            try:
                _d_frac = training_frac(step, training_time_ms + 1000.0 * (time.perf_counter() - t0))
                if _d_frac >= _distil_state.start_frac:
                    from submission.ideas.idea_059_distillation import kd_loss_from_logits

                    _d_x, _d_y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _student_logits = base_model.forward_logits(_d_x)
                    _saved_state = {k: v.clone() for k, v in base_model.state_dict().items()}
                    base_model.load_state_dict(
                        {k: v.to(device) for k, v in _distil_state.teacher_state.items()}, strict=True
                    )
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _teacher_logits = base_model.forward_logits(_d_x)
                    base_model.load_state_dict(_saved_state, strict=True)
                    _kd = kd_loss_from_logits(_student_logits, _teacher_logits.detach(), _distil_state.temp)
                    (_distil_state.alpha * _kd).backward()
                    del _saved_state, _student_logits, _teacher_logits, _d_x, _d_y
            except Exception:
                pass
        if _fuzzy_enabled:
            _cur_loss = train_loss.item()
            if _fuzzy_prev_loss is not None:
                _reward = _fuzzy_prev_loss - _cur_loss
                _fuzzy_counts[_fuzzy_arm_idx] += 1
                _fuzzy_means[_fuzzy_arm_idx] += (_reward - _fuzzy_means[_fuzzy_arm_idx]) / _fuzzy_counts[_fuzzy_arm_idx]
            _fuzzy_prev_loss = _cur_loss
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (
            step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1000.0)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    if _fuzzy_enabled:
        _best_arm = _fuzzy_means.index(max(_fuzzy_means))
        _total = sum(_fuzzy_counts) - len(_fuzzy_counts)
        log(
            f"FUZZY_LR_BANDIT summary: arms={_fuzzy_arms} means={[round(m, 4) for m in _fuzzy_means]} counts={[c - 1 for c in _fuzzy_counts]} total_steps={_total} best_arm={_fuzzy_arms[_best_arm]}"
        )
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return (base_model, compiled_model)


def prequant_ttt_adapt_adamw(h, base_model, device, val_tokens, rank=0, world_size=1):
    """Pre-Quant AdamW TTT (ported from PR #1485 / #1306).
    Fine-tunes the EMA-applied base_model on val tokens BEFORE GPTQ so the
    adaptation bakes into the quantized weights. Frontier (PR #1482) gives ~-0.014
    BPB on top of eval-time TTT. Modifies base_model in place.
    """
    seq_len = h.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = h.prequant_ttt_batch_seqs
    if h.prequant_ttt_freeze_blocks > 0:
        for i, block in enumerate(base_model.blocks):
            if i < h.prequant_ttt_freeze_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
    _shadow_active = False
    _prequant_ttt_epochs = h.prequant_ttt_epochs
    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    log(
        f"prequant_ttt:params trainable={sum((p.numel() for p in ttt_params))} frozen={sum((p.numel() for p in base_model.parameters() if not p.requires_grad))}"
    )
    _pg = None
    if _pg is not None:
        optimizer = torch.optim.AdamW(_pg, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(ttt_params, lr=h.prequant_ttt_lr, weight_decay=0.0)
    scheduler = None
    if h.prequant_ttt_cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=_prequant_ttt_epochs, eta_min=h.prequant_ttt_lr * 0.1
        )
    my_start = total_seqs * rank // world_size
    my_end = total_seqs * (rank + 1) // world_size
    base_model.train()
    t0 = time.perf_counter()
    _ttt_bma_mgr = None
    _ttt_sharpen = None
    for epoch in range(_prequant_ttt_epochs):
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_tokens = torch.zeros((), device=device, dtype=torch.float64)
        for bs in range(my_start, my_end, batch_seqs):
            be = min(bs + batch_seqs, my_end)
            raw_start = bs * seq_len
            raw_end = be * seq_len + 1
            if raw_end > val_tokens.numel():
                continue
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            if _ttt_sharpen is not None:
                try:
                    _sharpen_w = _ttt_sharpen.compute_weights(loss.detach().unsqueeze(0).unsqueeze(0))
                    loss = loss * _sharpen_w.mean().to(loss.device)
                except Exception:
                    pass
            loss.backward()
            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(ttt_params, h.prequant_ttt_grad_clip)
            optimizer.step()
            if _ttt_bma_mgr is not None:
                try:
                    if _ttt_bma_mgr.should_snapshot(epoch):
                        _ttt_bma_mgr.save_snapshot(base_model)
                except Exception:
                    pass
            epoch_loss_sum += loss.detach().to(torch.float64) * float(y.numel())
            epoch_tokens += float(y.numel())
        if world_size > 1:
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_tokens, op=dist.ReduceOp.SUM)
        epoch_avg = epoch_loss_sum.item() / max(epoch_tokens.item(), 1)
        if scheduler is not None:
            scheduler.step()
        log(
            f"prequant_ttt:epoch {epoch + 1}/{_prequant_ttt_epochs} loss:{epoch_avg:.4f} time:{time.perf_counter() - t0:.1f}s"
        )
    if _ttt_bma_mgr is not None and _ttt_bma_mgr.snapshots:
        try:
            _bma_weights = _ttt_bma_mgr.compute_weights()
            _bma_avg = {}
            for i, snap in enumerate(_ttt_bma_mgr.snapshots):
                for k, v in snap.items():
                    if k not in _bma_avg:
                        _bma_avg[k] = v.to(device).float() * _bma_weights[i]
                    else:
                        _bma_avg[k] += v.to(device).float() * _bma_weights[i]
            _final_w = 1.0 / (len(_bma_weights) + 1)
            _snap_w = 1.0 - _final_w
            _cur = base_model.state_dict()
            for k in _bma_avg:
                _bma_avg[k] = _snap_w * _bma_avg[k] + _final_w * _cur[k].float()
                _bma_avg[k] = _bma_avg[k].to(dtype=_cur[k].dtype)
            base_model.load_state_dict(_bma_avg, strict=True)
            log(f"[IDEA-022 bma_ttt] averaged {len(_ttt_bma_mgr.snapshots)} snapshots + final state")
        except Exception as e:
            log(f"[IDEA-022 bma_ttt] averaging failed: {e}")
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log(f"prequant_ttt:done elapsed={time.perf_counter() - t0:.1f}s")


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log("train_shards: " + str(len(list(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin")))))
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")
    _contrastive_pretrained_state = None
    base_model, compiled_model = train_model(h, device, val_data, contrastive_init=_contrastive_pretrained_state)
    torch._dynamo.reset()
    timed_eval("pre-quantization post-ema", eval_val, h, device, val_data, compiled_model)
    if h.prequant_ttt_enabled:
        prequant_ttt_adapt_adamw(h, base_model, device, val_data.val_tokens, rank=h.rank, world_size=h.world_size)
        torch._dynamo.reset()
        timed_eval("post-prequant-ttt", eval_val, h, device, val_data, base_model)
    _deq_was_unwrapped = False
    try:
        serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"), val_data=val_data)
    finally:
        pass
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    if getattr(eval_model, "_nlfi_enabled", False) and (not getattr(eval_model, "_nlfi_applied", False)):
        _sample = _load_train_sample_for_nlfi(h, device)
        if _sample is not None:
            eval_model._apply_nlfi_once(_sample)
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        timed_eval("quantized_sliding_window", eval_val_sliding, h, device, val_data, eval_model)
    if h.ttt_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval(
            "quantized_ttt",
            eval_val_sliding_ttt,
            h,
            ttt_model,
            h.rank,
            h.world_size,
            device,
            val_data,
            stride=h.eval_stride,
        )
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
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = bool(int(os.environ.get("USE_CUDNN_BENCHMARK", "1")))
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
            ).stdout,
            console=False,
        )
        log("=" * 100, console=False)
        try:
            import json as _json

            _hp = {
                k: v
                for k, v in vars(type(h)).items()
                if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
            }
            _meta = {
                "hyperparams": _hp,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": getattr(torch.version, "cuda", None),
                "env_toggles": {
                    k: os.environ[k]
                    for k in sorted(os.environ)
                    if k.startswith(
                        (
                            "USE_",
                            "GPTQ_",
                            "TTT_",
                            "PREQUANT_",
                            "SLIDING_",
                            "EMBED_",
                            "MATRIX_",
                            "NUM_",
                            "MODEL_",
                            "SEED",
                            "EXP_ID",
                            "TRAIN_",
                            "VAL_",
                            "WEIGHT_",
                            "BEZIER_",
                            "DRAFT_",
                            "SPEC_",
                            "DUAL_MLP_",
                            "FUSED_",
                            "INT6_",
                            "SIZE_",
                        )
                    )
                },
            }
            _out = os.path.dirname(h.model_path) or "."
            _json.dump(_meta, open(os.path.join(_out, "hyperparams.json"), "w"), indent=2, default=str)
        except Exception as _e:
            log(f"[hp dump] failed: {_e}", console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
