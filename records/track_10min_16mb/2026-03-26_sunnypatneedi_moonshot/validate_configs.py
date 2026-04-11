#!/usr/bin/env python3
"""
validate_configs.py — CPU-only config validator.

Estimates parameter counts and artifact sizes for each config in MOONSHOT_SPACE
without running training or needing GPU/data. Uses analytical formulas based on
the adaptive quantization scheme in train_gpt_v10_moonshot.py.

Quantization scheme (adaptive_quantize):
  - blocks.*.mlp.*  (large >65536): ternary int8  -> ~0.25 bytes/param post-zstd-22
  - blocks.*.attn.* (large >65536): int4 int8     -> ~0.50 bytes/param post-zstd-22
  - tok_emb / bigram.embed / ve_shared.embed / other large: int6 -> ~0.75 bytes/param
  - small tensors (<=65536, non-control): float16  -> ~1.80 bytes/param
  - control tensors (gates/scales/mix): float32   -> ~3.20 bytes/param

Compression ratios calibrated from:
  PR #486: ~30M params, uniform int6 -> 15.34MB artifact -> ~0.51 bytes/param
  Ternary (3 values): 5-7x better zstd than int6 -> ~0.20-0.25 bytes/param
  Int4 (15 values): ~2x better than int6 -> ~0.45-0.55 bytes/param

Fixed architecture:
  vocab_size=1024, num_layers=11, num_heads=8, num_kv_heads=4 (kv_dim=D/2)
  bigram_vocab_size=6144, bigram_dim=128, ve_dim=128, ve_layers=[7,8,9,10]
  gated_attention=True, INT8_KEEP_FLOAT_MAX_NUMEL=65536
"""
import json
import random
from pathlib import Path

# Architecture constants
NUM_LAYERS      = 11
VOCAB_SIZE      = 1024
NUM_HEADS       = 8
NUM_KV_HEADS    = 4
BIGRAM_VOCAB    = 6144
BIGRAM_DIM      = 128
VE_DIM          = 128
VE_LAYERS       = 4     # "7,8,9,10"
GATED_ATTN      = True
SMALL_THRESHOLD = 65_536

# Code file sizes (bytes)
CODE_BYTES_MOONSHOT = 82_935
CODE_BYTES_SAFE     = 82_956

# Compression bytes-per-param after zstd level 22
BPP_TERNARY  = 0.25
BPP_INT4     = 0.50
BPP_INT6     = 0.75
BPP_FLOAT16  = 1.80
BPP_FLOAT32  = 3.20

OVERHEAD_BYTES = 35_000  # torch.save + pickle + zstd frame overhead
LIMIT = 16_000_000


def estimate_artifact_size(model_dim, mlp_mult, script="moonshot"):
    D = model_dim
    M = mlp_mult
    head_dim = D // NUM_HEADS
    kv_dim   = NUM_KV_HEADS * head_dim  # = D/2

    # Large MLP matrices -> ternary
    mlp_fc_params   = int(D * M * D)   # D x (M*D)
    mlp_proj_params = int(M * D * D)   # (M*D) x D
    mlp_params_total = (mlp_fc_params + mlp_proj_params) * NUM_LAYERS
    # Scales: fc rows=D, proj rows=M*D -> D*(1+M) per block
    mlp_scale_params = int(D * (1 + M)) * NUM_LAYERS

    # Large attention matrices -> int4
    attn_weights = [D * D, D * kv_dim, D * kv_dim, D * D]  # cq, ck, cv, proj
    attn_int4_params   = sum(n for n in attn_weights if n > SMALL_THRESHOLD) * NUM_LAYERS
    attn_small_params  = sum(n for n in attn_weights if n <= SMALL_THRESHOLD) * NUM_LAYERS
    # Scales: 4*D rows per block
    attn_scale_params  = 4 * D * NUM_LAYERS
    # attn_gate: D*8 + 8 bias, always small -> float16
    attn_gate_params   = (D * 8 + 8) * NUM_LAYERS if GATED_ATTN else 0

    # Large embed/other -> int6
    tok_emb_n    = VOCAB_SIZE * D                           # e.g. 1024*640
    bigram_emb_n = BIGRAM_VOCAB * BIGRAM_DIM                # 6144*128=786432
    bigram_proj_n = BIGRAM_DIM * D                          # e.g. 128*640=81920
    ve_emb_n     = VOCAB_SIZE * VE_DIM                      # 1024*128=131072
    ve_proj_n    = VE_DIM * kv_dim                          # 128*(D/2)

    bigram_proj_int6 = bigram_proj_n if bigram_proj_n > SMALL_THRESHOLD else 0
    bigram_proj_f16  = bigram_proj_n if bigram_proj_n <= SMALL_THRESHOLD else 0
    ve_proj_int6 = ve_proj_n if ve_proj_n > SMALL_THRESHOLD else 0
    ve_proj_f16  = ve_proj_n if ve_proj_n <= SMALL_THRESHOLD else 0

    int6_params_total = tok_emb_n + bigram_emb_n + bigram_proj_int6 + ve_emb_n + ve_proj_int6
    # Scales for int6
    int6_scale_params = (VOCAB_SIZE + BIGRAM_VOCAB
                         + (BIGRAM_DIM if bigram_proj_int6 else 0)
                         + VOCAB_SIZE
                         + (VE_DIM if ve_proj_int6 else 0))

    # Small float16
    small_f16_params = (1  # bigram.scale
                        + bigram_proj_f16 + ve_proj_f16
                        + attn_small_params + attn_gate_params)

    # Control tensors (float32)
    # per block: attn_scale(D) + mlp_scale(D) + resid_mix(2D) = 4D
    ctrl_block = 4 * D * NUM_LAYERS
    num_skip = min(NUM_LAYERS // 2, NUM_LAYERS - NUM_LAYERS // 2)  # 5
    ctrl_global = (num_skip * D    # skip_weights
                   + D             # smear.gate
                   + 1             # ve_shared.scale
                   + VE_LAYERS     # ve_layer_scales
                   + 8 * NUM_LAYERS)  # q_gain
    ctrl_total = ctrl_block + ctrl_global

    total_params = (mlp_params_total + attn_int4_params + int6_params_total
                    + small_f16_params + ctrl_total
                    + mlp_scale_params + attn_scale_params + int6_scale_params)

    # Size estimates
    sz_ternary = mlp_params_total  * BPP_TERNARY
    sz_tscale  = mlp_scale_params  * BPP_FLOAT16
    sz_int4    = attn_int4_params  * BPP_INT4
    sz_ascale  = attn_scale_params * BPP_FLOAT16
    sz_int6    = int6_params_total * BPP_INT6
    sz_escale  = int6_scale_params * BPP_FLOAT16
    sz_f16     = small_f16_params  * BPP_FLOAT16
    sz_ctrl    = ctrl_total        * BPP_FLOAT32

    model_compressed = (sz_ternary + sz_tscale + sz_int4 + sz_ascale
                        + sz_int6 + sz_escale + sz_f16 + sz_ctrl
                        + OVERHEAD_BYTES)

    code_bytes  = CODE_BYTES_MOONSHOT if script == "moonshot" else CODE_BYTES_SAFE
    total_bytes = int(model_compressed) + code_bytes
    headroom    = LIMIT - total_bytes

    return {
        "model_dim": D, "mlp_mult": M,
        "total_params": total_params,
        "mlp_params": mlp_params_total,
        "attn_int4_params": attn_int4_params,
        "int6_params": int6_params_total,
        "model_compressed_bytes": int(model_compressed),
        "code_bytes": code_bytes,
        "total_bytes": total_bytes,
        "headroom_bytes": headroom,
        "fits_16mb": total_bytes < LIMIT,
        "pct_used": round(100.0 * total_bytes / LIMIT, 1),
        "breakdown_MB": {
            "ternary": round(sz_ternary / 1e6, 2),
            "int4":    round(sz_int4    / 1e6, 2),
            "int6":    round(sz_int6    / 1e6, 2),
            "scales":  round((sz_tscale + sz_ascale + sz_escale + sz_f16) / 1e6, 2),
            "ctrl":    round(sz_ctrl    / 1e6, 2),
        },
    }


SAFE_SPACE = {
    "hedge_eta":          [0.05, 0.1, 0.2, 0.3],
    "ngram_delta":        [0.001, 0.01, 0.05, 0.1],
    "ngram_alpha_center": [0.5, 0.6, 0.7, 0.8],
    "ngram_max_order":    [8, 10, 11],
    "swa_every":          [30, 50, 80],
    "ema_decay":          [0.995, 0.997, 0.999],
}
MOONSHOT_SPACE = {
    **SAFE_SPACE,
    "model_dim": [576, 608, 640],
    "mlp_mult":  [2.5, 3.0, 3.5],
}


def sample_config(space, seed):
    rng = random.Random(seed)
    return {k: rng.choice(v) for k, v in space.items()}


def main():
    HERE = Path(__file__).parent
    log_path = HERE / "experiments.jsonl"

    print("=" * 72)
    print("CONFIG VALIDATION — artifact size estimates (no GPU needed)")
    print("=" * 72)
    print()

    # 1. Unique (D, M) combos
    print("Unique (model_dim, mlp_mult) combos — moonshot adaptive quant:")
    print(f"  {'D':>4}  {'M':>5}  {'Params':>10}  {'Model MB':>9}  {'Total B':>11}  {'Headroom':>10}  {'%Used':>6}  Fits?")
    print("  " + "-" * 68)
    arch_results = {}
    for D in [576, 608, 640]:
        for M in [2.5, 3.0, 3.5]:
            r = estimate_artifact_size(D, M, "moonshot")
            arch_results[(D, M)] = r
            hkb = r["headroom_bytes"] / 1024
            tag = "YES" if r["fits_16mb"] else "NO ❌"
            print(f"  {D:>4}  {M:>5.1f}  {r['total_params']:>10,}  "
                  f"{r['model_compressed_bytes']/1e6:>8.2f}M  "
                  f"{r['total_bytes']:>11,}  "
                  f"{hkb:>+9.0f}K  "
                  f"{r['pct_used']:>5.1f}%  {tag}")
    print()

    # 2. Breakdown for D=640
    print("Size breakdown for D=640 configs:")
    for M in [2.5, 3.0, 3.5]:
        r = arch_results[(640, M)]
        b = r["breakdown_MB"]
        fits = "FITS" if r["fits_16mb"] else "TOO BIG"
        print(f"  D=640 M={M}: ternary={b['ternary']}MB  int4={b['int4']}MB  "
              f"int6={b['int6']}MB  scales={b['scales']}MB  ctrl={b['ctrl']}MB  [{fits}]")
    print()

    # 3. All 20 seeded moonshot configs
    print("All 20 seeded moonshot configs (--script train_gpt_v10_moonshot.py):")
    print(f"  {'Seed':>4}  {'D':>4}  {'M':>5}  {'Total B':>11}  {'%Used':>6}  {'Fits':>4}  Hyperparams")
    print("  " + "-" * 72)
    all_records = []
    for i in range(20):
        seed = 42 + i
        cfg  = sample_config(MOONSHOT_SPACE, seed)
        D, M = cfg["model_dim"], cfg["mlp_mult"]
        r    = arch_results[(D, M)]
        tag  = "YES" if r["fits_16mb"] else "NO"
        hstr = (f"n{cfg['ngram_max_order']} delta={cfg['ngram_delta']} "
                f"alpha={cfg['ngram_alpha_center']} eta={cfg['hedge_eta']} "
                f"ema={cfg['ema_decay']} swa={cfg['swa_every']}")
        print(f"  {seed:>4}  {D:>4}  {M:>5.1f}  {r['total_bytes']:>11,}  "
              f"{r['pct_used']:>5.1f}%  {tag:>4}  {hstr}")
        all_records.append({
            "seed": seed, "config": cfg,
            "estimate": {
                "total_params":    r["total_params"],
                "total_bytes":     r["total_bytes"],
                "fits_16mb":       r["fits_16mb"],
                "headroom_bytes":  r["headroom_bytes"],
                "pct_used":        r["pct_used"],
                "breakdown_MB":    r["breakdown_MB"],
            },
            "status": "dry_run_size_ok" if r["fits_16mb"] else "dry_run_too_large",
        })
    print()

    # 4. Safe script reference
    r_safe = estimate_artifact_size(512, 3.0, "safe")
    print("v10_safe reference (D=512, M=3.0, int6 quant for all large tensors):")
    print(f"  estimated total: {r_safe['total_bytes']:,} bytes  "
          f"headroom: {r_safe['headroom_bytes']/1024:+.0f}K  "
          f"{'FITS' if r_safe['fits_16mb'] else 'TOO BIG'}")
    print()

    # 5. Summary
    fits   = [r for r in all_records if r["estimate"]["fits_16mb"]]
    nobig  = [r for r in all_records if not r["estimate"]["fits_16mb"]]
    print("Summary:")
    print(f"  Configs fitting <16MB : {len(fits)}/20")
    print(f"  Configs too large     : {len(nobig)}/20")
    if nobig:
        print(f"  Too-large seeds       : {[r['seed'] for r in nobig]}")
    print()
    print("Top 5 by theoretical capacity (most params within 16MB):")
    for r in sorted(fits, key=lambda x: x["estimate"]["total_params"], reverse=True)[:5]:
        cfg = r["config"]
        e   = r["estimate"]
        print(f"  seed={r['seed']}  D={cfg['model_dim']}  M={cfg['mlp_mult']}  "
              f"params={e['total_params']:,}  bytes={e['total_bytes']:,}  ({e['pct_used']}%)")
    print()
    print("Max D per mlp_mult within budget:")
    for M in [2.5, 3.0, 3.5]:
        for D in [640, 608, 576]:
            r = arch_results.get((D, M))
            if r and r["fits_16mb"]:
                print(f"  M={M}: max D={D}  ({r['pct_used']}% used, "
                      f"{r['headroom_bytes']/1024:.0f}K headroom, "
                      f"{r['total_params']:,} params)")
                break
        else:
            print(f"  M={M}: no D in [576,608,640] fits")

    # Write experiments.jsonl
    with open(log_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    print()
    print(f"Wrote {len(all_records)} config estimates -> {log_path}")


if __name__ == "__main__":
    main()
