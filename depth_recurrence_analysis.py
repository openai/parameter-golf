"""
Depth Recurrence Parameter Budget Analysis
============================================
Computes parameter counts and compressed model sizes for various
depth-recurrence configurations of the parameter-golf transformer.

Architecture: GQA transformer with tied embeddings, U-Net skip connections.
Compression: int8 quantization + zlib (level 9).
"""

def compute_config(
    label: str,
    num_unique_blocks: int,
    loops: int,
    model_dim: int,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    mlp_mult: int = 2,
    vocab_size: int = 1024,
    use_int6_middle: bool = False,
):
    """Compute parameter budget and estimated compressed size."""

    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    hidden = mlp_mult * model_dim
    effective_depth = num_unique_blocks * loops

    # -- Per-block parameter counts --
    c_q = model_dim * model_dim       # dim -> dim
    c_k = model_dim * kv_dim          # dim -> kv_dim
    c_v = model_dim * kv_dim          # dim -> kv_dim
    proj = model_dim * model_dim      # dim -> dim
    fc = model_dim * hidden           # dim -> hidden
    mlp_proj = hidden * model_dim     # hidden -> dim
    attn_scale = model_dim
    mlp_scale = model_dim
    resid_mix = 2 * model_dim
    q_gain = num_heads

    matrix_params_per_block = c_q + c_k + c_v + proj + fc + mlp_proj
    scalar_params_per_block = attn_scale + mlp_scale + resid_mix + q_gain
    total_params_per_block = matrix_params_per_block + scalar_params_per_block

    # -- Per-block storage bytes (int8 payload) --
    # Matrix weights: int8 (1 byte/param) + per-row fp16 scales
    # c_q rows: model_dim, c_k rows: model_dim, c_v rows: model_dim
    # proj rows: model_dim, fc rows: model_dim, mlp_proj rows: hidden
    scale_rows_per_block = 5 * model_dim + hidden  # c_q,c_k,c_v,proj,fc have model_dim rows; mlp_proj has hidden rows
    matrix_bytes = matrix_params_per_block * 1 + scale_rows_per_block * 2  # int8 + fp16 scales

    # Scalar params: stored as fp16 passthrough (numel <= 65536)
    scalar_bytes = scalar_params_per_block * 2  # fp16

    bytes_per_block = matrix_bytes + scalar_bytes

    # -- Non-block parameters --
    embed_params = vocab_size * model_dim
    # Embedding: int8 quantized (since numel > 65536 for all our configs)
    embed_bytes = embed_params * 1 + vocab_size * 2  # int8 + per-row scales (vocab_size rows)

    # Check if embedding should be fp16 passthrough instead
    if embed_params <= 65536:
        embed_bytes = embed_params * 2  # fp16

    # Skip weights: for the EFFECTIVE depth (not unique blocks), since U-Net is over actual layers
    # Actually for recurrence, the skip connections would need to work over the effective depth.
    # With recurrence, we need to reconsider. The skip weights are per-effective-layer, not per-unique-block.
    # But since they are small (dim-sized vectors), they are negligible AND would be unique per position.
    # For recurrence, skip_weights would need to be over effective_depth.
    num_encoder = effective_depth // 2
    num_decoder = effective_depth - num_encoder
    num_skip = min(num_encoder, num_decoder)
    skip_params = num_skip * model_dim
    skip_bytes = skip_params * 2  # fp16 passthrough (always small enough)

    # -- Totals --
    total_unique_params = (
        num_unique_blocks * total_params_per_block
        + embed_params
        + skip_params
    )

    total_payload_bytes = (
        num_unique_blocks * bytes_per_block
        + embed_bytes
        + skip_bytes
    )

    # Add ~0.2% for torch serialization overhead (dicts, metadata)
    torch_overhead = int(total_payload_bytes * 0.002)
    total_payload_bytes += torch_overhead

    # -- zlib compression estimates --
    # From SOTA data:
    #   Pure int8 (no int6): payload ~19.03MB -> zlib ~17.6MB, ratio = 0.925
    #   With int6 middle: payload ~19.03MB -> zlib ~15.88MB, ratio = 0.834
    # For a new model with all int8, use the pure ratio of ~0.925
    # Smaller models may compress slightly better (less entropy), but let's be conservative.

    if use_int6_middle:
        zlib_ratio = 0.834
    else:
        zlib_ratio = 0.925

    zlib_compressed_bytes = int(total_payload_bytes * zlib_ratio)

    # Code size (from SOTA: ~49KB)
    code_bytes = 49000
    total_submission_bytes = zlib_compressed_bytes + code_bytes

    # Headroom
    limit = 16_000_000
    headroom = limit - total_submission_bytes
    headroom_pct = headroom / limit * 100

    # Training speed (relative to 10-layer baseline)
    speed_ratio = effective_depth / 10.0  # 1.0 = same as baseline

    return {
        "label": label,
        "unique_blocks": num_unique_blocks,
        "loops": loops,
        "effective_depth": effective_depth,
        "model_dim": model_dim,
        "params_per_block": total_params_per_block,
        "total_unique_params": total_unique_params,
        "embed_params": embed_params,
        "skip_params": skip_params,
        "payload_bytes": total_payload_bytes,
        "zlib_bytes": zlib_compressed_bytes,
        "total_submission": total_submission_bytes,
        "headroom": headroom,
        "headroom_pct": headroom_pct,
        "speed_ratio": speed_ratio,
        "use_int6": use_int6_middle,
    }


def find_max_dim(num_unique_blocks, loops, target_bytes=16_000_000, code_bytes=49000):
    """Binary search for maximum model_dim that fits in target."""
    lo, hi = 64, 2048
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        # Ensure divisible by num_heads=8
        mid = (mid // 8) * 8
        if mid < 64:
            lo = mid + 8
            continue
        try:
            r = compute_config(
                f"search_{mid}", num_unique_blocks, loops, mid,
                num_heads=max(1, mid // 64),  # keep head_dim=64
                num_kv_heads=max(1, mid // 128),  # keep kv_heads = heads/2
            )
            if r["total_submission"] <= target_bytes:
                best = mid
                lo = mid + 8
            else:
                hi = mid - 8
        except:
            hi = mid - 8
    return best


def fmt_bytes(b):
    if abs(b) >= 1_000_000:
        return f"{b/1_000_000:.2f}MB"
    elif abs(b) >= 1_000:
        return f"{b/1_000:.1f}KB"
    return f"{b}B"


def fmt_params(p):
    if p >= 1_000_000:
        return f"{p/1_000_000:.2f}M"
    elif p >= 1_000:
        return f"{p/1_000:.1f}K"
    return str(p)


def main():
    configs = [
        # Baseline SOTA for reference
        ("BASELINE: 10B x 1L (SOTA)", 10, 1, 512, 8, 4, False),
        # Depth recurrence configs
        ("Config 1: 5B x 4L", 5, 4, 512, 8, 4, False),
        ("Config 2: 7B x 3L", 7, 3, 512, 8, 4, False),
        ("Config 3: 10B x 2L", 10, 2, 512, 8, 4, False),
        ("Config 4: 5B x 4L dim=640", 5, 4, 640, 10, 5, False),
        ("Config 5: 5B x 4L dim=576", 5, 4, 576, 9, 4, False),
    ]

    results = []
    for label, blocks, loops, dim, nh, nkv, int6 in configs:
        r = compute_config(label, blocks, loops, dim, nh, nkv)
        results.append(r)

    # Print table
    print("=" * 130)
    print("DEPTH RECURRENCE PARAMETER BUDGET ANALYSIS")
    print("=" * 130)
    print()

    header = f"{'Configuration':<30} {'Dim':>4} {'Unique':>6} {'Eff.':>5} {'Params':>10} {'Payload':>10} {'zlib':>10} {'Total':>10} {'Headroom':>10} {'Speed':>6}"
    print(header)
    print(f"{'':30} {'':>4} {'Blocks':>6} {'Depth':>5} {'':>10} {'(int8)':>10} {'comp.':>10} {'+code':>10} {'vs 16MB':>10} {'ratio':>6}")
    print("-" * 130)

    for r in results:
        line = (
            f"{r['label']:<30} "
            f"{r['model_dim']:>4} "
            f"{r['unique_blocks']:>6} "
            f"{r['effective_depth']:>5} "
            f"{fmt_params(r['total_unique_params']):>10} "
            f"{fmt_bytes(r['payload_bytes']):>10} "
            f"{fmt_bytes(r['zlib_bytes']):>10} "
            f"{fmt_bytes(r['total_submission']):>10} "
            f"{fmt_bytes(r['headroom']):>10} "
            f"{r['speed_ratio']:>5.1f}x"
        )
        print(line)

    print()
    print("=" * 130)
    print("DETAILED BREAKDOWN")
    print("=" * 130)

    for r in results:
        print(f"\n--- {r['label']} ---")
        print(f"  Model dim:          {r['model_dim']}")
        print(f"  Unique blocks:      {r['unique_blocks']}")
        print(f"  Loops:              {r['loops']}")
        print(f"  Effective depth:    {r['effective_depth']} layers")
        print(f"  Params/block:       {fmt_params(r['params_per_block'])}")
        print(f"  Block params:       {fmt_params(r['unique_blocks'] * r['params_per_block'])}")
        print(f"  Embed params:       {fmt_params(r['embed_params'])}")
        print(f"  Skip params:        {fmt_params(r['skip_params'])}")
        print(f"  Total unique params:{fmt_params(r['total_unique_params'])}")
        print(f"  int8 payload:       {fmt_bytes(r['payload_bytes'])}")
        print(f"  zlib compressed:    {fmt_bytes(r['zlib_bytes'])}")
        print(f"  + code (~49KB):     {fmt_bytes(r['total_submission'])}")
        print(f"  Headroom vs 16MB:   {fmt_bytes(r['headroom'])} ({r['headroom_pct']:.1f}%)")
        print(f"  Training speed:     {r['speed_ratio']:.1f}x vs baseline (eff. depth {r['effective_depth']} vs 10)")
        print(f"  Steps in 10min:     ~{int(13100 / r['speed_ratio'])} (baseline gets ~13,100)")

    # Maximum dim analysis
    print()
    print("=" * 130)
    print("MAXIMUM DIM ANALYSIS (fitting in 16MB with pure int8 + zlib)")
    print("=" * 130)

    for blocks, loops in [(5, 4), (7, 3), (10, 2), (3, 7), (4, 5)]:
        max_dim = find_max_dim(blocks, loops)
        nh = max(1, max_dim // 64)
        nkv = max(1, max_dim // 128)
        r = compute_config(f"{blocks}B x {loops}L max", blocks, loops, max_dim, nh, nkv)
        print(f"\n  {blocks} blocks x {loops} loops (eff. depth {blocks*loops}):")
        print(f"    Max dim = {max_dim} (heads={nh}, kv_heads={nkv})")
        print(f"    Params: {fmt_params(r['total_unique_params'])}")
        print(f"    Payload: {fmt_bytes(r['payload_bytes'])} -> zlib: {fmt_bytes(r['zlib_bytes'])} -> total: {fmt_bytes(r['total_submission'])}")
        print(f"    Headroom: {fmt_bytes(r['headroom'])}")
        print(f"    Training speed: {r['speed_ratio']:.1f}x slower per step ({int(13100/r['speed_ratio'])} steps in 10min)")

    # Also check with int6 middle layers for tighter fit
    print()
    print("=" * 130)
    print("KEY TRADE-OFF ANALYSIS")
    print("=" * 130)
    print()
    print("  The fundamental trade-off with depth recurrence:")
    print("  - FEWER unique params (smaller artifact, more headroom for wider dim)")
    print("  - MORE effective depth (slower training, fewer steps in 10min)")
    print("  - Shared weights may limit expressiveness per-layer")
    print()
    print("  Sweet spots to explore:")
    print("  1. 5B x 4L at dim=640+: 2x fewer params, 2x deeper, significantly wider")
    print("  2. 7B x 3L at dim=512: ~30% fewer params, 2.1x deeper, same width")
    print("  3. 10B x 2L at dim=512: same params as SOTA, 2x deeper, 2x slower")
    print()

    # Comparison: what dim can we reach with various configs?
    print("=" * 130)
    print("DIM SCALING TABLE (all pure int8, what fits in 16MB)")
    print("=" * 130)
    print()
    print(f"  {'Config':<20} {'Max Dim':>8} {'Eff Depth':>10} {'Params':>10} {'Steps/10min':>12} {'Params x Steps':>15}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")

    for blocks, loops in [(10, 1), (10, 2), (7, 3), (5, 4), (4, 5), (3, 7)]:
        max_dim = find_max_dim(blocks, loops)
        nh = max(1, max_dim // 64)
        nkv = max(1, max_dim // 128)
        r = compute_config(f"{blocks}B x {loops}L", blocks, loops, max_dim, nh, nkv)
        steps = int(13100 / r['speed_ratio'])
        # "Param x Steps" is a rough proxy for total learning capacity
        capacity = r['total_unique_params'] * steps
        print(f"  {blocks}B x {loops}L{'':<13} {max_dim:>8} {blocks*loops:>10} {fmt_params(r['total_unique_params']):>10} {steps:>12,} {capacity:>15,}")


if __name__ == "__main__":
    main()
