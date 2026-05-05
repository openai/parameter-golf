#!/usr/bin/env python3
"""
Roofline model calculations for Parameter Golf on H100.
Computes arithmetic intensity and bounds for every operation.
"""

# ===== H100 SXM Specs =====
H100_BF16_TFLOPS = 1979.0   # Peak BF16 tensor core TFLOPS
H100_FP8_TFLOPS = 3958.0    # Peak FP8 tensor core TFLOPS
H100_HBM_BW_TBS = 3.35      # HBM3 bandwidth in TB/s
H100_L2_SIZE_MB = 50         # L2 cache size

# Roofline crossover points (FLOP/byte)
BF16_CROSSOVER = H100_BF16_TFLOPS / H100_HBM_BW_TBS  # 590.7
FP8_CROSSOVER = H100_FP8_TFLOPS / H100_HBM_BW_TBS    # 1181.5

# ===== Model Config =====
DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS  # 64
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 256
MLP_HIDDEN = 2 * DIM  # 1024
VOCAB = 1024
SEQ_LEN = 1024
NUM_LAYERS = 9

# ===== Batch Config (8 GPUs) =====
TOTAL_TOKENS = 524288
NUM_GPUS = 8
GRAD_ACCUM = 1
TOKENS_PER_GPU = TOTAL_TOKENS // (NUM_GPUS * GRAD_ACCUM)  # 65536
BATCH_SIZE = TOKENS_PER_GPU // SEQ_LEN  # 64

print("=" * 80)
print("ROOFLINE ANALYSIS: Parameter Golf on H100")
print("=" * 80)
print(f"\nModel: {NUM_LAYERS} layers, dim={DIM}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}")
print(f"MLP hidden: {MLP_HIDDEN}, vocab: {VOCAB}, seq_len: {SEQ_LEN}")
print(f"Per-GPU batch: {BATCH_SIZE} seqs × {SEQ_LEN} = {TOKENS_PER_GPU} tokens")
print(f"\nH100 BF16 peak: {H100_BF16_TFLOPS} TFLOPS")
print(f"H100 FP8 peak: {H100_FP8_TFLOPS} TFLOPS")
print(f"H100 HBM BW: {H100_HBM_BW_TBS} TB/s")
print(f"BF16 roofline crossover: {BF16_CROSSOVER:.1f} FLOP/byte")
print(f"FP8 roofline crossover: {FP8_CROSSOVER:.1f} FLOP/byte")

def gemm_analysis(name, M, K, N, bytes_per_elem=2):
    """Analyze a GEMM Y[M,N] = X[M,K] @ W[K,N]"""
    flops = 2 * M * K * N
    bytes_read = (M * K + K * N) * bytes_per_elem
    bytes_write = M * N * bytes_per_elem
    total_bytes = bytes_read + bytes_write
    oi = flops / total_bytes
    
    # Time at peak compute
    compute_time_ms = flops / (H100_BF16_TFLOPS * 1e12) * 1e3
    # Time at peak bandwidth
    bw_time_ms = total_bytes / (H100_HBM_BW_TBS * 1e12) * 1e3
    
    bound = "MEMORY" if oi < BF16_CROSSOVER else "COMPUTE"
    actual_time_ms = max(compute_time_ms, bw_time_ms)
    utilization = compute_time_ms / actual_time_ms * 100
    
    return {
        'name': name, 'M': M, 'K': K, 'N': N,
        'flops_G': flops / 1e9,
        'bytes_MB': total_bytes / 1e6,
        'OI': oi,
        'bound': bound,
        'compute_ms': compute_time_ms,
        'bw_ms': bw_time_ms,
        'actual_ms': actual_time_ms,
        'utilization': utilization,
    }

M = TOKENS_PER_GPU  # 65536

print("\n" + "=" * 80)
print("PER-LAYER GEMM ANALYSIS (BF16)")
print("=" * 80)
print(f"{'Operation':<25} {'Shape':<25} {'FLOPs(G)':>10} {'Bytes(MB)':>10} {'OI':>8} {'Bound':>8} {'Time(ms)':>10} {'MFU%':>8}")
print("-" * 110)

ops = [
    gemm_analysis("Q projection",  M, DIM, DIM),
    gemm_analysis("K projection",  M, DIM, KV_DIM),
    gemm_analysis("V projection",  M, DIM, KV_DIM),
    gemm_analysis("Out projection", M, DIM, DIM),
    gemm_analysis("MLP fc",        M, DIM, MLP_HIDDEN),
    gemm_analysis("MLP proj",      M, MLP_HIDDEN, DIM),
]

total_flops_G = 0
total_time_ms = 0
for op in ops:
    shape_str = f"[{op['M']},{op['K']}]×[{op['K']},{op['N']}]"
    print(f"{op['name']:<25} {shape_str:<25} {op['flops_G']:>10.1f} {op['bytes_MB']:>10.1f} {op['OI']:>8.0f} {op['bound']:>8} {op['actual_ms']:>10.4f} {op['utilization']:>7.1f}%")
    total_flops_G += op['flops_G']
    total_time_ms += op['actual_ms']

print("-" * 110)
print(f"{'Per-layer matmul total':<25} {'':25} {total_flops_G:>10.1f} {'':>10} {'':>8} {'':>8} {total_time_ms:>10.4f}")

# All 9 layers
layer_time = total_time_ms
all_layers_time = layer_time * NUM_LAYERS

# LM head
lm = gemm_analysis("LM head (tied)", M, DIM, VOCAB)
print(f"\n{'LM head (tied)':<25} {'['+str(M)+','+str(DIM)+']×['+str(DIM)+','+str(VOCAB)+']':<25} {lm['flops_G']:>10.1f} {lm['bytes_MB']:>10.1f} {lm['OI']:>8.0f} {lm['bound']:>8} {lm['actual_ms']:>10.4f} {lm['utilization']:>7.1f}%")

print("\n" + "=" * 80)
print("FULL MODEL COMPUTE BUDGET")
print("=" * 80)

# Attention FLOPs (approximate, FlashAttention changes memory pattern)
attn_flops_per_layer = 2 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM  # QK^T
attn_flops_per_layer += 2 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM  # AV

total_forward_flops = (total_flops_G * NUM_LAYERS + lm['flops_G']) * 1e9 + attn_flops_per_layer * NUM_LAYERS
fwd_bwd_flops = total_forward_flops * 3  # ~3x for forward + backward

print(f"Forward matmul FLOPs:     {total_flops_G * NUM_LAYERS + lm['flops_G']:.1f} GFLOPs")
print(f"Forward attention FLOPs:  {attn_flops_per_layer * NUM_LAYERS / 1e9:.1f} GFLOPs")
print(f"Total forward FLOPs:      {total_forward_flops / 1e9:.1f} GFLOPs = {total_forward_flops / 1e12:.3f} TFLOPs")
print(f"Fwd+Bwd+Opt (3x):        {fwd_bwd_flops / 1e12:.3f} TFLOPs")
print(f"\nAt H100 peak ({H100_BF16_TFLOPS} TFLOPS): {fwd_bwd_flops / (H100_BF16_TFLOPS * 1e12) * 1e3:.2f} ms")
print(f"Actual step time:         43.5 ms")
print(f"Model FLOPs Utilization:  {fwd_bwd_flops / (H100_BF16_TFLOPS * 1e12) / (43.5e-3) * 100:.1f}%")

# ===== FP8 Analysis =====
print("\n" + "=" * 80)
print("FP8 IMPACT ANALYSIS")
print("=" * 80)

for op_bf16 in ops:
    op_fp8 = gemm_analysis(op_bf16['name'] + " (FP8)", op_bf16['M'], op_bf16['K'], op_bf16['N'], bytes_per_elem=1)
    # FP8: reads at 1 byte/elem, but output is BF16 (2 bytes/elem)
    fp8_read_bytes = (op_bf16['M'] * op_bf16['K'] + op_bf16['K'] * op_bf16['N']) * 1  # FP8 input
    fp8_write_bytes = op_bf16['M'] * op_bf16['N'] * 2  # BF16 output
    fp8_total_bytes = fp8_read_bytes + fp8_write_bytes
    fp8_oi = op_bf16['flops_G'] * 1e9 / fp8_total_bytes
    fp8_bw_time = fp8_total_bytes / (H100_HBM_BW_TBS * 1e12) * 1e3
    fp8_compute_time = op_bf16['flops_G'] * 1e9 / (H100_FP8_TFLOPS * 1e12) * 1e3
    fp8_actual = max(fp8_bw_time, fp8_compute_time)
    speedup = op_bf16['actual_ms'] / fp8_actual
    
    bound = "MEMORY" if fp8_oi < FP8_CROSSOVER else "COMPUTE"
    print(f"{op_bf16['name']:<25} BF16: {op_bf16['actual_ms']:.4f}ms → FP8: {fp8_actual:.4f}ms  Speedup: {speedup:.2f}×  OI: {fp8_oi:.0f}  {bound}")

# ===== Wider Model Analysis =====
print("\n" + "=" * 80)
print("WIDER MODEL ARITHMETIC INTENSITY")
print("=" * 80)
print(f"{'Dim':<8} {'MLP OI':>10} {'Q/Out OI':>10} {'K/V OI':>10} {'MLP bound?':>12} {'Q bound?':>10}")
for d in [512, 576, 640, 672, 704, 768, 896, 1024, 1536, 2048]:
    kv_d = d // 2  # Assuming similar GQA ratio
    mlp_h = 2 * d
    mlp_oi = 2 * d * mlp_h / ((d + mlp_h) * 2)
    q_oi = 2 * d * d / ((d + d) * 2)
    kv_oi = 2 * d * kv_d / ((d + kv_d) * 2)
    mlp_bound = "COMPUTE" if mlp_oi >= BF16_CROSSOVER else "memory"
    q_bound = "COMPUTE" if q_oi >= BF16_CROSSOVER else "memory"
    print(f"{d:<8} {mlp_oi:>10.0f} {q_oi:>10.0f} {kv_oi:>10.0f} {mlp_bound:>12} {q_bound:>10}")

# ===== DDP Communication =====
print("\n" + "=" * 80)
print("DDP COMMUNICATION ANALYSIS")
print("=" * 80)
n_params = 17_000_000
grad_bytes = n_params * 4  # fp32 gradients
muon_bytes = 16_000_000 * 2  # bf16 Muon updates

nvlink_bw = 900  # GB/s bidirectional per GPU pair
pcie_bw = 128    # GB/s

# Ring allreduce: 2*(N-1)/N * data
ring_factor = 2 * (NUM_GPUS - 1) / NUM_GPUS

nvlink_time = ring_factor * grad_bytes / (nvlink_bw * 1e9) * 1e3
pcie_time = ring_factor * grad_bytes / (pcie_bw * 1e9) * 1e3

muon_nvlink = ring_factor * muon_bytes / (nvlink_bw * 1e9) * 1e3
muon_pcie = ring_factor * muon_bytes / (pcie_bw * 1e9) * 1e3

print(f"DDP gradient allreduce: {grad_bytes/1e6:.1f} MB")
print(f"  NVLink mesh: {nvlink_time:.2f} ms")
print(f"  PCIe:        {pcie_time:.2f} ms")
print(f"Muon allreduce:         {muon_bytes/1e6:.1f} MB")
print(f"  NVLink mesh: {muon_nvlink:.2f} ms")
print(f"  PCIe:        {muon_pcie:.2f} ms")
print(f"Total comm (NVLink):    {nvlink_time + muon_nvlink:.2f} ms")
print(f"Total comm (PCIe):      {pcie_time + muon_pcie:.2f} ms")

print("\n" + "=" * 80)
print("STEP TIME BUDGET BREAKDOWN")
print("=" * 80)
matmul_fwd_time = (total_time_ms * NUM_LAYERS + lm['actual_ms'])
attn_time = attn_flops_per_layer * NUM_LAYERS / (H100_BF16_TFLOPS * 1e12) * 1e3  # FA is optimized
fwd_time = matmul_fwd_time + attn_time
bwd_time = fwd_time * 2  # backward is ~2x forward
comm_time = nvlink_time + muon_nvlink  # assuming NVLink
optimizer_time = 3.0  # estimated for Muon NS + Adam
overhead_time = 43.5 - fwd_time - bwd_time - comm_time - optimizer_time

print(f"Forward matmuls:  {matmul_fwd_time:.2f} ms")
print(f"Forward attention:{attn_time:.2f} ms (FlashAttention)")
print(f"Backward (2x fwd):{bwd_time:.2f} ms")
print(f"Communication:    {comm_time:.2f} ms")
print(f"Optimizer:        {optimizer_time:.2f} ms (estimated)")
print(f"Other overhead:   {overhead_time:.2f} ms (kernel launch, elem-wise, sync)")
print(f"Total:            43.50 ms")
