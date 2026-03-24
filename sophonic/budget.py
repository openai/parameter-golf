"""
Byte budget calculator for Sophonic Quantization.

Helps determine feasible rank and k given the 16MB cap.
"""
from dataclasses import dataclass
from typing import List


BUDGET_CAP = 16_000_000  # 16 decimal MB


@dataclass
class LayerSpec:
    name: str
    rows: int
    cols: int


def default_11L_512d() -> List[LayerSpec]:
    """Weight matrices for the 11-layer 512-dim SOTA architecture."""
    layers = []
    for i in range(11):
        # Attention: Q, K, V, O projections (with GQA: K,V are smaller)
        layers.append(LayerSpec(f"layer{i}.attn.q", 512, 512))
        layers.append(LayerSpec(f"layer{i}.attn.k", 512, 256))  # 4 KV heads
        layers.append(LayerSpec(f"layer{i}.attn.v", 512, 256))
        layers.append(LayerSpec(f"layer{i}.attn.o", 512, 512))
        # MLP: up + down (3x expansion)
        layers.append(LayerSpec(f"layer{i}.mlp.up", 512, 1536))
        layers.append(LayerSpec(f"layer{i}.mlp.down", 1536, 512))
    # Embedding (tied, int8)
    layers.append(LayerSpec("embedding", 1024, 512))
    return layers


def estimate_budget(layers: List[LayerSpec], base_bits: int = 5,
                     rank: int = 4, residual_dtype_bytes: int = 2,
                     code_bytes: int = 50_000, zstd_ratio: float = 0.85):
    """
    Estimate total artifact size.
    
    Args:
        base_bits: quantization bits for base weights
        rank: rank of low-rank residuals
        residual_dtype_bytes: bytes per residual value (2 for fp16)
        code_bytes: estimated code size
        zstd_ratio: compression ratio (compressed/uncompressed)
    """
    base_bytes = 0
    residual_bytes = 0

    for layer in layers:
        n_weights = layer.rows * layer.cols
        # Base weights
        if "embedding" in layer.name:
            base_bytes += n_weights  # int8 = 1 byte
        else:
            base_bytes += int(n_weights * base_bits / 8)
            # Residual (low-rank): U is (rows, rank), V is (cols, rank)
            residual_bytes += (layer.rows + layer.cols) * rank * residual_dtype_bytes

    # Router
    hidden_dim = 512
    num_layers = 11
    router_bytes = hidden_dim * num_layers + num_layers * 4  # int8 weights + fp32 bias

    raw_total = base_bytes + residual_bytes + router_bytes + code_bytes
    compressed = int(raw_total * zstd_ratio)

    return {
        "base_weights_bytes": base_bytes,
        "residual_bytes": residual_bytes,
        "router_bytes": router_bytes,
        "code_bytes": code_bytes,
        "raw_total": raw_total,
        "estimated_compressed": compressed,
        "budget_remaining": BUDGET_CAP - compressed,
        "fits": compressed <= BUDGET_CAP,
    }


if __name__ == "__main__":
    layers = default_11L_512d()
    for rank in [1, 2, 4, 8]:
        for base_bits in [5, 6]:
            result = estimate_budget(layers, base_bits=base_bits, rank=rank)
            status = "FIT" if result["fits"] else "❌"
            print(f"{status} base=int{base_bits} rank={rank}: "
                  f"compressed={result['estimated_compressed']/1e6:.2f}MB "
                  f"(base={result['base_weights_bytes']/1e6:.2f}MB "
                  f"+ residuals={result['residual_bytes']/1e6:.2f}MB) "
                  f"remaining={result['budget_remaining']/1e6:.2f}MB")
