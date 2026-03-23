"""Analyze correlation between Gated Attention gate values and per-head quantization damage.

Hypothesis: if GA gates learn to dampen heads with high quant error, gate values
should negatively correlate with per-head quantization MSE.
"""

import torch
import sys

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pod_runs/no_ttt_run3/final_model_no_ttt_20260323_142955.pt"

print(f"Loading checkpoint: {CHECKPOINT}")
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

num_layers = max(int(k.split(".")[1]) for k in state if k.startswith("blocks.")) + 1
num_heads = 8
num_kv_heads = 4
head_dim = 64
print(f"Layers: {num_layers}, Heads: {num_heads}, KV Heads: {num_kv_heads}, Head dim: {head_dim}")


def quantize_int6(t):
    t32 = t.float()
    if t32.ndim != 2:
        return t32, t32, torch.tensor(0.0)
    row_max = t32.abs().amax(dim=1).clamp_min(1e-8)
    scale = row_max / 31.0
    q = torch.clamp(torch.round(t32 / scale[:, None]), -32, 31)
    deq = q * scale[:, None]
    mse = (t32 - deq).pow(2).mean()
    return t32, deq, mse


def corrcoef(a, b):
    a = a.float()
    b = b.float()
    a_m = a - a.mean()
    b_m = b - b.mean()
    num = (a_m * b_m).sum()
    den = (a_m.pow(2).sum() * b_m.pow(2).sum()).sqrt()
    return (num / den.clamp_min(1e-12)).item()


def fmt_tensor(t, precision=4):
    return "[" + ", ".join(f"{v:.{precision}f}" for v in t.tolist()) + "]"


# ─── PART 1: Per-head quantization damage vs Gated Attention gate values ───

print("\n" + "=" * 80)
print("PART 1: Per-head quantization damage vs Gated Attention gate values")
print("=" * 80)

gate_values = {}
head_quant_mse = {}

for i in range(num_layers):
    gate_bias_key = f"blocks.{i}.attn.attn_gate.bias"
    if gate_bias_key in state:
        bias = state[gate_bias_key].float()
        gate_values[i] = torch.sigmoid(bias)

    for proj_name in ["c_q", "c_k", "c_v", "proj"]:
        key = f"blocks.{i}.attn.{proj_name}.weight"
        if key not in state:
            continue
        w = state[key].float()
        _, deq, _ = quantize_int6(w)
        err = (w - deq).pow(2)

        n_heads = num_heads if proj_name in ("c_q", "proj") else num_kv_heads

        if proj_name == "proj":
            per_head = err.reshape(w.shape[0], n_heads, head_dim).mean(dim=(0, 2))
        else:
            per_head = err.reshape(n_heads, head_dim, w.shape[1]).mean(dim=(1, 2))

        if i not in head_quant_mse:
            head_quant_mse[i] = {}
        head_quant_mse[i][proj_name] = per_head

for i in range(num_layers):
    if i not in gate_values or i not in head_quant_mse:
        continue

    gate = gate_values[i]
    print(f"\nLayer {i}:")
    print(f"  Gate openings (sigmoid(bias)): {fmt_tensor(gate)}")

    for proj_name in ["c_q", "c_k", "c_v", "proj"]:
        if proj_name not in head_quant_mse[i]:
            continue
        mse = head_quant_mse[i][proj_name]

        if len(mse) == num_kv_heads and len(gate) == num_heads:
            group = num_heads // num_kv_heads
            mse_expanded = mse.repeat_interleave(group)
            corr = corrcoef(gate, mse_expanded)
        else:
            corr = corrcoef(gate, mse)
        print(f"  {proj_name:4s} per-head MSE: {fmt_tensor(mse, 6)} | corr(gate, mse) = {corr:+.4f}")


# ─── PART 2: Per-tensor quantization damage ranking ───

print("\n" + "=" * 80)
print("PART 2: Per-tensor quantization damage ranking")
print("=" * 80)

tensor_damage = []
for name, t in state.items():
    if t.ndim != 2 or t.numel() < 64:
        continue
    _, _, mse = quantize_int6(t)
    rel_mse = mse.item() / max(t.float().pow(2).mean().item(), 1e-12)
    tensor_damage.append((name, mse.item(), rel_mse, t.shape))

tensor_damage.sort(key=lambda x: x[2], reverse=True)
print("\nTop 20 most damaged tensors (by relative MSE):")
for name, mse, rel, shape in tensor_damage[:20]:
    print(f"  {rel:.6f} rel_mse | {mse:.8f} abs_mse | {str(shape):>15s} | {name}")

print("\nBottom 10 least damaged:")
for name, mse, rel, shape in tensor_damage[-10:]:
    print(f"  {rel:.6f} rel_mse | {mse:.8f} abs_mse | {str(shape):>15s} | {name}")


# ─── PART 3: fp32 control tensor analysis ───

print("\n" + "=" * 80)
print("PART 3: fp32 control tensor analysis (attn_scale, mlp_scale, resid_mix)")
print("=" * 80)

for i in range(num_layers):
    attn_s = state.get(f"blocks.{i}.attn_scale")
    mlp_s = state.get(f"blocks.{i}.mlp_scale")
    resid = state.get(f"blocks.{i}.resid_mix")

    parts = []
    if attn_s is not None:
        a = attn_s.float()
        parts.append(f"attn_scale: mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}")
    if mlp_s is not None:
        m = mlp_s.float()
        parts.append(f"mlp_scale: mean={m.mean():.4f} std={m.std():.4f} min={m.min():.4f} max={m.max():.4f}")
    if resid is not None:
        r = resid.float()
        parts.append(f"resid_mix: [{r[0].mean():.4f}, {r[1].mean():.4f}]")

    print(f"Layer {i:2d}: {' | '.join(parts)}")


# ─── PART 4: Value Residual lambda analysis ───

print("\n" + "=" * 80)
print("PART 4: Value Residual lambda analysis")
print("=" * 80)

for i in range(num_layers):
    vr_key = f"blocks.{i}.attn.vr_lambda"
    if vr_key in state:
        lam = state[vr_key].float()
        print(f"Layer {i:2d}: vr_lambda = [{lam[0]:.4f}, {lam[1]:.4f}] (v0={lam[0]:.4f}, v_cur={lam[1]:.4f})")


# ─── PART 5: Gate statistics across layers ───

print("\n" + "=" * 80)
print("PART 5: Gated Attention gate statistics across layers")
print("=" * 80)

if gate_values:
    print(f"\n{'Layer':>5s} | {'Mean':>6s} {'Std':>6s} {'Min':>6s} {'Max':>6s} | Gate openings per head")
    for i in sorted(gate_values.keys()):
        g = gate_values[i]
        print(f"  {i:3d}  | {g.mean():6.4f} {g.std():6.4f} {g.min():6.4f} {g.max():6.4f} | {fmt_tensor(g)}")

    means = torch.tensor([gate_values[i].mean().item() for i in sorted(gate_values.keys())])
    depths = torch.arange(len(means)).float()
    corr = corrcoef(depths, means)
    print(f"\nCorrelation(depth, mean_gate_opening) = {corr:+.4f}")
    print("Negative = deeper layers close gates more (dampen attention)")


# ─── PART 6: Cross-analysis: per-layer total quant damage vs gate dampening ───

print("\n" + "=" * 80)
print("PART 6: Per-layer total quant damage vs mean gate opening")
print("=" * 80)

layer_total_damage = []
layer_gate_mean = []
for i in range(num_layers):
    total_damage = 0.0
    count = 0
    for name, t in state.items():
        if f"blocks.{i}." in name and t.ndim == 2 and t.numel() >= 64:
            _, _, mse = quantize_int6(t)
            total_damage += mse.item()
            count += 1
    if count > 0:
        layer_total_damage.append(total_damage / count)
    else:
        layer_total_damage.append(0.0)
    if i in gate_values:
        layer_gate_mean.append(gate_values[i].mean().item())
    else:
        layer_gate_mean.append(0.0)

if gate_values:
    valid = [i for i in range(num_layers) if i in gate_values]
    d = torch.tensor([layer_total_damage[i] for i in valid])
    g = torch.tensor([layer_gate_mean[i] for i in valid])
    corr = corrcoef(d, g)
    print(f"\n{'Layer':>5s} | {'Quant Damage':>12s} | {'Gate Mean':>9s}")
    for i in range(num_layers):
        marker = " <--" if i in gate_values else ""
        print(f"  {i:3d}  | {layer_total_damage[i]:12.8f} | {layer_gate_mean[i]:9.4f}{marker}")
    print(f"\nCorrelation(layer_quant_damage, mean_gate_opening) = {corr:+.4f}")
    print("Negative = layers with higher quant damage have lower gate openings (GA compensating)")
