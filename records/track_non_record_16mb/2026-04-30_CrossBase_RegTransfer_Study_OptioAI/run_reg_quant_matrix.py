"""
Real-data reg × quant matrix analysis.

Runs after all 5 EmbStudy_* training cells complete. For each saved BF16 model:
  1. Load via PyTorch
  2. Run forward on a small val batch to capture last-block hidden states
  3. Apply 6 quantization schemes to those hidden states
  4. Compute per-cell metrics (L2 distortion, isoscore, silhouette, effective rank)
  5. Identify (reg, quant) pairs that "play nice"
"""

import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

# --- 6 quantization schemes ---
def quant_int_per_tensor(h, bits, asym=False):
    qmax = (1 << (bits-1)) - 1
    if asym:
        h_min, h_max = h.min().item(), h.max().item()
        scale = (h_max - h_min) / (2*qmax + 1)
        zp = round(-h_min / max(scale, 1e-8) - qmax - 1)
        h_q = (torch.round(h/scale) + zp).clamp(-qmax-1, qmax)
        return (h_q - zp) * scale
    scale = h.abs().max().clamp(min=1e-8) / qmax
    return torch.round(h/scale).clamp(-qmax-1, qmax) * scale

def quant_int_per_row(h, bits, asym=False):
    qmax = (1 << (bits-1)) - 1
    if asym:
        h_min = h.min(dim=-1, keepdim=True).values
        h_max = h.max(dim=-1, keepdim=True).values
        scale = (h_max - h_min).clamp(min=1e-8) / (2*qmax + 1)
        zp = (-h_min/scale - qmax - 1).round()
        h_q = (torch.round(h/scale) + zp).clamp(-qmax-1, qmax)
        return (h_q - zp) * scale
    scale = h.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8) / qmax
    return torch.round(h/scale).clamp(-qmax-1, qmax) * scale

def quant_awq_lite(h, bits=4):
    """Per-channel scaling (sqrt activation magnitude) before per-row int4."""
    chan_scale = (h.abs().mean(dim=0, keepdim=True).clamp(min=1e-8))**0.5
    return quant_int_per_row(h / chan_scale, bits) * chan_scale

def quant_gptq_lite(h, bits=4, damping=0.1):
    """Per-row scale, column-by-column with running residual (Hessian-free GPTQ proxy)."""
    qmax = (1 << (bits-1)) - 1
    n, d = h.shape
    h_q = torch.zeros_like(h)
    h_residual = h.clone()
    scale = h.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8) / qmax
    for col in range(d):
        col_q = torch.round(h_residual[:, col:col+1] / scale).clamp(-qmax-1, qmax) * scale
        h_q[:, col:col+1] = col_q
        if col+1 < d:
            err = h_residual[:, col:col+1] - col_q
            h_residual[:, col+1:] += err * damping
    return h_q

QUANT_SCHEMES = [
    ('int4 sym per-tensor', lambda h: quant_int_per_tensor(h, bits=4, asym=False)),
    ('int4 sym per-row',    lambda h: quant_int_per_row(h, bits=4, asym=False)),
    ('int4 asym per-row',   lambda h: quant_int_per_row(h, bits=4, asym=True)),
    ('int6 sym per-row',    lambda h: quant_int_per_row(h, bits=6, asym=False)),
    ('int8 sym per-row',    lambda h: quant_int_per_row(h, bits=8, asym=False)),
    ('AWQ-lite int4',       lambda h: quant_awq_lite(h, bits=4)),
    ('GPTQ-lite int4',      lambda h: quant_gptq_lite(h, bits=4)),
]

# --- Metrics ---
def isoscore(h):
    h_n = F.normalize(h, dim=-1, eps=1e-6)
    sim = h_n @ h_n.t()
    n = h_n.size(0)
    off = sim - torch.eye(n, device=h.device)
    return off.abs().mean().item()

def effective_rank(h):
    h_c = h - h.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(h_c, full_matrices=False)
    p = S / S.sum()
    p = p[p > 1e-10]
    return float(np.exp(-(p * p.log()).sum().item()))

def per_token_l2_distortion(h_pre, h_post):
    return (h_pre - h_post).pow(2).sum(dim=-1).sqrt().mean().item()

def cosine_shift(h_pre, h_post):
    return 1.0 - F.cosine_similarity(h_pre, h_post, dim=-1).mean().item()

def silhouette(h, labels, n_clusters):
    """Simplified silhouette (uses sample of pairwise distances)."""
    h_np = h.cpu().numpy() if isinstance(h, torch.Tensor) else h
    sil = 0.0
    for i in range(len(h_np)):
        same = (labels == labels[i]) & (np.arange(len(h_np)) != i)
        if not any(same): continue
        a = np.mean([np.linalg.norm(h_np[i] - x) for x in h_np[same]])
        b_min = float('inf')
        for c in range(n_clusters):
            if c == labels[i]: continue
            other = labels == c
            if any(other):
                b = np.mean([np.linalg.norm(h_np[i] - x) for x in h_np[other]])
                b_min = min(b_min, b)
        if max(a, b_min) > 0:
            sil += (b_min - a) / max(a, b_min)
    return sil / len(h_np)

# --- Hidden state extraction ---
def extract_hidden_states(model_dir, n_tokens=128, val_bin_path=None):
    """Load the trained BF16 model and run forward on val tokens.
    Captures the post-final-block hidden state for each token.
    """
    sys.path.insert(0, model_dir)
    # Avoid import collision
    if 'train_gpt' in sys.modules:
        del sys.modules['train_gpt']
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", os.path.join(model_dir, "train_gpt.py"))
    train_gpt = importlib.util.module_from_spec(spec)
    # Stub out distributed init; we just need the model class
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    spec.loader.exec_module(train_gpt)

    h_cls = train_gpt.Hyperparameters() if hasattr(train_gpt, 'Hyperparameters') else None
    # The model class might be 'GPT' or 'FinalMiniLM' depending on lineage
    model_cls = None
    for cls_name in ['GPT', 'FinalMiniLM', 'Model']:
        if hasattr(train_gpt, cls_name):
            model_cls = getattr(train_gpt, cls_name)
            break
    if model_cls is None:
        raise RuntimeError("no model class found in train_gpt.py")
    model = model_cls(h_cls)
    state_dict = torch.load(os.path.join(model_dir, "final_model.pt"), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).bfloat16()
    
    # Load some val tokens
    if val_bin_path and os.path.exists(val_bin_path):
        toks = np.fromfile(val_bin_path, dtype=np.uint16)[:n_tokens*8].astype(np.int64)
    else:
        # fallback: random tokens
        toks = np.random.randint(0, 8000, size=n_tokens*8)
    toks = torch.from_numpy(toks).reshape(8, n_tokens).to(device)
    
    # Forward through the model and extract last-block hidden state
    with torch.no_grad():
        # We need access to internal hidden states. Easiest: hook on the last block.
        captured = {}
        for name, mod in model.named_modules():
            if 'blocks' in name and isinstance(mod, torch.nn.Module):
                if name.endswith('.10') or name.endswith('.9'):  # last block
                    def make_hook(n):
                        def hook(m, inp, out):
                            captured[n] = out.detach().cpu().float()
                        return hook
                    mod.register_forward_hook(make_hook(name))
        # Run forward — need to call appropriate method
        try:
            _ = model.forward_logits(toks) if hasattr(model, 'forward_logits') else model(toks)
        except Exception as e:
            print(f"  forward error: {e}")
    
    # Get hidden states from the last captured layer
    if captured:
        h = list(captured.values())[-1]
        h = h.reshape(-1, h.size(-1))[:128]  # take 128 tokens
        return h
    return None

# --- Main matrix computation ---
def main():
    base_dirs = {
        'no-reg':   '/workspace/parameter-golf/candidate_pack/N18_baseA_nosimctg',  # SimCTG=0
        'SimCTG':   '/workspace/parameter-golf/candidate_pack/N18_baseA_baseline',  # SimCTG λ=0.3 only
        'SimCTG+QAHSP': '/workspace/parameter-golf/candidate_pack/N18_baseA_qahsp',
        'SimCTG+ES':    '/workspace/parameter-golf/candidate_pack/N18_baseA_es',
        'SimCTG+HSU':   '/workspace/parameter-golf/candidate_pack/N18_baseA_hsu',
        'SimCTG+AOS':   '/workspace/parameter-golf/candidate_pack/N18_baseA_aos',
    }
    
    # Find a val_bin
    val_bin = None
    for p in [
        '/workspace/parameter-golf/parameter-golf/data/datasets/datasets/fineweb10B_sp10240/fineweb_val_000000.bin',
        '/workspace/parameter-golf/parameter-golf/data/datasets/fineweb10B_sp10240/fineweb_val_000000.bin',
    ]:
        if os.path.exists(p):
            val_bin = p; break
    print(f"val_bin: {val_bin}")
    
    hidden_per_reg = {}
    for reg, dir_ in base_dirs.items():
        if not os.path.exists(os.path.join(dir_, "final_model.pt")):
            print(f"  {reg}: no final_model.pt yet — skipping")
            continue
        print(f"  {reg}: loading...")
        h = extract_hidden_states(dir_, n_tokens=128, val_bin_path=val_bin)
        if h is None:
            print(f"    extraction failed")
            continue
        hidden_per_reg[reg] = h
        print(f"    shape: {tuple(h.shape)}, mean L2: {h.pow(2).sum(-1).sqrt().mean().item():.3f}")
    
    if not hidden_per_reg:
        print("No models loaded. Exiting.")
        return
    
    # Compute the (reg × quant) matrix
    print()
    print("=== Real-data reg × quant matrix ===")
    n_tok = list(hidden_per_reg.values())[0].size(0)
    # Crude clustering: split tokens into 8 groups of equal size by their token ID
    labels = np.array([i // (n_tok // 8) for i in range(n_tok)])[:n_tok]
    n_clusters = 8
    
    results = {}
    for reg, h in hidden_per_reg.items():
        results[reg] = {}
        for qname, qfn in QUANT_SCHEMES:
            h_q = qfn(h)
            results[reg][qname] = {
                'l2_distortion': per_token_l2_distortion(h, h_q),
                'cos_shift':     cosine_shift(h, h_q),
                'isoscore_post': isoscore(h_q),
                'eff_rank_post': effective_rank(h_q),
            }
            print(f"  {reg:<10} {qname:<22} l2={results[reg][qname]['l2_distortion']:.4f} cos_shift={results[reg][qname]['cos_shift']:.4f}")
    
    # Save
    import json
    out_path = '/workspace/parameter-golf/submissions/C_CrossBase_RegTransfer_Study/real_reg_quant_matrix.json'
    open(out_path, 'w').write(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")

if __name__ == "__main__":
    main()
