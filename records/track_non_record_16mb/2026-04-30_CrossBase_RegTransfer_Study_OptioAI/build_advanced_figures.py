"""
Build 3 additional reviewer-grade figures for Sub C:

  1. fig_svd_spectrum.png        — singular value spectrum per reg, per matrix family.
                                    Mechanistic grounding for GPTQ-friendliness:
                                    flatter spectrum = more L2 mass in tail dims =
                                    harder to per-row int6 quantize.

  2. fig_depth_trajectory.png    — per-layer hidden-state mean ‖h‖ and excess kurtosis
                                    through depth, one curve per reg. Shows where
                                    outliers emerge in the depth dimension and gives
                                    the AOS / HSU / QAHSP motivation real grounding.

  3. fig_cka_heatmap.png         — pairwise CKA (Kornblith 2019) between final-block
                                    hidden states of the 6 reg variants. Tests whether
                                    the regs produce *meaningfully* different
                                    representations or just superficial perturbations.

All work is done on CPU to avoid contending with running training on GPU.
"""

import os, sys, json, math, gc
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# Monkey-patch flash_attn_3 with SDPA so we can run forward on CPU without
# touching the GPU (GPU is being used by training runs).
try:
    import flash_attn_interface as _fai
except ImportError:
    import types as _types
    _fai = _types.ModuleType("flash_attn_interface")
    sys.modules["flash_attn_interface"] = _fai

def _sdpa_fallback(q, k, v, causal=True, **_):
    # q,k,v: (B, T, H, D)  → SDPA wants (B, H, T, D)
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    # GQA: expand K/V heads to match Q heads
    if k_.size(1) != q_.size(1):
        rep = q_.size(1) // k_.size(1)
        k_ = k_.repeat_interleave(rep, dim=1)
        v_ = v_.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal)
    return out.transpose(1, 2).contiguous()

_fai.flash_attn_func = _sdpa_fallback

REG_DIRS = {
    'no-reg':         '/workspace/parameter-golf/candidate_pack/N18_baseA_nosimctg',
    'SimCTG':         '/workspace/parameter-golf/candidate_pack/N18_baseA_baseline',
    'SimCTG+QAHSP':   '/workspace/parameter-golf/candidate_pack/N18_baseA_qahsp',
    'SimCTG+ES':      '/workspace/parameter-golf/candidate_pack/N18_baseA_es',
    'SimCTG+HSU':     '/workspace/parameter-golf/candidate_pack/N18_baseA_hsu',
    'SimCTG+AOS':     '/workspace/parameter-golf/candidate_pack/N18_baseA_aos',
}

OUT_DIR = '/workspace/parameter-golf/submissions/C_CrossBase_RegTransfer_Study/figures'
os.makedirs(OUT_DIR, exist_ok=True)

REG_COLORS = {
    'no-reg':       '#6b7280',
    'SimCTG':       '#3b82f6',
    'SimCTG+QAHSP': '#10b981',
    'SimCTG+ES':    '#f59e0b',
    'SimCTG+HSU':   '#8b5cf6',
    'SimCTG+AOS':   '#ef4444',
}

# ───────────────────────────────────────────────────────────────────────────
# Figure 1: SVD spectrum
# ───────────────────────────────────────────────────────────────────────────
WEIGHT_FAMILIES = [
    ('attn.c_q.weight',  'Attention Q'),
    ('attn.c_k.weight',  'Attention K'),
    ('attn.c_v.weight',  'Attention V'),
    ('attn.proj.weight', 'Attention out'),
    ('mlp.fc.weight',    'MLP up-proj'),
    ('mlp.proj.weight',  'MLP down-proj'),
]

def collect_svd_spectra(state_dict):
    """For each weight family, compute SVD on each layer's weight,
    then average the (sorted, normalized) singular value curves across layers.
    Returns {family: ndarray of length min(in,out)}.
    """
    spectra = {fam: [] for fam, _ in WEIGHT_FAMILIES}
    for k, v in state_dict.items():
        if v.ndim != 2:
            continue
        for fam_substr, _ in WEIGHT_FAMILIES:
            if k.endswith(fam_substr):
                w = v.float().cpu()
                S = torch.linalg.svdvals(w)
                S = (S / S.max()).numpy()  # normalize to spectral norm
                spectra[fam_substr].append(S)
                break
    out = {}
    for fam, _ in WEIGHT_FAMILIES:
        if spectra[fam]:
            stacked = np.stack(spectra[fam], axis=0)
            out[fam] = stacked.mean(axis=0)
    return out

def plot_svd_spectrum(svd_per_reg):
    n_fam = len(WEIGHT_FAMILIES)
    n_cols = 3
    n_rows = (n_fam + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.6 * n_rows), squeeze=False)
    for i, (fam, label) in enumerate(WEIGHT_FAMILIES):
        ax = axes[i // n_cols][i % n_cols]
        for reg, spectra in svd_per_reg.items():
            if fam not in spectra: continue
            S = spectra[fam]
            ax.semilogy(np.arange(1, len(S)+1)/len(S), S, color=REG_COLORS[reg],
                        lw=1.7, alpha=0.9, label=reg)
        ax.set_xlabel('rank index (normalized)')
        ax.set_ylabel('σᵢ / σ₁  (log)')
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(bottom=1e-2)
    # legend on first axis
    axes[0][0].legend(loc='lower left', fontsize=8, framealpha=0.85)
    # hide unused panels
    for j in range(n_fam, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')
    fig.suptitle('Singular-value spectrum per regularizer, averaged across 11 layers',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_svd_spectrum.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out}")
    # also save a condensed bar chart of "spectrum flatness" — a single number per (reg, fam).
    # Flatness metric: -mean(log10(sigma_i / sigma_1)) — bigger = flatter spectrum =
    # more L2 mass distributed in tail, harder to per-row int6 quantize.
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    fams = [fam for fam, _ in WEIGHT_FAMILIES]
    fam_labels = [lbl for _, lbl in WEIGHT_FAMILIES]
    regs = list(svd_per_reg.keys())
    width = 0.8 / max(len(regs), 1)
    x = np.arange(len(fams))
    for k, reg in enumerate(regs):
        vals = []
        for fam in fams:
            S = svd_per_reg[reg].get(fam)
            if S is None or len(S) == 0:
                vals.append(np.nan); continue
            tail = S[max(1, len(S)//8):]   # exclude top 12.5% (head dims)
            vals.append(float(-np.log10(np.clip(tail, 1e-8, None)).mean()))
        ax2.bar(x + (k - (len(regs)-1)/2)*width, vals, width=width,
                color=REG_COLORS.get(reg, '#999'), label=reg, edgecolor='black', linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fam_labels, rotation=15, ha='right')
    ax2.set_ylabel('mean −log₁₀(σᵢ/σ₁) over tail dims\n(higher = flatter, harder to int6 per-row quantize)')
    ax2.set_title('Spectrum flatness per (regularizer, weight family) — tail dims only', fontsize=12)
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.85, ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')
    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, 'fig_svd_flatness.png')
    fig2.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close(fig2)
    print(f"  saved {out2}")

# ───────────────────────────────────────────────────────────────────────────
# Figure 2: per-layer depth trajectory of ‖h‖ and excess kurtosis
# ───────────────────────────────────────────────────────────────────────────
def excess_kurtosis(x, dim=-1):
    """Excess kurtosis (Fisher) along dim; positive = heavy tails."""
    x = x.float()
    m = x.mean(dim=dim, keepdim=True)
    x_c = x - m
    m4 = (x_c ** 4).mean(dim=dim)
    m2 = (x_c ** 2).mean(dim=dim)
    return (m4 / (m2 ** 2 + 1e-12) - 3.0)

def collect_depth_trajectory(model_dir, n_seq=8, seq_len=128):
    """Run a forward pass and capture hidden state after each block.
    Returns (mean_norm[L], mean_kurt[L]).
    """
    sys.path.insert(0, model_dir)
    if 'train_gpt' in sys.modules: del sys.modules['train_gpt']
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", os.path.join(model_dir, "train_gpt.py"))
    train_gpt = importlib.util.module_from_spec(spec)
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    spec.loader.exec_module(train_gpt)
    h_cls = train_gpt.Hyperparameters() if hasattr(train_gpt, 'Hyperparameters') else None
    model = train_gpt.GPT(h_cls)
    sd = torch.load(os.path.join(model_dir, "final_model.pt"), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval().float()  # CPU float32 for stability

    L = len(model.blocks)
    captured = [None] * L
    hooks = []
    for i, blk in enumerate(model.blocks):
        def make_hook(idx):
            def h(m, inp, out):
                captured[idx] = out.detach().float()
            return h
        hooks.append(blk.register_forward_hook(make_hook(i)))

    # Use deterministic synthetic tokens; we want consistent comparison across regs.
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 10000, size=(n_seq, seq_len)).astype(np.int64)
    toks = torch.from_numpy(toks)

    with torch.no_grad():
        try:
            _ = model.forward_logits(toks)
        except Exception as e:
            print(f"    forward error: {e}")
            for h in hooks: h.remove()
            return None, None

    for h in hooks: h.remove()

    norms, kurts = [], []
    for i, h in enumerate(captured):
        if h is None:
            norms.append(np.nan); kurts.append(np.nan); continue
        # h: (B, T, D); per-token norm + per-coord kurtosis
        hn = h.reshape(-1, h.size(-1))
        norms.append(hn.norm(dim=-1).mean().item())
        kurts.append(excess_kurtosis(hn, dim=-1).mean().item())
    del model, captured
    gc.collect()
    return np.array(norms), np.array(kurts)

def plot_depth_trajectory(traj_per_reg):
    fig, (axN, axK) = plt.subplots(1, 2, figsize=(14, 4.8))
    for reg, (norms, kurts) in traj_per_reg.items():
        if norms is None: continue
        x = np.arange(1, len(norms)+1)
        axN.plot(x, norms, marker='o', ms=4.5, color=REG_COLORS[reg], lw=1.7, label=reg)
        axK.plot(x, kurts, marker='o', ms=4.5, color=REG_COLORS[reg], lw=1.7, label=reg)
    axN.set_xlabel('block index (1 = closest to embedding)')
    axN.set_ylabel('mean ‖h‖₂ across tokens')
    axN.set_title('hidden-state norm trajectory through depth', fontsize=11)
    axN.grid(True, alpha=0.3)
    axK.set_xlabel('block index (1 = closest to embedding)')
    axK.set_ylabel('mean per-coord excess kurtosis')
    axK.set_title('hidden-state heavy-tail-ness through depth', fontsize=11)
    axK.grid(True, alpha=0.3)
    axK.axhline(0, color='k', lw=0.7, alpha=0.5)
    axK.legend(loc='best', fontsize=8, framealpha=0.85)
    fig.suptitle('Where in the depth do regularizers shape outliers?', fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_depth_trajectory.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out}")

# ───────────────────────────────────────────────────────────────────────────
# Figure 3: pairwise CKA heatmap
# ───────────────────────────────────────────────────────────────────────────
def linear_cka(X, Y):
    """Linear CKA from Kornblith et al. 2019.
    X: (N, dx)  Y: (N, dy) — same N."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    XtY = X.t() @ Y
    num = (XtY ** 2).sum()
    den = ((X.t() @ X) ** 2).sum().sqrt() * ((Y.t() @ Y) ** 2).sum().sqrt()
    return float(num / (den + 1e-12))

def get_last_hidden(model_dir, n_seq=8, seq_len=128):
    """Same setup as depth-trajectory but return only the final-block hidden state."""
    sys.path.insert(0, model_dir)
    if 'train_gpt' in sys.modules: del sys.modules['train_gpt']
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", os.path.join(model_dir, "train_gpt.py"))
    train_gpt = importlib.util.module_from_spec(spec)
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    spec.loader.exec_module(train_gpt)
    h_cls = train_gpt.Hyperparameters() if hasattr(train_gpt, 'Hyperparameters') else None
    model = train_gpt.GPT(h_cls)
    sd = torch.load(os.path.join(model_dir, "final_model.pt"), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval().float()
    captured = [None]
    last = model.blocks[-1]
    def hook(m, inp, out):
        captured[0] = out.detach().float()
    h = last.register_forward_hook(hook)
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 10000, size=(n_seq, seq_len)).astype(np.int64)
    toks = torch.from_numpy(toks)
    with torch.no_grad():
        try:
            _ = model.forward_logits(toks)
        except Exception as e:
            print(f"    forward error: {e}")
            h.remove()
            return None
    h.remove()
    out = captured[0]
    if out is None: return None
    out = out.reshape(-1, out.size(-1))
    del model
    gc.collect()
    return out

def plot_cka_heatmap(cka_matrix, regs):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(cka_matrix, vmin=0.0, vmax=1.0, cmap='magma')
    ax.set_xticks(range(len(regs)))
    ax.set_yticks(range(len(regs)))
    ax.set_xticklabels(regs, rotation=35, ha='right')
    ax.set_yticklabels(regs)
    for i in range(len(regs)):
        for j in range(len(regs)):
            txt_color = 'white' if cka_matrix[i, j] < 0.55 else 'black'
            ax.text(j, i, f'{cka_matrix[i, j]:.2f}', ha='center', va='center',
                    color=txt_color, fontsize=9)
    ax.set_title('Linear CKA between final-block hidden states\n(Kornblith et al. 2019)', fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('CKA  (1.0 = same representation)')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_cka_heatmap.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out}")

# ───────────────────────────────────────────────────────────────────────────
def main():
    # Figure 1: SVD spectrum (state-dict only, fast)
    print("[1/3] SVD spectra per reg (matrix-family-wise)")
    svd_per_reg = {}
    for reg, d in REG_DIRS.items():
        sd_path = os.path.join(d, 'final_model.pt')
        if not os.path.exists(sd_path):
            print(f"  {reg}: skipped (no final_model.pt)"); continue
        print(f"  {reg}: SVD")
        sd = torch.load(sd_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
        svd_per_reg[reg] = collect_svd_spectra(sd)
        del sd; gc.collect()
    plot_svd_spectrum(svd_per_reg)

    # Figure 2: depth trajectory (forward pass per reg)
    print("\n[2/3] depth trajectory per reg (per-layer ‖h‖ + kurtosis)")
    traj_per_reg = {}
    for reg, d in REG_DIRS.items():
        if not os.path.exists(os.path.join(d, 'final_model.pt')):
            traj_per_reg[reg] = (None, None); continue
        print(f"  {reg}: forward")
        norms, kurts = collect_depth_trajectory(d)
        traj_per_reg[reg] = (norms, kurts)
    plot_depth_trajectory(traj_per_reg)

    # Save trajectory numbers as JSON for the README to reference.
    traj_json = {reg: {'norms': (n.tolist() if n is not None else None),
                       'kurts': (k.tolist() if k is not None else None)}
                 for reg, (n, k) in traj_per_reg.items()}
    with open('/workspace/parameter-golf/submissions/C_CrossBase_RegTransfer_Study/depth_trajectory.json', 'w') as f:
        json.dump(traj_json, f, indent=2)

    # Figure 3: CKA heatmap
    print("\n[3/3] CKA pairwise heatmap")
    hidden_per_reg = {}
    for reg, d in REG_DIRS.items():
        if not os.path.exists(os.path.join(d, 'final_model.pt')):
            continue
        print(f"  {reg}: forward (last block only)")
        h = get_last_hidden(d)
        if h is not None:
            hidden_per_reg[reg] = h
    regs = list(hidden_per_reg.keys())
    n = len(regs)
    cka = np.zeros((n, n))
    for i, ri in enumerate(regs):
        for j, rj in enumerate(regs):
            if j < i:
                cka[i, j] = cka[j, i]
            else:
                cka[i, j] = linear_cka(hidden_per_reg[ri], hidden_per_reg[rj])
    plot_cka_heatmap(cka, regs)
    cka_json = {ri: {rj: float(cka[i, j]) for j, rj in enumerate(regs)} for i, ri in enumerate(regs)}
    with open('/workspace/parameter-golf/submissions/C_CrossBase_RegTransfer_Study/cka_pairwise.json', 'w') as f:
        json.dump(cka_json, f, indent=2)

    print("\nAll 3 figures + 2 JSON tables written.")

if __name__ == '__main__':
    main()
