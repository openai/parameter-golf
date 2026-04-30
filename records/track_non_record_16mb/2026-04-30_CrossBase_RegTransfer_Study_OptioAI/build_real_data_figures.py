"""Build visualizations from REAL captured hidden states across the 6 EmbStudy models."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

ROOT = '/workspace/parameter-golf'
FIG_DIR = f'{ROOT}/submissions/C_CrossBase_RegTransfer_Study/figures'
os.makedirs(FIG_DIR, exist_ok=True)

base_dirs = {
    'no-reg':       f'{ROOT}/candidate_pack/N18_baseA_nosimctg',
    'SimCTG':       f'{ROOT}/candidate_pack/N18_baseA_baseline',
    'SimCTG+QAHSP': f'{ROOT}/candidate_pack/N18_baseA_qahsp',
    'SimCTG+ES':    f'{ROOT}/candidate_pack/N18_baseA_es',
    'SimCTG+HSU':   f'{ROOT}/candidate_pack/N18_baseA_hsu',
    'SimCTG+AOS':   f'{ROOT}/candidate_pack/N18_baseA_aos',
}

def load_hidden(model_dir, n_tokens=128):
    """Load BF16 model + run forward + capture hidden states."""
    sys.path.insert(0, model_dir)
    if 'train_gpt' in sys.modules:
        del sys.modules['train_gpt']
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
    model_cls = None
    for cls_name in ['GPT', 'FinalMiniLM', 'Model']:
        if hasattr(train_gpt, cls_name):
            model_cls = getattr(train_gpt, cls_name)
            break
    model = model_cls(h_cls)
    state_dict = torch.load(os.path.join(model_dir, "final_model.pt"), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).bfloat16()
    
    toks = np.random.randint(0, 8000, size=8 * n_tokens)
    toks = torch.from_numpy(toks).reshape(8, n_tokens).long().to(device)
    
    captured = {}
    for name, mod in model.named_modules():
        if name.endswith('.10') or name.endswith('.9'):
            def make_hook(n):
                def hook(m, inp, out):
                    captured[n] = out.detach().cpu().float()
                return hook
            mod.register_forward_hook(make_hook(name))
    
    with torch.no_grad():
        try:
            _ = model.forward_logits(toks) if hasattr(model, 'forward_logits') else model(toks)
        except Exception as e:
            print(f"  forward error in {model_dir}: {e}")
    
    if captured:
        h = list(captured.values())[-1]
        return h.reshape(-1, h.size(-1))[:n_tokens]
    return None

# === Capture hidden states ===
np.random.seed(42)
torch.manual_seed(42)
hidden_per_reg = {}
for reg, d in base_dirs.items():
    if os.path.exists(os.path.join(d, "final_model.pt")):
        h = load_hidden(d, n_tokens=128)
        if h is not None:
            hidden_per_reg[reg] = h
            print(f"{reg:<14}: shape={tuple(h.shape)} mean_L2={h.pow(2).sum(-1).sqrt().mean().item():.2f}")

if not hidden_per_reg:
    print("No models loaded. Exit.")
    sys.exit(1)

# Save hidden states for downstream reuse
torch.save(hidden_per_reg, f'{ROOT}/submissions/C_CrossBase_RegTransfer_Study/real_hidden_states.pt')

# === FIG 1: Real-data 3D PCA scatter ===
fig = plt.figure(figsize=(20, 8))
fig.suptitle('Real Base A LM hidden states (128 tokens × 512 dims), 3D PCA per reg', fontsize=13, weight='bold')
n_tok = 128
colors = plt.cm.viridis(np.linspace(0, 1, n_tok))
for col, (name, h) in enumerate(hidden_per_reg.items()):
    h_np = h.numpy()
    h_c = h_np - h_np.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(h_c, full_matrices=False)
    proj = U[:, :3] * S[:3]
    ax = fig.add_subplot(1, len(hidden_per_reg), col+1, projection='3d')
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=colors, s=30, alpha=0.7, edgecolors='black', linewidths=0.3)
    ax.set_title(name, fontsize=11)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=45)
    # Annotate spread
    spread = (proj.max(0) - proj.min(0)).max()
    ax.set_xlabel(f'PC1 (sv={S[0]:.1f})', fontsize=8)
    ax.set_ylabel(f'PC2 (sv={S[1]:.1f})', fontsize=8)
    ax.set_zlabel(f'PC3 (sv={S[2]:.1f})', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_real_3d_pca.png', dpi=130, bbox_inches='tight')
plt.close()
print("saved fig_real_3d_pca.png")

# === FIG 2: Real per-coord distribution per reg ===
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Real per-coordinate hidden state value distributions (128 tokens × 512 dims)', fontsize=13, weight='bold')
for ax, (name, h) in zip(axes.flat, hidden_per_reg.items()):
    flat = h.numpy().flatten()
    ax.hist(flat, bins=60, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_title(f'{name}\nμ={flat.mean():.3f}, σ={flat.std():.3f}, max|h|={np.abs(flat).max():.2f}', fontsize=10)
    ax.set_xlabel('h_d value')
    ax.set_ylabel('count')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_real_coord_distribution.png', dpi=130)
plt.close()
print("saved fig_real_coord_distribution.png")

# === FIG 3: Real-data canonical metrics comparison ===
def isoscore(h):
    h_n = F.normalize(h, dim=-1)
    sim = h_n @ h_n.t()
    n = h_n.size(0)
    off = sim - torch.eye(n)
    return off.abs().mean().item()

def eff_rank(h):
    h_c = h - h.mean(0, keepdim=True)
    _, S, _ = torch.linalg.svd(h_c, full_matrices=False)
    p = S / S.sum()
    p = p[p > 1e-10]
    return float(np.exp(-(p * p.log()).sum().item()))

real_metrics = {}
for name, h in hidden_per_reg.items():
    real_metrics[name] = {
        'isoscore': isoscore(h),
        'eff_rank': eff_rank(h),
        'norm_var': h.pow(2).sum(-1).sqrt().var().item(),
        'norm_mean': h.pow(2).sum(-1).sqrt().mean().item(),
        'max_abs':  h.abs().max().item(),
    }

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Real-data canonical metrics: do regs change real LM hidden states the way synthetic predicts?', fontsize=13, weight='bold')
names = list(real_metrics.keys())
colors_p = plt.cm.tab10(np.arange(len(names)))

ax = axes[0, 0]
isos = [real_metrics[n]['isoscore'] for n in names]
ax.bar(names, isos, color=colors_p, edgecolor='black', alpha=0.85)
ax.set_ylabel('mean |cos(h_i, h_j)| off-diag')
ax.set_title('Isoscore (lower = more isotropic)')
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(isos): ax.text(i, v+0.001, f'{v:.4f}', ha='center', fontsize=9, weight='bold')

ax = axes[0, 1]
ers = [real_metrics[n]['eff_rank'] for n in names]
ax.bar(names, ers, color=colors_p, edgecolor='black', alpha=0.85)
ax.set_ylabel('exp(spectral entropy)')
ax.set_title('Effective rank (higher = more dimensions used)')
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(ers): ax.text(i, v+0.5, f'{v:.1f}', ha='center', fontsize=9, weight='bold')

ax = axes[1, 0]
nvs = [real_metrics[n]['norm_var'] for n in names]
ax.bar(names, nvs, color=colors_p, edgecolor='black', alpha=0.85)
ax.set_ylabel('variance of L2 norms')
ax.set_title('Per-token L2 norm variance (lower = more uniform)')
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(nvs): ax.text(i, v+0.05, f'{v:.2f}', ha='center', fontsize=9, weight='bold')

ax = axes[1, 1]
mxs = [real_metrics[n]['max_abs'] for n in names]
ax.bar(names, mxs, color=colors_p, edgecolor='black', alpha=0.85)
ax.set_ylabel('max |h| across all coords')
ax.set_title('Outlier coord magnitude (lower = AOS-like effect)')
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(mxs): ax.text(i, v+0.5, f'{v:.1f}', ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_real_canonical_metrics.png', dpi=130)
plt.close()
print("saved fig_real_canonical_metrics.png")

# === FIG 4: Real data per-token L2 norm distribution per reg ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Real per-token L2 norm distributions across 128 captured tokens', fontsize=13, weight='bold')
for ax, (name, h) in zip(axes.flat, hidden_per_reg.items()):
    norms = h.pow(2).sum(-1).sqrt().numpy()
    ax.hist(norms, bins=20, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.set_title(f'{name}\nμ={norms.mean():.2f}, σ={norms.std():.2f}', fontsize=10)
    ax.set_xlabel('‖h‖')
    ax.set_ylabel('count')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_real_l2norm_distribution.png', dpi=130)
plt.close()
print("saved fig_real_l2norm_distribution.png")

print()
print("=== Real-data canonical metric table ===")
print(f"{'reg':<14} {'isoscore':>10} {'eff_rank':>10} {'norm_var':>10} {'norm_mean':>10} {'max|h|':>10}")
for n in names:
    m = real_metrics[n]
    print(f"  {n:<14} {m['isoscore']:>10.4f} {m['eff_rank']:>10.2f} {m['norm_var']:>10.3f} {m['norm_mean']:>10.2f} {m['max_abs']:>10.2f}")

# Save metrics as JSON
import json
open(f'{ROOT}/submissions/C_CrossBase_RegTransfer_Study/real_canonical_metrics.json', 'w').write(json.dumps(real_metrics, indent=2))
print("saved real_canonical_metrics.json")
