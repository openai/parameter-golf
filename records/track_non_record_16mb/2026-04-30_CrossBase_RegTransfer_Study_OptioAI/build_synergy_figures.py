"""Build heatmap + synergy-detection figures from real_reg_quant_matrix.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('/workspace/parameter-golf/submissions/C_CrossBase_RegTransfer_Study/real_reg_quant_matrix.json') as f:
    data = json.load(f)

regs = list(data.keys())
quants = list(data[regs[0]].keys())
n_r, n_q = len(regs), len(quants)

# Build matrices for each metric
metrics = ['l2_distortion', 'cos_shift', 'isoscore_post', 'eff_rank_post']
mats = {m: np.zeros((n_r, n_q)) for m in metrics}
for ri, r in enumerate(regs):
    for qi, q in enumerate(quants):
        for m in metrics:
            mats[m][ri, qi] = data[r][q][m]

# === Figure: heatmaps ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Real (reg × quant) interaction matrix on REAL Base A LM hidden states\n(captured from forward pass on val tokens, 5 trained models)', fontsize=13, weight='bold')

cmaps = {'l2_distortion': 'RdYlGn_r', 'cos_shift': 'RdYlGn_r', 'isoscore_post': 'RdYlGn_r', 'eff_rank_post': 'RdYlGn'}
titles = {
    'l2_distortion': '(a) L2 distortion (lower = better)',
    'cos_shift': '(b) Cosine shift (lower = better)',
    'isoscore_post': '(c) Post-quant isoscore (lower = better)',
    'eff_rank_post': '(d) Post-quant effective rank (higher = better)',
}

for ax, m in zip(axes.flat, metrics):
    mat = mats[m]
    im = ax.imshow(mat, aspect='auto', cmap=cmaps[m])
    ax.set_xticks(range(n_q)); ax.set_xticklabels(quants, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_r)); ax.set_yticklabels(regs, fontsize=10)
    ax.set_title(titles[m], fontsize=11)
    # Annotate values
    for i in range(n_r):
        for j in range(n_q):
            ax.text(j, i, f'{mat[i,j]:.4f}' if 'l2' in m or 'shift' in m else f'{mat[i,j]:.3f}', 
                    ha='center', va='center', fontsize=8, color='black')
    plt.colorbar(im, ax=ax, fraction=0.04)
    # Mark best per quant (column) — for distortion/cos_shift, lowest; for eff_rank, highest
    for j in range(n_q):
        col = mat[:, j]
        best_row = np.argmin(col) if m != 'eff_rank_post' else np.argmax(col)
        ax.scatter(j, best_row, marker='*', s=200, c='gold', edgecolors='black', linewidths=1, zorder=10)

plt.tight_layout()
plt.savefig('/workspace/parameter-golf/submissions/C_CrossBase_RegTransfer_Study/figures/fig_reg_quant_matrix_real.png', dpi=130, bbox_inches='tight')
plt.close()
print("saved fig_reg_quant_matrix_real.png")

# === Synergy detection: which (reg, quant) pairs are unexpectedly good? ===
# For each metric, normalize each row (relative to that reg's mean) and each column (relative to that quant's mean)
# A "synergy" is a cell that's better than both its row mean AND column mean by margin
print()
print("=== SYNERGY detection (cells that play unusually nicely) ===")
for m in ['l2_distortion', 'cos_shift']:
    mat = mats[m]
    row_means = mat.mean(axis=1, keepdims=True)
    col_means = mat.mean(axis=0, keepdims=True)
    # Synergy: cell is BELOW both row mean and col mean (lower distortion is better here)
    rel_row = mat - row_means  # negative = better than row average
    rel_col = mat - col_means
    print(f"\nMetric: {m}")
    print(f"  Each cell: relative-to-row-mean / relative-to-col-mean")
    for ri, r in enumerate(regs):
        for qi, q in enumerate(quants):
            rr, rc = rel_row[ri, qi], rel_col[ri, qi]
            if rr < -mat.std()*0.3 and rc < -mat.std()*0.3:
                print(f"  ⭐ SYNERGY: {r:<10} × {q:<22} (row Δ {rr:+.4f}, col Δ {rc:+.4f}) — both reg AND quant outperform their means")

# === "Plays nice" summary table ===
print()
print("=== 'Plays nice' summary: best reg per quant + best quant per reg ===")
print()
print("For each QUANT scheme, which REG produces the smallest distortion?")
print(f"{'quant scheme':<22} {'best reg':<10} {'L2 dist':>9}")
for qi, q in enumerate(quants):
    col = mats['l2_distortion'][:, qi]
    best_r = np.argmin(col)
    print(f"  {q:<22} {regs[best_r]:<10} {col[best_r]:.4f}")
print()
print("For each REG, which QUANT gives smallest distortion?")
print(f"{'reg':<10} {'best quant':<22} {'L2 dist':>9}")
for ri, r in enumerate(regs):
    row = mats['l2_distortion'][ri, :]
    best_q = np.argmin(row)
    print(f"  {r:<10} {quants[best_q]:<22} {row[best_q]:.4f}")
