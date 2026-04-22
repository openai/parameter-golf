"""Plot per-100-step interval tok/s for 008 / 015 / 016 / 017 / 019 / 019b.
Writes research/scripts/throughput_interval.png.
"""
import re
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logs = {
    "008":  "runs/008-1736-reproduction/seed_42/train.log",
    "015":  "runs/015-recur-alpha/seed_42/train.log",
    "016":  "runs/016-recur-alpha-ones/seed_42/train.log",
    "017":  "runs/017-recur-alpha-full/seed_42/train.log",
    "019":  "runs/019-recur-alpha-constant-full/seed_42/train.log",
    "019b": "runs/019b-recur-alpha-manual-constant-full/seed_42/train.log",
}
pat = re.compile(r"^(\d+)/\d+ train_loss:.*tok/s:\s*(\d+)")

cum = {}
for r, log in logs.items():
    pts = []
    with open(log) as f:
        for line in f:
            m = pat.match(line)
            if m:
                pts.append((int(m.group(1)), int(m.group(2))))
    cum[r] = pts

def intervals(pts):
    out = []
    for i in range(1, len(pts)):
        s1, t1 = pts[i - 1]
        s2, t2 = pts[i]
        dt = s2 / t2 - s1 / t1
        if dt <= 0:
            continue
        rate = (s2 - s1) / dt
        out.append((s2, rate, (s1, s2)))  # end step, rate
    return out

iv = {r: intervals(cum[r]) for r in cum}

# Separate real intervals from 500-boundary logging artifact rows.
# Artifact rule: interval ending at a multiple of 500 (spike)
def is_artifact(end_step, run):
    # For dense runs (015-019), 500-step ends are artifacts.
    return run in {"015","016","017","019","019b"} and end_step % 500 == 0

colors = {
    "008":  "black",
    "015":  "tab:blue",
    "016":  "tab:cyan",
    "017":  "tab:green",
    "019":  "tab:red",
    "019b": "tab:purple",
}
markers = {
    "008":  "o",
    "015":  ".",
    "016":  ".",
    "017":  ".",
    "019":  ".",
    "019b": ".",
}

fig, ax = plt.subplots(figsize=(13, 6.5))

for r in ["008", "015", "016", "017", "019", "019b"]:
    data = iv[r]
    clean = [(s, v) for s, v, _ in data if not is_artifact(s, r)]
    artifact = [(s, v) for s, v, _ in data if is_artifact(s, r)]
    if clean:
        xs, ys = zip(*clean)
        ax.plot(xs, [y/1e6 for y in ys], label=r, color=colors[r],
                marker=markers[r], markersize=7 if r == "008" else 5,
                linewidth=2.2 if r == "008" else 1.4, alpha=0.9)
    if artifact:
        xs, ys = zip(*artifact)
        ax.scatter(xs, [y/1e6 for y in ys], color=colors[r],
                   marker="x", s=35, alpha=0.5)

ax.axvline(2142, color="gray", linestyle="--", alpha=0.6, label="loop activation (step 2142)")

ax.set_xlabel("step (end of 100-step interval)")
ax.set_ylabel("interval tok/s (M)")
ax.set_title("Per-100-step interval throughput — 008 / 015 / 016 / 017 / 019 / 019b\n"
             "(× markers = 500-step-boundary logging artifact; clean points are real rates)")
ax.set_ylim(4.0, 8.6)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9, ncol=2)

plt.tight_layout()
out = "research/scripts/throughput_interval.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")

# Also a post-activation zoom
fig2, ax2 = plt.subplots(figsize=(13, 5.5))
for r in ["008", "015", "016", "017", "019", "019b"]:
    data = iv[r]
    clean = [(s, v) for s, v, _ in data if not is_artifact(s, r) and s >= 2400]
    if not clean: continue
    xs, ys = zip(*clean)
    ax2.plot(xs, [y/1e6 for y in ys], label=r, color=colors[r],
             marker=markers[r], markersize=5 if r not in ("008","019b") else 7,
             linewidth=1.4 if r != "008" else 2.2, alpha=0.9)

ax2.set_xlabel("step")
ax2.set_ylabel("interval tok/s (M)")
ax2.set_title("Post-activation zoom (steps ≥ 2400, artifacts hidden)")
ax2.set_ylim(3.5, 5.9)
ax2.grid(alpha=0.3)
ax2.legend(loc="lower left", fontsize=9, ncol=3)
plt.tight_layout()
out2 = "research/scripts/throughput_interval_postact.png"
plt.savefig(out2, dpi=140)
print(f"wrote {out2}")
