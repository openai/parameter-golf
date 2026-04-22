#!/usr/bin/env python3
"""Task A: weight-delta analysis across flat-zone checkpoints."""
import json, os, re, sys, time
from collections import defaultdict
import torch

CKPTS = {
    1500: "/workspace/runs/000-sota-replication/checkpoints/ckpt_event_step1500.pt",
    2275: "/workspace/runs/000-sota-replication/checkpoints/ckpt_event_step2275.pt",
    3412: "/workspace/runs/000-sota-replication/checkpoints/ckpt_event_step3412.pt",
}
OUT_DIR = "/workspace/runs/005-weight-delta"
os.makedirs(OUT_DIR, exist_ok=True)

LOOP_LAYERS = {3, 4, 5}

def classify(name: str):
    m = re.search(r"blocks\.(\d+)\.", name)
    layer = int(m.group(1)) if m else -1
    lower = name.lower()
    if "attn" in lower and ("qkv" in lower or "q_proj" in lower or "k_proj" in lower or "v_proj" in lower):
        ptype = "attn_qkv"
    elif "attn" in lower and "proj" in lower:
        ptype = "attn_proj"
    elif "attn" in lower:
        ptype = "attn_other"
    elif "mlp" in lower and ("fc" in lower or "w1" in lower or "gate" in lower or "up" in lower):
        ptype = "mlp_fc"
    elif "mlp" in lower and ("proj" in lower or "w2" in lower or "down" in lower):
        ptype = "mlp_proj"
    elif "mlp" in lower:
        ptype = "mlp_other"
    elif "norm" in lower or "ln" in lower:
        ptype = "norm"
    elif "embed" in lower or "wte" in lower or "wpe" in lower:
        ptype = "embed"
    elif "head" in lower or "lm_head" in lower or "unembed" in lower:
        ptype = "head"
    else:
        ptype = "other"
    return layer, ptype

def load_sd(path):
    t0 = time.perf_counter()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    print(f"loaded {path} ({len(sd)} params) in {time.perf_counter()-t0:.1f}s", flush=True)
    return sd

def frob(t):
    return torch.linalg.vector_norm(t.float().flatten()).item()

def main():
    sd = {step: load_sd(p) for step, p in CKPTS.items()}
    keys = list(sd[1500].keys())
    print(f"\ncomputing deltas on {len(keys)} params...", flush=True)

    results = []
    for k in keys:
        t1, t2, t3 = sd[1500][k], sd[2275][k], sd[3412][k]
        if t1.dtype in (torch.int8, torch.int32, torch.int64, torch.uint8) or not t1.is_floating_point():
            continue
        n1 = frob(t1)
        n2 = frob(t2)
        n3 = frob(t3)
        d12 = frob(t2 - t1)
        d23 = frob(t3 - t2)
        rel12 = d12 / n1 if n1 > 0 else 0.0
        rel23 = d23 / n2 if n2 > 0 else 0.0
        layer, ptype = classify(k)
        results.append({
            "name": k, "layer": layer, "ptype": ptype,
            "numel": t1.numel(),
            "frob_1500": n1, "frob_2275": n2, "frob_3412": n3,
            "rel_delta_1500_2275": rel12,
            "rel_delta_2275_3412": rel23,
            "rel_per_step_first": rel12 / 775.0,
            "rel_per_step_second": rel23 / 1137.0,
            "loop_layer": layer in LOOP_LAYERS,
        })

    with open(os.path.join(OUT_DIR, "delta_layers.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote delta_layers.json ({len(results)} entries)", flush=True)

    # Group summaries
    by_layer_type = defaultdict(list)
    for r in results:
        by_layer_type[(r["layer"], r["ptype"])].append(r)

    lines = ["# Task A — weight-delta table\n",
             f"Checkpoints: step 1500 (pre-flat), step 2275 (flat-zone end), step 3412 (post-flat)",
             f"Intervals: first = 1500→2275 (775 steps), second = 2275→3412 (1137 steps)",
             f"Loop layers = blocks 3,4,5 (marked with *)\n",
             "## Per-(layer, type) aggregated (mean rel-delta across params in group)\n",
             "| layer | ptype | #params | totN | relΔ 1→2 | relΔ 2→3 | /step 1→2 (×1e-6) | /step 2→3 (×1e-6) | ratio 1st/2nd per-step |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (layer, ptype), group in sorted(by_layer_type.items()):
        tot_numel = sum(r["numel"] for r in group)
        # weighted by numel
        w = sum(r["numel"] for r in group) or 1
        m12 = sum(r["rel_delta_1500_2275"] * r["numel"] for r in group) / w
        m23 = sum(r["rel_delta_2275_3412"] * r["numel"] for r in group) / w
        ps1 = m12 / 775.0 * 1e6
        ps2 = m23 / 1137.0 * 1e6
        ratio = ps1 / ps2 if ps2 > 0 else float('inf')
        star = "*" if layer in LOOP_LAYERS else " "
        ll = f"{layer}{star}" if layer >= 0 else "(global)"
        lines.append(f"| {ll} | {ptype} | {len(group)} | {tot_numel:,} | {m12:.4f} | {m23:.4f} | {ps1:.2f} | {ps2:.2f} | {ratio:.3f} |")

    lines.append("\n## Loop-layer vs non-loop-layer aggregate (weighted by numel)\n")
    lines.append("| group | relΔ 1→2 | relΔ 2→3 | /step 1→2 (×1e-6) | /step 2→3 (×1e-6) |")
    lines.append("|---|---|---|---|---|")
    for label, pred in [("loop (3,4,5)", lambda r: r["loop_layer"]),
                         ("non-loop", lambda r: not r["loop_layer"] and r["layer"] >= 0),
                         ("global (no block)", lambda r: r["layer"] == -1)]:
        grp = [r for r in results if pred(r)]
        if not grp: continue
        w = sum(r["numel"] for r in grp) or 1
        m12 = sum(r["rel_delta_1500_2275"] * r["numel"] for r in grp) / w
        m23 = sum(r["rel_delta_2275_3412"] * r["numel"] for r in grp) / w
        ps1 = m12 / 775.0 * 1e6
        ps2 = m23 / 1137.0 * 1e6
        lines.append(f"| {label} | {m12:.4f} | {m23:.4f} | {ps1:.2f} | {ps2:.2f} |")

    lines.append("\n## Top 15 params by per-step movement in flat zone (1500→2275)\n")
    lines.append("| name | layer | ptype | numel | /step 1→2 (×1e-6) | /step 2→3 (×1e-6) | ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: -x["rel_per_step_first"])[:15]:
        star = "*" if r["loop_layer"] else ""
        ratio = r["rel_per_step_first"] / r["rel_per_step_second"] if r["rel_per_step_second"] > 0 else float('inf')
        lines.append(f"| `{r['name']}` | {r['layer']}{star} | {r['ptype']} | {r['numel']:,} | {r['rel_per_step_first']*1e6:.2f} | {r['rel_per_step_second']*1e6:.2f} | {ratio:.3f} |")

    with open(os.path.join(OUT_DIR, "delta_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote delta_table.md", flush=True)

if __name__ == "__main__":
    main()
