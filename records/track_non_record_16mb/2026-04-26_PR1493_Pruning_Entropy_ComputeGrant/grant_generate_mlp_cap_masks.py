#!/usr/bin/env python3
"""Generate per-block capped structured MLP pruning masks.

Ranks all MLP hidden channels by a saved importance score from
``mlp_channel_stats.pt`` and selects the globally cheapest channels subject to a
per-block cap.  With the default ``--cap-multiplier 1.0``, each MLP block can
contribute at most the requested global prune fraction.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import torch

SCORE_KEYS = ("activation_weighted_score", "norm_score")

def load_torch(path: Path) -> Any:
    try: return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: return torch.load(path, map_location="cpu")

def unwrap_state(obj: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict): obj = obj["model"]
    return {str(k): v for k, v in obj.items() if torch.is_tensor(v)} if isinstance(obj, dict) else None

def parse_floats(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals or any(v <= 0 or v >= 1 for v in vals): raise ValueError("fractions must be in (0,1)")
    return vals

def clean(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().cpu().float().flatten(); finite = torch.isfinite(y)
    if finite.all(): return y
    y = y.clone(); y[~finite] = y[finite].max() if finite.any() else 1.0; return y

def infer_shape(name: str, row: dict[str, torch.Tensor], state: dict[str, torch.Tensor] | None, default_dim: int) -> tuple[int, int]:
    score = next((row[k] for k in SCORE_KEYS if k in row), None)
    if score is None: raise KeyError(f"{name} has no score tensor")
    hidden = int(score.numel())
    if state:
        fc, proj = state.get(f"{name}.fc.weight"), state.get(f"{name}.proj.weight")
        if fc is not None and proj is not None: return int(fc.shape[1]), hidden
    return default_dim, hidden

def build(stats, score_key, state, default_dim):
    entries, shapes = [], {}
    for name in sorted(stats):
        row = stats[name]; model_dim, hidden = infer_shape(name, row, state, default_dim)
        score = clean(row[score_key]); assert score.numel() == hidden
        shapes[name] = {"model_dim": model_dim, "hidden": hidden}
        entries += [{"module": name, "channel": i, "score": float(score[i]), "model_dim": model_dim} for i in range(hidden)]
    entries.sort(key=lambda r: (r["score"], r["module"], r["channel"]))
    return entries, shapes

def make(entries, shapes, fraction, cap_multiplier):
    total = sum(s["hidden"] for s in shapes.values()); target = max(1, round(total * fraction))
    caps = {m: min(s["hidden"], max(1, math.ceil(s["hidden"] * fraction * cap_multiplier))) for m, s in shapes.items()}
    masks = {m: torch.ones(s["hidden"], dtype=torch.bool) for m, s in shapes.items()}
    per = {m: {"hidden": s["hidden"], "pruned": 0, "channels": []} for m, s in shapes.items()}
    selected = []
    for e in entries:
        if len(selected) >= target: break
        m, c = e["module"], int(e["channel"])
        if per[m]["pruned"] >= caps[m]: continue
        masks[m][c] = False; per[m]["pruned"] += 1; per[m]["channels"].append(c); selected.append(e)
    if len(selected) != target: raise RuntimeError("cap too tight; increase --cap-multiplier")
    removed = sum(2 * int(e["model_dim"]) for e in selected)
    summary = {"policy":"per_block_cap","fraction":fraction,"cap_multiplier":cap_multiplier,"total_channels":total,"pruned_channels":len(selected),"kept_channels":total-len(selected),"per_module_caps":caps,"per_module":per,"active_modules":sum(1 for x in per.values() if x["pruned"]),"estimated_fp16_bytes_removed":removed*2,"estimated_int6_packed_bytes_removed":math.ceil(removed*6/8),"estimated_current_quant_raw_bytes_removed":sum(2*int(e["model_dim"])+2 for e in selected),"score_mean_pruned":sum(float(e["score"]) for e in selected)/len(selected)}
    return masks, summary

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact-dir", type=Path, required=True); ap.add_argument("--fractions", default="0.01,0.02,0.05,0.08"); ap.add_argument("--score-key", choices=SCORE_KEYS, default="activation_weighted_score"); ap.add_argument("--cap-multiplier", type=float, default=1.0); ap.add_argument("--checkpoint", type=Path); ap.add_argument("--default-model-dim", type=int, default=512); ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args(); stats = load_torch(args.artifact_dir/"mlp_channel_stats.pt")
    ckpt = args.checkpoint or args.artifact_dir/"final_ema_fp.pt"; state = unwrap_state(load_torch(ckpt)) if ckpt.is_file() else None
    entries, shapes = build(stats, args.score_key, state, args.default_model_dim); out = args.output_dir or args.artifact_dir/"mlp_pruning_masks_cap"; out.mkdir(parents=True, exist_ok=True)
    summaries=[]
    for f in parse_floats(args.fractions):
        masks, summary = make(entries, shapes, f, args.cap_multiplier); pct=round(f*10000); path=out/f"mlp_cap_prune_{pct:04d}bp_masks.pt"
        torch.save({"policy":"per_block_cap","score_key":args.score_key,"fraction":f,"masks":masks,"summary":summary}, path); summary["mask_path"]=str(path); summaries.append(summary)
    (out/"summary.json").write_text(json.dumps({"summaries":summaries}, indent=2, sort_keys=True)); print(json.dumps({"output_dir":str(out),"generated":len(summaries)}, indent=2))
if __name__ == "__main__": main()
