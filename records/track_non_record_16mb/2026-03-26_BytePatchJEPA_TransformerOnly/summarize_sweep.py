from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SIMPLE_BASELINE_BPB = 1.22436570


def parse_variants(path: Path) -> dict[str, dict[str, object]]:
    variants: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return variants
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            variants[row["run_id"]] = row
    return variants


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(phase_root: Path) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    variants = parse_variants(phase_root / "variants.tsv")
    runs: dict[str, dict[str, object]] = {}
    for path in phase_root.glob("artifacts/*/backbone_run.json"):
        payload = load_json(path)
        run_id = str(payload["run_id"])
        runs[run_id] = {
            "variant": variants.get(run_id, {"run_id": run_id}),
            "backbone": payload,
            "probes": [],
        }
    probe_results: list[dict[str, object]] = []
    for path in phase_root.glob("artifacts/*/probe_results/*.json"):
        payload = load_json(path)
        probe_results.append(payload)
        run_id = str(payload["run_id"])
        if run_id not in runs:
            runs[run_id] = {"variant": variants.get(run_id, {"run_id": run_id}), "backbone": None, "probes": []}
        runs[run_id]["probes"].append(payload)
    return runs, probe_results


def best_probe(probes: list[dict[str, object]], kind: str, val_mode: str) -> dict[str, object] | None:
    eligible = [probe for probe in probes if probe.get("probe_kind") == kind and probe.get("probe_val_mode") == val_mode]
    if not eligible:
        return None
    return min(eligible, key=lambda probe: probe.get("best_val_bpb", float("inf")))


def ranking_key(run: dict[str, object]) -> tuple[float, float]:
    probes = run["probes"]
    full_strong = best_probe(probes, "strong", "full")
    proxy_strong = best_probe(probes, "strong", "proxy")
    cheap = best_probe(probes, "cheap", "proxy")
    if full_strong is not None:
        return float(full_strong["best_val_bpb"]), 0.0
    if proxy_strong is not None:
        return float(proxy_strong["best_val_bpb"]), 1.0
    if cheap is not None:
        return float(cheap["best_val_bpb"]), 2.0
    return float("inf"), 3.0


def family_rankings(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_family: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for run_id, run in runs.items():
        backbone = run.get("backbone") or {}
        variant = run.get("variant", {})
        backbone_kind = str(backbone.get("backbone_kind", variant.get("backbone_kind", "unknown")))
        patch_encoder_kind = str(
            backbone.get("patch_encoder_kind", variant.get("patch_encoder_kind", (backbone.get("config") or {}).get("patch_encoder_kind", "")))
        )
        objective_kind = str(variant.get("objective_kind", (backbone.get("config") or {}).get("objective_kind", "")))
        family_parts = [backbone_kind]
        if patch_encoder_kind:
            family_parts.append(patch_encoder_kind)
        if objective_kind:
            family_parts.append(objective_kind)
        family = "__".join(family_parts)
        by_family.setdefault(family, []).append((run_id, run))
    for family, items in by_family.items():
        ranked = sorted(items, key=lambda item: ranking_key(item[1]))
        best_run_id, best_run = ranked[0]
        best_metric, tier = ranking_key(best_run)
        best_backbone = best_run.get("backbone") or {}
        best_variant = best_run.get("variant", {})
        best_backbone_kind = str(best_backbone.get("backbone_kind", best_variant.get("backbone_kind", "unknown")))
        best_patch_encoder_kind = str(
            best_backbone.get(
                "patch_encoder_kind",
                best_variant.get("patch_encoder_kind", (best_backbone.get("config") or {}).get("patch_encoder_kind", "")),
            )
        )
        best_objective_kind = str(
            best_variant.get("objective_kind", (best_backbone.get("config") or {}).get("objective_kind", ""))
        )
        rows.append(
            {
                "family": family,
                "backbone_kind": best_backbone_kind,
                "patch_encoder_kind": best_patch_encoder_kind,
                "objective_kind": best_objective_kind,
                "best_run_id": best_run_id,
                "best_metric_bpb": best_metric,
                "ranking_tier": tier,
            }
        )
    rows.sort(key=lambda row: (math.isnan(row["best_metric_bpb"]), row["best_metric_bpb"], row["ranking_tier"]))
    return rows


def strong_full_points(runs: dict[str, dict[str, object]]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for run_id, run in runs.items():
        backbone = run.get("backbone")
        if not backbone:
            continue
        variant = run.get("variant", {})
        predict_horizons = str(variant.get("predict_horizons", backbone.get("config", {}).get("predict_horizons", "")))
        multiscale_groups = str(variant.get("multiscale_groups", backbone.get("config", {}).get("multiscale_groups", "")))
        train_shards = int(variant.get("train_shards", backbone.get("train_shards_used", 0)) or 0)
        if predict_horizons not in {"1", "(1,)"}:
            continue
        if multiscale_groups not in {"8", "(8,)"}:
            continue
        if train_shards != 10:
            continue
        for probe in run["probes"]:
            if probe.get("probe_kind") != "strong" or probe.get("probe_val_mode") != "full":
                continue
            points.append(
                {
                    "run_id": run_id,
                    "backbone_kind": str(backbone["backbone_kind"]),
                    "params": float(backbone["model_params"]),
                    "train_bytes_seen": float(probe["checkpoint_train_bytes"]),
                    "backbone_gpu_hours": float(backbone["elapsed_gpu_hours"]),
                    "probe_gpu_hours": float(probe["elapsed_gpu_hours"]),
                    "end_to_end_gpu_hours": float(backbone["elapsed_gpu_hours"]) + float(probe["elapsed_gpu_hours"]),
                    "full_val_bpb": float(probe["best_val_bpb"]),
                }
            )
    return points


def fit_two_variable_scaling(points: list[dict[str, float]], target_bpb: float) -> dict[str, object]:
    if len(points) < 4:
        return {"status": "insufficient_points", "num_points": len(points)}
    y = np.array([point["full_val_bpb"] for point in points], dtype=np.float64)
    p = np.array([point["params"] for point in points], dtype=np.float64)
    t = np.array([point["train_bytes_seen"] for point in points], dtype=np.float64)
    best: dict[str, object] | None = None
    l_candidates = np.linspace(max(0.0, float(y.min()) - 1.5), float(y.min()) - 1e-4, 30)
    alpha_candidates = np.linspace(0.05, 1.0, 24)
    beta_candidates = np.linspace(0.05, 1.0, 24)
    for l_inf in l_candidates:
        residual = y - l_inf
        if np.any(residual <= 0):
            continue
        for alpha in alpha_candidates:
            x1 = p ** (-alpha)
            for beta in beta_candidates:
                x2 = t ** (-beta)
                design = np.stack([x1, x2], axis=1)
                coeffs, _, _, _ = np.linalg.lstsq(design, residual, rcond=None)
                a, b = coeffs
                if a <= 0.0 or b <= 0.0:
                    continue
                pred = l_inf + design @ coeffs
                mse = float(np.mean((pred - y) ** 2))
                candidate = {
                    "status": "ok",
                    "l_inf": float(l_inf),
                    "a": float(a),
                    "b": float(b),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "mse": mse,
                    "num_points": len(points),
                }
                if best is None or mse < float(best["mse"]):
                    best = candidate
    if best is None:
        return {"status": "fit_failed", "num_points": len(points)}
    throughput_by_params: dict[float, list[float]] = {}
    for point in points:
        throughput = point["train_bytes_seen"] / max(point["backbone_gpu_hours"], 1e-9)
        throughput_by_params.setdefault(point["params"], []).append(throughput)
    reach_candidates: list[dict[str, float]] = []
    for params_value, throughputs in throughput_by_params.items():
        params_term = best["l_inf"] + best["a"] * (params_value ** (-best["alpha"]))
        remaining = target_bpb - params_term
        if remaining <= 0.0:
            continue
        required_bytes = (best["b"] / remaining) ** (1.0 / best["beta"])
        median_throughput = float(np.median(np.array(throughputs)))
        reach_candidates.append(
            {
                "params": float(params_value),
                "required_train_bytes": float(required_bytes),
                "estimated_backbone_gpu_hours": float(required_bytes / max(median_throughput, 1e-9)),
            }
        )
    best["reach_candidates"] = sorted(reach_candidates, key=lambda row: row["estimated_backbone_gpu_hours"])
    if reach_candidates:
        best["best_reach_candidate"] = best["reach_candidates"][0]
    else:
        best["best_reach_candidate"] = None
    return best


def seed_noise(points: list[dict[str, float]]) -> float:
    if len(points) < 2:
        return 0.0
    grouped: dict[tuple[str, float], list[float]] = {}
    for point in points:
        grouped.setdefault((point["backbone_kind"], point["params"]), []).append(point["full_val_bpb"])
    spreads = [float(np.std(np.array(values))) for values in grouped.values() if len(values) >= 2]
    return float(np.mean(np.array(spreads))) if spreads else 0.0


def fit_scaling_bundle(points: list[dict[str, float]], target_bpb: float) -> dict[str, object]:
    central = fit_two_variable_scaling(points, target_bpb)
    if central.get("status") != "ok":
        return {"target_bpb": target_bpb, "central": central}
    noise = seed_noise(points)
    optimistic_points = [{**point, "full_val_bpb": point["full_val_bpb"] - noise} for point in points]
    conservative_points = [{**point, "full_val_bpb": point["full_val_bpb"] + noise} for point in points]
    optimistic = fit_two_variable_scaling(optimistic_points, target_bpb)
    conservative = fit_two_variable_scaling(conservative_points, target_bpb)
    return {
        "target_bpb": target_bpb,
        "noise_bpb_std": noise,
        "central": central,
        "optimistic": optimistic,
        "conservative": conservative,
    }


def write_reach_report(path: Path, fit: dict[str, object], points: list[dict[str, float]]) -> None:
    lines = ["# Pure JEPA Reach Estimate", ""]
    lines.append(f"Target baseline `val_bpb`: `{SIMPLE_BASELINE_BPB:.8f}`")
    lines.append(f"Strong full-val scaling points: `{len(points)}`")
    lines.append("")
    central = fit.get("central", {})
    if central.get("status") != "ok":
        lines.append("Scaling fit status: unsupported")
        lines.append("")
        lines.append(f"Reason: `{central.get('status', 'unknown')}`")
    else:
        data_binding = "unknown"
        shard_runs = {}
        for point in points:
            shard_runs.setdefault(point["params"], []).append(point["full_val_bpb"])
        noise = float(fit.get("noise_bpb_std", 0.0))
        lines.append("Scaling fit status: supported")
        lines.append("")
        lines.append(
            f"Central fit: `L(P,T) = {central['l_inf']:.4f} + {central['a']:.4f} * P^-{central['alpha']:.3f} + "
            f"{central['b']:.4f} * T^-{central['beta']:.3f}`"
        )
        lines.append(f"Estimated fit MSE: `{central['mse']:.6f}`")
        lines.append(f"Observed seed/checkpoint noise proxy: `{noise:.4f} bpb`")
        lines.append("")
        for label in ("optimistic", "central", "conservative"):
            bundle = fit.get(label, {})
            candidate = bundle.get("best_reach_candidate") if isinstance(bundle, dict) else None
            if not candidate:
                lines.append(f"{label.title()} reach estimate: unsupported")
                continue
            lines.append(
                f"{label.title()} reach estimate: params `{int(candidate['params'])}`, "
                f"train_bytes `{candidate['required_train_bytes']:.3e}`, "
                f"backbone_gpu_hours `{candidate['estimated_backbone_gpu_hours']:.2f}`"
            )
        lines.append("")
        lines.append("Interpretation:")
        best_central = central.get("best_reach_candidate")
        if best_central is None:
            lines.append("The fitted curve does not reach the baseline within the tested size range.")
        elif best_central["estimated_backbone_gpu_hours"] <= (8 * 600 / 3600.0) * 8:
            lines.append("The fitted curve suggests the baseline may be reachable with additional scale and compute.")
        else:
            lines.append("The fitted curve suggests the baseline is still far away under the current pure-JEPA family.")
        lines.append(f"Data binding heuristic: `{data_binding}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_curves(path: Path, runs: dict[str, dict[str, object]], probe_results: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "run_id",
                "backbone_kind",
                "row_kind",
                "probe_kind",
                "probe_val_mode",
                "step",
                "train_bytes_seen",
                "loss",
                "bpb",
                "model_params",
                "backbone_gpu_hours",
                "probe_gpu_hours",
            ]
        )
        for run_id, run in runs.items():
            backbone = run.get("backbone")
            if backbone:
                for row in backbone.get("train_points", []):
                    writer.writerow(
                        [
                            run_id,
                            backbone.get("backbone_kind"),
                            "backbone_train",
                            "",
                            "",
                            row.get("step"),
                            row.get("train_bytes_seen"),
                            row.get("train_loss"),
                            "",
                            backbone.get("model_params"),
                            backbone.get("elapsed_gpu_hours"),
                            "",
                        ]
                    )
                for row in backbone.get("val_points", []):
                    writer.writerow(
                        [
                            run_id,
                            backbone.get("backbone_kind"),
                            "backbone_val",
                            "",
                            "",
                            row.get("step"),
                            row.get("train_bytes_seen"),
                            row.get("val_jepa_loss"),
                            "",
                            backbone.get("model_params"),
                            backbone.get("elapsed_gpu_hours"),
                            "",
                        ]
                    )
            for probe in run.get("probes", []):
                for row in probe.get("val_points", []):
                    writer.writerow(
                        [
                            run_id,
                            probe.get("backbone_kind"),
                            "probe_val",
                            probe.get("probe_kind"),
                            probe.get("probe_val_mode"),
                            row.get("step"),
                            probe.get("checkpoint_train_bytes"),
                            row.get("val_loss"),
                            row.get("val_bpb"),
                            backbone.get("model_params") if backbone else "",
                            backbone.get("elapsed_gpu_hours") if backbone else "",
                            probe.get("elapsed_gpu_hours"),
                        ]
                    )


def summary_payload(runs: dict[str, dict[str, object]], fit: dict[str, object]) -> dict[str, object]:
    ordered_runs = sorted(runs.items(), key=lambda item: ranking_key(item[1]))
    ranking = []
    for idx, (run_id, run) in enumerate(ordered_runs, start=1):
        full_strong = best_probe(run["probes"], "strong", "full")
        proxy_strong = best_probe(run["probes"], "strong", "proxy")
        cheap = best_probe(run["probes"], "cheap", "proxy")
        best_metric, tier = ranking_key(run)
        variant = run.get("variant", {})
        backbone = run.get("backbone") or {}
        ranking.append(
            {
                "rank": idx,
                "run_id": run_id,
                "backbone_kind": backbone.get("backbone_kind"),
                "patch_encoder_kind": variant.get("patch_encoder_kind", backbone.get("patch_encoder_kind")),
                "objective_kind": variant.get("objective_kind", (backbone.get("config") or {}).get("objective_kind")),
                "best_metric_bpb": best_metric,
                "ranking_tier": tier,
                "best_full_val_strong_bpb": full_strong.get("best_val_bpb") if full_strong else None,
                "best_proxy_strong_bpb": proxy_strong.get("best_val_bpb") if proxy_strong else None,
                "best_proxy_cheap_bpb": cheap.get("best_val_bpb") if cheap else None,
                "delta_vs_simple_baseline_bpb": (best_metric - SIMPLE_BASELINE_BPB) if math.isfinite(best_metric) else None,
            }
        )
    return {
        "simple_baseline_bpb": SIMPLE_BASELINE_BPB,
        "ranking": ranking,
        "family_ranking": family_rankings(runs),
        "runs": runs,
        "scaling_fit": fit,
    }


def self_test() -> None:
    synthetic_points = [
        {"run_id": "a", "backbone_kind": "transformer_rope_gqa_base", "params": 1e6, "train_bytes_seen": 1e8, "backbone_gpu_hours": 1.0, "probe_gpu_hours": 0.1, "end_to_end_gpu_hours": 1.1, "full_val_bpb": 3.2},
        {"run_id": "b", "backbone_kind": "transformer_rope_gqa_base", "params": 2e6, "train_bytes_seen": 1e8, "backbone_gpu_hours": 1.1, "probe_gpu_hours": 0.1, "end_to_end_gpu_hours": 1.2, "full_val_bpb": 2.9},
        {"run_id": "c", "backbone_kind": "transformer_rope_gqa_base", "params": 1e6, "train_bytes_seen": 3e8, "backbone_gpu_hours": 1.5, "probe_gpu_hours": 0.1, "end_to_end_gpu_hours": 1.6, "full_val_bpb": 2.8},
        {"run_id": "d", "backbone_kind": "transformer_rope_gqa_base", "params": 2e6, "train_bytes_seen": 3e8, "backbone_gpu_hours": 1.6, "probe_gpu_hours": 0.1, "end_to_end_gpu_hours": 1.7, "full_val_bpb": 2.5},
    ]
    fit = fit_scaling_bundle(synthetic_points, SIMPLE_BASELINE_BPB)
    central = fit["central"]
    if central.get("status") != "ok":
        raise AssertionError(f"Expected successful fit, got {central}")
    if central.get("mse", 1.0) <= 0.0:
        raise AssertionError("Expected positive fit error on synthetic data")
    print("self_test:ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", help="Phase result root, for example results/backbone_screen")
    parser.add_argument("--summary-out")
    parser.add_argument("--curves-out")
    parser.add_argument("--scaling-fit-out")
    parser.add_argument("--reach-out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.phase_root or not args.summary_out or not args.curves_out or not args.scaling_fit_out or not args.reach_out:
        raise SystemExit("Missing required output arguments")

    phase_root = Path(args.phase_root)
    runs, probe_results = load_results(phase_root)
    scaling_points = strong_full_points(runs)
    fit = fit_scaling_bundle(scaling_points, SIMPLE_BASELINE_BPB)

    write_curves(Path(args.curves_out), runs, probe_results)
    save_payload = summary_payload(runs, fit)
    Path(args.summary_out).write_text(json.dumps(save_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.scaling_fit_out).write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_reach_report(Path(args.reach_out), fit, scaling_points)


if __name__ == "__main__":
    main()
