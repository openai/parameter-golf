#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_SUBMISSION_BYTES = 16_000_000
FINAL_PREQUANT_RE = re.compile(r"final_prequant_exact val_loss:(?P<loss>\S+) val_bpb:(?P<bpb>\S+)")
FINAL_SLIDING_RE = re.compile(r"final_int6_sliding_window_exact val_loss:(?P<loss>\S+) val_bpb:(?P<bpb>\S+)")
FINAL_ROUNDTRIP_RE = re.compile(r"final_int6_roundtrip_exact val_loss:(?P<loss>\S+) val_bpb:(?P<bpb>\S+)")
TOTAL_SIZE_RE = re.compile(r"Total submission size int6\+\w+: (?P<bytes>\d+) bytes")
PAYLOAD_COMPONENTS_RE = re.compile(
    r"int6_payload_components quant_weight_bytes:(?P<quant>\d+) "
    r"scale_meta_bytes:(?P<scale>\d+) rescue_bytes:(?P<rescue>\d+) "
    r"passthrough_bytes:(?P<passthrough>\d+) payload_bytes:(?P<payload>\d+)"
)
OUTLIER_STATS_RE = re.compile(
    r"outlier_stats step:(?P<step>\d+) probe:(?P<probe>\S+) "
    r"excess_kurtosis:(?P<kurt>\S+) max_to_rms:(?P<maxrms>\S+) "
    r"channel_peak_ratio:(?P<peak>\S+)"
)


@dataclass(frozen=True)
class Choice:
    label: str
    env: dict[str, str]


MUTATION_GROUPS: OrderedDict[str, list[Choice]] = OrderedDict(
    [
        (
            "bigram",
            [
                Choice("off", {"BIGRAM_VOCAB_SIZE": "0", "BIGRAM_DIM": "0"}),
                Choice("small", {"BIGRAM_VOCAB_SIZE": "1024", "BIGRAM_DIM": "64"}),
                Choice("base", {"BIGRAM_VOCAB_SIZE": "2048", "BIGRAM_DIM": "96"}),
                Choice("large", {"BIGRAM_VOCAB_SIZE": "4096", "BIGRAM_DIM": "96"}),
            ],
        ),
        (
            "swa",
            [
                Choice("off", {"SWA_CHECKPOINTS": "0"}),
                Choice("mid", {"SWA_CHECKPOINTS": "5"}),
                Choice("base", {"SWA_CHECKPOINTS": "7"}),
            ],
        ),
        (
            "adam_wd",
            [
                Choice("zero", {"ADAM_WEIGHT_DECAY": "0.0"}),
                Choice("mid", {"ADAM_WEIGHT_DECAY": "0.005"}),
                Choice("base", {"ADAM_WEIGHT_DECAY": "0.01"}),
            ],
        ),
        (
            "muon_wd",
            [
                Choice("low", {"WEIGHT_DECAY": "0.034"}),
                Choice("base", {"WEIGHT_DECAY": "0.038"}),
                Choice("high", {"WEIGHT_DECAY": "0.042"}),
            ],
        ),
        (
            "outlier_penalty",
            [
                Choice("off", {"OUTLIER_PENALTY_COEF": "0.0"}),
                Choice(
                    "base",
                    {
                        "OUTLIER_PENALTY_COEF": "0.0001",
                        "OUTLIER_PENALTY_START_FRAC": "0.70",
                        "OUTLIER_TARGET_KURTOSIS": "6.0",
                    },
                ),
                Choice(
                    "strong",
                    {
                        "OUTLIER_PENALTY_COEF": "0.0003",
                        "OUTLIER_PENALTY_START_FRAC": "0.60",
                        "OUTLIER_TARGET_KURTOSIS": "5.0",
                    },
                ),
            ],
        ),
        (
            "rescue_mode",
            [
                Choice("tensor", {"AUTO_RESCUE_MODE": "tensor"}),
                Choice("base", {"AUTO_RESCUE_MODE": "off"}),
                Choice("row", {"AUTO_RESCUE_MODE": "row"}),
            ],
        ),
        (
            "rescue_bytes",
            [
                Choice("zero", {"AUTO_RESCUE_BYTES": "0"}),
                Choice("base", {"AUTO_RESCUE_BYTES": "65536"}),
                Choice("high", {"AUTO_RESCUE_BYTES": "131072"}),
            ],
        ),
        (
            "rescue_topk",
            [
                Choice("small", {"AUTO_RESCUE_TOPK": "8"}),
                Choice("base", {"AUTO_RESCUE_TOPK": "16"}),
                Choice("large", {"AUTO_RESCUE_TOPK": "32"}),
            ],
        ),
        (
            "scale_mode",
            [
                Choice("row", {"INT6_SCALE_MODE": "row"}),
                Choice("base", {"INT6_SCALE_MODE": "group8"}),
                Choice("group16", {"INT6_SCALE_MODE": "group16"}),
            ],
        ),
        (
            "grad_clip",
            [
                Choice("low", {"GRAD_CLIP_NORM": "0.2"}),
                Choice("base", {"GRAD_CLIP_NORM": "0.3"}),
                Choice("high", {"GRAD_CLIP_NORM": "0.4"}),
            ],
        ),
    ]
)

BASE_SELECTIONS = {group: "base" for group in MUTATION_GROUPS}


def candidate_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return candidate_dir().parents[2]


def parse_name_value(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty environment variable name in {item!r}")
        env[name] = value
    return env


def default_fixed_env(nproc_per_node: int, seed: int, max_wallclock_seconds: int) -> dict[str, str]:
    root = repo_root()
    conservative_single_gpu = nproc_per_node == 1
    return {
        "DATA_PATH": str(root / "data" / "datasets" / "fineweb10B_sp1024"),
        "TOKENIZER_PATH": str(root / "data" / "tokenizers" / "fineweb_1024_bpe.model"),
        "SEED": str(seed),
        "NUM_LAYERS": "11",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3.0",
        "TRAIN_SEQ_LEN": "2048",
        "TRAIN_BATCH_TOKENS": "262144" if conservative_single_gpu else "524288",
        "VAL_BATCH_SIZE": "131072" if conservative_single_gpu else "524288",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "50",
        "MATRIX_LR": "0.025",
        "SCALAR_LR": "0.025",
        "TIED_EMBED_LR": "0.035",
        "MUON_MOMENTUM": "0.99",
        "MUON_MOMENTUM_WARMUP_START": "0.92",
        "MUON_MOMENTUM_WARMUP_STEPS": "1500",
        "WARMDOWN_ITERS": "3000",
        "EVAL_STRIDE": "64",
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        "OUTLIER_LOG_EVERY": "100",
        "OUTLIER_PROBE_LAYERS": "auto",
        "AUTO_RESCUE_CANDIDATES": "auto",
        "AUTO_RESCUE_CALIB_BATCHES": "1",
        "INT6_SCALE_GROUP_ROWS": "8",
    }


def resolve_selection_map(selections: dict[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for group_name, choices in MUTATION_GROUPS.items():
        by_label = {choice.label: choice for choice in choices}
        env.update(by_label[selections[group_name]].env)
    return env


def config_fingerprint(env: dict[str, str]) -> str:
    payload = json.dumps(env, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def trial_name_from_selections(selections: dict[str, str]) -> str:
    parts: list[str] = []
    for group_name, base_label in BASE_SELECTIONS.items():
        label = selections[group_name]
        if label != base_label:
            parts.append(f"{group_name}={label}")
    return "base" if not parts else ",".join(parts)


def change_count(selections: dict[str, str]) -> int:
    return sum(int(selections[group] != base) for group, base in BASE_SELECTIONS.items())


def build_trial_env(
    selections: dict[str, str],
    fixed_env: dict[str, str],
    extra_env: dict[str, str],
    run_id: str,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(fixed_env)
    env.update(resolve_selection_map(selections))
    env.update(extra_env)
    env["RUN_ID"] = run_id
    return env


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return results


def append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")


def is_valid_result(result: dict[str, Any]) -> bool:
    return result.get("status") == "ok"


def sort_key(result: dict[str, Any]) -> tuple[int, float, int, float, float, float]:
    valid_rank = 0 if is_valid_result(result) else 1
    metrics = result.get("metrics", {})
    bpb = float(metrics.get("sliding_val_bpb", float("inf")))
    total_bytes = int(metrics.get("total_submission_bytes", 10**18))
    qid = float(metrics.get("qid", float("inf")))
    outlier_kurt = float(metrics.get("outlier_max_excess_kurtosis", float("inf")))
    elapsed = float(result.get("elapsed_seconds", 10**18))
    return (valid_rank, bpb, total_bytes, qid, outlier_kurt, elapsed)


def write_leaderboard(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Budget Autoresearch Leaderboard",
        "",
        "| Rank | Status | Slide | PreQ | QID | Bytes | Rescue | Kurt | Minutes | Cost | Trial | Changes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, result in enumerate(sorted(results, key=sort_key), 1):
        metrics = result.get("metrics", {})
        score = metrics.get("sliding_val_bpb")
        score_str = f"{score:.6f}" if isinstance(score, (int, float)) else "-"
        preq = metrics.get("prequant_val_bpb")
        preq_str = f"{preq:.6f}" if isinstance(preq, (int, float)) else "-"
        qid = metrics.get("qid")
        qid_str = f"{qid:.6f}" if isinstance(qid, (int, float)) else "-"
        bytes_total = metrics.get("total_submission_bytes")
        bytes_str = str(bytes_total) if bytes_total is not None else "-"
        rescue_bytes = metrics.get("rescue_bytes")
        rescue_str = str(rescue_bytes) if rescue_bytes is not None else "-"
        kurt = metrics.get("outlier_max_excess_kurtosis")
        kurt_str = f"{kurt:.3f}" if isinstance(kurt, (int, float)) else "-"
        minutes = result.get("elapsed_seconds")
        minutes_str = f"{minutes / 60.0:.1f}" if isinstance(minutes, (int, float)) else "-"
        cost = result.get("estimated_cost_usd")
        cost_str = f"{cost:.2f}" if isinstance(cost, (int, float)) else "-"
        trial = result.get("trial_name", "-")
        changes = result.get("change_count", 0)
        lines.append(
            f"| {idx} | {result.get('status', '-')} | {score_str} | {preq_str} | {qid_str} | {bytes_str} | {rescue_str} | {kurt_str} | {minutes_str} | {cost_str} | {trial} | {changes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_metrics(log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {"outlier_by_probe": {}}
    for line in log_text.splitlines():
        prequant = FINAL_PREQUANT_RE.search(line)
        if prequant:
            metrics["prequant_val_loss"] = float(prequant.group("loss"))
            metrics["prequant_val_bpb"] = float(prequant.group("bpb"))
        slide = FINAL_SLIDING_RE.search(line)
        if slide:
            metrics["sliding_val_loss"] = float(slide.group("loss"))
            metrics["sliding_val_bpb"] = float(slide.group("bpb"))
        roundtrip = FINAL_ROUNDTRIP_RE.search(line)
        if roundtrip:
            metrics["roundtrip_val_loss"] = float(roundtrip.group("loss"))
            metrics["roundtrip_val_bpb"] = float(roundtrip.group("bpb"))
        total = TOTAL_SIZE_RE.search(line)
        if total:
            metrics["total_submission_bytes"] = int(total.group("bytes"))
        payload = PAYLOAD_COMPONENTS_RE.search(line)
        if payload:
            metrics["quant_weight_bytes"] = int(payload.group("quant"))
            metrics["scale_meta_bytes"] = int(payload.group("scale"))
            metrics["rescue_bytes"] = int(payload.group("rescue"))
            metrics["passthrough_bytes"] = int(payload.group("passthrough"))
            metrics["payload_bytes"] = int(payload.group("payload"))
        outlier = OUTLIER_STATS_RE.search(line)
        if outlier:
            probe = outlier.group("probe")
            stats = {
                "step": int(outlier.group("step")),
                "excess_kurtosis": float(outlier.group("kurt")),
                "max_to_rms": float(outlier.group("maxrms")),
                "channel_peak_ratio": float(outlier.group("peak")),
            }
            metrics["outlier_by_probe"][probe] = stats
            metrics["outlier_max_excess_kurtosis"] = max(
                float(metrics.get("outlier_max_excess_kurtosis", float("-inf"))),
                stats["excess_kurtosis"],
            )
            metrics["outlier_max_max_to_rms"] = max(
                float(metrics.get("outlier_max_max_to_rms", float("-inf"))),
                stats["max_to_rms"],
            )
            metrics["outlier_max_channel_peak_ratio"] = max(
                float(metrics.get("outlier_max_channel_peak_ratio", float("-inf"))),
                stats["channel_peak_ratio"],
            )
    if "prequant_val_bpb" in metrics and "roundtrip_val_bpb" in metrics:
        metrics["qid"] = float(metrics["roundtrip_val_bpb"]) - float(metrics["prequant_val_bpb"])
    if not metrics["outlier_by_probe"]:
        metrics.pop("outlier_by_probe")
    return metrics


def trial_status(returncode: int, metrics: dict[str, Any]) -> str:
    if returncode != 0:
        return "failed"
    if "sliding_val_bpb" not in metrics:
        return "missing_metric"
    if metrics.get("total_submission_bytes", MAX_SUBMISSION_BYTES + 1) > MAX_SUBMISSION_BYTES:
        return "oversize"
    return "ok"


def estimate_run_cost(hourly_rate: float, max_wallclock_seconds: int, eval_buffer_seconds: int) -> float:
    return hourly_rate * ((max_wallclock_seconds + eval_buffer_seconds) / 3600.0)


def branch_should_prune(result: dict[str, Any], results_by_fingerprint: dict[str, dict[str, Any]]) -> bool:
    parent_fingerprint = result.get("parent_fingerprint")
    if not parent_fingerprint:
        return False
    parent = results_by_fingerprint.get(parent_fingerprint)
    if not parent or not is_valid_result(result) or not is_valid_result(parent):
        return False
    metrics = result.get("metrics", {})
    parent_metrics = parent.get("metrics", {})
    prequant = metrics.get("prequant_val_bpb")
    parent_prequant = parent_metrics.get("prequant_val_bpb")
    qid = metrics.get("qid")
    parent_qid = parent_metrics.get("qid")
    total_bytes = metrics.get("total_submission_bytes")
    parent_bytes = parent_metrics.get("total_submission_bytes")
    if not all(isinstance(value, (int, float)) for value in (prequant, parent_prequant, qid, parent_qid, total_bytes, parent_bytes)):
        return False
    improved_prequant = float(prequant) < float(parent_prequant) - 0.0005
    worse_qid = float(qid) > float(parent_qid) + 0.0005
    worse_bytes = int(total_bytes) > int(parent_bytes) + 8192
    return improved_prequant and worse_qid and worse_bytes


def top_beam(
    results: list[dict[str, Any]],
    beam_width: int,
    max_depth: int,
    results_by_fingerprint: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [
        r
        for r in results
        if is_valid_result(r)
        and int(r.get("change_count", 0)) < max_depth
        and not branch_should_prune(r, results_by_fingerprint)
    ]
    return sorted(candidates, key=sort_key)[:beam_width]


def next_states(
    results: list[dict[str, Any]],
    seen_fingerprints: set[str],
    fixed_env: dict[str, str],
    extra_env: dict[str, str],
    beam_width: int,
    max_depth: int,
) -> list[dict[str, Any]]:
    if not results:
        return [{"selections": dict(BASE_SELECTIONS), "parent_fingerprint": None}]

    valid_results = [r for r in results if is_valid_result(r)]
    if not valid_results:
        base_state = dict(BASE_SELECTIONS)
        children: list[dict[str, Any]] = []
        for group_name, choices in MUTATION_GROUPS.items():
            for choice in choices:
                if choice.label == BASE_SELECTIONS[group_name]:
                    continue
                state = dict(base_state)
                state[group_name] = choice.label
                env = {**fixed_env, **resolve_selection_map(state), **extra_env}
                if config_fingerprint(env) not in seen_fingerprints:
                    children.append({"selections": state, "parent_fingerprint": None})
        return children

    results_by_fingerprint = {config_fingerprint(r["env"]): r for r in results if "env" in r}
    parents = top_beam(results, beam_width=beam_width, max_depth=max_depth, results_by_fingerprint=results_by_fingerprint)
    proposed: list[dict[str, Any]] = []
    proposed_fingerprints: set[str] = set()
    for parent in parents:
        parent_state = dict(parent["selections"])
        parent_fingerprint = config_fingerprint(parent["env"])
        for group_name, choices in MUTATION_GROUPS.items():
            if parent_state[group_name] != BASE_SELECTIONS[group_name]:
                continue
            for choice in choices:
                if choice.label == BASE_SELECTIONS[group_name]:
                    continue
                state = dict(parent_state)
                state[group_name] = choice.label
                if change_count(state) > max_depth:
                    continue
                env = {**fixed_env, **resolve_selection_map(state), **extra_env}
                fingerprint = config_fingerprint(env)
                if fingerprint in seen_fingerprints or fingerprint in proposed_fingerprints:
                    continue
                proposed.append({"selections": state, "parent_fingerprint": parent_fingerprint})
                proposed_fingerprints.add(fingerprint)
    return proposed


def run_trial(
    trial_index: int,
    selections: dict[str, str],
    parent_fingerprint: str | None,
    args: argparse.Namespace,
    fixed_env: dict[str, str],
    extra_env: dict[str, str],
) -> dict[str, Any]:
    run_id = f"search_{trial_index:03d}_{int(time.time())}"
    trial_name = trial_name_from_selections(selections)
    env_for_fingerprint = {**fixed_env, **resolve_selection_map(selections), **extra_env}
    fingerprint = config_fingerprint(env_for_fingerprint)
    trial_dir = candidate_dir() / "autotune_runs" / f"trial_{trial_index:03d}_{fingerprint[:8]}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    env = build_trial_env(selections, fixed_env, extra_env, run_id)
    command = [
        args.torchrun,
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(candidate_dir() / "train_gpt.py"),
    ]

    print(f"\n=== Trial {trial_index}: {trial_name} ===")
    print(" ".join(command))
    print(f"cwd={trial_dir}")

    started_at = time.time()
    proc = subprocess.Popen(
        command,
        cwd=trial_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_lines.append(line)
    returncode = proc.wait()
    elapsed_seconds = time.time() - started_at
    full_log = "".join(captured_lines)
    metrics = parse_metrics(full_log)
    status = trial_status(returncode, metrics)
    estimated_cost = elapsed_seconds / 3600.0 * args.hourly_rate

    result = {
        "trial_index": trial_index,
        "trial_name": trial_name,
        "run_id": run_id,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "estimated_cost_usd": estimated_cost,
        "nproc_per_node": args.nproc_per_node,
        "selections": selections,
        "change_count": change_count(selections),
        "env": env_for_fingerprint,
        "parent_fingerprint": parent_fingerprint,
        "metrics": metrics,
        "trial_dir": str(trial_dir),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
    }
    (trial_dir / "trial_output.txt").write_text(full_log, encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-capped autoresearch loop for the 11L candidate.")
    parser.add_argument("--budget-dollars", type=float, default=10.0, help="Soft budget for the search loop.")
    parser.add_argument("--hourly-rate", type=float, default=2.69, help="GPU hourly rate in USD.")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun nproc_per_node.")
    parser.add_argument("--seed", type=int, default=1337, help="Search seed.")
    parser.add_argument("--max-wallclock-seconds", type=int, default=600, help="Per-trial training wallclock cap.")
    parser.add_argument("--eval-buffer-seconds", type=int, default=240, help="Extra budget reserved per trial before launching the next run.")
    parser.add_argument("--beam-width", type=int, default=3, help="How many top configs to expand at each depth.")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum number of mutated groups from the base config.")
    parser.add_argument("--max-runs", type=int, default=24, help="Hard cap on number of trials launched by this process.")
    parser.add_argument("--torchrun", default="torchrun", help="torchrun executable to use.")
    parser.add_argument("--extra-env", action="append", default=[], help="Extra NAME=VALUE overrides. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned next candidates and exit.")
    parser.add_argument("--reset", action="store_true", help="Clear previous autotune results before starting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = candidate_dir() / "autotune_runs"
    results_path = runs_dir / "results.jsonl"
    leaderboard_path = runs_dir / "leaderboard.md"
    extra_env = parse_name_value(args.extra_env)
    fixed_env = default_fixed_env(
        nproc_per_node=args.nproc_per_node,
        seed=args.seed,
        max_wallclock_seconds=args.max_wallclock_seconds,
    )

    if args.reset:
        if results_path.exists():
            results_path.unlink()
        if leaderboard_path.exists():
            leaderboard_path.unlink()

    results = load_results(results_path)
    spent = sum(float(r.get("estimated_cost_usd", 0.0)) for r in results)
    seen = {config_fingerprint(r["env"]) for r in results if "env" in r}
    planned_run_cost = estimate_run_cost(
        hourly_rate=args.hourly_rate,
        max_wallclock_seconds=args.max_wallclock_seconds,
        eval_buffer_seconds=args.eval_buffer_seconds,
    )

    print(f"candidate_dir={candidate_dir()}")
    print(f"budget_dollars={args.budget_dollars:.2f} spent_so_far={spent:.2f} planned_run_cost={planned_run_cost:.2f}")
    print(f"nproc_per_node={args.nproc_per_node} max_runs={args.max_runs} beam_width={args.beam_width} max_depth={args.max_depth}")

    next_queue = next_states(
        results=results,
        seen_fingerprints=seen,
        fixed_env=fixed_env,
        extra_env=extra_env,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
    )
    if args.dry_run:
        for idx, queued in enumerate(next_queue, 1):
            state = queued["selections"]
            env = {**fixed_env, **resolve_selection_map(state), **extra_env}
            print(
                f"{idx:02d} {trial_name_from_selections(state)} "
                f"parent={queued.get('parent_fingerprint')} {json.dumps(env, sort_keys=True)}"
            )
        return

    trial_index = len(results) + 1
    while next_queue and trial_index <= args.max_runs:
        if spent + planned_run_cost > args.budget_dollars:
            print(
                f"Stopping before trial {trial_index}: soft budget would be exceeded "
                f"({spent:.2f} + {planned_run_cost:.2f} > {args.budget_dollars:.2f})"
            )
            break

        queued = next_queue.pop(0)
        state = queued["selections"]
        result = run_trial(
            trial_index=trial_index,
            selections=state,
            parent_fingerprint=queued.get("parent_fingerprint"),
            args=args,
            fixed_env=fixed_env,
            extra_env=extra_env,
        )
        current_by_fingerprint = {config_fingerprint(r["env"]): r for r in results if "env" in r}
        result["pruned_branch"] = branch_should_prune(result, current_by_fingerprint)
        append_result(results_path, result)
        results.append(result)
        write_leaderboard(leaderboard_path, results)

        spent += float(result.get("estimated_cost_usd", 0.0))
        seen.add(config_fingerprint(result["env"]))
        trial_index += 1
        next_queue = next_states(
            results=results,
            seen_fingerprints=seen,
            fixed_env=fixed_env,
            extra_env=extra_env,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
        )

    print("\nTop results:")
    for idx, result in enumerate(sorted(results, key=sort_key)[:5], 1):
        metrics = result.get("metrics", {})
        score = metrics.get("sliding_val_bpb")
        bytes_total = metrics.get("total_submission_bytes")
        qid = metrics.get("qid")
        print(
            f"{idx}. {result.get('trial_name')} "
            f"status={result.get('status')} "
            f"val_bpb={score if score is not None else '-'} "
            f"qid={qid if qid is not None else '-'} "
            f"bytes={bytes_total if bytes_total is not None else '-'} "
            f"cost=${result.get('estimated_cost_usd', 0.0):.2f}"
        )
    print(f"\nLeaderboard written to {leaderboard_path}")


if __name__ == "__main__":
    main()
