#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SPACE = {
    "TIED_EMBED_LR": [0.04, 0.05, 0.06],
    "MATRIX_LR": [0.035, 0.04, 0.045],
    "SCALAR_LR": [0.03, 0.04, 0.05],
    "QK_GAIN_INIT": [1.25, 1.5, 1.75],
    "LOGIT_SOFTCAP": [20.0, 30.0, 40.0],
    "TIED_EMBED_INIT_STD": [0.003, 0.005, 0.008],
}

BASELINE = {
    "TIED_EMBED_LR": 0.05,
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.04,
    "QK_GAIN_INIT": 1.5,
    "LOGIT_SOFTCAP": 30.0,
    "TIED_EMBED_INIT_STD": 0.005,
}

ARCH_BASELINE = {
    "TIE_EMBEDDINGS": 1,
    "VOCAB_SIZE": 1024,
    "NUM_LAYERS": 9,
    "MODEL_DIM": 512,
    "NUM_HEADS": 8,
    "NUM_KV_HEADS": 4,
    "MLP_MULT": 2,
}

FINAL_METRIC_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
TRAIN_LINE_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+) train_loss:(?P<train_loss>[0-9.]+) "
    r"train_time:(?P<train_time_ms>[0-9.]+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")
INT8_ZLIB_RE = re.compile(
    r"serialized_model_int8_zlib:(?P<int8_zlib_bytes>\d+) bytes "
    r"\(payload:(?P<int8_payload_bytes>\d+) raw_(?:pickle|torch):(?P<int8_raw_bytes>\d+) "
    r"payload_ratio:(?P<payload_ratio>[0-9.]+)x\)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local MLX search harness for Parameter Golf")
    parser.add_argument("--train-script", default="train_gpt_mlx.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--out-dir", default="./experiments")
    parser.add_argument("--session-name", default="")
    parser.add_argument("--strategy", choices=("hybrid", "grid", "random"), default="hybrid")
    parser.add_argument("--space-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--train-batch-tokens", type=int, default=8192)
    parser.add_argument("--val-batch-size", type=int, default=65536)
    parser.add_argument("--full-val-batch-size", type=int, default=131072)
    parser.add_argument("--val-max-seqs", type=int, default=128)
    parser.add_argument("--train-max-shards", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--train-log-every", type=int, default=20)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--mlx-max-microbatch-tokens", type=int, default=8192)
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--promote-top-k", type=int, default=2)
    parser.add_argument("--promote-budget", type=int, default=4)
    parser.add_argument("--full-val-top-k", type=int, default=0)
    parser.add_argument("--max-param-count", type=int, default=0)
    parser.add_argument("--max-int8-zlib-bytes", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--space", action="append", default=[], metavar="KEY=V1,V2,V3")
    return parser


def coerce_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return "1" if lowered == "true" else "0"
    try:
        if "." in raw or "e" in lowered:
            value = float(raw)
            return int(value) if value.is_integer() else value
        return int(raw)
    except ValueError:
        return raw


def parse_overrides(items: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        overrides[key.strip().upper()] = coerce_value(raw_value.strip())
    return overrides


def parse_space_overrides(items: list[str]) -> dict[str, list[object]]:
    overrides: dict[str, list[object]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=V1,V2,V3, got {item!r}")
        key, raw_values = item.split("=", 1)
        values = [coerce_value(part.strip()) for part in raw_values.split(",") if part.strip()]
        if not values:
            raise ValueError(f"space override for {key.strip().upper()} must include at least one value")
        overrides[key.strip().upper()] = values
    return overrides


def value_key(value: object) -> tuple[int, object]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


def normalize_signature(candidate: dict[str, object]) -> tuple[tuple[str, object], ...]:
    effective = dict(ARCH_BASELINE)
    effective.update(BASELINE)
    effective.update(candidate)
    return tuple(sorted(effective.items()))


def make_session_dir(root: Path, session_name: str) -> Path:
    if session_name:
        name = session_name
    else:
        name = time.strftime("local_search_%Y%m%d_%H%M%S")
    session_dir = root / name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def candidate_name(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:03d}"


def generate_grid(space: dict[str, list[object]]) -> list[dict[str, object]]:
    keys = list(space)
    values = [space[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def merged_arch(candidate: dict[str, object]) -> dict[str, object]:
    arch = dict(ARCH_BASELINE)
    arch.update({key: value for key, value in candidate.items() if key in ARCH_BASELINE})
    return arch


def estimate_model_params(candidate: dict[str, object]) -> int | None:
    arch = merged_arch(candidate)
    try:
        tie_embeddings = int(arch["TIE_EMBEDDINGS"])
        vocab_size = int(arch["VOCAB_SIZE"])
        num_layers = int(arch["NUM_LAYERS"])
        model_dim = int(arch["MODEL_DIM"])
        num_heads = int(arch["NUM_HEADS"])
        num_kv_heads = int(arch["NUM_KV_HEADS"])
        mlp_mult = int(arch["MLP_MULT"])
    except (TypeError, ValueError):
        return None

    if tie_embeddings != 1:
        return None
    if min(vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult) <= 0:
        return None
    if model_dim % num_heads != 0:
        return None
    head_dim = model_dim // num_heads
    if head_dim % 2 != 0:
        return None
    if num_heads % num_kv_heads != 0:
        return None

    kv_dim = num_kv_heads * head_dim
    per_block_matrix = (
        model_dim * model_dim
        + model_dim * kv_dim
        + model_dim * kv_dim
        + model_dim * model_dim
        + model_dim * (model_dim * mlp_mult)
        + (model_dim * mlp_mult) * model_dim
    )
    per_block_control = 4 * model_dim + num_heads
    skip_params = (num_layers // 2) * model_dim
    return (
        vocab_size * model_dim
        + num_layers * per_block_matrix
        + num_layers * per_block_control
        + skip_params
    )


def validate_candidate(candidate: dict[str, object]) -> str | None:
    estimated = estimate_model_params(candidate)
    if estimated is None:
        return "invalid_architecture"
    return None


def within_budget(result: dict[str, object], args: argparse.Namespace) -> bool:
    if args.max_param_count > 0:
        model_params = result.get("model_params") or result.get("estimated_model_params")
        if model_params is not None and int(model_params) > args.max_param_count:
            return False
    if args.max_int8_zlib_bytes > 0:
        int8_zlib_bytes = result.get("int8_zlib_bytes")
        if int8_zlib_bytes is not None and int(int8_zlib_bytes) > args.max_int8_zlib_bytes:
            return False
    return True


def rankable_results(results: list[dict[str, object]], args: argparse.Namespace, *, full_val: bool) -> list[dict[str, object]]:
    return [
        result
        for result in results
        if result.get("status") == "ok" and bool(result.get("full_val")) == full_val and within_budget(result, args)
    ]


def sample_random(
    space: dict[str, list[object]],
    rng: random.Random,
    budget: int,
    seen: set[tuple[tuple[str, object], ...]],
) -> list[dict[str, object]]:
    keys = list(space)
    candidates: list[dict[str, object]] = []
    attempts = 0
    max_attempts = max(100, budget * 20)
    while len(candidates) < budget and attempts < max_attempts:
        attempts += 1
        candidate = {key: rng.choice(space[key]) for key in keys}
        sig = normalize_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(candidate)
    return candidates


def promote_neighbors(
    results: list[dict[str, object]],
    space: dict[str, list[object]],
    promote_top_k: int,
    budget: int,
    args: argparse.Namespace,
    seen: set[tuple[tuple[str, object], ...]],
) -> list[dict[str, object]]:
    ranked = rankable_results(results, args, full_val=False)
    ranked.sort(key=lambda item: item["val_bpb"])
    promoted: list[dict[str, object]] = []
    for result in ranked[:promote_top_k]:
        base = dict(result["candidate"])
        for key, values in space.items():
            ordered = sorted(values, key=value_key)
            current = base[key]
            try:
                idx = ordered.index(current)
            except ValueError:
                continue
            for neighbor_idx in (idx - 1, idx + 1):
                if not (0 <= neighbor_idx < len(ordered)):
                    continue
                candidate = dict(base)
                candidate[key] = ordered[neighbor_idx]
                sig = normalize_signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                promoted.append(candidate)
                if len(promoted) >= budget:
                    return promoted
    return promoted


def parse_metrics(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8")
    final_matches = list(FINAL_METRIC_RE.finditer(text))
    if not final_matches:
        raise ValueError(f"missing final roundtrip metric in {log_path}")
    final = final_matches[-1]
    train_matches = list(TRAIN_LINE_RE.finditer(text))
    train_line = train_matches[-1] if train_matches else None
    model_params_match = MODEL_PARAMS_RE.search(text)
    int8_match = INT8_ZLIB_RE.search(text)
    payload_ratio = float(int8_match.group("payload_ratio")) if int8_match else None
    return {
        "val_loss": float(final.group("val_loss")),
        "val_bpb": float(final.group("val_bpb")),
        "step": int(train_line.group("step")) if train_line else None,
        "train_loss": float(train_line.group("train_loss")) if train_line else None,
        "train_time_ms": float(train_line.group("train_time_ms")) if train_line else None,
        "step_avg_ms": float(train_line.group("step_avg_ms")) if train_line else None,
        "model_params": int(model_params_match.group("model_params")) if model_params_match else None,
        "int8_zlib_bytes": int(int8_match.group("int8_zlib_bytes")) if int8_match else None,
        "int8_payload_bytes": int(int8_match.group("int8_payload_bytes")) if int8_match else None,
        "int8_raw_bytes": int(int8_match.group("int8_raw_bytes")) if int8_match else None,
        "payload_ratio": payload_ratio,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(
    path: Path,
    results: list[dict[str, object]],
    session_dir: Path,
    args: argparse.Namespace,
) -> None:
    ranked = sorted(
        [result for result in results if result.get("status") == "ok"],
        key=lambda item: (item["val_bpb"], item["name"]),
    )
    lines = [
        "# Local Search Report",
        "",
        f"- Session: `{session_dir.name}`",
        f"- Strategy: `{args.strategy}`",
        f"- Search metric: `final_int8_zlib_roundtrip_exact val_bpb`",
        f"- Proxy validation subset: `{args.val_max_seqs}` sequences",
        f"- Full-val confirmations: `{args.full_val_top_k}`",
        f"- Max param count: `{args.max_param_count or 'off'}`",
        f"- Max int8+zlib bytes: `{args.max_int8_zlib_bytes or 'off'}`",
        "",
        "## Ranked Runs",
        "",
        "| Rank | Run | Score (val_bpb) | Params | Int8+zlib bytes | Budget | Full Val | Status | Candidate |",
        "| ---: | --- | ---: | ---: | ---: | :---: | :---: | --- | --- |",
    ]
    for idx, result in enumerate(ranked, start=1):
        model_params = result.get("model_params") or result.get("estimated_model_params")
        int8_zlib_bytes = result.get("int8_zlib_bytes")
        budget_ok = "yes" if within_budget(result, args) else "no"
        lines.append(
            "| "
            f"{idx} | {result['name']} | {result['val_bpb']:.6f} | {model_params if model_params is not None else '-'} | "
            f"{int8_zlib_bytes if int8_zlib_bytes is not None else '-'} | {budget_ok} | "
            f"{'yes' if result.get('full_val') else 'no'} | {result['status']} | "
            f"`{json.dumps(result['candidate'], sort_keys=True)}` |"
        )
    if not ranked:
        lines.append("| - | - | - | - | - | - | - | no successful runs | - |")

    skipped = [result for result in results if result.get("status") != "ok"]
    if skipped:
        lines.extend(["", "## Skipped / Failed", ""])
        for result in skipped:
            lines.append(
                f"- `{result['name']}`: `{result['status']}` candidate=`{json.dumps(result['candidate'], sort_keys=True)}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidate(
    *,
    repo_root: Path,
    session_dir: Path,
    args: argparse.Namespace,
    candidate: dict[str, object],
    name: str,
    full_val: bool,
) -> dict[str, object]:
    log_dir = session_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = name if not full_val else f"{name}_fullval"
    log_path = log_dir / f"{run_id}.txt"
    launcher_path = log_dir / f"{run_id}.stdout.txt"
    estimated_model_params = estimate_model_params(candidate)
    invalid_reason = validate_candidate(candidate)
    if invalid_reason is not None:
        return {
            "name": run_id,
            "candidate": candidate,
            "status": invalid_reason,
            "returncode": None,
            "full_val": full_val,
            "duration_sec": 0.0,
            "log_path": str(log_path),
            "stdout_path": str(launcher_path),
            "estimated_model_params": estimated_model_params,
        }
    if args.max_param_count > 0 and estimated_model_params is not None and estimated_model_params > args.max_param_count:
        return {
            "name": run_id,
            "candidate": candidate,
            "status": "skipped_param_budget",
            "returncode": None,
            "full_val": full_val,
            "duration_sec": 0.0,
            "log_path": str(log_path),
            "stdout_path": str(launcher_path),
            "estimated_model_params": estimated_model_params,
        }

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "OUT_DIR": str(log_dir),
            "DATA_PATH": args.data_path,
            "TOKENIZER_PATH": args.tokenizer_path,
            "ITERATIONS": str(args.iterations),
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "VAL_BATCH_SIZE": str(args.full_val_batch_size if full_val else args.val_batch_size),
            "VAL_LOSS_EVERY": "0",
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "WARMUP_STEPS": str(args.warmup_steps),
            "TRAIN_MAX_SHARDS": str(args.train_max_shards),
            "VAL_MAX_SEQS": "0" if full_val else str(args.val_max_seqs),
            "GRAD_ACCUM_STEPS": str(args.grad_accum_steps),
            "MLX_MAX_MICROBATCH_TOKENS": str(args.mlx_max_microbatch_tokens),
            "MAX_WALLCLOCK_SECONDS": "0",
        }
    )
    for key, value in candidate.items():
        env[key] = str(value)

    cmd = [args.python, args.train_script]
    started_at = time.time()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    launcher_path.write_text(proc.stdout, encoding="utf-8")
    result = {
        "name": run_id,
        "candidate": candidate,
        "status": "ok" if proc.returncode == 0 and log_path.is_file() else "failed",
        "returncode": proc.returncode,
        "full_val": full_val,
        "duration_sec": round(time.time() - started_at, 3),
        "log_path": str(log_path),
        "stdout_path": str(launcher_path),
        "estimated_model_params": estimated_model_params,
    }
    if result["status"] == "ok":
        result.update(parse_metrics(log_path))
    return result


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    session_root = Path(args.out_dir).resolve()
    session_dir = make_session_dir(session_root, args.session_name)
    baseline = dict(BASELINE)
    overrides = parse_overrides(args.set)
    space_overrides = parse_space_overrides(args.space)
    space = {} if args.space_only else {key: list(values) for key, values in DEFAULT_SPACE.items()}
    for key, values in space_overrides.items():
        space[key] = values
    for key, value in overrides.items():
        baseline[key] = value
        if key not in space:
            space[key] = [value]
        elif value not in space[key]:
            space[key] = sorted(space[key] + [value], key=value_key)

    rng = random.Random(args.seed)
    results: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, object], ...]] = {normalize_signature(baseline)}
    queue: list[dict[str, object]] = [dict(baseline)]

    if args.strategy == "grid":
        for candidate in generate_grid(space):
            sig = normalize_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            queue.append(candidate)
    elif args.strategy == "random":
        queue.extend(sample_random(space, rng, max(args.max_runs - 1, 0), seen))
    else:
        queue.extend(sample_random(space, rng, max(args.max_runs - 1, 0), seen))

    queue = queue[: args.max_runs]
    for idx, candidate in enumerate(queue, start=1):
        result = run_candidate(
            repo_root=repo_root,
            session_dir=session_dir,
            args=args,
            candidate=candidate,
            name=candidate_name("search", idx),
            full_val=False,
        )
        results.append(result)
        write_json(session_dir / "results.json", results)
        write_markdown_report(session_dir / "REPORT.md", results, session_dir, args)

    if args.strategy == "hybrid":
        promoted = promote_neighbors(results, space, args.promote_top_k, args.promote_budget, args, seen)
        for idx, candidate in enumerate(promoted, start=1):
            result = run_candidate(
                repo_root=repo_root,
                session_dir=session_dir,
                args=args,
                candidate=candidate,
                name=candidate_name("promote", idx),
                full_val=False,
            )
            results.append(result)
            write_json(session_dir / "results.json", results)
            write_markdown_report(session_dir / "REPORT.md", results, session_dir, args)

    ranked = sorted(
        [result for result in results if result.get("status") == "ok" and not result.get("full_val")],
        key=lambda item: item["val_bpb"],
    )
    for idx, result in enumerate(ranked[: args.full_val_top_k], start=1):
        full_val_result = run_candidate(
            repo_root=repo_root,
            session_dir=session_dir,
            args=args,
            candidate=dict(result["candidate"]),
            name=candidate_name("confirm", idx),
            full_val=True,
        )
        results.append(full_val_result)
        write_json(session_dir / "results.json", results)
        write_markdown_report(session_dir / "REPORT.md", results, session_dir, args)

    summary = {
        "session_dir": str(session_dir),
        "best_proxy": min(
            rankable_results(results, args, full_val=False),
            key=lambda item: item["val_bpb"],
            default=None,
        ),
        "best_proxy_any": min(
            (result for result in results if result.get("status") == "ok" and not result.get("full_val")),
            key=lambda item: item["val_bpb"],
            default=None,
        ),
        "best_full_val": min(
            rankable_results(results, args, full_val=True),
            key=lambda item: item["val_bpb"],
            default=None,
        ),
        "best_full_val_any": min(
            (result for result in results if result.get("status") == "ok" and result.get("full_val")),
            key=lambda item: item["val_bpb"],
            default=None,
        ),
        "run_count": len(results),
    }
    write_json(session_dir / "summary.json", summary)
    write_markdown_report(session_dir / "REPORT.md", results, session_dir, args)
    print(session_dir)


if __name__ == "__main__":
    main()
