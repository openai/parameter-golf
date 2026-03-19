#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.workspace import best_run_for_candidate, dump_json, iso_utc, repo_root, slugify, utc_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a best run into records/")
    parser.add_argument("--candidate", required=True, help="Candidate id")
    parser.add_argument("--tier", required=True, help="Tier to source from")
    parser.add_argument("--track", required=True, help="Track directory under records/")
    parser.add_argument("--name", required=True, help="Record run name")
    parser.add_argument("--author", required=True, help="Submission author")
    parser.add_argument("--github-id", required=True, help="Submission GitHub id")
    parser.add_argument("--blurb", help="Submission blurb")
    return parser


def render_readme(run_payload: dict, args: argparse.Namespace) -> str:
    objective = run_payload["objective"]
    env_lines = [f"{key}={value}" for key, value in sorted(run_payload["env"].items())]
    command = " ".join(run_payload["command"])
    return "\n".join(
        [
            f"This record was generated from `autoresearch_pg` candidate `{args.candidate}`.",
            "",
            "Source run:",
            f"- Tier: `{args.tier}`",
            f"- Run id: `{run_payload['run_id']}`",
            f"- Run dir: `{run_payload['run_dir']}`",
            "",
            "Key metrics:",
            f"- Post-quant eval: `val_loss:{objective.get('post_quant_val_loss')}` `val_bpb:{objective.get('post_quant_val_bpb')}`",
            f"- Pre-quant eval: `val_loss:{objective.get('pre_quant_val_loss')}` `val_bpb:{objective.get('pre_quant_val_bpb')}`",
            f"- Quant gap: `loss:{objective.get('quant_gap_loss')}` `bpb:{objective.get('quant_gap_bpb')}`",
            f"- Total bytes: `{objective.get('bytes_total')}`",
            f"- Code bytes: `{objective.get('bytes_code')}`",
            f"- Model bytes int8+zlib: `{objective.get('bytes_model_int8_zlib')}`",
            f"- Step avg: `{objective.get('step_avg_ms')}ms`",
            f"- Train time: `{objective.get('train_time_ms')}ms`",
            "",
            "Command:",
            "```bash",
            command,
            "```",
            "",
            "Environment:",
            "```bash",
            "\n".join(env_lines),
            "```",
            "",
            "Included files:",
            "- `train_gpt.py`",
            "- `train.log`",
            "- `submission.json`",
        ]
    ) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    best = best_run_for_candidate(args.candidate, tier=args.tier, require_valid=True)
    if best is None:
        print(f"no valid run found for candidate={args.candidate} tier={args.tier}")
        return 1

    date_prefix = utc_now().strftime("%Y-%m-%d")
    record_slug = slugify(args.name)
    record_dir = repo_root() / "records" / args.track / f"{date_prefix}_{record_slug}"
    if record_dir.exists():
        raise FileExistsError(f"record directory already exists: {record_dir}")
    record_dir.mkdir(parents=True, exist_ok=False)

    source_run_dir = repo_root() / best["run_dir"]
    shutil.copy2(source_run_dir / "train_gpt.py", record_dir / "train_gpt.py")
    shutil.copy2(source_run_dir / "train.log", record_dir / "train.log")

    objective = best["objective"]
    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.name,
        "blurb": args.blurb or f"Generated from autoresearch_pg candidate {args.candidate}.",
        "date": iso_utc(),
        "val_loss": objective.get("post_quant_val_loss"),
        "val_bpb": objective.get("post_quant_val_bpb"),
        "bytes_total": objective.get("bytes_total"),
        "bytes_code": objective.get("bytes_code"),
    }
    dump_json(record_dir / "submission.json", submission)
    (record_dir / "README.md").write_text(render_readme(best, args), encoding="utf-8")

    print(record_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
