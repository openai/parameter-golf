#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PR-ready records folder.")
    parser.add_argument("--track", choices=["10min_16mb", "non_record_16mb"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--author-name", required=True)
    parser.add_argument("--github-id", required=True)
    parser.add_argument("--val-bpb", type=float, required=True)
    parser.add_argument("--source-train-script", default="train_gpt.py")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    resolved_repo_root = repo_root.resolve()
    date = dt.datetime.now().strftime("%Y-%m-%d")
    display_run_name = args.run_name.strip()
    if not display_run_name:
        display_run_name = "run"
    # Ensure run name is safe for use as a single path component.
    safe_run_name = re.sub(r"[^A-Za-z0-9._-]", "_", args.run_name).strip("._")
    if not safe_run_name:
        safe_run_name = "run"
    slug = f"{date}_{safe_run_name}"

    if args.track == "10min_16mb":
        track_dir = repo_root / "records" / "track_10min_16mb"
    else:
        track_dir = repo_root / "records" / "track_non_record_16mb"

    out_dir = track_dir / slug
    # Prevent path traversal by enforcing final location under selected track dir.
    if track_dir.resolve() not in out_dir.resolve().parents:
        raise ValueError(f"Unsafe output path resolved outside track dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    template_dir = repo_root / "starter_kit" / "templates"
    readme_tpl = (template_dir / "README_submission_template.md").read_text(encoding="utf-8")
    readme = (
        readme_tpl
        .replace("{{RUN_NAME}}", display_run_name)
        .replace("{{DATE}}", date)
        .replace("{{TRACK}}", args.track)
        .replace("{{AUTHOR_NAME}}", args.author_name)
        .replace("{{GITHUB_ID}}", args.github_id)
        .replace("{{VAL_BPB}}", f"{args.val_bpb:.4f}")
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    submission = {
        "author": args.author_name,
        "name": display_run_name,
        "blurb": "Fill out details and attach train logs.",
        "github_id": args.github_id,
        "track": args.track,
        "val_bpb": round(args.val_bpb, 4),
        "date": date
    }
    (out_dir / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")

    source_script_arg = Path(args.source_train_script)
    if source_script_arg.is_absolute():
        raise ValueError("--source-train-script must be a relative path within the repository")
    source_script = (resolved_repo_root / source_script_arg).resolve()
    if source_script != resolved_repo_root and resolved_repo_root not in source_script.parents:
        raise ValueError(
            f"Unsafe source train script resolved outside repository: {args.source_train_script}"
        )
    if not source_script.exists():
        raise FileNotFoundError(f"Could not find train script: {source_script}")
    shutil.copy2(source_script, out_dir / "train_gpt.py")

    (out_dir / "train.log").write_text("# Paste or copy real run logs here\n", encoding="utf-8")

    print(f"Created: {out_dir}")
    print("Next: copy your actual log into train.log and complete README details.")


if __name__ == "__main__":
    main()
