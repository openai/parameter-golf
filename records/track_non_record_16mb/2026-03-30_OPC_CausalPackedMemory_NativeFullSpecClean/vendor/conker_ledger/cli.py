from __future__ import annotations

import argparse
from pathlib import Path

from .ledger import (
    dumps_json,
    lineage_rows,
    render_table,
    scan_results,
    sort_records,
    survival_rows,
    write_report_bundle,
    write_validity_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validity packaging and backlog-analysis tool for Conker-style experiment outputs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bundle = sub.add_parser("bundle", help="Assemble a manifest-first validity bundle")
    p_bundle.add_argument("manifest")
    p_bundle.add_argument("out_dir")
    p_bundle.add_argument("--json")

    p_scan = sub.add_parser("scan", help="Scan a directory of experiment JSON outputs")
    p_scan.add_argument("root")
    p_scan.add_argument("--json")

    p_table = sub.add_parser("table", help="Show a ranked table of normalized records")
    p_table.add_argument("root")
    p_table.add_argument("--kind", choices=["all", "bridge", "full_eval", "study"], default="all")
    p_table.add_argument("--metric", default="bpb")
    p_table.add_argument("--top", type=int, default=20)
    p_table.add_argument("--descending", action="store_true")
    p_table.add_argument("--json")

    p_survival = sub.add_parser("survival", help="Compare bridge rows with their full-eval descendants")
    p_survival.add_argument("root")
    p_survival.add_argument("--top", type=int, default=50)
    p_survival.add_argument("--json")

    p_lineage = sub.add_parser("lineage", help="Trace warm-start ancestry from loaded_state_path to saved_state_path")
    p_lineage.add_argument("root")
    p_lineage.add_argument("--top", type=int, default=50)
    p_lineage.add_argument("--json")

    p_report = sub.add_parser("report", help="Write a public report bundle with JSON, CSV, and SVG outputs")
    p_report.add_argument("root")
    p_report.add_argument("out_dir")
    p_report.add_argument("--top", type=int, default=20)
    p_report.add_argument("--json")

    return parser


def write_output(text: str, json_path: str | None) -> None:
    print(text)
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bundle":
        result = write_validity_bundle(Path(args.manifest), Path(args.out_dir))
        write_output(dumps_json(result), args.json)
        return

    root = Path(args.root)
    scanned = scan_results(root)

    if args.command == "scan":
        write_output(dumps_json(scanned), args.json)
        return

    if args.command == "table":
        records = scanned["records"]
        if args.kind != "all":
            records = [record for record in records if record["kind"] == args.kind]
        records = sort_records(records, args.metric, ascending=not args.descending)
        if args.json:
            write_output(dumps_json({"rows": records[: args.top]}), args.json)
        else:
            columns = ["kind", "family_id", "run_id", "seed", args.metric, "path"]
            write_output(render_table(records, columns, top=args.top), None)
        return

    if args.command == "survival":
        rows = survival_rows(scanned["records"])
        rows = sort_records(rows, "full_fp16", ascending=True)
        if args.json:
            write_output(dumps_json({"rows": rows[: args.top]}), args.json)
        else:
            columns = ["family_id", "run_id", "seed", "bridge_fp16", "full_fp16", "bridge_int6", "full_int6", "status"]
            write_output(render_table(rows, columns, top=args.top), None)
        return

    if args.command == "lineage":
        rows = lineage_rows(scanned["records"])
        if args.json:
            write_output(dumps_json({"rows": rows[: args.top]}), args.json)
        else:
            columns = ["parent_run_id", "child_run_id", "seed", "child_bpb", "family_id"]
            write_output(render_table(rows, columns, top=args.top), None)
        return

    if args.command == "report":
        result = write_report_bundle(root, Path(args.out_dir), top=args.top)
        write_output(dumps_json(result), args.json)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
