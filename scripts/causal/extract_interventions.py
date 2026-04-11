"""C2: extract_interventions.py -- Record Parser.

Parses records/track_10min_16mb/*/README.md and submission.json to build
a structured intervention-outcome dataset (interventions.json).

Three-tier extraction:
  Format A: Change/Impact ablation table  (delta_bpb from Impact column)
  Format B: Base/This comparison table    (delta_bpb computed from val_bpb row)
  Format C: Prose only                    (interventions from blurb, delta_bpb = null)
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Category classifier
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("quantization", ["quant", "int6", "int8", "int5", "gptq", "ste", "clip", "fake-quant", "dequant"]),
    ("architecture", ["layer", "head", "dim", "mlp", "rope", "xsa", "attention", "skip", "u-net",
                       "unet", "smear", "bigram", "embed", "ln scale", "rmsnorm", "sliding",
                       "partial rope", "kv head", "gqa", "softcap"]),
    ("optimization", ["ema", "swa", "warmdown", "warmup", "lr", "learning rate", "momentum",
                       "muon", "adam", "weight decay", "wd", "grad clip", "ortho", "init",
                       "scheduler", "decay", "batch"]),
    ("data", ["shard", "data", "tokenizer", "sequence length", "seq_len", "context"]),
    ("encoding", ["zstd", "compress", "lossless"]),
]


def classify_category(name: str) -> str:
    """Map an intervention name to a category string."""
    lower = name.lower()
    for cat, keywords in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in lower:
                return cat
    return "optimization"  # default


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_FORMAT_A_PATTERN = re.compile(
    r"\|\s*Change\s*\|.*\|\s*Impact\s*\|", re.IGNORECASE
)

_FORMAT_B_PATTERN = re.compile(
    r"\|\s*\|.*(?:PR|Base|Previous|Ref)\s*[#\d]*.*\|\s*This\s*\|", re.IGNORECASE
)

# Alternate: header row with 3 columns where first is empty
_FORMAT_B_ALT_PATTERN = re.compile(
    r"###?\s*Changes?\s+from", re.IGNORECASE
)


def detect_format(readme: str) -> str:
    """Detect the README format: 'A', 'B', or 'C'."""
    if _FORMAT_A_PATTERN.search(readme):
        return "A"
    if _FORMAT_B_PATTERN.search(readme) or (
        _FORMAT_B_ALT_PATTERN.search(readme)
        and re.search(r"\|.*\|.*\|.*\|", readme)
    ):
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Format A parser: Change/Impact table
# ---------------------------------------------------------------------------

_DELTA_RE = re.compile(r"([+-]?\d+\.\d+)\s*BPB", re.IGNORECASE)


def _strip_bold(s: str) -> str:
    return s.replace("**", "").strip()


def parse_format_a(readme: str) -> list[dict]:
    """Parse Format A (Change/Impact ablation table).

    Returns list of intervention dicts with name, category, delta_bpb, delta_source.
    """
    interventions: list[dict] = []
    lines = readme.splitlines()
    in_table = False

    for line in lines:
        if _FORMAT_A_PATTERN.search(line):
            in_table = True
            continue
        if in_table and line.strip().startswith("|---"):
            continue
        if in_table and "|" in line:
            cells = [c.strip() for c in line.split("|")]
            # Filter empty strings from leading/trailing pipes
            cells = [c for c in cells if c]
            if len(cells) < 4:
                continue
            name = _strip_bold(cells[0])
            impact = cells[-1]

            # Skip "Total" summary row
            if name.lower() in ("total", "sum", "net"):
                continue

            delta_match = _DELTA_RE.search(impact)
            delta_bpb = float(delta_match.group(1)) if delta_match else None

            interventions.append({
                "name": name,
                "category": classify_category(name),
                "delta_bpb": delta_bpb,
                "delta_source": "ablation_table",
            })
        elif in_table and line.strip() == "":
            # End of table
            in_table = False

    return interventions


def extract_base_bpb_format_a(readme: str) -> float | None:
    """Extract base BPB from the Total row of a Format A table."""
    lines = readme.splitlines()
    in_table = False

    for line in lines:
        if _FORMAT_A_PATTERN.search(line):
            in_table = True
            continue
        if in_table and "|" in line:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if len(cells) >= 3:
                name = _strip_bold(cells[0])
                if name.lower() in ("total", "sum", "net"):
                    # Base value is in the second column (PR #xxx column)
                    base_str = _strip_bold(cells[1])
                    try:
                        return float(base_str)
                    except ValueError:
                        pass
        elif in_table and line.strip() == "":
            in_table = False

    return None


# ---------------------------------------------------------------------------
# Format B parser: Base/This comparison table
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"\d+\.\d+")


def parse_format_b(readme: str) -> list[dict]:
    """Parse Format B (Base/This comparison table).

    Extracts intervention names from "What's new" or numbered list sections.
    """
    interventions: list[dict] = []

    # Look for "What's new" section or numbered items after the table
    whats_new = re.findall(
        r"(?:^|\n)\d+\.\s+\*\*(.+?)\*\*",
        readme,
    )

    # Also look for table rows that describe changes (non-bpb, non-artifact rows)
    lines = readme.splitlines()
    table_started = False
    header_seen = False

    for line in lines:
        if _FORMAT_B_PATTERN.search(line):
            table_started = True
            continue
        if table_started and line.strip().startswith("|---"):
            header_seen = True
            continue
        if table_started and header_seen and "|" in line:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if len(cells) >= 3:
                row_label = _strip_bold(cells[0])
                base_val = _strip_bold(cells[1])
                this_val = _strip_bold(cells[2])

                # Skip metric rows and "same" rows
                lower_label = row_label.lower()
                if any(skip in lower_label for skip in [
                    "val_bpb", "val bpb", "artifact", "everything else",
                    "same", "total"
                ]):
                    continue
                # Skip if base and this are the same
                if base_val.lower() == this_val.lower():
                    continue
                # This is a change row — extract as intervention
                if row_label and base_val.lower() not in ("same",):
                    interventions.append({
                        "name": row_label,
                        "category": classify_category(row_label),
                        "delta_bpb": None,
                        "delta_source": "readme_prose",
                    })
        elif table_started and header_seen and line.strip() == "":
            table_started = False
            header_seen = False

    # If we didn't find interventions from the table diff rows, use "What's new"
    if not interventions and whats_new:
        for item in whats_new:
            # Clean up: take text before the first period or parenthesis
            name = re.split(r"[.(]", item)[0].strip()
            interventions.append({
                "name": name,
                "category": classify_category(name),
                "delta_bpb": None,
                "delta_source": "readme_prose",
            })

    # If still empty, fall back to "What's new" items even if we found table rows
    if not interventions:
        # Try bold items in numbered lists
        bold_items = re.findall(r"\*\*(.+?)\*\*", readme)
        for item in bold_items:
            name = re.split(r"[.(]", item)[0].strip()
            if len(name) > 3 and not _FLOAT_RE.fullmatch(name) and name.lower() not in (
                "total", "mean", "std"
            ):
                interventions.append({
                    "name": name,
                    "category": classify_category(name),
                    "delta_bpb": None,
                    "delta_source": "readme_prose",
                })
                if len(interventions) >= 5:
                    break

    return interventions


def extract_base_bpb_format_b(readme: str) -> float | None:
    """Extract base BPB from a Format B comparison table (val_bpb row, Base column)."""
    lines = readme.splitlines()
    table_started = False
    header_seen = False

    for line in lines:
        if _FORMAT_B_PATTERN.search(line):
            table_started = True
            continue
        if table_started and line.strip().startswith("|---"):
            header_seen = True
            continue
        if table_started and header_seen and "|" in line:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if len(cells) >= 3:
                label = _strip_bold(cells[0]).lower()
                if "bpb" in label or "val_bpb" in label:
                    base_str = _strip_bold(cells[1])
                    floats = _FLOAT_RE.findall(base_str)
                    if floats:
                        return float(floats[0])
        elif table_started and header_seen and line.strip() == "":
            table_started = False
            header_seen = False

    return None


# ---------------------------------------------------------------------------
# Format C parser: prose only
# ---------------------------------------------------------------------------

def parse_format_c(readme: str, submission: dict) -> list[dict]:
    """Parse Format C (prose only).

    Extracts intervention names from submission blurb and README headings/bullets.
    """
    interventions: list[dict] = []
    blurb = submission.get("blurb", "")

    # Use the blurb as a single intervention description
    if blurb:
        # Try to extract key technique names from the blurb
        name = submission.get("name", blurb[:80])
        interventions.append({
            "name": name,
            "category": classify_category(name),
            "delta_bpb": None,
            "delta_source": "submission_blurb",
        })

    # Also check README for ### headings that might name techniques
    headings = re.findall(r"^###?\s+(.+)$", readme, re.MULTILINE)
    for heading in headings:
        heading_clean = heading.strip()
        # Skip generic headings
        if heading_clean.lower() in (
            "configuration", "training", "results", "reproducibility",
            "architecture", "key metrics", "included files", "run command",
            "summary", "quantization",
        ):
            continue
        if len(heading_clean) > 3:
            interventions.append({
                "name": heading_clean,
                "category": classify_category(heading_clean),
                "delta_bpb": None,
                "delta_source": "submission_blurb",
            })

    if not interventions:
        # Absolute fallback: use submission name
        interventions.append({
            "name": submission.get("name", "unknown"),
            "category": "optimization",
            "delta_bpb": None,
            "delta_source": "submission_blurb",
        })

    return interventions


# ---------------------------------------------------------------------------
# Field coverage computation
# ---------------------------------------------------------------------------

_COVERAGE_FIELDS = [
    "submission_id",
    "date",
    "author",
    "base_bpb",
    "final_bpb",
    "interventions",  # non-empty list counts as present
]


def compute_field_coverage(submissions: list[dict]) -> float:
    """Compute field_coverage = (non-null fields) / (submissions x 6)."""
    if not submissions:
        return 0.0

    total = len(submissions) * len(_COVERAGE_FIELDS)
    present = 0
    for sub in submissions:
        for field in _COVERAGE_FIELDS:
            val = sub.get(field)
            if field == "interventions":
                if val and len(val) > 0:
                    present += 1
            elif val is not None:
                present += 1

    return present / total


# ---------------------------------------------------------------------------
# Append experiment mode
# ---------------------------------------------------------------------------

def append_experiment(existing: dict, raw_runs_path: str) -> dict:
    """Append experiment results from raw_runs.json as a new submission entry."""
    raw = json.loads(Path(raw_runs_path).read_text(encoding="utf-8"))

    treatment_bpb = raw.get("treatment_bpb", [])
    control_bpb = raw.get("control_bpb", [])

    mean_treatment = sum(treatment_bpb) / len(treatment_bpb) if treatment_bpb else None
    mean_control = sum(control_bpb) / len(control_bpb) if control_bpb else None

    intervention_name = raw.get("intervention", "unknown")
    delta = None
    if mean_treatment is not None and mean_control is not None:
        delta = round(mean_treatment - mean_control, 6)

    entry = {
        "submission_id": raw.get("experiment_id", "experiment"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "author": "causal_pipeline",
        "base_bpb": mean_control,
        "final_bpb": mean_treatment,
        "interventions": [
            {
                "name": intervention_name,
                "category": classify_category(intervention_name),
                "delta_bpb": delta,
                "delta_source": "experiment",
            }
        ],
        "parse_quality": "structured",
    }

    existing["submissions"].append(entry)
    existing["metadata"]["total_submissions"] = len(existing["submissions"])
    existing["field_coverage"] = compute_field_coverage(existing["submissions"])
    return existing


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

# Ordered list of (compiled_regex, group_index) for base_bpb extraction.
# Tried in priority order; first valid match wins.
_BASE_BPB_PATTERNS = [
    re.compile(r"[Bb]aseline.*?\|\s*(\d+\.\d{3,})\s*\|"),                      # table row
    re.compile(r"[Nn]aive\s+[Bb]aseline.*?(\d+\.\d{3,})"),                     # prose
    re.compile(r"(?:baseline|previous|base)[:\s]*(\d+\.\d{3,})", re.IGNORECASE),# generic
    re.compile(r"\(default\).*?\|\s*(\d+\.\d{3,})"),                           # sweep table
    re.compile(r"SOTA\s*\(\s*(\d+\.\d{3,})", re.IGNORECASE),                   # SOTA ref
    re.compile(r"[Nn]aive\s+[Bb]aseline\s*\|.*?\|\s*\*?\*?(\d+\.\d{3,})"),    # table w/ bold
    re.compile(r"(\d+\.\d{3,})\s*\)?\s*(?:->|→)\s*(?:this|$)", re.IGNORECASE), # arrow chain
]


def _extract_base_bpb_from_prose(readme: str) -> float | None:
    """Try to extract base_bpb from prose references. Checks patterns in priority order."""
    for pattern in _BASE_BPB_PATTERNS:
        m = pattern.search(readme)
        if m:
            val = float(m.group(1))
            if 1.0 < val < 2.0:  # reasonable BPB range
                return val
    return None


def _parse_one_record(record_dir: Path) -> dict | None:
    """Parse a single record directory into a submission dict."""
    sub_path = record_dir / "submission.json"
    readme_path = record_dir / "README.md"

    if not sub_path.exists():
        return None

    submission = json.loads(sub_path.read_text(encoding="utf-8"))
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    # Core fields from submission.json (with fallbacks for variant schemas)
    submission_id = record_dir.name
    date = submission.get("date")
    author = submission.get("author")
    final_bpb = submission.get("val_bpb")

    # Fallback: some records use mean_val_bpb instead of val_bpb
    if final_bpb is None:
        final_bpb = submission.get("mean_val_bpb")

    # If submission.json has a "techniques" list, use it for interventions
    techniques_list = submission.get("techniques", [])

    # Detect format and parse
    fmt = detect_format(readme)
    base_bpb = None
    interventions: list[dict] = []
    parse_quality = "minimal"

    if fmt == "A":
        interventions = parse_format_a(readme)
        base_bpb = extract_base_bpb_format_a(readme)
        parse_quality = "structured"
    elif fmt == "B":
        interventions = parse_format_b(readme)
        base_bpb = extract_base_bpb_format_b(readme)
        if interventions:
            parse_quality = "structured"
        else:
            # Format B detected but no interventions extracted from table;
            # fall back to Format C parsing for interventions
            if techniques_list:
                interventions = [
                    {
                        "name": t,
                        "category": classify_category(t),
                        "delta_bpb": None,
                        "delta_source": "submission_blurb",
                    }
                    for t in techniques_list
                ]
            else:
                interventions = parse_format_c(readme, submission)
            parse_quality = "prose"
    else:
        # If techniques list is available in submission.json, use it
        if techniques_list:
            interventions = [
                {
                    "name": t,
                    "category": classify_category(t),
                    "delta_bpb": None,
                    "delta_source": "submission_blurb",
                }
                for t in techniques_list
            ]
            parse_quality = "prose"
        else:
            interventions = parse_format_c(readme, submission)
            parse_quality = "prose"

    # Try to extract base_bpb from prose references if not already found
    if base_bpb is None:
        base_bpb = _extract_base_bpb_from_prose(readme)

    # Cross-reference: compute delta from base/final if no per-intervention deltas
    if base_bpb is not None and final_bpb is not None:
        total_delta = round(final_bpb - base_bpb, 6)
        # If Format B interventions have null deltas, distribute evenly (rough approx)
        null_delta_interventions = [i for i in interventions if i["delta_bpb"] is None]
        if null_delta_interventions and len(null_delta_interventions) > 0:
            per_intervention = round(total_delta / len(null_delta_interventions), 6)
            for i in null_delta_interventions:
                i["delta_bpb"] = per_intervention

    return {
        "submission_id": submission_id,
        "date": date,
        "author": author,
        "base_bpb": base_bpb,
        "final_bpb": final_bpb,
        "interventions": interventions,
        "parse_quality": parse_quality,
    }


def extract_all(records_dir: str) -> dict:
    """Extract interventions from all records in a directory."""
    records_path = Path(records_dir)
    submissions: list[dict] = []
    format_counts = {"A": 0, "B": 0, "C": 0}

    for record_dir in sorted(records_path.iterdir()):
        if not record_dir.is_dir():
            continue
        sub_path = record_dir / "submission.json"
        if not sub_path.exists():
            continue

        result = _parse_one_record(record_dir)
        if result is not None:
            submissions.append(result)
            # Track format
            readme_path = record_dir / "README.md"
            readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
            fmt = detect_format(readme)
            format_counts[fmt] += 1

    coverage = compute_field_coverage(submissions)

    structured_count = format_counts["A"] + format_counts["B"]

    return {
        "submissions": submissions,
        "field_coverage": round(coverage, 4),
        "metadata": {
            "total_submissions": len(submissions),
            "structured_count": structured_count,
            "prose_only_count": format_counts["C"],
            "format_a_count": format_counts["A"],
            "format_b_count": format_counts["B"],
            "format_c_count": format_counts["C"],
            "extraction_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract interventions from leaderboard records.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to records directory (e.g. records/track_10min_16mb/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file (e.g. results/causal/interventions.json)",
    )
    parser.add_argument(
        "--append-experiment",
        default=None,
        help="Path to raw_runs.json to append as a new experiment entry",
    )
    args = parser.parse_args()

    result = extract_all(args.input)

    if args.append_experiment:
        result = append_experiment(result, args.append_experiment)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Extracted {result['metadata']['total_submissions']} submissions")
    print(f"  Format A: {result['metadata'].get('format_a_count', 0)}")
    print(f"  Format B: {result['metadata'].get('format_b_count', 0)}")
    print(f"  Format C: {result['metadata'].get('format_c_count', 0)}")
    print(f"  field_coverage: {result['field_coverage']:.4f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
