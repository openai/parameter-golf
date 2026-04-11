"""Identifiability Check (C4) — Data Quality Assessment.

Analyzes the intervention matrix to determine whether causal effects
are identifiable from the available submissions. Classifies records
as single-variable or multi-variable, computes an identifiability
score, detects confounded pairs, and recommends proceed/skip.

CLI:
  python scripts/causal/identifiability_check.py \
    --interventions results/causal/interventions.json \
    --dag results/causal/cycle_0/causal_dag.json \
    --output results/causal/cycle_0/identifiability_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Classify submissions as single-variable or multi-variable
# ---------------------------------------------------------------------------


def classify_submissions(submissions: list[dict]) -> dict:
    """Count single-variable (==1 intervention) and multi-variable (3+) records.

    Returns dict with single_variable_records, multi_variable_records,
    total_records.
    """
    single = 0
    multi = 0
    for sub in submissions:
        n_interventions = len(sub.get("interventions", []))
        if n_interventions == 1:
            single += 1
        elif n_interventions >= 3:
            multi += 1
    return {
        "single_variable_records": single,
        "multi_variable_records": multi,
        "total_records": len(submissions),
    }


# ---------------------------------------------------------------------------
# 2. Identifiability score
# ---------------------------------------------------------------------------


def compute_identifiability_score(
    submissions: list[dict],
    dag: dict,
) -> float:
    """Fraction of DAG edges where at least one single-variable record tests that edge.

    An edge (source -> target) is "testable" if there exists a submission
    with exactly 1 intervention whose category matches the edge's source node.
    """
    edges = dag.get("edges", [])
    if not edges:
        return 1.0  # No edges to test -> trivially identifiable

    # Collect categories present in single-variable submissions
    single_var_categories: set[str] = set()
    for sub in submissions:
        interventions = sub.get("interventions", [])
        if len(interventions) == 1:
            single_var_categories.add(interventions[0]["category"])

    testable = 0
    for edge in edges:
        source = edge["source"]
        if source in single_var_categories:
            testable += 1

    return testable / len(edges)


# ---------------------------------------------------------------------------
# 3. Confounded pairs
# ---------------------------------------------------------------------------


def compute_confounded_pairs(submissions: list[dict]) -> list[dict]:
    """Detect intervention category pairs that always co-occur.

    A pair (A, B) is confounded if:
    - They co-occur in at least one submission, AND
    - Neither A nor B ever appears without the other.
    """
    # For each category, track which submissions it appears in
    category_subs: dict[str, set[int]] = {}
    for i, sub in enumerate(submissions):
        for iv in sub.get("interventions", []):
            cat = iv["category"]
            if cat not in category_subs:
                category_subs[cat] = set()
            category_subs[cat].add(i)

    confounded = []
    categories = sorted(category_subs.keys())
    for a, b in combinations(categories, 2):
        subs_a = category_subs[a]
        subs_b = category_subs[b]
        # They co-occur if intersection is non-empty
        co_occur = subs_a & subs_b
        if not co_occur:
            continue
        # Confounded: A never appears without B, and B never appears without A
        a_without_b = subs_a - subs_b
        b_without_a = subs_b - subs_a
        if not a_without_b and not b_without_a:
            confounded.append({
                "a": a,
                "b": b,
                "co_occurrence_count": len(co_occur),
            })

    return confounded


# ---------------------------------------------------------------------------
# 4. Unexplored combinations
# ---------------------------------------------------------------------------


def compute_unexplored_combinations(
    submissions: list[dict],
    dag: dict,
) -> list[dict]:
    """Enumerate all non-outcome node pairs from DAG, find pairs not co-occurring
    in any submission, rank by sum of individual marginal effects x interaction prior.
    """
    nodes = [n for n in dag.get("nodes", []) if n != "bpb"]
    if len(nodes) < 2:
        return []

    # Build marginal effect lookup
    effect_map: dict[str, float] = {}
    for eff in dag.get("estimated_effects", []):
        effect_map[eff["node"]] = abs(eff.get("marginal_effect_on_bpb", 0.0))

    # Collect co-occurring category pairs across submissions
    explored_pairs: set[frozenset[str]] = set()
    for sub in submissions:
        cats = [iv["category"] for iv in sub.get("interventions", [])]
        for a, b in combinations(cats, 2):
            explored_pairs.add(frozenset([a, b]))

    # Enumerate all node pairs, find unexplored
    unexplored = []
    for a, b in combinations(nodes, 2):
        pair_key = frozenset([a, b])
        if pair_key not in explored_pairs:
            interaction_prior = 1.0
            expected_effect = (effect_map.get(a, 0.0) + effect_map.get(b, 0.0)) * interaction_prior
            unexplored.append({
                "pair": sorted([a, b]),
                "expected_effect": round(expected_effect, 6),
                "interaction_prior": interaction_prior,
                "prior_rationale": None,
            })

    # Sort by expected_effect descending
    unexplored.sort(key=lambda x: x["expected_effect"], reverse=True)
    return unexplored


# ---------------------------------------------------------------------------
# 5. Full identifiability check
# ---------------------------------------------------------------------------


def identifiability_check(
    submissions: list[dict],
    dag: dict,
) -> dict:
    """Run full identifiability assessment.

    Returns the I4 output schema:
    {
      single_variable_records, multi_variable_records, total_records,
      identifiability_score, confounded_pairs, recommendation,
      unexplored_combinations
    }
    """
    counts = classify_submissions(submissions)
    score = compute_identifiability_score(submissions, dag)
    confounded = compute_confounded_pairs(submissions)
    unexplored = compute_unexplored_combinations(submissions, dag)

    # Recommendation: >50% multi-variable -> skip
    total = counts["total_records"]
    multi_frac = counts["multi_variable_records"] / total if total > 0 else 0.0
    recommendation = "skip" if multi_frac > 0.5 else "proceed"

    return {
        "single_variable_records": counts["single_variable_records"],
        "multi_variable_records": counts["multi_variable_records"],
        "total_records": counts["total_records"],
        "identifiability_score": round(score, 6),
        "confounded_pairs": confounded,
        "recommendation": recommendation,
        "unexplored_combinations": unexplored,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identifiability Check (C4) — Data Quality Assessment",
    )
    parser.add_argument(
        "--interventions", required=True,
        help="Path to interventions.json",
    )
    parser.add_argument(
        "--dag", required=True,
        help="Path to causal_dag.json",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for identifiability_report.json output",
    )
    args = parser.parse_args()

    # Load inputs
    interventions_data = json.loads(
        Path(args.interventions).read_text(encoding="utf-8")
    )
    submissions = interventions_data.get("submissions", interventions_data)
    if isinstance(submissions, dict):
        submissions = submissions.get("submissions", [])

    dag = json.loads(Path(args.dag).read_text(encoding="utf-8"))

    # Run check
    report = identifiability_check(submissions, dag)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Summary
    print(
        f"Identifiability: {report['identifiability_score']:.2%} of edges testable, "
        f"{report['single_variable_records']} single-var / "
        f"{report['multi_variable_records']} multi-var / "
        f"{report['total_records']} total"
    )
    print(f"Confounded pairs: {len(report['confounded_pairs'])}")
    print(f"Unexplored combinations: {len(report['unexplored_combinations'])}")
    print(f"Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
