"""Tests for identifiability_check.py (C4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/causal is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "causal"))

from identifiability_check import (
    classify_submissions,
    compute_confounded_pairs,
    compute_identifiability_score,
    compute_unexplored_combinations,
    identifiability_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_submission(interventions: list[dict], final_bpb: float = 1.10) -> dict:
    """Helper to build a minimal submission dict."""
    return {
        "interventions": interventions,
        "final_bpb": final_bpb,
    }


def _make_intervention(category: str, name: str) -> dict:
    return {"category": category, "name": name}


def _make_dag(edges: list[tuple[str, str]], nodes: list[str] | None = None) -> dict:
    """Build a minimal DAG dict matching estimate_dag.py output format."""
    if nodes is None:
        node_set = set()
        for s, t in edges:
            node_set.add(s)
            node_set.add(t)
        nodes = sorted(node_set)
    return {
        "nodes": nodes,
        "edges": [
            {"source": s, "target": t, "tag": "expert_imposed", "weight": 0.01}
            for s, t in edges
        ],
        "estimated_effects": [
            {"node": n, "marginal_effect_on_bpb": -0.01, "n_observations": 5}
            for n in nodes
            if n != "bpb"
        ],
    }


# ---------------------------------------------------------------------------
# T11a: test_single_variable_count
# ---------------------------------------------------------------------------


class TestSingleVariableCount:
    def test_single_variable_counted(self):
        """Submissions with exactly 1 intervention are classified as single-variable."""
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            _make_submission([_make_intervention("mlp_mult", "3")]),
            _make_submission([_make_intervention("quant_method", "int8")]),
        ]
        result = classify_submissions(subs)
        assert result["single_variable_records"] == 3
        assert result["multi_variable_records"] == 0

    def test_two_interventions_is_neither_single_nor_multi(self):
        """2 interventions: not single (!=1) and not multi (<3)."""
        subs = [
            _make_submission([
                _make_intervention("num_layers", "11"),
                _make_intervention("mlp_mult", "3"),
            ]),
        ]
        result = classify_submissions(subs)
        assert result["single_variable_records"] == 0
        assert result["multi_variable_records"] == 0


# ---------------------------------------------------------------------------
# T11a: test_multi_variable_count
# ---------------------------------------------------------------------------


class TestMultiVariableCount:
    def test_three_plus_interventions_are_multi(self):
        """Submissions with 3+ interventions are classified as multi-variable."""
        subs = [
            _make_submission([
                _make_intervention("num_layers", "11"),
                _make_intervention("mlp_mult", "3"),
                _make_intervention("quant_method", "int8"),
            ]),
            _make_submission([
                _make_intervention("num_layers", "12"),
                _make_intervention("mlp_mult", "4"),
                _make_intervention("quant_method", "gptq"),
                _make_intervention("compression", "pruning"),
            ]),
        ]
        result = classify_submissions(subs)
        assert result["multi_variable_records"] == 2
        assert result["single_variable_records"] == 0

    def test_mixed_submissions(self):
        """Mix of single and multi-variable submissions."""
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            _make_submission([
                _make_intervention("num_layers", "11"),
                _make_intervention("mlp_mult", "3"),
                _make_intervention("quant_method", "int8"),
            ]),
            _make_submission([_make_intervention("mlp_mult", "4")]),
        ]
        result = classify_submissions(subs)
        assert result["single_variable_records"] == 2
        assert result["multi_variable_records"] == 1
        assert result["total_records"] == 3


# ---------------------------------------------------------------------------
# T11a: test_proceed_recommendation
# ---------------------------------------------------------------------------


class TestProceedRecommendation:
    def test_proceed_when_few_multi_variable(self):
        """< 50% multi-variable records -> proceed."""
        # 3 single, 1 multi = 25% multi
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            _make_submission([_make_intervention("mlp_mult", "3")]),
            _make_submission([_make_intervention("quant_method", "int8")]),
            _make_submission([
                _make_intervention("num_layers", "12"),
                _make_intervention("mlp_mult", "4"),
                _make_intervention("quant_method", "gptq"),
            ]),
        ]
        dag = _make_dag([("num_layers", "bpb"), ("mlp_mult", "bpb")])
        report = identifiability_check(subs, dag)
        assert report["recommendation"] == "proceed"


# ---------------------------------------------------------------------------
# T11a: test_skip_recommendation
# ---------------------------------------------------------------------------


class TestSkipRecommendation:
    def test_skip_when_majority_multi_variable(self):
        """> 50% multi-variable records -> skip."""
        # 1 single, 3 multi = 75% multi
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            _make_submission([
                _make_intervention("num_layers", "12"),
                _make_intervention("mlp_mult", "4"),
                _make_intervention("quant_method", "gptq"),
            ]),
            _make_submission([
                _make_intervention("num_layers", "11"),
                _make_intervention("mlp_mult", "3"),
                _make_intervention("quant_method", "int8"),
            ]),
            _make_submission([
                _make_intervention("mlp_mult", "2"),
                _make_intervention("quant_method", "int8"),
                _make_intervention("compression", "pruning"),
            ]),
        ]
        dag = _make_dag([("num_layers", "bpb"), ("mlp_mult", "bpb")])
        report = identifiability_check(subs, dag)
        assert report["recommendation"] == "skip"


# ---------------------------------------------------------------------------
# T11a: test_confounded_pairs
# ---------------------------------------------------------------------------


class TestConfoundedPairs:
    def test_always_co_occurring_detected(self):
        """Interventions that always appear together are confounded."""
        # num_layers and mlp_mult always co-occur, never independently
        subs = [
            _make_submission([
                _make_intervention("num_layers", "11"),
                _make_intervention("mlp_mult", "3"),
            ]),
            _make_submission([
                _make_intervention("num_layers", "12"),
                _make_intervention("mlp_mult", "4"),
            ]),
            _make_submission([
                _make_intervention("num_layers", "11"),
                _make_intervention("mlp_mult", "2"),
            ]),
        ]
        pairs = compute_confounded_pairs(subs)
        pair_sets = [{p["a"], p["b"]} for p in pairs]
        assert {"num_layers", "mlp_mult"} in pair_sets

    def test_independent_interventions_not_confounded(self):
        """Interventions appearing independently are NOT confounded."""
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            _make_submission([_make_intervention("mlp_mult", "3")]),
            _make_submission([
                _make_intervention("num_layers", "12"),
                _make_intervention("mlp_mult", "4"),
            ]),
        ]
        pairs = compute_confounded_pairs(subs)
        pair_sets = [{p["a"], p["b"]} for p in pairs]
        assert {"num_layers", "mlp_mult"} not in pair_sets


# ---------------------------------------------------------------------------
# T11a: test_unexplored_combinations
# ---------------------------------------------------------------------------


class TestUnexploredCombinations:
    def test_missing_pairs_enumerated(self):
        """Node pairs from DAG not appearing in any submission are listed."""
        dag = _make_dag(
            [("num_layers", "bpb"), ("mlp_mult", "bpb"), ("quant_method", "bpb")],
            nodes=["num_layers", "mlp_mult", "quant_method", "bpb"],
        )
        # Only num_layers+mlp_mult co-occur; quant_method never paired
        subs = [
            _make_submission([
                _make_intervention("num_layers", "11"),
                _make_intervention("mlp_mult", "3"),
            ]),
            _make_submission([_make_intervention("num_layers", "12")]),
        ]
        unexplored = compute_unexplored_combinations(subs, dag)
        unexplored_pair_sets = [set(u["pair"]) for u in unexplored]
        # num_layers+quant_method and mlp_mult+quant_method should be unexplored
        assert {"num_layers", "quant_method"} in unexplored_pair_sets
        assert {"mlp_mult", "quant_method"} in unexplored_pair_sets
        # num_layers+mlp_mult IS explored (co-occur in first submission)
        assert {"num_layers", "mlp_mult"} not in unexplored_pair_sets

    def test_unexplored_ranked_by_expected_effect(self):
        """Unexplored combinations are sorted by expected_effect descending."""
        dag = _make_dag(
            [("num_layers", "bpb"), ("mlp_mult", "bpb"), ("quant_method", "bpb")],
            nodes=["num_layers", "mlp_mult", "quant_method", "bpb"],
        )
        # Override marginal effects to get predictable ranking
        dag["estimated_effects"] = [
            {"node": "num_layers", "marginal_effect_on_bpb": -0.05, "n_observations": 5},
            {"node": "mlp_mult", "marginal_effect_on_bpb": -0.03, "n_observations": 5},
            {"node": "quant_method", "marginal_effect_on_bpb": -0.01, "n_observations": 5},
        ]
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
        ]
        unexplored = compute_unexplored_combinations(subs, dag)
        # All pairs are unexplored (no co-occurrences)
        effects = [u["expected_effect"] for u in unexplored]
        assert effects == sorted(effects, reverse=True)

    def test_unexplored_has_interaction_prior(self):
        """Each unexplored combination has interaction_prior and prior_rationale."""
        dag = _make_dag(
            [("num_layers", "bpb"), ("mlp_mult", "bpb")],
            nodes=["num_layers", "mlp_mult", "bpb"],
        )
        subs = [_make_submission([_make_intervention("num_layers", "11")])]
        unexplored = compute_unexplored_combinations(subs, dag)
        for u in unexplored:
            assert "interaction_prior" in u
            assert u["interaction_prior"] == 1.0
            assert "prior_rationale" in u
            assert u["prior_rationale"] is None  # default


# ---------------------------------------------------------------------------
# T11a: Integration via identifiability_check()
# ---------------------------------------------------------------------------


class TestIdentifiabilityScore:
    def test_full_coverage(self):
        """All edges testable by single-variable records -> score 1.0."""
        dag = _make_dag(
            [("num_layers", "bpb"), ("mlp_mult", "bpb")],
            nodes=["num_layers", "mlp_mult", "bpb"],
        )
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            _make_submission([_make_intervention("mlp_mult", "3")]),
        ]
        score = compute_identifiability_score(subs, dag)
        assert score == 1.0

    def test_partial_coverage(self):
        """Some edges not covered -> score < 1.0."""
        dag = _make_dag(
            [("num_layers", "bpb"), ("mlp_mult", "bpb"), ("quant_method", "bpb")],
            nodes=["num_layers", "mlp_mult", "quant_method", "bpb"],
        )
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            # mlp_mult and quant_method have no single-variable records
        ]
        score = compute_identifiability_score(subs, dag)
        # 1 of 3 edges covered
        assert abs(score - 1.0 / 3.0) < 1e-9


class TestFullReport:
    def test_output_schema(self):
        """identifiability_check returns all required fields."""
        dag = _make_dag(
            [("num_layers", "bpb"), ("mlp_mult", "bpb")],
            nodes=["num_layers", "mlp_mult", "bpb"],
        )
        subs = [
            _make_submission([_make_intervention("num_layers", "11")]),
            _make_submission([_make_intervention("mlp_mult", "3")]),
        ]
        report = identifiability_check(subs, dag)
        assert "single_variable_records" in report
        assert "multi_variable_records" in report
        assert "total_records" in report
        assert "identifiability_score" in report
        assert "confounded_pairs" in report
        assert "recommendation" in report
        assert "unexplored_combinations" in report
        assert isinstance(report["confounded_pairs"], list)
        assert isinstance(report["unexplored_combinations"], list)
