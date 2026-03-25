"""Tests for scripts/causal/estimate_dag.py — Causal Structure Discovery (C3).

Covers T7 test specs:
  - test_expert_skeleton_edges
  - test_binary_encoding_dimensions
  - test_fci_synthetic_data
  - test_fci_near_degenerate
  - test_linalg_error_fallback
  - test_edge_tagging_logic
  - test_previous_dag_diff
  - test_dot_renders
  - test_next_intervention_recommendation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure scripts/causal is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "causal"))

import estimate_dag as ed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXPERT_EDGES = [
    ("num_layers", "bpb"),
    ("mlp_mult", "bpb"),
    ("quant_method", "bpb"),
    ("weight_avg_method", "bpb"),
    ("attention_variant", "bpb"),
    ("rope_variant", "bpb"),
    ("compression", "bpb"),
]


def _make_interventions(n: int = 20, seed: int = 42) -> dict:
    """Create a synthetic interventions.json payload with *n* submissions."""
    rng = np.random.default_rng(seed)
    categories = {
        "num_layers": ["10", "11", "12"],
        "mlp_mult": ["2", "3", "4"],
        "quant_method": ["none", "int8", "gptq"],
        "weight_avg_method": ["none", "ema", "swa"],
        "attention_variant": ["standard", "gqa"],
        "rope_variant": ["standard", "yarn"],
        "compression": ["none", "pruning"],
    }
    submissions = []
    for i in range(n):
        interventions = []
        for cat, vals in categories.items():
            chosen = rng.choice(vals)
            if chosen != vals[0]:
                interventions.append(
                    {
                        "name": chosen,
                        "category": cat,
                        "delta_bpb": round(float(rng.normal(-0.005, 0.01)), 4),
                        "delta_source": "synthetic",
                    }
                )
        bpb = round(1.15 + float(rng.normal(0, 0.02)), 4)
        submissions.append(
            {
                "submission_id": f"synth_{i}",
                "date": "2026-01-01",
                "author": "test",
                "base_bpb": 1.15,
                "final_bpb": bpb,
                "interventions": interventions,
                "parse_quality": "structured",
            }
        )
    return {
        "submissions": submissions,
        "field_coverage": 1.0,
        "metadata": {"total_submissions": n},
    }


@pytest.fixture
def interventions_path(tmp_path: Path) -> Path:
    p = tmp_path / "interventions.json"
    p.write_text(json.dumps(_make_interventions(20)), encoding="utf-8")
    return p


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cycle_0"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# T7 tests
# ---------------------------------------------------------------------------


class TestExpertSkeleton:
    """test_expert_skeleton_edges: known causal edges present, tagged expert_imposed."""

    def test_expert_skeleton_edges(self):
        skeleton = ed.build_expert_skeleton()
        edge_set = {(e["source"], e["target"]) for e in skeleton}
        for src, tgt in EXPERT_EDGES:
            assert (src, tgt) in edge_set, f"Missing expert edge: {src} -> {tgt}"
        # All edges should be tagged expert_imposed
        for e in skeleton:
            assert e["tag"] == "expert_imposed"


class TestBinaryEncoding:
    """test_binary_encoding_dimensions: correct matrix shape."""

    def test_binary_encoding_dimensions(self):
        data = _make_interventions(20)
        matrix, node_names = ed.binary_encode(data["submissions"])
        # 20 rows, one col per category (7) minus bpb (bpb is not encoded)
        assert matrix.shape[0] == 20
        # At least 7 binary columns (could be more if one-hot for multi-valued)
        assert matrix.shape[1] >= 7
        assert len(node_names) == matrix.shape[1]


class TestFCISyntheticData:
    """test_fci_synthetic_data: small synthetic dataset with known structure."""

    def test_fci_synthetic_data(self):
        # Create a dataset with clear causal structure:
        # X causes Y (correlated), Z is independent noise
        rng = np.random.default_rng(123)
        n = 200
        x = rng.standard_normal(n)
        y = 0.8 * x + 0.2 * rng.standard_normal(n)
        z = rng.standard_normal(n)
        data = np.column_stack([x, y, z])
        node_names = ["X", "Y", "Z"]

        result = ed.run_fci_validation(data, node_names, alpha=0.05)
        # FCI should find some relationship between X and Y
        assert result is not None
        edges = result["edges"]
        # X-Y should have an edge, Z should be mostly independent
        xy_edge = any(
            (e["source"] == "X" and e["target"] == "Y")
            or (e["source"] == "Y" and e["target"] == "X")
            for e in edges
        )
        assert xy_edge, "FCI should detect X-Y relationship"


class TestFCINearDegenerate:
    """test_fci_near_degenerate: n=20, highly correlated binary columns."""

    def test_fci_near_degenerate(self):
        # Highly correlated binary columns -> degenerate
        rng = np.random.default_rng(99)
        n = 20
        col = rng.integers(0, 2, size=n).astype(float)
        # All columns are near-identical
        data = np.column_stack([col, col, col + 0.01 * rng.standard_normal(n)])
        node_names = ["A", "B", "C"]

        result = ed.run_fci_validation(data, node_names, alpha=0.01)
        # Should detect degeneracy and return None or mark as degenerate
        assert result is None or result.get("degenerate") is True


class TestLinAlgErrorFallback:
    """test_linalg_error_fallback: mock LinAlgError -> falls back to expert DAG."""

    def test_linalg_error_fallback(self):
        data = _make_interventions(20)
        subs = data["submissions"]

        with patch.object(ed, "run_fci_validation", side_effect=np.linalg.LinAlgError("singular")):
            dag = ed.estimate_dag(subs, alpha=0.01)

        # Should fall back to expert-only
        assert dag["metadata"]["fci_degenerate"] is True
        # Expert edges should still be present
        assert dag["metadata"]["expert_edges_count"] == len(EXPERT_EDGES)


class TestEdgeTaggingLogic:
    """test_edge_tagging_logic: expert_imposed -> data_confirmed/data_contradicted/uncertain."""

    def test_edge_tagging_logic(self):
        expert_edges = [
            {"source": "A", "target": "B", "tag": "expert_imposed"},
            {"source": "C", "target": "D", "tag": "expert_imposed"},
            {"source": "E", "target": "F", "tag": "expert_imposed"},
        ]

        # Case 1: FCI found A->B (confirmed), not C->D or E->F (contradicted),
        # and found G->H which is not in expert (uncertain)
        fci_edges = [
            {"source": "A", "target": "B"},  # confirms expert edge
            # C->D missing -> data_contradicted
            # E->F missing -> data_contradicted
            {"source": "G", "target": "H"},  # new from data, not in expert -> uncertain
        ]

        tagged = ed.tag_edges(expert_edges, fci_edges)
        tag_map = {(e["source"], e["target"]): e["tag"] for e in tagged}

        assert tag_map[("A", "B")] == "data_confirmed"
        assert tag_map[("C", "D")] == "data_contradicted"
        assert tag_map[("E", "F")] == "data_contradicted"
        assert tag_map[("G", "H")] == "uncertain"

        # Case 2: FCI returns None (e.g., degenerate) -> all expert edges uncertain
        tagged_no_fci = ed.tag_edges(expert_edges, None)
        tag_map_no_fci = {(e["source"], e["target"]): e["tag"] for e in tagged_no_fci}
        assert tag_map_no_fci[("A", "B")] == "uncertain"
        assert tag_map_no_fci[("C", "D")] == "uncertain"
        assert tag_map_no_fci[("E", "F")] == "uncertain"


class TestPreviousDagDiff:
    """test_previous_dag_diff: --previous-dag mode computes edge diff correctly."""

    def test_previous_dag_diff(self, tmp_path: Path):
        old_dag = {
            "edges": [
                {"source": "A", "target": "B", "weight": 0.5},
                {"source": "C", "target": "D", "weight": 0.3},
            ]
        }
        new_dag = {
            "edges": [
                {"source": "A", "target": "B", "weight": 0.8},  # strengthened
                {"source": "E", "target": "F", "weight": 0.4},  # added
                # C->D removed
            ]
        }
        old_path = tmp_path / "old_dag.json"
        new_path = tmp_path / "new_dag.json"
        old_path.write_text(json.dumps(old_dag))
        new_path.write_text(json.dumps(new_dag))

        # Use dag_diff from common.py
        sys.path.insert(0, str(ROOT / "scripts" / "causal"))
        from common import dag_diff

        diff = dag_diff(str(old_path), str(new_path))
        assert "E -> F" in diff["edges_added"]
        assert "C -> D" in diff["edges_removed"]
        assert "A -> B" in diff["edges_strengthened"]


class TestDotRenders:
    """test_dot_renders: output dag.png exists and is non-empty."""

    def test_dot_renders(self, interventions_path: Path, output_dir: Path):
        dag = ed.estimate_dag(
            json.loads(interventions_path.read_text())["submissions"],
            alpha=0.01,
        )
        ed.render_dot(dag, output_dir / "dag.png")
        png = output_dir / "dag.png"
        assert png.exists()
        assert png.stat().st_size > 0


class TestNextInterventionRecommendation:
    """test_next_intervention_recommendation: non-null, valid format."""

    def test_next_intervention_recommendation(self, interventions_path: Path):
        data = json.loads(interventions_path.read_text())
        dag = ed.estimate_dag(data["submissions"], alpha=0.01)
        rec = dag.get("next_intervention")
        # Should provide a recommendation
        assert rec is not None
        assert "variable" in rec
        assert "expected_bpb_delta" in rec
        assert "rationale" in rec
        assert isinstance(rec["variable"], str)
        assert isinstance(rec["expected_bpb_delta"], (int, float))
