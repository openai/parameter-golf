"""Causal Structure Discovery (C3).

Builds an expert-guided causal DAG from intervention data,
optionally validates with FCI, produces versioned JSON output
and DOT visualization.

CLI:
  python scripts/causal/estimate_dag.py \
    --input results/causal/interventions.json \
    --output-dir results/causal/cycle_0/ \
    [--previous-dag results/causal/cycle_N-1/causal_dag.json] \
    [--alpha 0.01]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expert-guided skeleton (known causal edges)
# ---------------------------------------------------------------------------

_EXPERT_EDGES: list[tuple[str, str]] = [
    ("num_layers", "bpb"),
    ("mlp_mult", "bpb"),
    ("quant_method", "bpb"),
    ("weight_avg_method", "bpb"),
    ("attention_variant", "bpb"),
    ("rope_variant", "bpb"),
    ("compression", "bpb"),
]

# All node names (treatment variables + outcome)
_NODES = [
    "num_layers",
    "mlp_mult",
    "quant_method",
    "weight_avg_method",
    "attention_variant",
    "rope_variant",
    "compression",
    "bpb",
]

# Category -> possible values for binary encoding
_CATEGORY_VALUES: dict[str, list[str]] = {
    "num_layers": ["10", "11", "12"],
    "mlp_mult": ["2", "3", "4"],
    "quant_method": ["none", "int8", "gptq"],
    "weight_avg_method": ["none", "ema", "swa"],
    "attention_variant": ["standard", "gqa"],
    "rope_variant": ["standard", "yarn"],
    "compression": ["none", "pruning"],
}


def build_expert_skeleton() -> list[dict[str, str]]:
    """Return a list of expert-imposed directed edge dicts."""
    return [
        {"source": src, "target": tgt, "tag": "expert_imposed"}
        for src, tgt in _EXPERT_EDGES
    ]


# ---------------------------------------------------------------------------
# Binary encoding
# ---------------------------------------------------------------------------


def binary_encode(
    submissions: list[dict],
) -> tuple[np.ndarray, list[str]]:
    """Convert submissions to a binary presence/absence matrix.

    Each submission maps to a row. Each (category, value) pair becomes a
    binary column indicating whether that intervention was applied.
    Returns (matrix, node_names) where node_names lists column labels.
    """
    # Collect all (category, value) pairs across submissions
    col_pairs: list[tuple[str, str]] = []
    for cat, vals in sorted(_CATEGORY_VALUES.items()):
        for v in vals:
            col_pairs.append((cat, v))

    node_names = [f"{cat}={val}" for cat, val in col_pairs]
    n = len(submissions)
    m = len(col_pairs)
    matrix = np.zeros((n, m), dtype=np.float64)

    for i, sub in enumerate(submissions):
        present: dict[str, str] = {}
        for iv in sub.get("interventions", []):
            present[iv["category"]] = iv["name"]
        for j, (cat, val) in enumerate(col_pairs):
            if present.get(cat) == val:
                matrix[i, j] = 1.0

    return matrix, node_names


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------


def estimate_marginal_effects(
    submissions: list[dict],
) -> list[dict[str, Any]]:
    """Estimate marginal effect of each node category on BPB.

    For each category, compare mean BPB of submissions with any
    non-default intervention vs those without.
    """
    # Gather BPB values
    effects: list[dict[str, Any]] = []
    for cat in sorted(_CATEGORY_VALUES.keys()):
        with_intervention: list[float] = []
        without_intervention: list[float] = []
        for sub in submissions:
            bpb = sub.get("final_bpb")
            if bpb is None:
                continue
            cats_present = {iv["category"] for iv in sub.get("interventions", [])}
            if cat in cats_present:
                with_intervention.append(bpb)
            else:
                without_intervention.append(bpb)

        if with_intervention and without_intervention:
            effect = float(np.mean(with_intervention) - np.mean(without_intervention))
        else:
            effect = 0.0

        n_obs = len(with_intervention) + len(without_intervention)
        confidence = "medium"  # expert_imposed baseline
        effects.append(
            {
                "node": cat,
                "marginal_effect_on_bpb": round(effect, 6),
                "n_observations": n_obs,
                "confidence": confidence,
            }
        )
    return effects


# ---------------------------------------------------------------------------
# FCI validation
# ---------------------------------------------------------------------------


def run_fci_validation(
    data: np.ndarray,
    node_names: list[str],
    alpha: float = 0.01,
) -> dict | None:
    """Run FCI from causal-learn on the data matrix.

    Returns a dict with 'edges' list and 'degenerate' flag, or None
    if the result is degenerate (empty or fully-connected).
    """
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    n_samples, n_vars = data.shape

    # Degenerate detection: check for near-zero variance columns
    col_stds = np.std(data, axis=0)
    if np.any(col_stds < 1e-10):
        logger.warning("Near-zero variance columns detected, skipping FCI")
        return {"edges": [], "degenerate": True}

    # Check for nearly identical columns (high correlation)
    if n_vars >= 2:
        corr = np.corrcoef(data, rowvar=False)
        # Upper triangle excluding diagonal
        triu_mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        max_corr = np.max(np.abs(corr[triu_mask])) if np.any(triu_mask) else 0.0
        if max_corr > 0.999:
            logger.warning("Near-degenerate correlation matrix (max_corr=%.4f), skipping FCI", max_corr)
            return {"edges": [], "degenerate": True}

    try:
        g, edges_result = fci(data, fisherz, alpha, verbose=False)
    except np.linalg.LinAlgError:
        logger.warning("LinAlgError in FCI, returning degenerate")
        return None
    except Exception as exc:
        logger.warning("FCI failed with %s: %s", type(exc).__name__, exc)
        return None

    # Parse the FCI graph result
    adj = g.graph  # shape (n_vars, n_vars)
    fci_edges = []

    # Degenerate check: empty graph or fully-connected
    n_edges = 0
    max_possible = n_vars * (n_vars - 1) // 2
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if adj[i, j] != 0 or adj[j, i] != 0:
                n_edges += 1

    if n_edges == 0:
        logger.warning("FCI produced empty graph, marking degenerate")
        return {"edges": [], "degenerate": True}

    if max_possible > 0 and n_edges == max_possible:
        logger.warning("FCI produced fully-connected graph, marking degenerate")
        return {"edges": [], "degenerate": True}

    # Extract edges
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if adj[i, j] != 0 or adj[j, i] != 0:
                # Determine edge direction based on FCI endpoint marks:
                #  -1 = tail, 1 = arrowhead, 2 = circle
                mark_ij = adj[i, j]
                mark_ji = adj[j, i]

                if mark_ij == -1 and mark_ji == 1:
                    # i -> j
                    fci_edges.append({"source": node_names[i], "target": node_names[j]})
                elif mark_ij == 1 and mark_ji == -1:
                    # j -> i
                    fci_edges.append({"source": node_names[j], "target": node_names[i]})
                else:
                    # bidirected, circle, or uncertain
                    fci_edges.append({"source": node_names[i], "target": node_names[j]})

    return {"edges": fci_edges, "degenerate": False}


# ---------------------------------------------------------------------------
# Edge tagging
# ---------------------------------------------------------------------------


def tag_edges(
    expert_edges: list[dict],
    fci_edges: list[dict] | None,
) -> list[dict]:
    """Compare expert skeleton with FCI results and tag each edge.

    Tags: data_confirmed, data_contradicted, uncertain.
    Expert edges not in FCI -> data_contradicted.
    Expert edges also in FCI -> data_confirmed.
    Expert edges when FCI is None -> uncertain.
    FCI edges not in expert -> uncertain (new from data).
    """
    if fci_edges is None:
        # No FCI results: all expert edges stay uncertain
        return [
            {**e, "tag": "uncertain"} for e in expert_edges
        ]

    fci_pairs = {(e["source"], e["target"]) for e in fci_edges}
    # Also consider reversed direction as a match for undirected relationships
    fci_pairs_either = set()
    for e in fci_edges:
        fci_pairs_either.add((e["source"], e["target"]))
        fci_pairs_either.add((e["target"], e["source"]))

    expert_pairs = {(e["source"], e["target"]) for e in expert_edges}

    tagged = []

    for e in expert_edges:
        pair = (e["source"], e["target"])
        if pair in fci_pairs_either:
            tagged.append({**e, "tag": "data_confirmed"})
        elif len(fci_edges) == 0:
            # FCI ran but found nothing -> keep as uncertain
            tagged.append({**e, "tag": "uncertain"})
        else:
            tagged.append({**e, "tag": "data_contradicted"})

    # Add FCI-only edges as uncertain
    for e in fci_edges:
        pair = (e["source"], e["target"])
        rev = (e["target"], e["source"])
        if pair not in expert_pairs and rev not in expert_pairs:
            tagged.append({"source": e["source"], "target": e["target"], "tag": "uncertain"})

    return tagged


# ---------------------------------------------------------------------------
# Next intervention recommendation
# ---------------------------------------------------------------------------


def recommend_next_intervention(
    effects: list[dict[str, Any]],
    tagged_edges: list[dict],
) -> dict | None:
    """Recommend the next intervention for highest expected BPB improvement.

    Prioritises uncertain edges with the largest absolute marginal effect.
    """
    uncertain_nodes = set()
    for e in tagged_edges:
        if e["tag"] == "uncertain" and e["source"] != "bpb":
            uncertain_nodes.add(e["source"])

    if not uncertain_nodes:
        # Also consider data_contradicted as candidates
        for e in tagged_edges:
            if e["tag"] == "data_contradicted" and e["source"] != "bpb":
                uncertain_nodes.add(e["source"])

    if not uncertain_nodes:
        return None

    # Find the effect with the largest absolute impact among uncertain nodes
    best: dict | None = None
    best_abs = -1.0
    for eff in effects:
        if eff["node"] in uncertain_nodes:
            abs_effect = abs(eff["marginal_effect_on_bpb"])
            if abs_effect > best_abs:
                best_abs = abs_effect
                best = eff

    if best is None:
        return None

    # Suggest a value: the most common non-default value from category
    cat = best["node"]
    vals = _CATEGORY_VALUES.get(cat, [])
    suggested = vals[1] if len(vals) > 1 else vals[0] if vals else "unknown"

    return {
        "variable": best["node"],
        "suggested_value": suggested,
        "expected_bpb_delta": best["marginal_effect_on_bpb"],
        "rationale": (
            f"Uncertain edge {best['node']} -> bpb with marginal effect "
            f"{best['marginal_effect_on_bpb']:.4f}; needs experimental validation"
        ),
    }


# ---------------------------------------------------------------------------
# Main DAG estimation
# ---------------------------------------------------------------------------


def estimate_dag(
    submissions: list[dict],
    alpha: float = 0.01,
    previous_dag: dict | None = None,
) -> dict:
    """Build the causal DAG from submissions.

    1. Build expert skeleton
    2. Binary encode submissions
    3. Run FCI validation (with fallback)
    4. Tag edges
    5. Estimate marginal effects
    6. Produce recommendation
    """
    # Step 1: Expert skeleton
    expert_edges = build_expert_skeleton()

    # Step 2: Binary encoding + BPB column
    matrix, col_names = binary_encode(submissions)
    bpb_col = np.array(
        [s.get("final_bpb", 1.15) for s in submissions], dtype=np.float64
    ).reshape(-1, 1)
    full_data = np.hstack([matrix, bpb_col])
    full_names = col_names + ["bpb"]

    # Step 3: FCI validation
    fci_degenerate = False
    fci_result = None
    try:
        fci_result = run_fci_validation(full_data, full_names, alpha=alpha)
        if fci_result is None or fci_result.get("degenerate"):
            fci_degenerate = True
            fci_result = None
    except np.linalg.LinAlgError:
        logger.warning("LinAlgError during FCI, falling back to expert DAG")
        fci_degenerate = True
    except Exception as exc:
        logger.warning("FCI failed (%s), falling back to expert DAG", exc)
        fci_degenerate = True

    # Step 4: Tag edges
    fci_edges = fci_result["edges"] if fci_result else None
    tagged_edges = tag_edges(expert_edges, fci_edges)

    # Step 5: Marginal effects
    effects = estimate_marginal_effects(submissions)

    # Update confidence in effects based on edge tags
    tag_map = {e["source"]: e["tag"] for e in tagged_edges if e["target"] == "bpb"}
    for eff in effects:
        node_tag = tag_map.get(eff["node"], "uncertain")
        if node_tag == "data_confirmed":
            eff["confidence"] = "high"
        elif node_tag == "expert_imposed":
            eff["confidence"] = "medium"
        else:
            eff["confidence"] = "low"

    # Step 6: Recommendation
    recommendation = recommend_next_intervention(effects, tagged_edges)

    # Count tag categories
    confirmed_count = sum(1 for e in tagged_edges if e["tag"] == "data_confirmed")
    contradicted_count = sum(1 for e in tagged_edges if e["tag"] == "data_contradicted")

    # Build adjacency matrix (for _NODES ordering)
    n_nodes = len(_NODES)
    adj_matrix = [[0] * n_nodes for _ in range(n_nodes)]
    node_idx = {n: i for i, n in enumerate(_NODES)}
    for e in tagged_edges:
        src_i = node_idx.get(e["source"])
        tgt_i = node_idx.get(e["target"])
        if src_i is not None and tgt_i is not None:
            adj_matrix[src_i][tgt_i] = 1

    # Compute stability from previous_dag
    prev_edge_set: set[tuple[str, str]] = set()
    if previous_dag:
        for e in previous_dag.get("edges", []):
            prev_edge_set.add((e.get("source", e.get("from", "")), e.get("target", e.get("to", ""))))

    edges_out = []
    for e in tagged_edges:
        pair = (e["source"], e["target"])
        stability = 1
        if previous_dag:
            if pair in prev_edge_set:
                # Find previous stability
                for pe in previous_dag.get("edges", []):
                    psrc = pe.get("source", pe.get("from", ""))
                    ptgt = pe.get("target", pe.get("to", ""))
                    if (psrc, ptgt) == pair:
                        stability = pe.get("stability", 0) + 1
                        break

        # Find effect estimate
        effect_est = None
        for eff in effects:
            if eff["node"] == e["source"]:
                effect_est = eff["marginal_effect_on_bpb"]
                break

        edge_type = "directed"
        edges_out.append(
            {
                "source": e["source"],
                "target": e["target"],
                "type": edge_type,
                "tag": e["tag"],
                "stability": stability,
                "effect_estimate": effect_est,
                "weight": abs(effect_est) if effect_est else 0.0,
            }
        )

    # Determine cycle number
    cycle = 0
    if previous_dag:
        cycle = previous_dag.get("cycle", 0) + 1

    method = "expert_guided"
    if not fci_degenerate and fci_result:
        method = "expert_guided+fci_validation"

    dag = {
        "cycle": cycle,
        "method": method,
        "nodes": _NODES,
        "adjacency_matrix": adj_matrix,
        "edges": edges_out,
        "estimated_effects": effects,
        "next_intervention": recommendation,
        "metadata": {
            "n_samples": len(submissions),
            "alpha": alpha,
            "fci_degenerate": fci_degenerate,
            "expert_edges_count": len(_EXPERT_EDGES),
            "data_confirmed_count": confirmed_count,
            "data_contradicted_count": contradicted_count,
        },
    }

    return dag


# ---------------------------------------------------------------------------
# DOT visualization
# ---------------------------------------------------------------------------


def render_dot(dag: dict, output_path: str | Path) -> Path:
    """Render the DAG as a PNG via graphviz."""
    import graphviz

    output_path = Path(output_path)
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR")

    for node in dag.get("nodes", []):
        if node == "bpb":
            dot.node(node, shape="doublecircle", style="filled", fillcolor="lightblue")
        else:
            dot.node(node, shape="box")

    tag_colors = {
        "data_confirmed": "green",
        "data_contradicted": "red",
        "expert_imposed": "blue",
        "uncertain": "gray",
    }

    for edge in dag.get("edges", []):
        color = tag_colors.get(edge.get("tag", ""), "black")
        label = edge.get("tag", "")
        if edge.get("effect_estimate") is not None:
            label += f"\n{edge['effect_estimate']:.4f}"
        dot.edge(edge["source"], edge["target"], color=color, label=label)

    # Render: graphviz appends .png to the filename, so we strip it
    stem = str(output_path).removesuffix(".png")
    dot.render(stem, cleanup=True)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal Structure Discovery (C3)")
    parser.add_argument("--input", required=True, help="Path to interventions.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for cycle results")
    parser.add_argument("--previous-dag", default=None, help="Path to previous cycle causal_dag.json")
    parser.add_argument("--alpha", type=float, default=0.01, help="FCI significance threshold")
    args = parser.parse_args()

    # Load input
    interventions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    submissions = interventions["submissions"]

    # Load previous DAG if specified
    previous_dag = None
    if args.previous_dag:
        previous_dag = json.loads(Path(args.previous_dag).read_text(encoding="utf-8"))

    # Estimate DAG
    dag = estimate_dag(submissions, alpha=args.alpha, previous_dag=previous_dag)

    # Write output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dag_path = out_dir / "causal_dag.json"
    dag_path.write_text(json.dumps(dag, indent=2), encoding="utf-8")
    logger.info("Wrote %s", dag_path)

    # Render DOT
    png_path = out_dir / "dag.png"
    render_dot(dag, png_path)
    logger.info("Rendered %s", png_path)

    # Compute diff if previous DAG provided
    if args.previous_dag:
        from scripts.causal.common import dag_diff

        diff = dag_diff(args.previous_dag, str(dag_path))
        diff_path = out_dir / "dag_diff.json"
        diff_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
        logger.info("Wrote diff: %s", diff_path)

    # Summary
    meta = dag["metadata"]
    print(f"DAG estimated: {meta['n_samples']} samples, "
          f"alpha={meta['alpha']}, "
          f"FCI degenerate={meta['fci_degenerate']}, "
          f"confirmed={meta['data_confirmed_count']}, "
          f"contradicted={meta['data_contradicted_count']}")
    if dag["next_intervention"]:
        rec = dag["next_intervention"]
        print(f"Next intervention: {rec['variable']} = {rec['suggested_value']} "
              f"(expected delta: {rec['expected_bpb_delta']:.4f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
