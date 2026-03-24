"""Shared utilities for the causal inference training pipeline.

Provides model loading, BPB computation, statistical testing,
experiment logging, and DAG comparison helpers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. load_submission_json
# ---------------------------------------------------------------------------

def load_submission_json(path: str) -> dict:
    """Parse a submission.json file and return its contents as a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"submission.json not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 2. load_model
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    config_overrides: dict[str, Any] | None = None,
) -> tuple:
    """Load a GPT model from a checkpoint and return (model, tokenizer).

    Uses train_gpt_mlx.py's GPT class and Hyperparameters.
    The checkpoint_path should point to a .safetensors weights file.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece as spm
    import train_gpt_mlx as tgm

    args = tgm.Hyperparameters()
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(args, k):
                object.__setattr__(args, k, v)

    model = tgm.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    weights = mx.load(checkpoint_path)
    model.load_weights(list(weights.items()))

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)

    return model, sp


# ---------------------------------------------------------------------------
# 3. compute_bpb
# ---------------------------------------------------------------------------

def compute_bpb(
    model,
    val_tokens: np.ndarray,
    sp_model,
    seq_len: int = 1024,
    batch_tokens: int = 524_288,
    grad_accum_steps: int = 8,
) -> float:
    """Compute bits-per-byte (BPB) on validation tokens.

    Reuses the eval_val logic from train_gpt_mlx.py:
        bpb = (mean_loss / ln(2)) * (total_tokens / total_bytes)
    """
    import mlx.core as mx
    import train_gpt_mlx as tgm

    args = tgm.Hyperparameters()
    args_dict = {
        "val_batch_size": batch_tokens,
        "grad_accum_steps": grad_accum_steps,
        "train_seq_len": seq_len,
    }
    for k, v in args_dict.items():
        object.__setattr__(args, k, v)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        tgm.build_sentencepiece_luts(sp_model, args.vocab_size)
    )

    compiled_loss = mx.compile(
        lambda x, y: model.loss(x, y),
        inputs=model.state,
        outputs=model.state,
    )

    _val_loss, val_bpb = tgm.eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    return val_bpb


# ---------------------------------------------------------------------------
# 4. paired_ttest
# ---------------------------------------------------------------------------

def paired_ttest(
    treatment: list[float],
    control: list[float],
    n_bootstrap: int = 10_000,
) -> tuple[float, float, float, float]:
    """Paired t-test with bootstrap 95% CI.

    Returns (mean_diff, ci_lo, ci_hi, p_value).
    mean_diff = mean(treatment) - mean(control).
    """
    t_arr = np.array(treatment, dtype=np.float64)
    c_arr = np.array(control, dtype=np.float64)
    diffs = t_arr - c_arr
    mean_diff = float(np.mean(diffs))

    # Paired t-test
    if np.all(diffs == 0):
        p_value = 1.0
    else:
        _t_stat, p_value = stats.ttest_rel(t_arr, c_arr)
        p_value = float(p_value)

    # Bootstrap 95% CI for mean difference
    rng = np.random.default_rng(seed=42)
    boot_means = np.empty(n_bootstrap)
    n = len(diffs)
    for i in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    return mean_diff, ci_lo, ci_hi, p_value


# ---------------------------------------------------------------------------
# 5. holm_bonferroni
# ---------------------------------------------------------------------------

def holm_bonferroni(
    p_values: list[float],
    alpha: float = 0.01,
) -> tuple[list[bool], list[float]]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns (reject_mask, adjusted_p_values).
    """
    from statsmodels.stats.multitest import multipletests

    reject, adjusted, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return list(reject), list(adjusted)


# ---------------------------------------------------------------------------
# 6. decision_gate
# ---------------------------------------------------------------------------

def decision_gate(
    effect_size: float,
    p_value: float,
    mde: float = 0.002,
) -> str:
    """Apply the decision gate protocol.

    Returns:
        "confirmed"  -- |effect_size| >= mde AND p_value < 0.01
        "suggestive" -- |effect_size| >= mde AND p_value >= 0.01
        "null"       -- |effect_size| < mde
    """
    if abs(effect_size) >= mde:
        if p_value < 0.01:
            return "confirmed"
        return "suggestive"
    return "null"


# ---------------------------------------------------------------------------
# 7. log_experiment
# ---------------------------------------------------------------------------

def log_experiment(path: str, entry: dict) -> None:
    """Append an experiment entry to the master experiment_log.json.

    Creates the file with {"experiments": []} if it does not exist.
    """
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"experiments": []}

    data["experiments"].append(entry)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 8. get_cycle_dir
# ---------------------------------------------------------------------------

def get_cycle_dir(base_path: str, cycle: int) -> Path:
    """Return Path(base_path) / 'cycle_{cycle}', creating it if needed."""
    d = Path(base_path) / f"cycle_{cycle}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 9. dag_diff
# ---------------------------------------------------------------------------

def dag_diff(old_dag_path: str, new_dag_path: str) -> dict:
    """Compare two causal_dag.json files.

    Returns {edges_added: [...], edges_removed: [...], edges_strengthened: [...]}.
    Edge format: "source -> target".
    """
    old = json.loads(Path(old_dag_path).read_text(encoding="utf-8"))
    new = json.loads(Path(new_dag_path).read_text(encoding="utf-8"))

    def edge_map(dag: dict) -> dict[str, float]:
        result = {}
        for e in dag.get("edges", []):
            key = f"{e['source']} -> {e['target']}"
            result[key] = e.get("weight", 0.0)
        return result

    old_edges = edge_map(old)
    new_edges = edge_map(new)

    old_keys = set(old_edges)
    new_keys = set(new_edges)

    edges_added = sorted(new_keys - old_keys)
    edges_removed = sorted(old_keys - new_keys)

    # Strengthened: edge exists in both but weight increased (absolute value)
    edges_strengthened = []
    for key in sorted(old_keys & new_keys):
        if abs(new_edges[key]) > abs(old_edges[key]):
            edges_strengthened.append(key)

    return {
        "edges_added": edges_added,
        "edges_removed": edges_removed,
        "edges_strengthened": edges_strengthened,
    }
