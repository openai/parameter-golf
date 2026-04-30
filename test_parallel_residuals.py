"""Local smoke for the PARALLEL_RESIDUALS / PARALLEL_RESIDUALS_START flags in train_gpt.py.

Runs on CPU. Imports GPT from train_gpt.py, builds three tiny models (default,
PR=on with default start, PR=on with start=0), runs one forward+backward each,
and checks: shapes match, loss finite, param count unchanged across configs,
parallel-residuals branch is actually exercised.

Usage:
    .venv/bin/python test_parallel_residuals.py
"""

from __future__ import annotations

import sys
import torch

import train_gpt as tg


def build(num_layers: int, parallel_residuals: bool, parallel_residuals_start: int) -> tg.GPT:
    model = tg.GPT(
        vocab_size=128,
        num_layers=num_layers,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        parallel_residuals=parallel_residuals,
        parallel_residuals_start=parallel_residuals_start,
    )
    # The model uses _zero_init=True on attn.proj and mlp.proj so the network starts as
    # identity, which means parallel- and sequential-residual branches are byte-identical
    # at init. Re-init those projections with small nonzero weights so the smoke can
    # actually distinguish the two branches.
    with torch.no_grad():
        for blk in model.blocks:
            torch.nn.init.normal_(blk.attn.proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(blk.mlp.proj.weight, mean=0.0, std=0.02)
    return model


def run_one(name: str, model: tg.GPT, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    grad_norm_sq = sum(p.grad.detach().float().pow(2).sum().item() for p in model.parameters() if p.grad is not None)
    pr_flags = [bool(b.parallel_residuals) for b in model.blocks]
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "name": name,
        "loss": loss.item(),
        "loss_finite": torch.isfinite(loss).item(),
        "grad_norm": grad_norm_sq ** 0.5,
        "n_params": n_params,
        "pr_flags": pr_flags,
    }


def main() -> int:
    torch.manual_seed(0)
    num_layers = 6
    bsz, seqlen = 2, 16

    x = torch.randint(0, 128, (bsz, seqlen), dtype=torch.long)
    y = torch.randint(0, 128, (bsz, seqlen), dtype=torch.long)

    configs = [
        ("baseline (PR=0)", False, -1),
        ("PR=1 default-start (= num_layers//2)", True, -1),
        ("PR=1 start=0 (all layers)", True, 0),
        ("PR=1 start=num_layers (no layers)", True, num_layers),
    ]

    results = []
    for name, pr, start in configs:
        torch.manual_seed(0)
        model = build(num_layers, pr, start)
        results.append(run_one(name, model, x, y))

    print(f"{'config':<42} {'loss':>10} {'finite':>7} {'grad_norm':>12} {'n_params':>10} pr_flags")
    print("-" * 110)
    for r in results:
        print(
            f"{r['name']:<42} "
            f"{r['loss']:>10.4f} {str(r['loss_finite']):>7} {r['grad_norm']:>12.4f} {r['n_params']:>10} "
            f"{r['pr_flags']}"
        )

    fail = []
    n_params_set = {r["n_params"] for r in results}
    if len(n_params_set) != 1:
        fail.append(f"param counts differ across configs: {n_params_set}")
    for r in results:
        if not r["loss_finite"]:
            fail.append(f"non-finite loss in {r['name']!r}")
    expected_flags = [
        [False] * num_layers,
        [False] * (num_layers // 2) + [True] * (num_layers - num_layers // 2),
        [True] * num_layers,
        [False] * num_layers,
    ]
    for r, ef in zip(results, expected_flags):
        if r["pr_flags"] != ef:
            fail.append(f"pr_flags mismatch in {r['name']!r}: got {r['pr_flags']} expected {ef}")
    base_loss = results[0]["loss"]
    pr_loss = results[1]["loss"]
    if abs(base_loss - pr_loss) < 1e-6:
        fail.append(
            f"baseline and PR-on losses identical ({base_loss:.6f}); the parallel-residuals branch may not be exercised"
        )

    if fail:
        print("\nFAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nOK: shapes/finite/param-count/branch-exercise all pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
