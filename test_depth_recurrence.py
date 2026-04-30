"""Local smoke for the depth-recurrence flags in train_gpt.py.

Runs on CPU. Imports GPT from train_gpt.py, builds tiny models with various
combinations of TRAINING/EVAL_DEPTH_RECURRENCE and span knobs, and checks:
  * recurrence=1 is byte-identical to baseline (no-op verified)
  * recurrence=3 changes the output
  * train vs eval modes use the right recurrence value
  * layer-span knobs limit which layers loop (verified by counting block calls)
  * U-Net skips still balance (forward completes; final shape correct)
  * state_dict round-trips through save/load on a fresh module (export/reload sanity)

Usage:
    .venv/bin/python test_depth_recurrence.py
"""

from __future__ import annotations

import io
import sys
import torch

import train_gpt as tg


def build(
    num_layers: int,
    *,
    parallel_residuals: bool = False,
    parallel_residuals_start: int = -1,
    training_depth_recurrence: int = 1,
    eval_depth_recurrence: int = 1,
    depth_recurrence_layer_start: int = 0,
    depth_recurrence_layer_end: int = -1,
) -> tg.GPT:
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
        training_depth_recurrence=training_depth_recurrence,
        eval_depth_recurrence=eval_depth_recurrence,
        depth_recurrence_layer_start=depth_recurrence_layer_start,
        depth_recurrence_layer_end=depth_recurrence_layer_end,
    )
    # Make _zero_init projections nonzero so block calls actually change x.
    with torch.no_grad():
        for blk in model.blocks:
            torch.nn.init.normal_(blk.attn.proj.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(blk.mlp.proj.weight, mean=0.0, std=0.02)
    return model


def call_count_per_block(model: tg.GPT, x: torch.Tensor, y: torch.Tensor) -> list[int]:
    """Wrap each block's forward to count how many times it's called in one full forward."""
    counts = [0] * len(model.blocks)
    originals = []
    for i, blk in enumerate(model.blocks):
        originals.append(blk.forward)

        def make_wrap(i_local, orig):
            def wrap(x_in, x0):
                counts[i_local] += 1
                return orig(x_in, x0)
            return wrap

        blk.forward = make_wrap(i, blk.forward)
    try:
        _ = model(x, y)
    finally:
        for blk, orig in zip(model.blocks, originals):
            blk.forward = orig
    return counts


def main() -> int:
    torch.manual_seed(0)
    num_layers = 6
    bsz, seqlen = 2, 16

    x = torch.randint(0, 128, (bsz, seqlen), dtype=torch.long)
    y = torch.randint(0, 128, (bsz, seqlen), dtype=torch.long)

    fail: list[str] = []

    # --- Test 1: recurrence=1 is a no-op vs baseline (byte-identical loss) ---
    torch.manual_seed(0); m_base = build(num_layers); m_base.train()
    torch.manual_seed(0); m_r1 = build(num_layers, training_depth_recurrence=1, eval_depth_recurrence=1); m_r1.train()
    loss_base = m_base(x, y).item()
    loss_r1 = m_r1(x, y).item()
    if abs(loss_base - loss_r1) > 1e-9:
        fail.append(f"recurrence=1 no-op broken: baseline={loss_base} vs r1={loss_r1}")
    print(f"Test 1 baseline=recurrence=1 no-op:                          base={loss_base:.6f} r1={loss_r1:.6f}")

    # --- Test 2: recurrence=3 changes the loss ---
    torch.manual_seed(0); m_r3 = build(num_layers, training_depth_recurrence=3); m_r3.train()
    loss_r3 = m_r3(x, y).item()
    if abs(loss_base - loss_r3) < 1e-6:
        fail.append(f"training_depth_recurrence=3 produced identical loss to baseline ({loss_r3})")
    if not torch.isfinite(torch.tensor(loss_r3)):
        fail.append(f"training_depth_recurrence=3 loss non-finite: {loss_r3}")
    print(f"Test 2 recurrence=3 changes loss + finite:                    r3={loss_r3:.6f} (base={loss_base:.6f})")

    # --- Test 3: train vs eval recurrence selection ---
    torch.manual_seed(0); m_split = build(num_layers, training_depth_recurrence=1, eval_depth_recurrence=3)
    m_split.train(); loss_train = m_split(x, y).item()
    m_split.eval();
    with torch.no_grad():
        loss_eval = m_split(x, y).item()
    # In train mode it should match baseline; in eval it should differ.
    if abs(loss_train - loss_base) > 1e-9:
        fail.append(f"train mode recurrence=1 should match baseline: got {loss_train} vs {loss_base}")
    if abs(loss_eval - loss_train) < 1e-6:
        fail.append(f"eval mode recurrence=3 should differ from train mode: train={loss_train} eval={loss_eval}")
    print(f"Test 3 train uses training_dr / eval uses eval_dr:            train={loss_train:.6f} eval={loss_eval:.6f}")

    # --- Test 4: layer-span knobs ---
    # span = [2, 4) with recurrence=3 should call only blocks 2 and 3 three times each, others once.
    torch.manual_seed(0)
    m_span = build(num_layers, training_depth_recurrence=3, depth_recurrence_layer_start=2, depth_recurrence_layer_end=4)
    m_span.train()
    counts = call_count_per_block(m_span, x, y)
    expected = [1, 1, 3, 3, 1, 1]
    if counts != expected:
        fail.append(f"span [2,4) recurrence=3 expected {expected} block calls, got {counts}")
    print(f"Test 4 layer-span [2,4) recurrence=3 calls:                   {counts} (expected {expected})")

    # --- Test 5: U-Net skip stack still balances + final shape correct ---
    torch.manual_seed(0)
    m_full = build(num_layers, training_depth_recurrence=3); m_full.train()
    out = m_full(x, y)
    if out.ndim != 0:
        fail.append(f"loss should be scalar, got shape {tuple(out.shape)}")
    if not torch.isfinite(out):
        fail.append(f"recurrence=3 full-span loss non-finite: {out}")
    print(f"Test 5 U-Net skip balance + scalar loss:                      loss={out.item():.6f} ndim={out.ndim}")

    # --- Test 6: state_dict round-trip (export/reload still works) ---
    torch.manual_seed(0)
    m_a = build(num_layers, training_depth_recurrence=3, depth_recurrence_layer_start=1, depth_recurrence_layer_end=5)
    buf = io.BytesIO()
    torch.save(m_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    m_b = build(num_layers, training_depth_recurrence=3, depth_recurrence_layer_start=1, depth_recurrence_layer_end=5)
    m_b.load_state_dict(sd, strict=True)
    m_a.train(); m_b.train()
    la = m_a(x, y).item()
    lb = m_b(x, y).item()
    if abs(la - lb) > 1e-6:
        fail.append(f"state_dict round-trip changed loss: {la} vs {lb}")
    print(f"Test 6 state_dict round-trip preserves loss:                  before={la:.6f} after={lb:.6f}")

    # --- Test 7: param count unchanged ---
    n_base = sum(p.numel() for p in m_base.parameters())
    n_r3 = sum(p.numel() for p in m_r3.parameters())
    n_span = sum(p.numel() for p in m_span.parameters())
    if not (n_base == n_r3 == n_span):
        fail.append(f"param counts differ: base={n_base} r3={n_r3} span={n_span}")
    print(f"Test 7 param counts equal across configs:                     {n_base}")

    if fail:
        print("\nFAIL:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nOK: depth-recurrence smoke passes (no-op / change / mode-switch / span / U-Net / round-trip / params).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
