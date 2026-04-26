"""Verification script for 0032 (Mamba-2 / SSD smoke).

Run with the experiment env sourced, e.g.:
    cd experiments/0032_mamba2_ssd_smoke
    set -a; source env.sh; set +a
    python scratch_verify.py

Does NOT run training. Performs the following checks:
  1. Constructs the full GPT model with MAMBA2_LAYER_POSITIONS=1.
  2. Prints all named parameters under each Mamba2Block instance.
  3. Prints which of those params are matched by CONTROL_TENSOR_NAME_PATTERNS.
     Verifies A_log, D_skip, conv1d.weight, conv1d.bias, dt_bias all matched
     (fp32-protected).
  4. Numerical oracle: at small input scale, the SSD chunkwise scan output
     equals a sequential per-step selective-scan reference to atol=1e-3,
     rtol=1e-2.
  5. Runs one B=2, L=8 forward through the full GPT and prints loss.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

sys.path.insert(0, str(HERE))
import train_gpt  # noqa: E402
from train_gpt import (  # noqa: E402
    CONTROL_TENSOR_NAME_PATTERNS,
    GPT,
    Hyperparameters,
    Mamba2Block,
    ssd_minimal_discrete,
)


def _build_model(args: Hyperparameters, device: torch.device) -> GPT:
    attn_layer_positions = {
        int(x) for x in args.attn_layer_positions.split(",") if x.strip()
    }
    mamba2_layer_positions = {
        int(x) for x in args.mamba2_layer_positions.split(",") if x.strip()
    }
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        num_unique_layers=args.num_unique_layers,
        num_loops=args.num_loops,
        attn_layer_positions=attn_layer_positions,
        mamba2_layer_positions=mamba2_layer_positions,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    )
    return model.to(device)


def _sequential_ssd_reference(
    X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """Sequential per-step selective-SSM scan with scalar A per head.

    Mirrors the SSD recurrence:
        h_t = exp(A_t) * h_{t-1} + B_t * X_t
        y_t = C_t^T h_t
    where A_t is the dt-scaled scalar decay per head, and X already absorbs
    dt (matches `ssd_minimal_discrete`'s calling convention).

    Shapes:
        X: (b, L, h, p)  -- dt-scaled inputs
        A: (b, L, h)     -- dt-scaled scalar decay
        B: (b, L, h, n)
        C: (b, L, h, n)
    Returns:
        Y: (b, L, h, p)
    """
    b, L, h, p = X.shape
    n = B.size(-1)
    state = torch.zeros((b, h, p, n), dtype=X.dtype, device=X.device)
    ys = []
    for t in range(L):
        a_t = torch.exp(A[:, t]).unsqueeze(-1).unsqueeze(-1)  # (b, h, 1, 1)
        # outer(X_t, B_t) over (p, n): (b, h, p) * (b, h, n) -> (b, h, p, n)
        outer = X[:, t].unsqueeze(-1) * B[:, t].unsqueeze(-2)
        state = a_t * state + outer
        # y_t = sum_n C_t[h, n] * state[h, p, n]  -> (b, h, p)
        y_t = torch.einsum("bhpn,bhn->bhp", state, C[:, t])
        ys.append(y_t)
    return torch.stack(ys, dim=1)  # (b, L, h, p)


def check_2_3_named_params(model: GPT) -> None:
    print("\n=== Check 2: named parameters on Mamba2Block instances ===")
    print(f"CONTROL_TENSOR_NAME_PATTERNS = {CONTROL_TENSOR_NAME_PATTERNS}")
    mamba2_param_names: list[str] = []
    for block_idx, block in enumerate(model.blocks):
        if isinstance(block.attn, Mamba2Block):
            for n, p in block.attn.named_parameters():
                full = f"blocks.{block_idx}.attn.{n}"
                mamba2_param_names.append(full)
                print(f"  {full:50s} shape={tuple(p.shape)} dtype={p.dtype}")
    if not mamba2_param_names:
        raise RuntimeError(
            "No Mamba2Block found in model.blocks. Did MAMBA2_LAYER_POSITIONS get set?"
        )

    print("\n=== Check 3: which Mamba-2 params match CONTROL_TENSOR_NAME_PATTERNS? ===")
    matched_names: set[str] = set()
    for name in mamba2_param_names:
        hit = [pat for pat in CONTROL_TENSOR_NAME_PATTERNS if pat in name]
        if hit:
            matched_names.add(name)
            print(f"  MATCH   {name:50s} via {hit}")
        else:
            print(f"  no-hit  {name}")

    # Required substrings: A_log, D_skip, conv1d.weight, conv1d.bias, dt_bias.
    # (conv1d.bias is also auto-restored as 1D, but we check via the pattern
    # for symmetry with conv1d.weight.)
    required = ["A_log", "D_skip", "conv1d.weight", "conv1d.bias", "dt_bias"]
    missing = [req for req in required if not any(req in n for n in matched_names)]
    if missing:
        raise AssertionError(
            f"Required fp32-protected substrings missing from matched set: {missing}.\n"
            f"matched={sorted(matched_names)}"
        )
    print("  PASS: A_log, D_skip, conv1d.weight, conv1d.bias, dt_bias all matched.")


def check_4_chunkwise_vs_sequential() -> None:
    """SSD chunkwise scan vs sequential reference at small scale."""
    print("\n=== Check 4: ssd_minimal_discrete vs sequential reference ===")
    torch.manual_seed(0)
    b, L, h, p = 2, 16, 4, 16  # nheads=4, headdim=16
    n = 8  # d_state
    chunk = 8  # block_len divides L=16

    X = torch.randn(b, L, h, p, dtype=torch.float32)
    # dt sampled small + softplus to keep stable.
    dt = torch.nn.functional.softplus(
        torch.randn(b, L, h, dtype=torch.float32) - 2.0
    )
    A_base = -(torch.rand(h, dtype=torch.float32) + 0.1)  # negative reals
    A = A_base[None, None, :].expand(b, L, h) * dt
    X_disc = X * dt.unsqueeze(-1)
    B = torch.randn(b, L, h, n, dtype=torch.float32)
    C = torch.randn(b, L, h, n, dtype=torch.float32)

    # Chunkwise.
    Y_chunk, _ = ssd_minimal_discrete(X_disc, A, B, C, block_len=chunk)
    # Sequential.
    Y_seq = _sequential_ssd_reference(X_disc, A, B, C)

    max_abs = (Y_chunk - Y_seq).abs().max().item()
    print(f"  Y_chunk.shape = {tuple(Y_chunk.shape)}, Y_seq.shape = {tuple(Y_seq.shape)}")
    print(f"  max abs diff: {max_abs:.3e}")
    if not torch.allclose(Y_chunk, Y_seq, atol=1e-3, rtol=1e-2):
        raise AssertionError(
            f"SSD chunkwise disagrees with sequential reference: "
            f"max abs diff {max_abs:.3e}"
        )
    print("  PASS: torch.allclose(atol=1e-3, rtol=1e-2)")


def check_5_forward(model: GPT, device: torch.device) -> None:
    print("\n=== Check 5: one forward (B=2, L=8) ===")
    # NOTE: Mamba2Block requires L % chunk_size == 0 (chunk_size=64). For
    # this quick correctness check we briefly drop chunk_size to 8 on the
    # constructed Mamba2Block(s) so a tiny L=8 forward executes cleanly,
    # then restore the original setting before exiting.
    _orig_chunks: list[tuple[Mamba2Block, int]] = []
    for block in model.blocks:
        if isinstance(block.attn, Mamba2Block):
            _orig_chunks.append((block.attn, block.attn.chunk_size))
            block.attn.chunk_size = 8
    try:
        model.eval()
        B, L = 2, 8
        input_ids = torch.randint(0, model.tok_emb.num_embeddings, (B, L), device=device)
        target_ids = torch.randint(0, model.tok_emb.num_embeddings, (B, L), device=device)
        with torch.no_grad():
            loss = model(input_ids, target_ids)
        print(f"  loss = {loss.item():.6f}, loss.shape = {tuple(loss.shape)}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is non-finite: {loss}")
    finally:
        for blk, orig in _orig_chunks:
            blk.chunk_size = orig


def main() -> None:
    args = Hyperparameters()
    print(f"RUN_ID = {args.run_id}")
    print(f"ATTN_LAYER_POSITIONS = {args.attn_layer_positions!r}")
    print(f"MAMBA2_LAYER_POSITIONS = {args.mamba2_layer_positions!r}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device = {device}")

    print("\n=== Check 1: constructing GPT ===")
    model = _build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model_params = {n_params}")
    print(f"  recurrent = {getattr(model, 'recurrent', False)}")
    print(f"  num_blocks = {len(model.blocks)}")
    for i, block in enumerate(model.blocks):
        kind = type(block.attn).__name__
        print(f"  block[{i}].attn = {kind}")

    check_2_3_named_params(model)
    # Numerical oracle runs on CPU/fp32 (independent of model device).
    check_4_chunkwise_vs_sequential()
    check_5_forward(model, device)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
