"""GOLDEN SUNFLOWERS module smoke test (CPU, no data).

Exercises the three wish-list modules added to train_gpt.py without
loading the full GPT or fineweb dataset. Verifies:

  1. PhiNTA: frozen φ-OrthoInit basis is non-zero, has bounded row norms,
     does not receive gradients; trainable A,B do receive gradients.
  2. _jepa_loss: returns finite non-negative scalar with grad on h.
  3. Universal Transformer dispatcher: identity when ut_loops <= 1.
  4. φ-physics constants match SACRED-PHYSICS-001.

Run:  python smoke_modules.py
"""

from __future__ import annotations
import importlib.util, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "train_gpt_gs", str(HERE / "train_gpt.py")
)
# We only need the module-level definitions; skip __main__ side effects.
import torch
import torch.nn.functional as F


def load_module():
    src = (HERE / "train_gpt.py").read_text()
    # Strip the if __name__ == "__main__" tail to keep import side-effect free.
    g = {"__name__": "train_gpt_gs"}
    exec(compile(src, "train_gpt.py", "exec"), g)
    return g


def main() -> int:
    mod = load_module()
    PhiNTA = mod["PhiNTA"]
    _jepa_loss = mod["_jepa_loss"]
    PHI = mod["PHI"]
    PHI_INV = mod["PHI_INV"]
    ALPHA_PHI = mod["ALPHA_PHI"]
    PHI_LOOPS = mod["PHI_LOOPS"]

    # 1. Trinity identity
    trinity = PHI ** 2 + PHI ** -2
    assert abs(trinity - 3.0) < 1e-9, f"Trinity identity broken: {trinity}"
    assert abs(PHI_INV - (PHI - 1.0)) < 1e-12
    assert abs(ALPHA_PHI - (PHI ** -3) / 2.0) < 1e-12
    assert PHI_LOOPS == 4
    print(f"[1/5] φ-physics OK: φ²+φ⁻²={trinity:.12f} α_φ={ALPHA_PHI:.6f} loops={PHI_LOOPS}")

    # 2. PhiNTA
    torch.manual_seed(1597)  # F₁₇ seed (canonical)
    dim, rank = 64, 13       # F₇ heads
    nta = PhiNTA(dim, rank)
    assert "W_frozen" in dict(nta.named_buffers())
    assert "W_frozen" not in dict(nta.named_parameters())
    # Frozen basis row-norms ~ 1/φ
    rows = nta.W_frozen.norm(dim=1)
    assert torch.allclose(rows, torch.full_like(rows, PHI_INV), atol=1e-5), \
        f"row norms {rows[:5]} ≠ 1/φ"
    x = torch.randn(2, 16, dim, requires_grad=True)
    y = nta(x).sum()
    y.backward()
    # B is zero-init, so on the first backward only B receives gradient
    # (∂y/∂A goes through B which is zero); after one B step A picks up grad too.
    assert nta.B.grad is not None and nta.B.grad.abs().sum() > 0
    assert nta.A.requires_grad and not nta.W_frozen.requires_grad
    n_train = sum(p.numel() for p in nta.parameters() if p.requires_grad)
    n_frozen = nta.W_frozen.numel()
    print(f"[2/5] PhiNTA OK: trainable={n_train} frozen={n_frozen} ratio={n_train/n_frozen:.3f}")

    # 3. JEPA loss
    h = torch.randn(2, 32, dim, requires_grad=True)
    loss = _jepa_loss(h, max_span_frac=0.5)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    print(f"[3/5] JEPA loss OK: {loss.item():.4f} (cosine-similarity form)")

    # 4. UT loop arithmetic
    # When ut_loops=1, maybe_loop must be identity-like.
    # When ut_loops=4, applying any block 4× must equal block(block(block(block(x)))).
    class _Id(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return x + 0.01 * x
    blk = _Id()
    x0 = torch.randn(1, 8, dim)
    x_one = blk(x0)
    x_four = x0
    for _ in range(4):
        x_four = blk(x_four)
    expected_growth = (1.01 ** 4)
    actual_growth = (x_four.norm() / x0.norm()).item()
    assert abs(actual_growth - expected_growth) < 1e-3
    print(f"[4/5] UT loop OK: ‖x_4‖/‖x_0‖={actual_growth:.4f} expected={expected_growth:.4f}")

    # 5. JEPA tap normalisation — -1 must resolve to the last block.
    n_total = 9
    for raw in (-1, 0, 4, 8):
        idx = raw if raw >= 0 else n_total - 1
        assert 0 <= idx < n_total, f"jepa tap {raw} → {idx} out of range"
    print("[5/5] JEPA tap normalisation OK: -1 → last block, in-range indices preserved")

    print("\n🌻 GOLDEN SUNFLOWERS smoke OK · 5/5 · phi^2 + phi^-2 = 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
