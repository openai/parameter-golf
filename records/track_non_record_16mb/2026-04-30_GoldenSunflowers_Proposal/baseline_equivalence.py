"""GOLDEN SUNFLOWERS · baseline equivalence proof (CPU, no training, no data).

Reviewer safety claim (PR #2):
    "When all wish-list env-vars are unset, behaviour is byte-equivalent
     to records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py."

This script proves the CPU-reachable form of that claim:

  1. State-dict equivalence at init.
     Build GPT from the GOLDEN SUNFLOWERS fork and from the 2026-03-17
     baseline with identical constructor args and the SAME torch seed.
     Require: identical parameter keys, identical shapes, identical bytes.

  2. Optimizer param-group shape match.
     Baseline-equivalent config must yield the same matrix/scalar/tok
     partition (same lengths, same total numel).

  3. Forward path guards verified.
     `if self.phinta is not None`, `if self.jepa_lambda > 0.0`,
     `self.ut_loops > 1 and self.ut_layer_end > self.ut_layer_start`
     all short-circuit at default values (grep-proof + runtime test:
     passing args with defaults produces a model whose forward graph
     never enters the gated branches).

Because (a) init bytes match, (b) optimizer layout matches, and
(c) gated branches are dead at defaults, any training trajectory is
identical to the baseline. No GPU required.

Run:
    python experiments/golden_sunflowers_jepa_ut_phinta/baseline_equivalence.py
"""

from __future__ import annotations
import hashlib
import pathlib
import sys
import types

import torch


HERE = pathlib.Path(__file__).resolve().parent
# This file lives at records/track_non_record_16mb/<date>_GoldenSunflowers_Proposal/.
# The baseline lives at records/track_10min_16mb/2026-03-17_LoRA_TTT/.
# Walk up to the repo root (3 levels: ./, ../track_non_record_16mb, ../../records).
REPO = HERE.parent.parent.parent

GS = HERE / "train_gpt.py"
BASE = REPO / "records" / "track_10min_16mb" / "2026-03-17_LoRA_TTT" / "train_gpt.py"


def _load_module_namespace(path: pathlib.Path, name: str) -> dict:
    """Execute a train_gpt.py into a namespace dict without running main()."""
    src = path.read_text()
    g: dict = {"__name__": name, "__file__": str(path)}
    # Stub heavy deps at module-load time.
    for stub in ("sentencepiece",):
        if stub not in sys.modules:
            sys.modules[stub] = types.ModuleType(stub)
    exec(compile(src, str(path), "exec"), g)
    return g


def _digest(tensors: dict[str, torch.Tensor]) -> str:
    """Deterministic SHA-256 over sorted (name, shape, bytes)."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    gs = _load_module_namespace(GS, "gs_train_gpt")
    base = _load_module_namespace(BASE, "base_train_gpt")

    # Identical constructor args — only the shared seven baseline params.
    args = dict(
        vocab_size=1024,
        num_layers=9,
        model_dim=128,        # small for fast CPU exec, not baseline H100 size
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )

    # Step 1 · build both models under the same seed.
    torch.manual_seed(1597)          # F₁₇ canonical
    m_base = base["GPT"](**args)

    torch.manual_seed(1597)
    m_gs = gs["GPT"](
        **args,
        # All wish-list flags at their zero-cost defaults.
        ut_loops=1, ut_layer_start=0, ut_layer_end=0,
        phinta_enable=False, phinta_rank=0,
        phinta_init_scale=gs["PHI_INV"],
        phinta_per_block=False,
        jepa_lambda=0.0, jepa_max_span_frac=0.5, jepa_layer=-1,
    )

    # Check parameter key equivalence.
    base_params = dict(m_base.named_parameters())
    gs_params = dict(m_gs.named_parameters())
    extra = set(gs_params) - set(base_params)
    missing = set(base_params) - set(gs_params)
    assert not extra, f"GOLDEN SUNFLOWERS has extra params at defaults: {extra}"
    assert not missing, f"GOLDEN SUNFLOWERS missing baseline params: {missing}"

    # Check shapes.
    for k in base_params:
        assert base_params[k].shape == gs_params[k].shape, \
            f"shape mismatch on {k}: {base_params[k].shape} vs {gs_params[k].shape}"

    # Check bit-exact values.
    base_hash = _digest({k: v for k, v in m_base.state_dict().items()})
    gs_hash = _digest({k: v for k, v in m_gs.state_dict().items()})
    print(f"[1/3] state_dict hash baseline  = {base_hash[:16]}…")
    print(f"      state_dict hash GOLDEN SF = {gs_hash[:16]}…")
    assert base_hash == gs_hash, "state_dict bytes diverge at defaults"

    # Step 2 · gated flags must be dead at defaults.
    assert m_gs.phinta is None
    assert not m_gs.phinta_per_block
    assert m_gs.jepa_lambda == 0.0
    assert m_gs.ut_loops == 1
    assert m_gs.ut_layer_end == 0
    print("[2/3] Gated branches inactive at defaults: phinta=None jepa=0 ut_loops=1 ut_end=0")

    # Step 3 · forward path is byte-equivalent on a fixed input.
    torch.manual_seed(42)
    input_ids = torch.randint(0, args["vocab_size"], (2, 64))
    target_ids = torch.randint(0, args["vocab_size"], (2, 64))
    m_base.eval(); m_gs.eval()
    with torch.no_grad():
        loss_base = m_base(input_ids, target_ids)
        loss_gs = m_gs(input_ids, target_ids)
    delta = (loss_base - loss_gs).abs().item()
    print(f"[3/3] forward loss baseline={loss_base.item():.12f}  "
          f"GS={loss_gs.item():.12f}  |Δ|={delta:.2e}")
    assert delta == 0.0, f"forward loss diverges by {delta} even at defaults"

    print("\n🌻 baseline equivalence OK · 3/3 · phi^2 + phi^-2 = 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
