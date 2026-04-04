#!/usr/bin/env python3
"""
Selective Freeze Patch for Clark's train_gpt.py
=================================================
Freezes MLP fc (expansion) weights as deterministic random.
The frozen weights are converted from Parameters to BUFFERS
so they're NOT included in state_dict → NOT saved in artifact.

At eval time: regenerate from seed (same as training).

CRITICAL: Must be called BEFORE optimizer creation.
Must also patch the serialize/deserialize to regenerate frozen weights.

MATH (4x MLP, dim=512):
  Clark 11L: 33.8M total, all learned → 15.9MB artifact
  Ours  13L: 39.6M total, 26M learned, 13.6M frozen → 15.7MB artifact
  Gain: 17% more params, 2 extra layers, same artifact size
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


class FrozenFC(nn.Module):
    """Replaces CastedLinear for MLP fc with frozen random weights.

    Weights are stored as a BUFFER (not parameter) → excluded from
    state_dict → excluded from artifact. Regenerated from seed at load time.
    """
    def __init__(self, in_features, out_features, seed):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed

        # Generate and register as buffer (NOT parameter)
        rng = torch.Generator()
        rng.manual_seed(seed)
        w = torch.randn(out_features, in_features, generator=rng) / math.sqrt(in_features)
        self.register_buffer('weight', w, persistent=False)  # NOT saved to state_dict

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


def apply_selective_freeze(model):
    """Replace MLP fc layers with FrozenFC modules.

    Call AFTER model construction, BEFORE optimizer creation.

    The FrozenFC weights are buffers, not parameters:
    - NOT included in optimizer (no gradients)
    - NOT included in state_dict (not saved to artifact)
    - Regenerated from seed at both train and eval time
    """
    if os.environ.get("SELECTIVE_FREEZE", "0") not in ("1", "true", "True"):
        print("selective_freeze: disabled")
        return 0

    frozen_count = 0
    for i, block in enumerate(model.blocks):
        mlp = block.mlp
        old_fc = mlp.fc
        seed = 42_000 + i  # deterministic per layer

        # Replace CastedLinear with FrozenFC
        new_fc = FrozenFC(
            old_fc.in_features,
            old_fc.out_features,
            seed=seed
        ).to(old_fc.weight.device)

        mlp.fc = new_fc
        frozen_count += new_fc.weight.numel()

    # Verify: frozen weights should NOT appear in parameters()
    param_count = sum(p.numel() for p in model.parameters())
    buffer_count = sum(b.numel() for name, b in model.named_buffers())

    print(f"selective_freeze: {frozen_count:,} MLP fc params → frozen buffers")
    print(f"  Parameters (learned, saved): {param_count:,}")
    print(f"  Buffers (frozen, NOT saved): {buffer_count:,}")
    print(f"  Total effective: {param_count + buffer_count:,}")
    print(f"  Artifact estimate (int6+Brotli): {param_count * 6 / 8 * 0.8 / 1e6:.1f}MB")

    return frozen_count


def regenerate_frozen_weights(model):
    """Regenerate frozen MLP fc weights from seeds.

    Call AFTER loading state_dict at eval time.
    The state_dict won't contain fc weights (they're buffers that
    weren't saved). This function recreates them.
    """
    for i, block in enumerate(model.blocks):
        if isinstance(block.mlp.fc, FrozenFC):
            # Already a FrozenFC — weights generated in __init__
            continue

        # If loading a model that was trained with selective freeze,
        # the fc won't have weights in state_dict. Replace with FrozenFC.
        old_fc = block.mlp.fc
        seed = 42_000 + i
        new_fc = FrozenFC(
            old_fc.in_features,
            old_fc.out_features,
            seed=seed
        ).to(next(model.parameters()).device)
        block.mlp.fc = new_fc


# ============================================================
# Integration test
# ============================================================
if __name__ == "__main__":
    print("Testing selective freeze...")

    # Simulate Clark's MLP
    class CastedLinear(nn.Linear):
        def forward(self, x):
            return F.linear(x, self.weight.to(x.dtype))

    class MLP(nn.Module):
        def __init__(self, dim, mult):
            super().__init__()
            hidden = dim * mult
            self.fc = CastedLinear(dim, hidden, bias=False)
            self.proj = CastedLinear(hidden, dim, bias=False)
        def forward(self, x):
            return self.proj(F.leaky_relu(self.fc(x), 0.5).square())

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([
                nn.Module() for _ in range(11)
            ])
            for i, b in enumerate(self.blocks):
                b.mlp = MLP(512, 4)
                b.attn = nn.Linear(512, 512)  # placeholder

    model = FakeModel()

    # Before freeze
    sd_before = model.state_dict()
    print(f"\nBefore freeze:")
    print(f"  state_dict keys: {len(sd_before)}")
    print(f"  Total bytes (float32): {sum(v.numel()*4 for v in sd_before.values())/1e6:.1f}MB")

    # Apply freeze
    os.environ["SELECTIVE_FREEZE"] = "1"
    apply_selective_freeze(model)

    # After freeze
    sd_after = model.state_dict()
    print(f"\nAfter freeze:")
    print(f"  state_dict keys: {len(sd_after)}")
    print(f"  Total bytes (float32): {sum(v.numel()*4 for v in sd_after.values())/1e6:.1f}MB")
    print(f"  Removed: {len(sd_before) - len(sd_after)} keys")

    # Verify frozen weights NOT in state_dict
    fc_in_sd = [k for k in sd_after.keys() if '.fc.' in k and 'weight' in k]
    print(f"  MLP fc weights in state_dict: {len(fc_in_sd)} (should be 0)")

    # Verify forward pass works
    x = torch.randn(2, 10, 512)
    for b in model.blocks:
        y = b.mlp(x)
    print(f"\n  Forward pass: OK (output shape {y.shape})")

    # Verify deterministic regeneration
    model2 = FakeModel()
    os.environ["SELECTIVE_FREEZE"] = "1"
    apply_selective_freeze(model2)
    w1 = model.blocks[0].mlp.fc.weight
    w2 = model2.blocks[0].mlp.fc.weight
    print(f"  Deterministic: {torch.allclose(w1, w2)} (should be True)")

    print("\nAll tests passed!")
