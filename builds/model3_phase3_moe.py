from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as baseline

from builds.model3_lib import SSMOnlyGPT, env_int, run_smoke_training


class Hyperparameters(baseline.Hyperparameters):
    num_layers = int(os.environ.get("NUM_LAYERS", 8))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    state_dim = int(os.environ.get("STATE_DIM", model_dim))
    selector_hidden = int(os.environ.get("SELECTOR_HIDDEN", 128))
    num_experts = int(os.environ.get("NUM_EXPERTS", 4))


class Phase3MoEGPT(SSMOnlyGPT):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("state_dim", Hyperparameters.state_dim)
        kwargs.setdefault("selector_hidden", Hyperparameters.selector_hidden)
        kwargs.setdefault("use_moe", True)
        kwargs.setdefault("num_experts", Hyperparameters.num_experts)
        super().__init__(*args, **kwargs)


def run_smoke() -> None:
    model = Phase3MoEGPT(
        vocab_size=128,
        num_layers=8,
        model_dim=256,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.0,
        state_dim=256,
        selector_hidden=96,
        num_experts=4,
    )
    result = run_smoke_training(
        model,
        steps=env_int("SMOKE_STEPS", 8),
        batch_size=env_int("SMOKE_BATCH", 16),
        seq_len=env_int("SMOKE_SEQ_LEN", 64),
        vocab_size=128,
    )
    print(f"phase3_smoke losses:{result.losses}")
    print(f"phase3_smoke avg_step_ms:{result.step_ms:.2f}")
    print(f"phase3_smoke loss_delta:{result.losses[0] - result.losses[-1]:.4f}")
    print(f"phase3_routing fractions:{result.routing_fractions}")


if __name__ == "__main__":
    if os.environ.get("MODEL3_SMOKE") == "1":
        run_smoke()
    else:
        if os.environ.get("MODEL3_ENABLE_COMPILE", "0") != "1":
            baseline.torch.compile = lambda fn, **kwargs: fn
        baseline.GPT = Phase3MoEGPT
        baseline.Hyperparameters = Hyperparameters
        baseline.main()
