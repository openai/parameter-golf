from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as baseline

from builds.model3_lib import HybridGPT, benchmark_step_time, env_int, run_smoke_training


class Hyperparameters(baseline.Hyperparameters):
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    state_dim = int(os.environ.get("STATE_DIM", model_dim))
    selector_hidden = int(os.environ.get("SELECTOR_HIDDEN", 128))
    ssm_layers = int(os.environ.get("SSM_LAYERS", 8))


class Phase2HybridGPT(HybridGPT):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("state_dim", Hyperparameters.state_dim)
        kwargs.setdefault("selector_hidden", Hyperparameters.selector_hidden)
        kwargs.setdefault("ssm_layers", Hyperparameters.ssm_layers)
        kwargs.setdefault("use_moe", False)
        super().__init__(*args, **kwargs)


def run_smoke() -> None:
    hybrid = Phase2HybridGPT(
        vocab_size=128,
        num_layers=11,
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
        ssm_layers=8,
    )
    ssm_only = Phase2HybridGPT(
        vocab_size=128,
        num_layers=9,
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
        ssm_layers=8,
    )
    result = run_smoke_training(
        hybrid,
        steps=env_int("SMOKE_STEPS", 8),
        batch_size=env_int("SMOKE_BATCH", 8),
        seq_len=env_int("SMOKE_SEQ_LEN", 64),
        vocab_size=128,
    )
    print(f"phase2_smoke losses:{result.losses}")
    print(f"phase2_smoke avg_step_ms:{result.step_ms:.2f}")
    print(f"phase2_smoke loss_delta:{result.losses[0] - result.losses[-1]:.4f}")
    print(
        "phase2_timing "
        f"hybrid_ms:{benchmark_step_time(hybrid):.2f} "
        f"mostly_ssm_ms:{benchmark_step_time(ssm_only):.2f}"
    )


if __name__ == "__main__":
    if os.environ.get("MODEL3_SMOKE") == "1":
        run_smoke()
    else:
        if os.environ.get("MODEL3_ENABLE_COMPILE", "0") != "1":
            baseline.torch.compile = lambda fn, **kwargs: fn
        baseline.GPT = Phase2HybridGPT
        baseline.Hyperparameters = Hyperparameters
        baseline.main()
