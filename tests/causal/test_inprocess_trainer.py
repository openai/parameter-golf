"""Tests for scripts/causal/inprocess_trainer.py -- in-process training runner."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.causal.inprocess_trainer import (
    SharedTrainingContext,
    _build_hyperparameters,
    _env_override,
    create_shared_context,
    run_condition_inprocess,
    train_single_run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = str(_REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")
_TOKENIZER_PATH = str(_REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")
_DATA_EXISTS = Path(_DATA_PATH).exists() and Path(_TOKENIZER_PATH).exists()

_skip_no_data = pytest.mark.skipif(
    not _DATA_EXISTS,
    reason="Training data not available on this machine",
)


# ---------------------------------------------------------------------------
# Unit tests: _env_override context manager
# ---------------------------------------------------------------------------

class TestEnvOverride:
    def test_sets_inside_and_restores_after(self):
        key = "_INPROCESS_TRAINER_TEST_KEY"
        # Ensure key is absent
        os.environ.pop(key, None)
        assert key not in os.environ

        with _env_override({key: "hello"}):
            assert os.environ[key] == "hello"

        assert key not in os.environ

    def test_restores_original_value(self):
        key = "_INPROCESS_TRAINER_TEST_KEY2"
        os.environ[key] = "original"
        try:
            with _env_override({key: "override"}):
                assert os.environ[key] == "override"
            assert os.environ[key] == "original"
        finally:
            os.environ.pop(key, None)

    def test_restores_on_exception(self):
        key = "_INPROCESS_TRAINER_TEST_KEY3"
        os.environ.pop(key, None)

        with pytest.raises(RuntimeError):
            with _env_override({key: "temp"}):
                assert os.environ[key] == "temp"
                raise RuntimeError("boom")

        assert key not in os.environ


# ---------------------------------------------------------------------------
# Unit tests: _build_hyperparameters
# ---------------------------------------------------------------------------

class TestBuildHyperparameters:
    def test_applies_env_overrides(self):
        args = _build_hyperparameters(
            env_overrides={"NUM_LAYERS": "11"},
            seed=42,
            iterations=5,
        )
        assert args.num_layers == 11
        assert args.seed == 42
        assert args.iterations == 5

    def test_restores_env(self):
        key = "NUM_LAYERS"
        original = os.environ.get(key)
        try:
            _build_hyperparameters(
                env_overrides={key: "11"},
                seed=42,
                iterations=5,
            )
            assert os.environ.get(key) == original
        finally:
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    def test_sets_iteration_based_stopping(self):
        args = _build_hyperparameters(
            env_overrides={"MAX_WALLCLOCK_SECONDS": "0"},
            seed=1,
            iterations=100,
        )
        assert args.max_wallclock_seconds == 0.0

    def test_default_warmup_and_warmdown(self):
        args = _build_hyperparameters(
            env_overrides={},
            seed=1,
            iterations=10,
            warmup_steps=1,
            warmdown_iters=0,
        )
        assert args.warmup_steps == 1
        assert args.warmdown_iters == 0


# ---------------------------------------------------------------------------
# Integration tests: SharedTrainingContext
# ---------------------------------------------------------------------------

@_skip_no_data
class TestSharedContext:
    def test_loads_context(self):
        ctx = create_shared_context(
            data_path=_DATA_PATH,
            tokenizer_path=_TOKENIZER_PATH,
            vocab_size=1024,
            train_seq_len=1024,
        )
        assert isinstance(ctx, SharedTrainingContext)
        assert ctx.val_tokens.size > 0
        assert ctx.base_bytes_lut.shape[0] >= 1024
        assert ctx.has_leading_space_lut.shape[0] >= 1024
        assert ctx.is_boundary_token_lut.shape[0] >= 1024
        assert "fineweb_train_*.bin" in ctx.train_files_pattern


# ---------------------------------------------------------------------------
# Integration tests: train_single_run
# ---------------------------------------------------------------------------

@_skip_no_data
class TestTrainSingleRun:
    @pytest.fixture(scope="class")
    def ctx(self):
        return create_shared_context(
            data_path=_DATA_PATH,
            tokenizer_path=_TOKENIZER_PATH,
            vocab_size=1024,
            train_seq_len=1024,
        )

    def test_returns_result_dict(self, ctx):
        result = train_single_run(
            ctx=ctx,
            env_overrides={"MAX_WALLCLOCK_SECONDS": "0"},
            seed=42,
            iterations=5,
            val_loss_every=0,
            warmup_steps=1,
            warmdown_iters=0,
        )
        assert "seed" in result
        assert result["seed"] == 42
        assert "val_bpb" in result
        assert isinstance(result["val_bpb"], float)
        assert result["val_bpb"] > 0
        assert "val_loss" in result
        assert isinstance(result["val_loss"], float)
        assert "wall_time_s" in result
        assert result["wall_time_s"] > 0

    def test_same_seed_reproducible(self, ctx):
        kwargs = dict(
            ctx=ctx,
            env_overrides={"MAX_WALLCLOCK_SECONDS": "0"},
            seed=42,
            iterations=5,
            val_loss_every=0,
            warmup_steps=1,
            warmdown_iters=0,
        )
        r1 = train_single_run(**kwargs)
        r2 = train_single_run(**kwargs)
        # MLX bfloat16 compiled graphs can produce small non-deterministic
        # differences across runs due to Metal shader scheduling and
        # floating-point accumulation order. We allow 0.1% relative tolerance.
        assert r1["val_bpb"] == pytest.approx(r2["val_bpb"], rel=1e-3), (
            f"Same seed should produce near-identical val_bpb: {r1['val_bpb']} vs {r2['val_bpb']}"
        )

    def test_different_seeds_differ(self, ctx):
        base_kwargs = dict(
            ctx=ctx,
            env_overrides={"MAX_WALLCLOCK_SECONDS": "0"},
            iterations=5,
            val_loss_every=0,
            warmup_steps=1,
            warmdown_iters=0,
        )
        r1 = train_single_run(seed=42, **base_kwargs)
        r2 = train_single_run(seed=137, **base_kwargs)
        assert r1["val_bpb"] != r2["val_bpb"], (
            "Different seeds should produce different val_bpb"
        )


# ---------------------------------------------------------------------------
# Integration tests: run_condition_inprocess
# ---------------------------------------------------------------------------

@_skip_no_data
class TestRunConditionInprocess:
    @pytest.fixture(scope="class")
    def ctx(self):
        return create_shared_context(
            data_path=_DATA_PATH,
            tokenizer_path=_TOKENIZER_PATH,
            vocab_size=1024,
            train_seq_len=1024,
        )

    def test_returns_list_of_results(self, ctx):
        cfg = {
            "script": "train_gpt_mlx.py",
            "env_overrides": {"MAX_WALLCLOCK_SECONDS": "0"},
        }
        results = run_condition_inprocess(
            ctx=ctx,
            cfg=cfg,
            seeds=[42],
            iterations=5,
            warmup_steps=1,
            warmdown_iters=0,
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["seed"] == 42
        assert results[0]["val_bpb"] is not None
