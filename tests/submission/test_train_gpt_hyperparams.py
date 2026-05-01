"""Tests for train_gpt.py Section 1 — Hyperparameters class."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

SUBMISSION_DIR = (
    Path(__file__).parent.parent.parent
    / "records"
    / "track_10min_16mb"
    / "2026-05-01_SemanticEngine_CareSSM"
)
TRAIN_GPT_PATH = (
    SUBMISSION_DIR / "train_gpt.py"
)
SUBMISSION_JSON_PATH = SUBMISSION_DIR / "submission.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_gpt", TRAIN_GPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vocab_size():
    mod = _load_module()
    assert mod.Hyperparameters.vocab_size == 16384


def test_model_dim_default():
    # Ensure MODEL_DIM env var is not set so we get the default.
    env_backup = os.environ.pop("MODEL_DIM", None)
    try:
        mod = _load_module()
        assert mod.Hyperparameters.model_dim == 384
    finally:
        if env_backup is not None:
            os.environ["MODEL_DIM"] = env_backup


def test_batch_size_default_matches_successful_h100_shape():
    env_backup = os.environ.pop("BATCH_SIZE", None)
    try:
        mod = _load_module()
        assert mod.Hyperparameters.batch_size == 800
    finally:
        if env_backup is not None:
            os.environ["BATCH_SIZE"] = env_backup


def test_crct_memory_write_tokens_per_step():
    env_backup = os.environ.pop("CRCT_MEMORY_WRITE_TOKENS_PER_STEP", None)
    try:
        mod = _load_module()
        assert mod.Hyperparameters.crct_memory_write_tokens_per_step == 256
    finally:
        if env_backup is not None:
            os.environ["CRCT_MEMORY_WRITE_TOKENS_PER_STEP"] = env_backup


def test_online_episodic_write_tokens_per_chunk():
    env_backup = os.environ.pop("ONLINE_EPISODIC_WRITE_TOKENS_PER_CHUNK", None)
    try:
        mod = _load_module()
        assert mod.Hyperparameters.online_episodic_write_tokens_per_chunk == 64
    finally:
        if env_backup is not None:
            os.environ["ONLINE_EPISODIC_WRITE_TOKENS_PER_CHUNK"] = env_backup


def test_packet_eval_defaults_are_batched_and_live_writing():
    backups = {
        key: os.environ.pop(key, None)
        for key in (
            "PACKET_EVAL_BATCH_DOCS",
            "PACKET_EVAL_BATCH_TOKEN_BUDGET",
            "PACKET_EVAL_WRITE_TOKENS_PER_CHUNK",
            "PACKET_EVAL_CONTROLLER_READ",
            "PACKET_EVAL_CONTROLLER_TOPK_K",
            "PACKET_EVAL_CONTROLLER_SCORE_MODE",
            "STOP_MARGIN_SECONDS",
        )
    }
    try:
        mod = _load_module()
        assert mod.Hyperparameters.packet_eval_batch_docs == 48
        assert mod.Hyperparameters.packet_eval_batch_token_budget == 49152
        assert mod.Hyperparameters.packet_eval_write_tokens_per_chunk == 1
        assert mod.Hyperparameters.packet_eval_controller_read_enabled is False
        assert mod.Hyperparameters.packet_eval_controller_topk_k == 16
        assert (
            mod.Hyperparameters.packet_eval_controller_score_mode
            == "cosine_survival"
        )
        assert mod.Hyperparameters.stop_margin_seconds == 32.0
    finally:
        for key, value in backups.items():
            if value is not None:
                os.environ[key] = value


def test_seed_env_override(monkeypatch):
    monkeypatch.setenv("SEED", "99")
    mod = _load_module()
    assert mod.Hyperparameters.seed == 99


def test_log_a_beta_coupling_env_one(monkeypatch):
    monkeypatch.setenv("LOG_A_BETA_COUPLING", "1")
    mod = _load_module()
    assert mod.Hyperparameters.log_a_beta_coupling is True


def test_score_summary_keys_present():
    """Regression: the summary block must contain val_bpb and artifact_bytes."""
    source = TRAIN_GPT_PATH.read_text()
    assert "val_bpb" in source
    assert "artifact_bytes" in source
    assert "packet_online_cache" in source
    assert "score each chunk" in source.lower() or "score-before-write" in source.lower()


def test_submission_artifact_accounting_is_not_raw_bf16_size():
    """Public artifact field must be the under-cap compressed payload estimate."""
    data = json.loads(SUBMISSION_JSON_PATH.read_text())
    assert data["artifact_submit_valid"] is True
    assert data["artifact_bytes_estimate"] < data["artifact_bytes_limit"]
    assert data["raw_bf16_weight_bytes"] > data["artifact_bytes_limit"]
    assert "int6/LZMA" in data["artifact_accounting_note"]
