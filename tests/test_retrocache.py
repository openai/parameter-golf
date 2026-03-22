from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "records"
    / "track_10min_16mb"
    / "2026-03-21_v38_TightSWA_RetroCache"
    / "train_gpt.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("retrocache_train_gpt", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_forward_eval_features_matches_forward_logits():
    mod = load_module()
    torch.manual_seed(0)
    model = mod.GPT(
        vocab_size=32,
        num_layers=2,
        model_dim=16,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2.0,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.0,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=8,
        bigram_dim=8,
        xsa_last_n=0,
        rope_dims=0,
        ln_scale=False,
        dtg=False,
        ve_enabled=False,
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)

    logits_ref = model.forward_logits(tokens)
    logits_eval, keys = model.forward_eval_features(tokens)

    torch.testing.assert_close(logits_eval, logits_ref)
    assert keys.shape == (1, 5, 16)
    key_norms = torch.linalg.vector_norm(keys.float(), dim=-1)
    torch.testing.assert_close(key_norms, torch.ones_like(key_norms), atol=1e-3, rtol=1e-3)


def test_retrocached_nll_only_changes_after_append():
    mod = load_module()
    args = SimpleNamespace(
        cache_warmup_tokens=0,
        cache_lambda_max=0.35,
        cache_topk=1,
        cache_beta=24.0,
    )
    memory = mod.RetroCacheMemory(
        key_dim=4,
        device=torch.device("cpu"),
        max_tokens=8,
        recent_tokens=4,
        old_stride=2,
        key_dtype=torch.float32,
    )
    logits = torch.zeros(1, 16, dtype=torch.float32)
    targets = torch.tensor([7], dtype=torch.int64)
    query_keys = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    nll_before = mod._retrocached_nll(logits, targets, query_keys, memory, args)
    memory.append(query_keys, targets)
    nll_after = mod._retrocached_nll(logits, targets, query_keys, memory, args)

    assert nll_after.item() < nll_before.item()


def test_append_retrocache_entries_resets_after_bos():
    mod = load_module()
    memory = mod.RetroCacheMemory(
        key_dim=4,
        device=torch.device("cpu"),
        max_tokens=8,
        recent_tokens=4,
        old_stride=2,
        key_dtype=torch.float32,
    )
    keys = torch.eye(4, dtype=torch.float32)[:3]
    vals = torch.tensor([5, mod.BOS_ID, 9], dtype=torch.int64)

    mod._append_retrocache_entries(memory, keys, vals, reset_on_bos=True)
    stored_keys, stored_vals = memory.candidate_tensors()

    assert stored_keys.shape[0] == 1
    assert stored_vals.tolist() == [9]
