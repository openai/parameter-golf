from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "records"
    / "track_10min_16mb"
    / "2026-03-19_TrainingOptSeq4096"
    / "train_gpt.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("seq4096_train_gpt", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Args:
    advanced_loss_scale = 1.0
    graphcg_loss_weight = 0.05
    toric_tropical_loss_weight = 0.03
    slepian_loss_weight = 0.02
    koszul_bgg_loss_weight = 0.01
    analogy_loss_weight = 0.01
    advanced_loss_sample_tokens = 16
    toric_tropical_fan_bins = 8
    advanced_loss_log_only = False
    advanced_loss_start_step = 0
    advanced_loss_end_step = 0
    advanced_loss_every = 1
    advanced_loss_warmup_steps = 0
    advanced_loss_min_best_val_bpb = 0.0
    advanced_loss_max_ce_ratio = 0.0


def tiny_model(module):
    kwargs = dict(
        vocab_size=32,
        num_layers=2,
        model_dim=16,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        fineweb_graphify=True,
        tokengt_first_class=True,
        tokengt_graph_radius=2,
        tokengt_token_class_buckets=8,
        tokengt_position_buckets=32,
        tokengt_structural_weight=0.05,
        tokengt_edge_weight=0.03,
        tokengt_torus_weight=0.01,
        tokengt_identifier_dim=8,
        tokengt_identifier_weight=0.01,
        tokengt_endpoint_weight=0.01,
        tokengt_edge_token_weight=0.01,
        graph_output_flattening=True,
        graph_output_edge_radius=2,
        graph_output_node_weight=0.05,
        graph_output_edge_weight=0.05,
        graph_output_virtual_edge_tokens=True,
        graph_output_edge_token_weight=0.03,
        graph_output_score_correction=True,
        graph_output_score_correction_weight=0.02,
    )
    signature = inspect.signature(module.GPT.__init__)
    return module.GPT(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def test_advanced_embedding_losses_are_finite_and_backpropagate() -> None:
    module = load_module()
    model = tiny_model(module)
    input_ids = torch.arange(48, dtype=torch.long).reshape(2, 24) % 32

    loss, metrics = module.advanced_embedding_losses(model, Args, input_ids)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert metrics["advanced/graphcg_loss"].isfinite()
    assert metrics["advanced/toric_tropical_loss"].isfinite()
    assert metrics["advanced/slepian_pollak_loss"].isfinite()
    assert metrics["advanced/koszul_bgg_loss"].isfinite()
    assert metrics["advanced/analogy_loss"].isfinite()
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert torch.isfinite(model.tok_emb.weight.grad).all()


def test_advanced_embedding_losses_can_be_disabled() -> None:
    module = load_module()
    model = tiny_model(module)
    input_ids = torch.arange(16, dtype=torch.long).reshape(1, 16) % 32

    class Disabled(Args):
        advanced_loss_scale = 0.0

    loss, metrics = module.advanced_embedding_losses(model, Disabled, input_ids)

    assert loss.item() == 0.0
    assert metrics == {}


def test_advanced_embedding_losses_log_only_without_backprop() -> None:
    module = load_module()
    model = tiny_model(module)
    input_ids = torch.arange(48, dtype=torch.long).reshape(2, 24) % 32

    class LogOnly(Args):
        advanced_loss_scale = 0.0
        advanced_loss_log_only = True

    loss, metrics = module.advanced_embedding_losses(model, LogOnly, input_ids, runtime_scale=0.0)

    assert loss.item() == 0.0
    assert metrics["advanced/graphcg_loss"].isfinite()
    assert metrics["advanced/weighted_unscaled_loss"].isfinite()
    assert metrics["advanced/runtime_scale"].item() == 0.0
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert model.tok_emb.weight.grad.abs().sum().item() == 0.0


def test_advanced_loss_runtime_scale_uses_schedule_and_best_val_gate() -> None:
    module = load_module()

    class Scheduled(Args):
        advanced_loss_scale = 0.03
        advanced_loss_start_step = 100
        advanced_loss_end_step = 110
        advanced_loss_every = 2
        advanced_loss_warmup_steps = 4
        advanced_loss_min_best_val_bpb = 1.205

    assert module.advanced_loss_runtime_scale(Scheduled, 99, 1.200) == 0.0
    assert module.advanced_loss_runtime_scale(Scheduled, 100, 1.210) == 0.0
    assert module.advanced_loss_runtime_scale(Scheduled, 101, 1.200) == 0.0
    assert module.advanced_loss_runtime_scale(Scheduled, 110, 1.200) == 0.0
    assert module.advanced_loss_runtime_scale(Scheduled, 100, 1.200) == 0.03 / 4.0
    assert module.advanced_loss_runtime_scale(Scheduled, 106, 1.200) == 0.03
