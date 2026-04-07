from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F

from my_submission import original_train_gpt as original_tg
from my_submission import train_gpt as tg


def copy_original_gpt_to_new(new_model: tg.GPT, old_model: original_tg.GPT) -> None:
    with torch.no_grad():
        new_model.tok_emb.weight.copy_(old_model.tok_emb.weight)
        new_model.skip_weights.copy_(old_model.skip_weights)
        for new_block, old_block in zip(new_model.blocks, old_model.blocks):
            new_block.attn.c_qkv.weight.copy_(
                torch.cat(
                    (old_block.attn.c_q.weight, old_block.attn.c_k.weight, old_block.attn.c_v.weight),
                    dim=0,
                )
            )
            new_block.attn.proj.weight.copy_(old_block.attn.proj.weight)
            new_block.attn.q_gain.copy_(old_block.attn.q_gain)
            new_block.mlp.fc.weight.copy_(old_block.mlp.fc.weight)
            new_block.mlp.proj.weight.copy_(old_block.mlp.proj.weight)
            new_block.attn_scale.copy_(old_block.attn_scale)
            new_block.mlp_scale.copy_(old_block.mlp_scale)
            new_block.resid_mix.copy_(old_block.resid_mix)
        if old_model.lm_head is not None and new_model.lm_head is not None:
            new_model.lm_head.weight.copy_(old_model.lm_head.weight)


def run_hidden_stack(model: tg.GPT | original_tg.GPT, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[torch.Tensor] = []
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)
    return model.final_norm(x).reshape(-1, x.size(-1))


def old_loss_path(model: original_tg.GPT, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if model.tie_embeddings:
        logits_proj = F.linear(hidden, model.tok_emb.weight)
    else:
        logits_proj = model.lm_head(hidden)
    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return F.cross_entropy(logits.float(), targets, reduction="mean")


def new_loss_path(model: tg.GPT, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    projection_weight = model.tok_emb.weight if model.tie_embeddings else model.lm_head.weight
    return model.fused_loss_fn(projection_weight, hidden, targets)


def liger_runtime_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        weight = torch.randn(16, 8, device="cuda")
        hidden = torch.randn(4, 8, device="cuda")
        targets = torch.randint(0, 16, (4,), device="cuda")
        tg.LigerFusedLinearCrossEntropyLoss().cuda()(weight, hidden, targets)
    except Exception:
        return False
    return True


def configure_profile_loss(model: tg.GPT) -> None:
    if not liger_runtime_available():
        pytest.skip("no liger runtime available")


def bench(fn, runs: int = 5, warmup: int = 5, iters: int = 10) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        samples: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            samples.append((time.perf_counter() - start) / iters)
    return min(samples)


def test_liger_loss_path_is_faster_than_original_loss_path():
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        vocab_size = 8192
        model_dim = 256
        hidden = torch.randn(16 * 128, model_dim)
        targets = torch.randint(0, vocab_size, (16 * 128,))
        old_model = original_tg.GPT(
            vocab_size=vocab_size,
            num_layers=2,
            model_dim=model_dim,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
        )
        new_model = tg.GPT(
            vocab_size=vocab_size,
            num_layers=2,
            model_dim=model_dim,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
        )
        copy_original_gpt_to_new(new_model, old_model)
        configure_profile_loss(new_model)

        old_s = bench(lambda: old_loss_path(old_model, hidden, targets), iters=12)
        new_s = bench(lambda: new_loss_path(new_model, hidden, targets), iters=12)

        assert new_s < old_s
    finally:
        torch.set_num_threads(threads_before)


def test_new_gpt_forward_is_faster_than_original_gpt_forward():
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        vocab_size = 8192
        old_model = original_tg.GPT(
            vocab_size=vocab_size,
            num_layers=2,
            model_dim=256,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
        )
        new_model = tg.GPT(
            vocab_size=vocab_size,
            num_layers=2,
            model_dim=256,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
        )
        copy_original_gpt_to_new(new_model, old_model)
        configure_profile_loss(new_model)
        input_ids = torch.randint(0, vocab_size, (16, 128))
        target_ids = torch.randint(0, vocab_size, (16, 128))

        old_s = bench(lambda: old_model(input_ids, target_ids), iters=6)
        new_s = bench(lambda: new_model(input_ids, target_ids), iters=6)

        assert new_s < old_s
    finally:
        torch.set_num_threads(threads_before)
