from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from my_submission import train_gpt as tg


def write_shard(path, tokens: list[int]) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with open(path, "wb") as f:
        header.tofile(f)
        np.asarray(tokens, dtype="<u2").tofile(f)


def test_hyperparameters_defaults_are_sane():
    args = tg.Hyperparameters()
    assert args.vocab_size > 0
    assert args.train_seq_len > 0
    assert args.num_layers > 0
    assert args.model_dim % args.num_heads == 0


def test_muon_step_updates_parameter(monkeypatch):
    param = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32))
    opt = tg.Muon([param], lr=0.1, momentum=0.0, backend_steps=1)
    monkeypatch.setattr(tg, "zeropower_via_newtonschulz5", lambda G, steps=10, eps=1e-7: G)
    param.grad = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32)
    before = param.detach().clone()
    opt.step()
    assert "momentum_buffer" in opt.state[param]
    assert not torch.allclose(param, before)


def test_token_stream_wraps_across_shards(tmp_path):
    write_shard(tmp_path / "train_a.bin", [1, 2, 3])
    write_shard(tmp_path / "train_b.bin", [4, 5])
    stream = tg.TokenStream(str(tmp_path / "train_*.bin"))
    taken = stream.take(7)
    assert taken.tolist() == [1, 2, 3, 4, 5, 1, 2]


def test_distributed_token_loader_returns_shifted_batches(tmp_path):
    write_shard(tmp_path / "train.bin", [10, 11, 12, 13, 14, 15])
    loader = tg.DistributedTokenLoader(str(tmp_path / "*.bin"), rank=1, world_size=2, device=torch.device("cpu"))
    x, y = loader.next_batch(global_tokens=4, seq_len=2, grad_accum_steps=1)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.tolist() == [[13, 14]]
    assert y.tolist() == [[14, 15]]


def test_rmsnorm_matches_functional():
    mod = tg.RMSNorm(eps=1e-6)
    x = torch.randn(2, 3, 4)
    y = mod(x)
    expected = F.rms_norm(x, (x.size(-1),), eps=1e-6)
    assert torch.allclose(y, expected)


def test_casted_linear_uses_input_dtype():
    mod = tg.CastedLinear(3, 2, bias=True)
    x = torch.randn(4, 3, dtype=torch.float64)
    y = mod(x)
    expected = F.linear(x, mod.weight.to(x.dtype), mod.bias.to(x.dtype))
    assert y.dtype == torch.float64
    assert torch.allclose(y, expected)


def test_rotary_caches_tables():
    rotary = tg.Rotary(4)
    cos1, sin1 = rotary(5, torch.device("cpu"), torch.float32)
    cos2, sin2 = rotary(5, torch.device("cpu"), torch.float32)
    assert cos1.shape == (1, 1, 5, 2)
    assert sin1.shape == (1, 1, 5, 2)
    assert rotary._seq_len_cached == 5
    assert torch.allclose(cos1, cos2)
    assert torch.allclose(sin1, sin2)


def test_causal_self_attention_runs_and_preserves_shape():
    attn = tg.CausalSelfAttention(dim=12, num_heads=3, num_kv_heads=1, rope_base=10000.0, qk_gain_init=1.5)
    x = torch.randn(2, 4, 12)
    y = attn(x)
    assert attn.c_k.weight.shape == (4, 12)
    assert attn.c_v.weight.shape == (4, 12)
    q = attn.c_q(x).reshape(2, 4, 3, 4).transpose(1, 2)
    k = attn.c_k(x).reshape(2, 4, 1, 4).transpose(1, 2)
    v = attn.c_v(x).reshape(2, 4, 1, 4).transpose(1, 2)
    assert q.shape == (2, 3, 4, 4)
    assert k.shape == (2, 1, 4, 4)
    assert v.shape == (2, 1, 4, 4)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mlp_runs_and_preserves_shape():
    mlp = tg.MLP(dim=8, mlp_mult=2)
    x = torch.randn(2, 4, 8)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_block_runs_and_preserves_shape():
    block = tg.Block(dim=12, num_heads=3, num_kv_heads=1, mlp_mult=2, rope_base=10000.0, qk_gain_init=1.5)
    x = torch.randn(2, 4, 12)
    y = block(x, x.clone())
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_gpt_forward_returns_scalar_loss(tie_embeddings):
    model = tg.GPT(
        vocab_size=32,
        num_layers=4,
        model_dim=8,
        num_heads=2,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=tie_embeddings,
        tied_embed_init_std=0.01,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    )
    input_ids = torch.randint(0, 32, (2, 4))
    target_ids = torch.randint(0, 32, (2, 4))
    loss = model(input_ids, target_ids)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
