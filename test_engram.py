from __future__ import annotations

import torch
from torch import nn

import train_gpt_sota as tg


class DummyBlock(nn.Module):
    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=None, v0=None):
        if v_embed is not None:
            x = x + v_embed.new_zeros(x.shape)
        return x + 0.01, None


def build_model(engram: bool) -> tg.GPT:
    model = tg.GPT(
        vocab_size=128,
        num_layers=4,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_vocab_size=0,
        xsa_last_n=0,
        rope_dims=0,
        ln_scale=False,
        ve_enabled=False,
        engram_enabled=engram,
        engram_vocab=64,
        engram_heads=2,
        engram_dim=8,
        engram_max_n=3,
        engram_layers="0,2",
    )
    model.blocks = nn.ModuleList([DummyBlock() for _ in range(4)])
    return model


def check(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)
    print(f"PASS: {msg}")


def main() -> None:
    torch.manual_seed(123)
    token_ids = torch.randint(0, 128, (3, 9))
    hidden = torch.randn(3, 9, 32)
    module = tg.EngramModule(max_n=3, engram_vocab=64, n_heads=2, engram_dim=8, model_dim=32)
    out0 = module(token_ids, hidden)
    check(out0.shape == hidden.shape, "engram output shape matches hidden state")
    check(torch.isfinite(out0).all().item(), "engram output is finite")
    check(out0.abs().max().item() == 0.0, "engram starts as no-op")
    nn.init.normal_(module.val_proj.weight, std=0.02)
    out1 = module(token_ids, hidden)
    check(out1.abs().sum().item() > 0.0, "engram becomes active when value projection is nonzero")
    check(module._hash(token_ids, 2).shape == (3, 9, 2), "bigram multi-head hash shape is correct")
    check(module._hash(token_ids, 3).shape == (3, 9, 2), "trigram multi-head hash shape is correct")

    base = build_model(False)
    eng = build_model(True)
    x = torch.randint(0, 128, (2, 8))
    y = torch.randint(0, 128, (2, 8))
    loss0 = base(x, y)
    loss1 = eng(x, y)
    check(loss0.ndim == 0 and loss1.ndim == 0, "GPT losses are scalar")
    check(torch.isfinite(loss0).item() and torch.isfinite(loss1).item(), "GPT losses are finite")
    loss1.backward()
    check(eng.engram is not None, "GPT stores Engram module")
    check(eng.engram.val_proj.weight.grad is not None, "Engram value projection receives gradient")
    check(eng.engram.tables[0].grad is not None, "Engram table receives gradient")
    hessian_model = tg._HessianGPT(
        vocab_size=128,
        num_layers=4,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_vocab_size=0,
        xsa_last_n=0,
        rope_dims=0,
        ln_scale=False,
        ve_enabled=False,
        engram_enabled=True,
        engram_vocab=64,
        engram_heads=2,
        engram_dim=8,
        engram_max_n=3,
        engram_layers="0,2",
    )
    unbanked = tg._unbank_state_dict(eng.state_dict(), 4)
    missing, unexpected = hessian_model.load_state_dict(
        {k: v for k, v in unbanked.items() if k in hessian_model.state_dict()},
        strict=False,
    )
    check(not unexpected, "Hessian Engram load has no unexpected keys")
    check("engram.tables.0" not in missing and "engram.val_proj.weight" not in missing, "Hessian Engram keys load")
    print("OK: Engram smoke passes.")


if __name__ == "__main__":
    main()
