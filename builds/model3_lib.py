from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as baseline

Muon = baseline.Muon
DistributedTokenLoader = baseline.DistributedTokenLoader
build_sentencepiece_luts = baseline.build_sentencepiece_luts
load_validation_tokens = baseline.load_validation_tokens
eval_val = baseline.eval_val
quantize_state_dict_int8 = baseline.quantize_state_dict_int8
dequantize_state_dict_int8 = baseline.dequantize_state_dict_int8
restore_low_dim_params_to_fp32 = baseline.restore_low_dim_params_to_fp32
CONTROL_TENSOR_NAME_PATTERNS = baseline.CONTROL_TENSOR_NAME_PATTERNS
RMSNorm = baseline.RMSNorm
CastedLinear = baseline.CastedLinear
MLP = baseline.MLP
AttentionBlock = baseline.Block


class SelectiveSSMExpert(nn.Module):
    def __init__(self, dim: int, state_dim: int, selector_hidden: int):
        super().__init__()
        self.state_dim = state_dim
        self.in_proj = CastedLinear(dim, state_dim, bias=False)
        self.selector_in = CastedLinear(dim, selector_hidden, bias=False)
        self.selector_out = CastedLinear(selector_hidden, 4 * state_dim, bias=False)
        self.out_proj = CastedLinear(state_dim, dim, bias=False)
        self.out_proj._zero_init = True
        self.d = nn.Parameter(torch.ones(state_dim, dtype=torch.float32))

    def step(self, x_t: Tensor, h_prev: Tensor) -> tuple[Tensor, Tensor]:
        u_t = self.in_proj(x_t)
        selector_hidden = F.silu(self.selector_in(x_t))
        a_raw, b_raw, c_raw, d_raw = self.selector_out(selector_hidden).chunk(4, dim=-1)
        a_t = torch.sigmoid(a_raw)
        b_t = torch.tanh(b_raw)
        c_t = torch.tanh(c_raw)
        d_t = torch.sigmoid(d_raw)
        h_t = a_t * h_prev + b_t * u_t
        y_t = c_t * h_t + (self.d.to(dtype=x_t.dtype)[None, :] * d_t) * u_t
        return self.out_proj(y_t), h_t

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        h_t = x.new_zeros(batch, self.state_dim)
        outputs: list[Tensor] = []
        for t in range(seq_len):
            y_t, h_t = self.step(x[:, t], h_t)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class SelectiveSSMLayer(nn.Module):
    def __init__(self, dim: int, state_dim: int, selector_hidden: int):
        super().__init__()
        self.expert = SelectiveSSMExpert(dim, state_dim, selector_hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.expert(x)


def murmurhash3_finalizer(token_ids: Tensor) -> Tensor:
    x = token_ids.to(dtype=torch.int64)
    x = x ^ (x >> 16)
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x = x ^ (x >> 13)
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return x


class HashRoutedSSMLayer(nn.Module):
    def __init__(self, dim: int, state_dim: int, selector_hidden: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.state_dim = state_dim
        self.experts = nn.ModuleList(
            [SelectiveSSMExpert(dim, state_dim, selector_hidden) for _ in range(num_experts)]
        )
        self.register_buffer("routing_counts", torch.zeros(num_experts, dtype=torch.long), persistent=False)

    def forward(self, x: Tensor, token_ids: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        routes = murmurhash3_finalizer(token_ids) % self.num_experts
        states = [x.new_zeros(batch, self.state_dim) for _ in range(self.num_experts)]
        outputs: list[Tensor] = []
        if self.routing_counts.device != x.device:
            self.routing_counts = self.routing_counts.to(x.device)
        self.routing_counts.zero_()
        for t in range(seq_len):
            x_t = x[:, t]
            route_t = routes[:, t]
            y_t = x.new_zeros(batch, dim)
            for expert_id, expert in enumerate(self.experts):
                mask = route_t == expert_id
                count = int(mask.sum().item())
                if count == 0:
                    continue
                self.routing_counts[expert_id] += count
                idx = mask.nonzero(as_tuple=False).squeeze(-1)
                prev_state = states[expert_id].index_select(0, idx)
                out_sel, next_state = expert.step(x_t.index_select(0, idx), prev_state)
                states[expert_id].index_copy_(0, idx, next_state)
                y_t.index_copy_(0, idx, out_sel)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def get_routing_fractions(self) -> Tensor:
        total = self.routing_counts.sum().clamp_min(1)
        return self.routing_counts.float() / total.float()


class SSMBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int, selector_hidden: int, mlp_mult: int):
        super().__init__()
        self.ssm_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.ssm = SelectiveSSMLayer(dim, state_dim, selector_hidden)
        self.mlp = MLP(dim, mlp_mult)
        self.ssm_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, token_ids: Tensor | None = None) -> Tensor:
        del token_ids
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        ssm_out = self.ssm(self.ssm_norm(x))
        x = x + self.ssm_scale.to(dtype=x.dtype)[None, None, :] * ssm_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class HashRoutedSSMBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int, selector_hidden: int, mlp_mult: int, num_experts: int = 4):
        super().__init__()
        self.ssm_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.ssm = HashRoutedSSMLayer(dim, state_dim, selector_hidden, num_experts=num_experts)
        self.mlp = MLP(dim, mlp_mult)
        self.ssm_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, token_ids: Tensor | None = None) -> Tensor:
        if token_ids is None:
            raise ValueError("HashRoutedSSMBlock requires token_ids for routing")
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        ssm_out = self.ssm(self.ssm_norm(x), token_ids)
        x = x + self.ssm_scale.to(dtype=x.dtype)[None, None, :] * ssm_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class AttentionAdapterBlock(nn.Module):
    def __init__(
        self,
        block_ctor: Callable[..., nn.Module],
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
    ):
        super().__init__()
        kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            rope_base=rope_base,
            qk_gain_init=qk_gain_init,
        )
        if "layer_idx" in block_ctor.__init__.__code__.co_varnames:
            kwargs["layer_idx"] = layer_idx
        if "rope_dims" in block_ctor.__init__.__code__.co_varnames:
            kwargs["rope_dims"] = rope_dims
        if "ln_scale" in block_ctor.__init__.__code__.co_varnames:
            kwargs["ln_scale"] = ln_scale
        self.block = block_ctor(**kwargs)

    def forward(self, x: Tensor, x0: Tensor, token_ids: Tensor | None = None) -> Tensor:
        del token_ids
        return self.block(x, x0)


class SSMOnlyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        state_dim: int | None = None,
        selector_hidden: int = 128,
        use_moe: bool = False,
        num_experts: int = 4,
        **_: object,
    ):
        super().__init__()
        del num_heads, num_kv_heads, rope_base, qk_gain_init
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.skip_weights = nn.Parameter(torch.zeros(0, model_dim, dtype=torch.float32))
        self.state_dim = state_dim or model_dim
        block_cls = HashRoutedSSMBlock if use_moe else SSMBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    model_dim,
                    self.state_dim,
                    selector_hidden,
                    mlp_mult,
                    num_experts=num_experts,
                )
                if use_moe
                else block_cls(model_dim, self.state_dim, selector_hidden, mlp_mult)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def embed(self, input_ids: Tensor) -> Tensor:
        return F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        x0 = x
        for block in self.blocks:
            x = block(x, x0, input_ids)
        return self.final_norm(x)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.forward_hidden(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight.to(dtype=x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids).reshape(-1, self.tok_emb.num_embeddings)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

    def routing_fractions(self) -> list[Tensor]:
        fractions: list[Tensor] = []
        for block in self.blocks:
            if isinstance(block, HashRoutedSSMBlock):
                fractions.append(block.ssm.get_routing_fractions().detach().cpu())
        return fractions


class HybridGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        state_dim: int | None = None,
        selector_hidden: int = 128,
        ssm_layers: int = 8,
        use_moe: bool = False,
        num_experts: int = 4,
        attn_block_ctor: Callable[..., nn.Module] = AttentionBlock,
        rope_dims: int = 0,
        ln_scale: bool = False,
        **_: object,
    ):
        super().__init__()
        if ssm_layers >= num_layers:
            raise ValueError("ssm_layers must be smaller than num_layers for hybrid models")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.skip_weights = nn.Parameter(torch.zeros(0, model_dim, dtype=torch.float32))
        self.state_dim = state_dim or model_dim
        block_cls = HashRoutedSSMBlock if use_moe else SSMBlock
        self.blocks = nn.ModuleList()
        for i in range(ssm_layers):
            if use_moe:
                block = block_cls(model_dim, self.state_dim, selector_hidden, mlp_mult, num_experts=num_experts)
            else:
                block = block_cls(model_dim, self.state_dim, selector_hidden, mlp_mult)
            self.blocks.append(block)
        for i in range(ssm_layers, num_layers):
            self.blocks.append(
                AttentionAdapterBlock(
                    attn_block_ctor,
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                    layer_idx=i,
                    rope_dims=rope_dims,
                    ln_scale=ln_scale,
                )
            )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def embed(self, input_ids: Tensor, bigram: nn.Module | None = None, smear: nn.Module | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        if bigram is not None:
            x = x + bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if smear is not None:
            x = smear(x)
        return x

    def forward_hidden(
        self,
        input_ids: Tensor,
        bigram: nn.Module | None = None,
        smear: nn.Module | None = None,
    ) -> Tensor:
        x = self.embed(input_ids, bigram=bigram, smear=smear)
        x0 = x
        for block in self.blocks:
            x = block(x, x0, input_ids)
        return self.final_norm(x)

    def logits_from_hidden(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight.to(dtype=x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self.logits_from_hidden(self.forward_hidden(input_ids))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids).reshape(-1, self.tok_emb.num_embeddings)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

    def routing_fractions(self) -> list[Tensor]:
        fractions: list[Tensor] = []
        for block in self.blocks:
            if isinstance(block, HashRoutedSSMBlock):
                fractions.append(block.ssm.get_routing_fractions().detach().cpu())
        return fractions


@dataclass
class SmokeResult:
    losses: list[float]
    step_ms: float
    routing_fractions: list[list[float]]


def synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len, device=device)[None, :]
    x = (starts + offsets) % vocab_size
    y = (x + 1) % vocab_size
    return x, y


def run_smoke_training(
    model: nn.Module,
    *,
    steps: int = 8,
    batch_size: int = 8,
    seq_len: int = 64,
    vocab_size: int = 128,
    lr: float = 3e-3,
    device: str | None = None,
) -> SmokeResult:
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(chosen)
    model = model.to(torch_device)
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        x, y = synthetic_batch(batch_size, seq_len, vocab_size, torch_device)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(steps, 1)
    routing = []
    if hasattr(model, "routing_fractions"):
        routing = [frac.tolist() for frac in model.routing_fractions()]
    return SmokeResult(losses=losses, step_ms=elapsed_ms, routing_fractions=routing)


def benchmark_step_time(model: nn.Module, *, batch_size: int = 4, seq_len: int = 128, vocab_size: int = 256) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    x, y = synthetic_batch(batch_size, seq_len, vocab_size, device)
    for _ in range(2):
        _ = model(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        _ = model(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / 5.0


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))
