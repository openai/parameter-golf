from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _env_int(name: str) -> int:
    return int(os.environ.get(name, "0"))


def _env_float(name: str) -> float:
    return float(os.environ.get(name, "0"))


def _apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    cos = freqs.cos()[None, :, None, :].to(x.dtype)
    sin = freqs.sin()[None, :, None, :].to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat(
        [x1 * cos[..., :half] - x2 * sin[..., half:], x1 * sin[..., :half] + x2 * cos[..., half:]], dim=-1
    )


def _patch_memory_block(block: nn.Module, mem_tokens: int) -> None:
    if mem_tokens <= 0 or not hasattr(block, "qkv") or hasattr(block, "mem_k"):
        return
    std = 0.02 / max(block.head_dim, 1) ** 0.5
    device = block.qkv.weight.device
    block.mem_tokens = mem_tokens
    block.register_parameter("mem_k", nn.Parameter(torch.empty(block.n_heads * mem_tokens, block.head_dim, device=device)))
    block.register_parameter("mem_v", nn.Parameter(torch.empty(block.n_heads * mem_tokens, block.head_dim, device=device)))
    nn.init.normal_(block.mem_k, mean=0.0, std=std)
    nn.init.normal_(block.mem_v, mean=0.0, std=std)

    def forward(self: nn.Module, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        residual = x
        x = self.norm(x)
        bsz, seq_len, d_model = x.shape
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.kv_heads * self.head_dim
        q, k, v = self.qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        q = self.q_norm(q.reshape(bsz, seq_len, self.n_heads, self.head_dim))
        k = self.k_norm(k.reshape(bsz, seq_len, self.kv_heads, self.head_dim))
        v = v.reshape(bsz, seq_len, self.kv_heads, self.head_dim)
        freqs = self.rope_freqs[:seq_len]
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=2)
            v = v.repeat_interleave(self.kv_groups, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        mem_k = self.mem_k.view(self.n_heads, self.mem_tokens, self.head_dim).to(x.dtype).unsqueeze(0).expand(bsz, -1, -1, -1)
        mem_v = self.mem_v.view(self.n_heads, self.mem_tokens, self.head_dim).to(x.dtype).unsqueeze(0).expand(bsz, -1, -1, -1)
        k = torch.cat([mem_k, k], dim=2)
        v = torch.cat([mem_v, v], dim=2)
        mem_prefix = torch.ones((seq_len, self.mem_tokens), dtype=torch.bool, device=x.device)
        token_mask = self.attn_mask[:seq_len, :seq_len]
        if reset_mask is not None:
            seg_ids = reset_mask.to(torch.int64).cumsum(dim=1)
            token_mask = (seg_ids[:, :, None] == seg_ids[:, None, :]) & token_mask
            attn_mask = torch.cat(
                [torch.ones((bsz, seq_len, self.mem_tokens), dtype=torch.bool, device=x.device), token_mask], dim=-1
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[:, None, :, :], is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.cat([mem_prefix, token_mask], dim=-1), is_causal=False)
        y = y.transpose(1, 2).reshape(bsz, seq_len, d_model)
        x = residual + self.out(y)
        return self.ffn(x) if self.ffn is not None else x

    block.forward = types.MethodType(forward, block)


def _tie_attention_layers(model: nn.Module, loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor], physical_layers: int) -> None:
    total_layers = len(model.blocks)
    if physical_layers <= 0 or physical_layers >= total_layers:
        return
    model.blocks = nn.ModuleList(list(model.blocks[:physical_layers]))
    pattern = tuple(i % physical_layers for i in range(total_layers))
    model._tied_layer_pattern = pattern

    def forward(self: nn.Module, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        reset_mask = input_ids.eq(self.bos_id) if self.reset_on_bos else None
        for idx in self._tied_layer_pattern:
            x = self.blocks[idx](x, reset_mask)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return loss_fn(logits.float(), targets, self._train_step)

    model.forward = types.MethodType(forward, model)


def apply_attention_playground(model: nn.Module, loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor]) -> None:
    mem_tokens = _env_int("MEMORY_TOKENS")
    tied_layers = _env_int("ATTN_TIED_LAYERS")
    if mem_tokens > 0:
        for block in model.blocks:
            _patch_memory_block(block, mem_tokens)
    if tied_layers > 0:
        _tie_attention_layers(model, loss_fn, tied_layers)


def maybe_load_init_ckpt(model: nn.Module, log_fn: Callable[[str], None]) -> None:
    ckpt_path = os.environ.get("INIT_CKPT", "").strip()
    if not ckpt_path:
        return
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"INIT_CKPT={ckpt_path} did not contain a state dict")
    missing, unexpected = model.load_state_dict(payload, strict=False)
    log_fn(f"init_ckpt:{ckpt_path} missing:{len(missing)} unexpected:{len(unexpected)}")


def maybe_init_ema(model: nn.Module) -> dict[str, object] | None:
    decay = _env_float("EMA_DECAY")
    if not 0.0 < decay < 1.0:
        return None
    params = {name: param.detach().clone() for name, param in model.named_parameters() if param.dtype.is_floating_point}
    return {"decay": decay, "params": params}


def maybe_update_ema(ema_state: dict[str, object] | None, model: nn.Module) -> None:
    if ema_state is None:
        return
    decay = float(ema_state["decay"])
    shadow = ema_state["params"]
    for name, param in model.named_parameters():
        if name in shadow:
            shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def maybe_swap_to_ema(model: nn.Module, ema_state: dict[str, object] | None, log_fn: Callable[[str], None]) -> None:
    if ema_state is None:
        return
    shadow = ema_state["params"]
    for name, param in model.named_parameters():
        if name in shadow:
            param.data.copy_(shadow[name])
    log_fn(f"ema_swap:decay:{float(ema_state['decay']):.6f}")
