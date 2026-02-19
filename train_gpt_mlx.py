#!/usr/bin/env python3
"""
MLX port that tracks train_gpt.py behavior as closely as practical on a single Mac device.

Notable scope:
- No distributed collectives/comms paths.
- No Triton/FlashAttention kernels; attention is implemented with dense masks.
- MTP loss is approximated with shifted-target CE terms.
"""

from __future__ import annotations

import argparse
import glob
import math
import time
from dataclasses import dataclass
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
except ImportError as exc:
    raise SystemExit("Missing MLX. Install with: python3 -m pip install mlx") from exc


MAGIC_NUMBER = 20240520
VERSION = 1
HEADER_INTS = 256
BOS_ID = 50256
RAND_INT_1 = 36313
RAND_INT_2 = 27191


def next_multiple_of_n(v: int, n: int) -> int:
    return ((v + n - 1) // n) * n


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def linear_no_bias(x: mx.array, weight_out_in: mx.array) -> mx.array:
    return x @ mx.transpose(weight_out_in, (1, 0))


def softcap_logits(logits: mx.array) -> mx.array:
    return 23.0 * mx.sigmoid((logits + 5.0) / 7.5)


def load_data_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if len(header) != HEADER_INTS:
        raise ValueError(f"{path}: shard header is truncated")
    if int(header[0]) != MAGIC_NUMBER:
        raise ValueError(f"{path}: bad magic number {int(header[0])}, expected {MAGIC_NUMBER}")
    if int(header[1]) != VERSION:
        raise ValueError(f"{path}: unsupported shard version {int(header[1])}")

    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype=np.uint16, count=num_tokens, offset=HEADER_INTS * 4)
    if len(tokens) != num_tokens:
        raise ValueError(f"{path}: expected {num_tokens} tokens, found {len(tokens)}")
    return tokens.astype(np.int32, copy=False)


def get_bigram_hash_np(x: np.ndarray, bigram_vocab_size: int) -> np.ndarray:
    mod = bigram_vocab_size - 1
    out = x.astype(np.int32, copy=True)
    out[0] = mod
    out[1:] = np.bitwise_xor(RAND_INT_1 * out[1:], RAND_INT_2 * out[:-1]) % mod
    return out


class Shard:
    def __init__(self, tokens: np.ndarray):
        self.tokens = tokens
        self.size = int(tokens.size)
        self.i = 0
        self.bos_idx = np.flatnonzero(tokens == BOS_ID).astype(np.int64)
        if len(self.bos_idx) == 0:
            raise ValueError("Shard has no BOS tokens; cannot build BOS-aligned batches")

    def next_batch(self, num_tokens_local: int, max_seq_len: int) -> tuple[list[int], list[int]]:
        starts: list[int] = []
        ends: list[int] = []

        idx = self.i
        cur_len = 0
        n = len(self.bos_idx)
        while cur_len <= num_tokens_local:
            if idx >= n:
                raise StopIteration("Insufficient BOS ahead; reached end of shard")
            cur = int(self.bos_idx[idx])
            starts.append(cur)
            nxt = int(self.bos_idx[idx + 1]) if idx + 1 < n else self.size
            end = min(nxt, cur + max_seq_len, cur + num_tokens_local - cur_len + 1)
            ends.append(end)
            cur_len += end - cur
            idx += 1

        if cur_len != num_tokens_local + 1:
            raise RuntimeError(f"Expected {num_tokens_local + 1} tokens, got {cur_len}")
        self.i = idx
        return starts, ends


class BosAlignedDataLoader:
    def __init__(
        self,
        filename_pattern: str,
        bigram_vocab_size: int,
    ):
        self.files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
        self.bigram_vocab_size = bigram_vocab_size
        self.file_idx = 0
        self.shard = Shard(load_data_shard(self.files[self.file_idx]))

    def _advance_shard(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.shard = Shard(load_data_shard(self.files[self.file_idx]))

    def next_batch(self, num_tokens_local: int, max_seq_len: int) -> tuple[mx.array, mx.array, np.ndarray, mx.array]:
        while True:
            try:
                starts, ends = self.shard.next_batch(num_tokens_local, max_seq_len)
                break
            except StopIteration:
                self._advance_shard()

        chunks = [self.shard.tokens[s:e] for s, e in zip(starts, ends)]
        buf = np.concatenate(chunks, axis=0)
        x_np = buf[:-1].astype(np.int32, copy=False)
        y_np = buf[1:].astype(np.int32, copy=False)

        doc_lens = [e - s for s, e in zip(starts, ends)]
        doc_lens[-1] -= 1
        if sum(doc_lens) != num_tokens_local:
            raise RuntimeError("Document length accounting mismatch in BOS loader")
        seq_ids = np.repeat(np.arange(len(doc_lens), dtype=np.int32), doc_lens)
        bigram_np = get_bigram_hash_np(x_np, self.bigram_vocab_size)

        return (
            mx.array(x_np, dtype=mx.int32),
            mx.array(y_np, dtype=mx.int32),
            seq_ids,
            mx.array(bigram_np, dtype=mx.int32),
        )


class ContiguousDataLoader:
    def __init__(self, filename_pattern: str, bigram_vocab_size: int):
        self.files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
        self.bigram_vocab_size = bigram_vocab_size
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def _advance_shard(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def next_batch(self, num_tokens_local: int, _max_seq_len: int = -1) -> tuple[mx.array, mx.array, np.ndarray, mx.array]:
        while self.pos + num_tokens_local + 1 >= len(self.tokens):
            self._advance_shard()

        buf = self.tokens[self.pos : self.pos + num_tokens_local + 1]
        self.pos += num_tokens_local

        x_np = buf[:-1].astype(np.int32, copy=False)
        y_np = buf[1:].astype(np.int32, copy=False)

        seq_ids = np.cumsum(x_np == BOS_ID, dtype=np.int32)
        if len(seq_ids) > 0 and seq_ids[0] > 0:
            seq_ids = seq_ids - seq_ids[0]

        bigram_np = get_bigram_hash_np(x_np, self.bigram_vocab_size)
        return (
            mx.array(x_np, dtype=mx.int32),
            mx.array(y_np, dtype=mx.int32),
            seq_ids,
            mx.array(bigram_np, dtype=mx.int32),
        )


def build_doc_causal_mask(seq_ids: np.ndarray, window: Optional[int]) -> mx.array:
    t = int(len(seq_ids))
    i = np.arange(t, dtype=np.int32)[:, None]
    j = np.arange(t, dtype=np.int32)[None, :]
    allowed = j <= i
    if window is not None:
        allowed &= (i - j) <= int(window)
    allowed &= seq_ids[:, None] == seq_ids[None, :]
    mask = np.where(allowed, 0.0, -1e9).astype(np.float32)
    return mx.array(mask)


class Yarn:
    def __init__(self, head_dim: int, max_seq_len: int, paired: bool = False):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def reset(self) -> None:
        angular_freq = (1.0 / 1024.0) ** mx.linspace(0.0, 1.0, num=self.head_dim // 4, dtype=mx.float32)
        angular_freq = mx.repeat(angular_freq, 2)
        angular_freq = mx.concatenate([angular_freq, mx.zeros((self.head_dim // 2,), dtype=mx.float32)], axis=0)

        t = mx.arange(2 * self.max_seq_len, dtype=mx.float32)
        if not self.paired:
            theta = mx.outer(t, angular_freq)
            factor1 = mx.cos(theta)
            factor2 = mx.sin(theta)
        else:
            t_even = 2.0 * t
            t_odd = 2.0 * t + 1.0
            theta1 = mx.outer(t_even, angular_freq)
            theta2 = mx.outer(t_odd, angular_freq)
            factor1 = mx.concatenate([mx.cos(theta1), mx.cos(theta2)], axis=-1)
            factor2 = mx.concatenate([mx.sin(theta1), mx.sin(theta2)], axis=-1)

        # Match train_gpt sign convention
        idx = np.arange(factor2.shape[-1])
        sign = np.ones_like(idx, dtype=np.float32)
        sign[idx % 2 == 1] = -1.0
        factor2 = factor2 * mx.array(sign, dtype=mx.float32)

        self.angular_freq = angular_freq
        self.factor1 = factor1.astype(mx.float32)
        self.factor2 = factor2.astype(mx.float32)
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32) -> None:
        rotations = old_window * self.angular_freq / (2.0 * math.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = mx.clip((rotations - alpha) / (beta - alpha), 0.0, 1.0)
        self.angular_freq = self.angular_freq * (scaling_factor + interpolation_weight * (1.0 - scaling_factor))

        t = mx.arange(2 * self.max_seq_len, dtype=mx.float32)
        if not self.paired:
            theta = mx.outer(t, self.angular_freq)
            factor1 = mx.cos(theta)
            factor2 = mx.sin(theta)
        else:
            t_even = 2.0 * t
            t_odd = 2.0 * t + 1.0
            theta1 = mx.outer(t_even, self.angular_freq)
            theta2 = mx.outer(t_odd, self.angular_freq)
            factor1 = mx.concatenate([mx.cos(theta1), mx.cos(theta2)], axis=-1)
            factor2 = mx.concatenate([mx.sin(theta1), mx.sin(theta2)], axis=-1)

        idx = np.arange(factor2.shape[-1])
        sign = np.ones_like(idx, dtype=np.float32)
        sign[idx % 2 == 1] = -1.0
        self.factor1 = factor1.astype(mx.float32)
        self.factor2 = (factor2 * mx.array(sign, dtype=mx.float32)).astype(mx.float32)
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1.0

    def rotary(self, x_bthd: mx.array) -> mx.array:
        t = x_bthd.shape[1]
        factor1 = self.factor1[None, :t, None, :]
        factor2 = self.factor2[None, :t, None, :]

        x = x_bthd.reshape(*x_bthd.shape[:-1], x_bthd.shape[-1] // 2, 2)
        x_flip = mx.concatenate([x[..., 1:2], x[..., 0:1]], axis=-1).reshape(x_bthd.shape)
        return factor1 * x_bthd + factor2 * x_flip


@dataclass
class ForwardScheduleConfig:
    ws_short: int
    ws_long: int
    train_max_seq_len: int


@dataclass
class TrainingStage:
    lr_mul: float
    batch_size_tokens: int
    window_sizes: tuple[int, int]
    train_max_seq_len: int
    duration: Optional[float] = None


class TrainingSchedule:
    def __init__(
        self,
        stages: list[TrainingStage],
        scheduled_iterations: int,
        extension_iterations: int,
        cooldown_frac: float = 0.55,
        split_embed_stage: int = 2,
        ws_post_yarn_ext: int = 20,
    ):
        self.stages = stages
        self.scheduled_iterations = scheduled_iterations
        self.cooldown_frac = cooldown_frac
        self.ws_post_yarn_ext = ws_post_yarn_ext
        self.total_steps = scheduled_iterations + extension_iterations

        ends = [0] + [round(c * scheduled_iterations) for c in accumulate(s.duration for s in stages[:-1])] + [self.total_steps]
        if self.scheduled_iterations != ends[-2]:
            raise ValueError("Scheduled iteration boundary mismatch")
        self.boundaries = list(pairwise(ends))
        self.split_step = self.boundaries[split_embed_stage][0] | 1

    def lookup(self, step: int) -> tuple[TrainingStage, float]:
        for i, (start, end) in enumerate(self.boundaries):
            if step < end:
                t = (step - start) / max(1, (end - start))
                return self.stages[i], t
        return self.stages[-1], 1.0

    def get_lr_mul(self, step: int) -> float:
        stage, _ = self.lookup(step)
        lr = stage.lr_mul
        cd_start = int(self.scheduled_iterations * (1.0 - self.cooldown_frac))
        if step >= cd_start and self.scheduled_iterations > cd_start:
            t = min(1.0, (step - cd_start) / (self.scheduled_iterations - cd_start))
            lr = lr * (1.0 - t) + 0.15 * t
        return float(lr)


@dataclass
class ParamConfig:
    lr_mul: float = 1.0
    wd_mul: float = 1.0
    adam_betas: tuple[float, float] = (0.9, 0.95)


class AdamMLX:
    def __init__(
        self,
        model: "GPTMLX",
        adam_lr: float,
        adam_eps: float,
        adam_weight_decay: float,
    ):
        self.adam_lr = float(adam_lr)
        self.adam_eps = float(adam_eps)
        self.adam_weight_decay = float(adam_weight_decay)

        self.split_embed = False
        self.adam_step = 0

        params = dict(tree_flatten(model.parameters()))
        self.param_cfgs = {key: self._cfg_for_key(key) for key in params.keys()}
        self.adam_m = {k: mx.zeros_like(v) for k, v in params.items()}
        self.adam_v = {k: mx.zeros_like(v) for k, v in params.items()}

    def _cfg_for_key(self, key: str) -> ParamConfig:
        if key == "scalars":
            return ParamConfig(lr_mul=5.0, wd_mul=0.0, adam_betas=(0.9, 0.99))
        if key == "smear_gate_weight":
            return ParamConfig(lr_mul=0.01, wd_mul=0.0, adam_betas=(0.9, 0.99))
        if key == "skip_gate_weight":
            return ParamConfig(lr_mul=0.05, wd_mul=0.0, adam_betas=(0.9, 0.99))
        if key in {"attn_gate_bank", "ve_gate_bank"}:
            return ParamConfig(adam_betas=(0.9, 0.99))
        if key == "x0_lambdas":
            return ParamConfig(lr_mul=5.0, wd_mul=0.0, adam_betas=(0.65, 0.95))
        if key == "bigram_embed.weight":
            return ParamConfig(lr_mul=75.0, wd_mul=5.0, adam_betas=(0.75, 0.95))
        if key == "lm_head_weight":
            return ParamConfig(wd_mul=150.0, adam_betas=(0.5, 0.95))
        if key == "value_embeds":
            return ParamConfig(lr_mul=75.0, wd_mul=5.0, adam_betas=(0.75, 0.95))
        if key == "embed.weight":
            return ParamConfig(wd_mul=150.0, adam_betas=(0.5, 0.95))
        return ParamConfig()

    def _adam_update(self, key: str, param: mx.array, grad: mx.array, step_lr_mul: float) -> mx.array:
        cfg = self.param_cfgs[key]
        beta1, beta2 = cfg.adam_betas
        lr = self.adam_lr * cfg.lr_mul * step_lr_mul
        wd = self.adam_weight_decay * cfg.wd_mul

        m_prev = self.adam_m[key]
        v_prev = self.adam_v[key]

        m = beta1 * m_prev + (1.0 - beta1) * grad
        v = beta2 * v_prev + (1.0 - beta2) * (grad * grad)
        self.adam_m[key] = m
        self.adam_v[key] = v

        bc1 = 1.0 - beta1 ** self.adam_step
        bc2 = 1.0 - beta2 ** self.adam_step
        m_hat = m / bc1
        v_hat = v / bc2

        upd = m_hat / (mx.sqrt(v_hat) + self.adam_eps)
        if wd > 0:
            upd = upd + wd * param
        return param - lr * upd.astype(param.dtype)

    def copy_lm_state_to_embed(self) -> None:
        if "lm_head_weight" not in self.adam_m or "embed.weight" not in self.adam_m:
            return
        self.adam_m["embed.weight"] = mx.transpose(self.adam_m["lm_head_weight"], (1, 0))
        self.adam_v["embed.weight"] = mx.transpose(self.adam_v["lm_head_weight"], (1, 0))
        self.split_embed = True

    def step(
        self,
        model: "GPTMLX",
        grads: dict,
        step_lr_mul: float,
    ) -> None:
        self.adam_step += 1

        params_flat = dict(tree_flatten(model.parameters()))
        grads_flat = dict(tree_flatten(grads))

        updated: dict[str, mx.array] = {}
        for key, param in params_flat.items():
            cfg = self.param_cfgs.get(key)
            grad = grads_flat.get(key)
            if cfg is None or grad is None:
                updated[key] = param
                continue

            if key == "embed.weight" and not self.split_embed:
                updated[key] = param
                continue
            updated[key] = self._adam_update(key, param, grad, step_lr_mul=step_lr_mul)

        model.update(tree_unflatten(updated))

    def state_for_eval(self):
        return self.adam_m, self.adam_v


class GPTMLX(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        model_dim: int,
        max_seq_len: int,
        bigram_vocab_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = next_multiple_of_n(vocab_size, 128)
        self.bigram_vocab_size = bigram_vocab_size

        if self.num_heads * self.head_dim != self.model_dim:
            raise ValueError("num_heads * head_dim must equal model_dim")

        self.tied_phase = True

        self.smear_gate_weight = mx.zeros((1, 12), dtype=mx.float32)
        self.skip_gate_weight = mx.zeros((1, 12), dtype=mx.float32)

        self.value_embeds = 0.01 * mx.random.normal((5 * self.vocab_size, model_dim), dtype=mx.float32)

        self.attn_layer_indices = [i for i in range(num_layers) if i != 6]
        self.mlp_layer_indices = list(range(num_layers))
        self.layer_to_attn_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.attn_layer_indices)}
        self.layer_to_mlp_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.mlp_layer_indices)}

        num_attn_layers = len(self.attn_layer_indices)
        self.attn_gate_bank = mx.zeros((num_attn_layers, num_heads, 12), dtype=mx.float32)
        self.ve_gate_bank = mx.zeros((5, num_heads, 12), dtype=mx.float32)

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim
        self.attn_bank = mx.zeros((num_attn_layers, 4 * model_dim, hdim), dtype=mx.float32)

        num_mlp_with_padding = len(self.mlp_layer_indices) + 1
        self.mlp_bank = mx.zeros((num_mlp_with_padding, 2, mlp_hdim, model_dim), dtype=mx.float32)

        std = 0.5 * model_dim ** -0.5
        bound = math.sqrt(3.0) * std
        self.attn_bank = mx.random.uniform(low=-bound, high=bound, shape=self.attn_bank.shape, dtype=mx.float32)
        c_fc = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(num_mlp_with_padding, mlp_hdim, model_dim),
            dtype=mx.float32,
        )
        c_proj = mx.zeros((num_mlp_with_padding, mlp_hdim, model_dim), dtype=mx.float32)
        self.mlp_bank = mx.stack([c_fc, c_proj], axis=1)

        self.lm_head_weight = 0.005 * mx.random.normal((model_dim, self.vocab_size), dtype=mx.float32)
        self.embed = nn.Embedding(self.vocab_size, model_dim)
        self.embed.weight = mx.transpose(self.lm_head_weight, (1, 0))

        self.bigram_embed = nn.Embedding(self.bigram_vocab_size, model_dim)
        self.bigram_embed.weight = mx.zeros_like(self.bigram_embed.weight)

        self.x0_lambdas = mx.zeros((num_layers,), dtype=mx.float32)

        pad = 0
        scalars = [
            1.1 * mx.ones((num_layers,), dtype=mx.float32),
            mx.array([0.5, 1.0] * num_layers, dtype=mx.float32),
            0.1 * mx.ones((num_layers,), dtype=mx.float32),
            mx.array([0.0], dtype=mx.float32),
            0.5 * mx.ones((1,), dtype=mx.float32),
            -1.5 * mx.ones((1,), dtype=mx.float32),
            mx.ones((pad,), dtype=mx.float32),
        ]
        self.scalars = mx.concatenate(scalars, axis=0)

        self.paired_head_layers = [i for i in [0, 2, 5, 9] if i < num_layers]
        self.yarn = Yarn(head_dim, max_seq_len)
        self.yarn_paired_head = Yarn(head_dim, max_seq_len, paired=True)

    def split_embed_from_lm(self) -> None:
        self.embed.weight = mx.transpose(self.lm_head_weight, (1, 0))
        self.tied_phase = False

    def _attention(
        self,
        x: mx.array,
        seq_ids: np.ndarray,
        qkvo_w: mx.array,
        sa_lambdas: mx.array,
        ve: Optional[mx.array],
        key_offset: bool,
        attn_gate_w: mx.array,
        ve_gate_w: Optional[mx.array],
        bm_size: Optional[int],
        yarn: Yarn,
        mask_cache: dict[tuple[Optional[int], bool], mx.array],
    ) -> mx.array:
        bsz, t, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        qkv = linear_no_bias(x, sa_lambdas[0] * qkvo_w[: 3 * self.model_dim, :])
        qkv = qkv.reshape(bsz, t, 3 * h, d)
        q, k, v = mx.split(qkv, 3, axis=2)

        q = rms_norm(q)
        k = rms_norm(k)

        paired = yarn.paired
        if not paired:
            q = yarn.rotary(q)
            k = yarn.rotary(k)
            if key_offset:
                stationary = k[:, :, :, d // 2 :]
                shifted = mx.concatenate([stationary[:, :1, :, :], stationary[:, :-1, :, :]], axis=1)
                k = mx.concatenate([k[:, :, :, : d // 2], shifted], axis=-1)

            if ve is not None and ve_gate_w is not None:
                gate_in = mx.concatenate([x[:, :, :6], ve[None, :, :6]], axis=-1)
                ve_gate = 2.0 * mx.sigmoid(linear_no_bias(gate_in, ve_gate_w)).reshape(bsz, t, h, 1)
                v = v + ve_gate * ve.reshape(bsz, t, h, d)

            mask_key = (bm_size, False)
            if mask_key not in mask_cache:
                mask_cache[mask_key] = build_doc_causal_mask(seq_ids, bm_size)
            mask = mask_cache[mask_key]
        else:
            q = q.reshape(bsz, t, h // 2, d * 2)
            k = k.reshape(bsz, t, h // 2, d * 2)
            v = v.reshape(bsz, t * 2, h // 2, d)

            q = yarn.rotary(q)
            k = yarn.rotary(k)
            q = q.reshape(bsz, t * 2, h // 2, d)
            k = k.reshape(bsz, t * 2, h // 2, d)

            if ve is not None and ve_gate_w is not None:
                gate_in = x[:, :, :12]
                ve_gate_raw = linear_no_bias(gate_in, ve_gate_w)  # [B, T, H]
                ve_gate = 2.0 * mx.sigmoid(ve_gate_raw).reshape(bsz, t * 2, h // 2, 1)
                v = v + ve_gate * ve.reshape(bsz, t * 2, h // 2, d)

            seq_ids2 = np.repeat(seq_ids, 2)
            mask_key = (bm_size, True)
            if mask_key not in mask_cache:
                mask_cache[mask_key] = build_doc_causal_mask(seq_ids2, bm_size)
            mask = mask_cache[mask_key]

        qh = mx.transpose(q, (0, 2, 1, 3))
        kh = mx.transpose(k, (0, 2, 1, 3))
        vh = mx.transpose(v, (0, 2, 1, 3))

        scores = (qh @ mx.transpose(kh, (0, 1, 3, 2))) * yarn.attn_scale
        scores = scores + mask[None, None, :, :]
        probs = mx.softmax(scores, axis=-1)
        yh = probs @ vh
        y = mx.transpose(yh, (0, 2, 1, 3))

        if paired:
            y = y.reshape(bsz, t, h, d)

        attn_gate = mx.sigmoid(linear_no_bias(x[:, :, :12], attn_gate_w)).reshape(bsz, t, h, 1)
        y = y * attn_gate
        y = y.reshape(bsz, t, self.model_dim)
        y = linear_no_bias(y, sa_lambdas[1] * qkvo_w[3 * self.model_dim :, :])
        return y

    def _mlp(self, x: mx.array, c_fc: mx.array, c_proj: mx.array) -> mx.array:
        h = linear_no_bias(x, c_fc)
        h = mx.maximum(h, 0.0)
        h = h * h
        return h @ c_proj

    def forward(self, input_seq: mx.array, bigram_input_seq: mx.array, seq_ids: np.ndarray, cfg: ForwardScheduleConfig) -> mx.array:
        if input_seq.ndim != 1:
            raise ValueError("input_seq must be 1D")

        ws_short = cfg.ws_short
        ws_long = cfg.ws_long

        skip_connections: list[mx.array] = []
        skip_in = [i for i in [3] if i < self.num_layers]
        skip_out = [i for i in [6] if i < self.num_layers]
        backout_layer = 7 if self.num_layers > 7 else self.num_layers - 1
        x_backout = None

        resid_lambdas = self.scalars[: self.num_layers]
        sa_lambdas = self.scalars[self.num_layers : 3 * self.num_layers].reshape(self.num_layers, 2)
        bigram_lambdas = self.scalars[3 * self.num_layers : 4 * self.num_layers]
        smear_lambda = self.scalars[4 * self.num_layers]
        backout_lambda = self.scalars[4 * self.num_layers + 1]
        skip_lambda = self.scalars[4 * self.num_layers + 2]

        bm_sizes: list[Optional[int]] = [ws_short] * self.num_layers
        if self.num_layers > 3:
            bm_sizes[3] = ws_long
        if self.num_layers > 0:
            bm_sizes[-1] = ws_long
        if self.num_layers > 6:
            bm_sizes[6] = None
        key_offset = [b == ws_long for b in bm_sizes]

        if self.tied_phase:
            embed_w = mx.transpose(self.lm_head_weight, (1, 0))
            x = mx.take(embed_w, input_seq, axis=0)
        else:
            x = self.embed(input_seq)

        x0_bigram = self.bigram_embed(bigram_input_seq)[None, :, :]

        ve_table = self.value_embeds.reshape(5, self.vocab_size, self.model_dim)
        ve_tok = ve_table[:, input_seq]
        ve_layers: list[Optional[mx.array]] = [None] * self.num_layers
        if self.num_layers > 1:
            ve_layers[1] = ve_tok[0]
        if self.num_layers > 2:
            ve_layers[2] = ve_tok[1]
        if self.num_layers > 0:
            ve_layers[-1] = ve_tok[4]
        if self.num_layers > 1:
            ve_layers[-2] = ve_tok[3]
        if self.num_layers > 2:
            ve_layers[-3] = ve_tok[2]

        smear_gate = smear_lambda * mx.sigmoid(linear_no_bias(x[1:, :12], self.smear_gate_weight))
        x = mx.concatenate([x[:1], x[1:] + smear_gate * x[:-1]], axis=0)
        x = x[None, :, :]
        x = x0 = rms_norm(x)

        attn_gates: list[Optional[mx.array]] = [None] * self.num_layers
        for layer_idx in self.attn_layer_indices:
            attn_gates[layer_idx] = self.attn_gate_bank[self.layer_to_attn_idx[layer_idx]]

        ve_gates: list[Optional[mx.array]] = [None] * self.num_layers
        idx_map = [1, 2, self.num_layers - 3, self.num_layers - 2, self.num_layers - 1]
        for gate_idx, layer_idx in enumerate(idx_map):
            if 0 <= layer_idx < self.num_layers:
                ve_gates[layer_idx] = self.ve_gate_bank[gate_idx]

        mask_cache: dict[tuple[Optional[int], bool], mx.array] = {}

        for i in range(self.num_layers):
            yarn = self.yarn_paired_head if i in self.paired_head_layers else self.yarn

            if i in skip_out and skip_connections:
                skip_gate_out = mx.sigmoid(skip_lambda) * 2.0 * mx.sigmoid(
                    linear_no_bias(x0[:, :, :12], self.skip_gate_weight)
                )
                x = x + skip_gate_out * skip_connections.pop()

            if i == 0:
                x = (resid_lambdas[0] + self.x0_lambdas[0]) * x + bigram_lambdas[0] * x0_bigram
            else:
                x = resid_lambdas[i] * x + self.x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram

            if i in self.layer_to_attn_idx:
                qkvo_w = self.attn_bank[self.layer_to_attn_idx[i]]
                x = x + self._attention(
                    rms_norm(x),
                    seq_ids,
                    qkvo_w=qkvo_w,
                    sa_lambdas=sa_lambdas[i],
                    ve=ve_layers[i],
                    key_offset=key_offset[i],
                    attn_gate_w=attn_gates[i],
                    ve_gate_w=ve_gates[i],
                    bm_size=bm_sizes[i],
                    yarn=yarn,
                    mask_cache=mask_cache,
                )

            if i in self.layer_to_mlp_idx:
                mlp_idx = self.layer_to_mlp_idx[i]
                c_fc = self.mlp_bank[mlp_idx, 0]
                c_proj = self.mlp_bank[mlp_idx, 1]
                x = x + self._mlp(rms_norm(x), c_fc, c_proj)

            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - backout_lambda * x_backout
        x = rms_norm(x)
        logits = x @ self.lm_head_weight
        return logits[0]


def compute_loss(logits: mx.array, targets: mx.array) -> mx.array:
    logits = softcap_logits(logits)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


class TrainingManager:
    def __init__(self, model: GPTMLX, args: argparse.Namespace):
        self.model = model
        self.args = args
        self.block_size = 128

        self.stages = [
            TrainingStage(
                duration=1 / 3,
                train_max_seq_len=896,
                batch_size_tokens=8 * 2048 * 8,
                window_sizes=(1, 3),
                lr_mul=1.0,
            ),
            TrainingStage(
                duration=1 / 3,
                train_max_seq_len=2048,
                batch_size_tokens=16 * 2048 * 8,
                window_sizes=(3, 7),
                lr_mul=1.52,
            ),
            TrainingStage(
                duration=1 / 3,
                train_max_seq_len=2048,
                batch_size_tokens=24 * 2048 * 8,
                window_sizes=(5, 11),
                lr_mul=1.73,
            ),
            TrainingStage(
                train_max_seq_len=2048,
                batch_size_tokens=24 * 2048 * 8,
                window_sizes=(6, 13),
                lr_mul=1.0,
            ),
        ]

        self.schedule = TrainingSchedule(
            self.stages,
            scheduled_iterations=args.num_scheduled_iterations,
            extension_iterations=args.num_extension_iterations,
            cooldown_frac=args.cooldown_frac,
            split_embed_stage=2,
            ws_post_yarn_ext=args.ws_post_yarn_ext,
        )

        self.optimizer = AdamMLX(
            model,
            adam_lr=args.adam_lr,
            adam_eps=args.adam_eps,
            adam_weight_decay=args.adam_weight_decay,
        )

        self.split_step = self.schedule.split_step
        self.ws_short, self.ws_long = self.stages[0].window_sizes
        self.batch_tokens = 0
        self.train_max_seq_len = self.stages[0].train_max_seq_len

    def advance_schedule(self, step: int) -> ForwardScheduleConfig:
        stage, _ = self.schedule.lookup(step)

        old_ws_long = self.ws_long
        self.ws_short, self.ws_long = stage.window_sizes
        self.train_max_seq_len = min(stage.train_max_seq_len, self.args.max_seq_len_cap)

        if self.ws_long != old_ws_long:
            self.model.yarn.apply(old_ws_long * self.block_size, self.ws_long * self.block_size)
            self.model.yarn_paired_head.apply(old_ws_long * self.block_size, self.ws_long * self.block_size)

        raw_batch_tokens = int(round(stage.batch_size_tokens * self.args.batch_token_scale))
        if self.args.batch_tokens_override > 0:
            raw_batch_tokens = self.args.batch_tokens_override
        raw_batch_tokens = max(raw_batch_tokens, self.args.min_batch_tokens)

        gs = max(1, self.args.grad_accum_steps)
        self.batch_tokens = max(gs, (raw_batch_tokens // gs) * gs)

        return ForwardScheduleConfig(
            ws_short=self.ws_short * self.block_size,
            ws_long=self.ws_long * self.block_size,
            train_max_seq_len=self.train_max_seq_len,
        )

    def step_optimizers(self, step: int, grads: dict) -> None:
        step_lr_mul = self.schedule.get_lr_mul(step)
        self.optimizer.step(
            self.model,
            grads,
            step_lr_mul=step_lr_mul,
        )

        if (not self.optimizer.split_embed) and (not self.args.disable_split_embed) and step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()
            self.model.split_embed_from_lm()


@dataclass
class EvalStats:
    val_loss: float
    elapsed_sec: float


def evaluate(
    model: GPTMLX,
    val_loader: ContiguousDataLoader,
    cfg: ForwardScheduleConfig,
    val_steps: int,
    val_batch_tokens: int,
) -> EvalStats:
    if val_steps <= 0:
        return EvalStats(val_loss=float("nan"), elapsed_sec=0.0)

    start = time.time()
    losses = []
    for _ in range(val_steps):
        x, y, seq_ids, bigram = val_loader.next_batch(val_batch_tokens, -1)
        logits = model.forward(x, bigram, seq_ids, cfg)
        loss = compute_loss(logits, y)
        mx.eval(loss)
        losses.append(float(loss))

    return EvalStats(val_loss=float(np.mean(losses)), elapsed_sec=time.time() - start)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLX train_gpt-like trainer")

    p.add_argument("--train-files", default="data/fineweb10B/fineweb_train_*.bin")
    p.add_argument("--val-files", default="data/fineweb10B/fineweb_val_*.bin")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--synthetic-train-tokens", type=int, default=4_000_000)
    p.add_argument("--synthetic-val-tokens", type=int, default=400_000)

    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--bigram-vocab-size", type=int, default=50304 * 5)

    p.add_argument("--num-layers", type=int, default=11)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--model-dim", type=int, default=768)
    p.add_argument("--max-seq-len-cap", type=int, default=2048)

    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--batch-token-scale", type=float, default=1 / 64)
    p.add_argument("--batch-tokens-override", type=int, default=0)
    p.add_argument("--min-batch-tokens", type=int, default=1024)

    p.add_argument("--num-scheduled-iterations", type=int, default=1490)
    p.add_argument("--num-extension-iterations", type=int, default=40)
    p.add_argument("--cooldown-frac", type=float, default=0.55)
    p.add_argument("--ws-post-yarn-ext", type=int, default=20)

    p.add_argument("--val-loss-every", type=int, default=25)
    p.add_argument("--val-tokens", type=int, default=1_048_576)
    p.add_argument("--val-batch-tokens", type=int, default=8192)
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--adam-lr", type=float, default=0.008)
    p.add_argument("--adam-eps", type=float, default=1e-10)
    p.add_argument("--adam-weight-decay", type=float, default=0.005)

    p.add_argument("--disable-split-embed", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")

    return p


def maybe_build_synthetic_dataset(args: argparse.Namespace) -> None:
    if not args.allow_synthetic:
        return

    train_matches = sorted(glob.glob(args.train_files))
    val_matches = sorted(glob.glob(args.val_files))
    if train_matches and val_matches:
        return

    synth_root = Path("/tmp/train_gpt_mlx_synth")
    synth_root.mkdir(parents=True, exist_ok=True)

    def write_shard(path: Path, toks: np.ndarray) -> None:
        header = np.zeros(HEADER_INTS, dtype=np.int32)
        header[0] = MAGIC_NUMBER
        header[1] = VERSION
        header[2] = toks.size
        with path.open("wb") as f:
            f.write(header.tobytes())
            f.write(toks.astype(np.uint16).tobytes())

    rng = np.random.default_rng(args.seed)

    def gen_tokens(n: int) -> np.ndarray:
        toks = rng.integers(0, min(args.vocab_size, 2**16 - 1), size=n, dtype=np.int32)
        step = max(32, n // 2000)
        toks[::step] = min(BOS_ID, args.vocab_size - 1)
        return toks

    train_path = synth_root / "fineweb_train_000001.bin"
    val_path = synth_root / "fineweb_val_000000.bin"
    write_shard(train_path, gen_tokens(args.synthetic_train_tokens))
    write_shard(val_path, gen_tokens(args.synthetic_val_tokens))

    args.train_files = str(synth_root / "fineweb_train_*.bin")
    args.val_files = str(synth_root / "fineweb_val_*.bin")


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)

    maybe_build_synthetic_dataset(args)

    model = GPTMLX(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        model_dim=args.model_dim,
        max_seq_len=args.max_seq_len_cap,
        bigram_vocab_size=args.bigram_vocab_size,
    )

    train_loader = BosAlignedDataLoader(args.train_files, bigram_vocab_size=args.bigram_vocab_size)
    val_loader = ContiguousDataLoader(args.val_files, bigram_vocab_size=args.bigram_vocab_size)

    manager = TrainingManager(model, args)

    def loss_fn(
        m: GPTMLX,
        input_seq: mx.array,
        target_seq: mx.array,
        bigram_seq: mx.array,
        seq_ids: np.ndarray,
        cfg: ForwardScheduleConfig,
    ) -> mx.array:
        logits = m.forward(input_seq, bigram_seq, seq_ids, cfg)
        return compute_loss(logits, target_seq)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(
        f"device={mx.default_device()} "
        f"layers={args.num_layers} heads={args.num_heads} head_dim={args.head_dim} model_dim={args.model_dim}"
    )
    print(
        f"train_files='{args.train_files}' val_files='{args.val_files}' "
        f"steps={args.steps} grad_accum={args.grad_accum_steps} batch_token_scale={args.batch_token_scale}"
    )

    def format_hms(seconds: float) -> str:
        seconds_i = max(0, int(round(seconds)))
        h = seconds_i // 3600
        m = (seconds_i % 3600) // 60
        s = seconds_i % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    global_start = time.time()
    window_start = global_start
    window_step = 0

    for step in range(args.steps):
        cfg = manager.advance_schedule(step)
        micro_tokens = manager.batch_tokens // max(1, args.grad_accum_steps)
        grad_scale = 1.0 / max(1, args.grad_accum_steps)

        accum_grads = None
        loss_sum = 0.0
        for _ in range(max(1, args.grad_accum_steps)):
            x, y, seq_ids, bigram = train_loader.next_batch(micro_tokens, cfg.train_max_seq_len)
            loss, grads = loss_and_grad(model, x, y, bigram, seq_ids, cfg)
            mx.eval(loss)
            loss_sum += float(loss)

            if accum_grads is None:
                accum_grads = tree_map(lambda g: g * grad_scale, grads)
            else:
                accum_grads = tree_map(lambda a, g: a + g * grad_scale, accum_grads, grads)

        assert accum_grads is not None
        if args.grad_clip > 0:
            accum_grads, grad_norm = optim.clip_grad_norm(accum_grads, args.grad_clip)
        else:
            grad_norm = mx.array(0.0, dtype=mx.float32)

        manager.step_optimizers(step, accum_grads)
        mx.eval(model.parameters(), manager.optimizer.state_for_eval(), grad_norm)

        step1 = step + 1
        if step1 == 1 or step1 % max(1, args.log_every) == 0:
            now = time.time()
            step_delta = max(1, step1 - window_step)
            elapsed = max(1e-6, now - window_start)
            tokens_done = step_delta * manager.batch_tokens
            tok_per_s = tokens_done / elapsed
            avg_step_sec_window = elapsed / step_delta
            elapsed_total = max(1e-6, now - global_start)
            avg_step_sec_total = elapsed_total / step1
            remaining_steps = max(0, args.steps - step1)
            eta_sec = remaining_steps * avg_step_sec_total
            print(
                f"step={step1:5d}/{args.steps} "
                f"loss={(loss_sum / max(1, args.grad_accum_steps)):.4f} "
                f"lr_mul={manager.schedule.get_lr_mul(step):.4f} "
                f"grad_norm={float(grad_norm):.4f} "
                f"tokens/step={manager.batch_tokens} "
                f"tok/s={tok_per_s:,.0f} "
                f"avg_step_s={avg_step_sec_window:.2f} "
                f"eta={format_hms(eta_sec)}"
            )
            window_start = now
            window_step = step1

        if step1 % args.val_loss_every == 0 or step1 == args.steps:
            val_steps = max(1, args.val_tokens // max(1, args.val_batch_tokens))
            stats = evaluate(
                model,
                val_loader,
                cfg,
                val_steps=val_steps,
                val_batch_tokens=args.val_batch_tokens,
            )
            print(
                f"step={step1:5d}/{args.steps} val_loss={stats.val_loss:.4f} "
                f"val_steps={val_steps} val_time={stats.elapsed_sec:.2f}s"
            )

    total = time.time() - global_start
    print(f"done steps={args.steps} wall_time_sec={total:.1f}")


if __name__ == "__main__":
    main()
