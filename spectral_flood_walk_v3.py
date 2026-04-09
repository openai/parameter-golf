#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_flood_walk_v0 import maybe_reset_cuda_peak_memory, maybe_sync_cuda
from spectral_flood_walk_v2a import batch_from_starts, build_lm_starts


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def maybe_autocast(device: torch.device) -> Any:
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def maybe_cache_tokens_on_device(tokens: torch.Tensor, *, device: torch.device, enabled: bool) -> torch.Tensor:
    if enabled and device.type == "cuda" and tokens.device != device:
        return tokens.to(device=device, dtype=torch.long)
    return tokens


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class V3Config:
    enwik8_path: str
    output_json: str | None = None
    device: str = "auto"
    seed: int = 1337

    vocab_size: int = 256
    seq_len: int = 128
    stride: int = 64
    batch_size: int = 4
    train_steps: int = 16
    eval_batches: int = 8
    report_every: int = 4
    cache_dataset_on_device: bool = True

    recurrent_dim: int = 64
    recurrent_heads: int = 4
    num_distinct_blocks: int = 1
    view_count: int = 2
    view_combination: str = "average"
    cross_token_mode: str = "floor"
    block_has_residual: bool = True
    block_nonlinearity: str = "gelu"
    recurrence_step_size: float = 1.0
    state_decay: float = 1.0
    contraction_target: float = 0.995
    train_recurrence_steps: int = 16
    eval_recurrence_steps: int = 32
    tbptt_chunk: int = 16
    norm_interval_k: int = 16
    train_floor_interval: int = 8
    floor_min_interval: int = 4
    floor_max_interval: int = 16
    floor_threshold: float = 0.05
    kernel_feature_map: str = "elu_plus_1"
    accumulator_decay: float = 0.999
    state_core: str = "scalar_decay"
    hippo_delta_scale: float = 0.0
    hippo_rank: int = 1
    quantization: str = "ternary"
    jacobian_lambda: float = 1e-2
    stochastic_round_p: float = 0.0

    base_lr: float = 2e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    warmup_steps: int = 4
    min_lr_scale: float = 0.10


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def activation_fn(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "swish":
        return F.silu(x)
    raise ValueError(f"unsupported activation: {name}")


def quantization_levels(name: str) -> int:
    levels = {
        "ternary": 3,
        "int4": 15,
        "int6": 63,
        "fp16": 0,
    }.get(name)
    if levels is None:
        raise ValueError(f"unsupported quantization: {name}")
    return levels


def stochastic_round_tensor(
    tensor: torch.Tensor,
    *,
    quantization: str,
    probability: float,
) -> torch.Tensor:
    levels = quantization_levels(quantization)
    if probability <= 0.0 or levels <= 1:
        return tensor
    half_span = (levels - 1) / 2.0
    max_abs = tensor.detach().abs().amax()
    if float(max_abs) <= 1e-12:
        return tensor
    step = max_abs / max(half_span, 1.0)
    scaled = (tensor.float() / step).clamp(-half_span, half_span)
    lower = scaled.floor()
    upper = scaled.ceil()
    pick_upper = torch.rand_like(scaled).lt(scaled - lower)
    rounded = torch.where(pick_upper, upper, lower)
    quantized = (rounded * step).to(tensor.dtype)
    ste_quantized = tensor + (quantized - tensor).detach()
    if probability >= 1.0:
        return ste_quantized
    mask = torch.rand_like(scaled).lt(probability)
    return torch.where(mask, ste_quantized, tensor)


def estimate_spectral_norm(weight: torch.Tensor, *, num_power_iters: int = 4) -> torch.Tensor:
    matrix = weight.float().reshape(weight.shape[0], -1)
    v = torch.ones((matrix.shape[1],), device=matrix.device, dtype=matrix.dtype)
    v = F.normalize(v, dim=0)
    sigma = torch.tensor(0.0, device=matrix.device, dtype=matrix.dtype)
    for _ in range(max(num_power_iters, 1)):
        u = F.normalize(matrix @ v, dim=0)
        v = F.normalize(matrix.transpose(0, 1) @ u, dim=0)
        sigma = torch.dot(u, matrix @ v)
    return sigma.abs().clamp_min(1e-6)


def clamp_spectral_gain(weight: torch.Tensor, target_gain: float) -> torch.Tensor:
    if target_gain <= 0.0:
        raise ValueError("target_gain must be positive")
    sigma = estimate_spectral_norm(weight).detach()
    gain = torch.tensor(target_gain, device=weight.device, dtype=weight.dtype)
    scale = torch.clamp(gain / sigma.to(dtype=weight.dtype), max=1.0)
    return weight * scale


def build_hippo_legs_matrix(dim: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")
    index = torch.arange(dim, device=device, dtype=torch.float32)
    scales = torch.sqrt(2.0 * index + 1.0)
    lower = torch.tril(torch.outer(scales, scales), diagonal=-1)
    diagonal = torch.diag(index + 1.0)
    matrix = -(lower + diagonal)
    return matrix.to(dtype=dtype)


class DeepFloorRecurrentBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        step_size: float,
        state_decay: float,
        contraction_target: float,
        has_residual: bool,
        nonlinearity: str,
        quantization: str,
        stochastic_round_p: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.step_size = step_size
        self.state_decay = state_decay
        self.contraction_target = contraction_target
        self.has_residual = has_residual
        self.nonlinearity = nonlinearity
        self.quantization = quantization
        self.stochastic_round_p = stochastic_round_p

    def prepare_runtime(self) -> dict[str, torch.Tensor]:
        per_weight_gain = self.contraction_target ** 0.25
        return {
            "q_weight": clamp_spectral_gain(self.q_proj.weight, per_weight_gain),
            "k_weight": clamp_spectral_gain(self.k_proj.weight, per_weight_gain),
            "v_weight": clamp_spectral_gain(self.v_proj.weight, per_weight_gain),
            "o_weight": clamp_spectral_gain(self.o_proj.weight, per_weight_gain),
        }

    def forward(
        self,
        state: torch.Tensor,
        *,
        runtime: dict[str, torch.Tensor] | None = None,
        apply_quant_noise: bool = True,
    ) -> torch.Tensor:
        normed = self.norm(state)
        runtime = runtime or self.prepare_runtime()
        q = torch.sigmoid(F.linear(normed, runtime["q_weight"]))
        k = activation_fn(self.nonlinearity, F.linear(normed, runtime["k_weight"]))
        v = F.linear(normed, runtime["v_weight"])
        update = F.linear(q * (k * v), runtime["o_weight"])
        if apply_quant_noise:
            update = stochastic_round_tensor(
                update,
                quantization=self.quantization,
                probability=self.stochastic_round_p,
            )
        if self.has_residual:
            return self.state_decay * state + self.step_size * update
        return self.step_size * update

    def jacobian_proxy_penalty(
        self,
        state: torch.Tensor,
        *,
        runtime: dict[str, torch.Tensor] | None = None,
        epsilon: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(state)
        noise_norm = torch.linalg.vector_norm(noise.reshape(noise.size(0), -1), dim=-1, keepdim=True).clamp_min(1e-6)
        noise = noise / noise_norm.view(noise.size(0), *([1] * (noise.dim() - 1)))
        noise = noise * epsilon
        runtime = runtime or self.prepare_runtime()
        base = self.forward(state, runtime=runtime, apply_quant_noise=False)
        perturbed = self.forward(state + noise, runtime=runtime, apply_quant_noise=False)
        delta = perturbed - base
        delta_norm = torch.linalg.vector_norm(delta.reshape(delta.size(0), -1), dim=-1)
        perturb_norm = torch.linalg.vector_norm(noise.reshape(noise.size(0), -1), dim=-1).clamp_min(1e-6)
        gain = delta_norm / perturb_norm
        mean_gain = gain.mean()
        penalty = F.relu(mean_gain - self.contraction_target).pow(2)
        return penalty, mean_gain


class DeepFloorAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, contraction_target: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("recurrent_dim must be divisible by recurrent_heads")
        self.norm = RMSNorm()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_norm = RMSNorm()
        self.heads = heads
        self.head_dim = dim // heads
        self.contraction_target = contraction_target

    def prepare_runtime(self) -> dict[str, torch.Tensor]:
        projection_gain = math.sqrt(self.contraction_target)
        return {
            "q_weight": clamp_spectral_gain(self.q_proj.weight, projection_gain),
            "k_weight": clamp_spectral_gain(self.k_proj.weight, projection_gain),
            "v_weight": clamp_spectral_gain(self.v_proj.weight, projection_gain),
            "out_weight": clamp_spectral_gain(self.out_proj.weight, projection_gain),
        }

    def forward(self, state: torch.Tensor, *, runtime: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        normed = self.norm(state)
        runtime = runtime or self.prepare_runtime()
        q = F.linear(normed, runtime["q_weight"])
        k = F.linear(normed, runtime["k_weight"])
        v = F.linear(normed, runtime["v_weight"])
        batch, seq_len, dim = q.shape
        q = q.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        mixed = mixed.transpose(1, 2).reshape(batch, seq_len, dim)
        mixed = F.linear(mixed, runtime["out_weight"])
        return self.out_norm(state + mixed)


class DeepFloorFusedMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        feature_map: str,
        decay: float,
        *,
        state_core: str,
        hippo_delta_scale: float,
        hippo_rank: int,
        contraction_target: float,
        quantization: str,
        stochastic_round_p: float,
    ) -> None:
        super().__init__()
        if hippo_rank <= 0:
            raise ValueError("hippo_rank must be positive")
        self.norm = RMSNorm()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.selection_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.feature_map = feature_map
        self.decay = decay
        self.state_core = state_core
        self.hippo_delta_scale = hippo_delta_scale
        self.hippo_rank = hippo_rank
        self.contraction_target = contraction_target
        self.quantization = quantization
        self.stochastic_round_p = stochastic_round_p
        self.register_buffer("hippo_matrix", build_hippo_legs_matrix(dim), persistent=False)
        if state_core == "hippo_plus_lowrank":
            self.hippo_left = nn.Parameter(torch.zeros((dim, hippo_rank), dtype=torch.float32))
            self.hippo_right = nn.Parameter(torch.zeros((dim, hippo_rank), dtype=torch.float32))
            nn.init.normal_(self.hippo_left, mean=0.0, std=0.02)
            nn.init.normal_(self.hippo_right, mean=0.0, std=0.02)
        else:
            self.register_parameter("hippo_left", None)
            self.register_parameter("hippo_right", None)

    def _transition_base(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dim = self.q_proj.weight.shape[0]
        if self.state_core == "scalar_decay":
            return torch.eye(dim, device=device, dtype=dtype)
        hippo = self.hippo_matrix.to(device=device, dtype=dtype)
        if self.state_core == "hippo":
            return hippo
        if self.state_core == "hippo_plus_lowrank":
            if self.hippo_left is None or self.hippo_right is None:
                raise RuntimeError("hippo_plus_lowrank requires low-rank parameters")
            delta = torch.matmul(self.hippo_left.to(dtype=dtype), self.hippo_right.to(dtype=dtype).transpose(0, 1))
            return hippo + (self.hippo_delta_scale * delta)
        raise ValueError(f"unsupported state_core: {self.state_core}")

    def prepare_runtime(self) -> dict[str, torch.Tensor]:
        projection_gain = math.sqrt(self.contraction_target)
        transition_base = self._transition_base(device=self.q_proj.weight.device, dtype=self.q_proj.weight.dtype)
        return {
            "q_weight": clamp_spectral_gain(self.q_proj.weight, projection_gain),
            "k_weight": clamp_spectral_gain(self.k_proj.weight, projection_gain),
            "v_weight": clamp_spectral_gain(self.v_proj.weight, projection_gain),
            "select_weight": clamp_spectral_gain(self.selection_proj.weight, 1.0),
            "out_weight": clamp_spectral_gain(self.out_proj.weight, projection_gain),
            "transition_weight": clamp_spectral_gain(transition_base, self.decay),
        }

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu_plus_1":
            return F.elu(x) + 1.0
        if self.feature_map == "identity":
            return x
        raise ValueError(f"unsupported kernel feature map: {self.feature_map}")

    def forward(
        self,
        state: torch.Tensor,
        accumulator: torch.Tensor | None,
        *,
        runtime: dict[str, torch.Tensor] | None = None,
        apply_quant_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(state)
        runtime = runtime or self.prepare_runtime()
        q = self._phi(F.linear(normed, runtime["q_weight"]))
        k = self._phi(F.linear(normed, runtime["k_weight"]))
        v = F.linear(normed, runtime["v_weight"])
        selection = torch.sigmoid(F.linear(normed, runtime["select_weight"]))
        token_contributions = (k * selection).unsqueeze(-1) * (v * selection).unsqueeze(-2)
        if apply_quant_noise:
            token_contributions = stochastic_round_tensor(
                token_contributions,
                quantization=self.quantization,
                probability=self.stochastic_round_p,
            )
        step_accumulator = token_contributions.cumsum(dim=1)
        if accumulator is None:
            accumulator = step_accumulator
        else:
            accumulator = torch.einsum("ij,btjk->btik", runtime["transition_weight"], accumulator) + step_accumulator
        mixed = torch.einsum("bti,btij->btj", q, accumulator)
        mixed = F.linear(mixed, runtime["out_weight"])
        return state + mixed, accumulator


class DeepFloorModel(nn.Module):
    def __init__(self, cfg: V3Config) -> None:
        super().__init__()
        if cfg.view_count <= 0:
            raise ValueError("view_count must be positive")
        if cfg.num_distinct_blocks <= 0:
            raise ValueError("num_distinct_blocks must be positive")
        self.cfg = cfg
        self.view_embeddings = nn.ModuleList(
            nn.Embedding(cfg.vocab_size, cfg.recurrent_dim) for _ in range(cfg.view_count)
        )
        self.blocks = nn.ModuleList(
            DeepFloorRecurrentBlock(
                cfg.recurrent_dim,
                step_size=cfg.recurrence_step_size,
                state_decay=cfg.state_decay,
                contraction_target=cfg.contraction_target,
                has_residual=cfg.block_has_residual,
                nonlinearity=cfg.block_nonlinearity,
                quantization=cfg.quantization,
                stochastic_round_p=cfg.stochastic_round_p,
            )
            for _ in range(cfg.num_distinct_blocks)
        )
        self.floor_attention = DeepFloorAttentionBlock(
            cfg.recurrent_dim,
            cfg.recurrent_heads,
            contraction_target=cfg.contraction_target,
        )
        self.fused_mixer = DeepFloorFusedMixer(
            cfg.recurrent_dim,
            cfg.kernel_feature_map,
            cfg.accumulator_decay,
            state_core=cfg.state_core,
            hippo_delta_scale=cfg.hippo_delta_scale,
            hippo_rank=cfg.hippo_rank,
            contraction_target=cfg.contraction_target,
            quantization=cfg.quantization,
            stochastic_round_p=cfg.stochastic_round_p,
        )
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(cfg.recurrent_dim, cfg.vocab_size, bias=False)
        if cfg.view_combination == "weighted":
            self.view_weights = nn.Parameter(torch.zeros((cfg.view_count,), dtype=torch.float32))
        else:
            self.register_parameter("view_weights", None)
        if cfg.view_combination == "project":
            self.view_project = nn.Linear(cfg.view_count * cfg.recurrent_dim, cfg.vocab_size, bias=False)
        else:
            self.register_module("view_project", None)

    def _normalize_recurrent_state(self, state: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(state, (state.size(-1),), eps=1e-6)

    def _normalize_accumulator(self, accumulator: torch.Tensor) -> torch.Tensor:
        rms = accumulator.float().pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        return (accumulator.float() / rms).to(dtype=accumulator.dtype)

    def _should_fire_floor(
        self,
        *,
        step_idx: int,
        last_floor_step: int,
        state: torch.Tensor,
        previous_state: torch.Tensor,
        adaptive: bool,
    ) -> bool:
        interval = step_idx + 1 - last_floor_step
        if interval < self.cfg.floor_min_interval:
            return False
        if interval >= self.cfg.floor_max_interval:
            return True
        if not adaptive:
            return (step_idx + 1) % self.cfg.train_floor_interval == 0
        delta = torch.linalg.vector_norm((state - previous_state).reshape(-1, state.size(-1)), dim=-1).mean()
        baseline = torch.linalg.vector_norm(previous_state.reshape(-1, previous_state.size(-1)), dim=-1).mean()
        relative_delta = float((delta / baseline.clamp_min(1e-6)).detach().cpu())
        return relative_delta >= self.cfg.floor_threshold

    def _combine_views(self, view_states: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(view_states, dim=0)
        if self.view_weights is None:
            return stacked.mean(dim=0)
        weights = F.softmax(self.view_weights, dim=0).view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def _compute_logits(self, view_states: list[torch.Tensor]) -> torch.Tensor:
        if self.view_project is not None:
            normalized = [self.final_norm(state) for state in view_states]
            return self.view_project(torch.cat(normalized, dim=-1))
        combined = self._combine_views(view_states)
        return self.lm_head(self.final_norm(combined))

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        adaptive_floor: bool,
        recurrence_steps: int,
        tbptt_chunk: int | None = None,
        return_metadata: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        view_states: list[torch.Tensor] = []
        metadata: dict[str, Any] = {
            "floor_steps": [],
            "floor_counts": [],
            "state_norm_counts": [],
            "accumulator_norm_counts": [],
            "tbptt_detach_counts": [],
        }
        tbptt_chunk = self.cfg.tbptt_chunk if tbptt_chunk is None else tbptt_chunk
        block_runtimes = [block.prepare_runtime() for block in self.blocks]
        floor_runtime = self.floor_attention.prepare_runtime() if self.cfg.cross_token_mode == "floor" else None
        fused_runtime = self.fused_mixer.prepare_runtime() if self.cfg.cross_token_mode == "fused" else None
        for view_idx, embedding in enumerate(self.view_embeddings):
            state = embedding(inputs)
            previous_recurrence_state = state
            accumulator: torch.Tensor | None = None
            floor_steps: list[int] = []
            last_floor_step = 0
            state_norm_count = 0
            accumulator_norm_count = 0
            tbptt_detach_count = 0
            for step_idx in range(recurrence_steps):
                block = self.blocks[step_idx % len(self.blocks)]
                block_runtime = block_runtimes[step_idx % len(self.blocks)]
                state = block(state, runtime=block_runtime)
                recurrent_state = state
                if self.cfg.cross_token_mode == "floor":
                    if self._should_fire_floor(
                        step_idx=step_idx,
                        last_floor_step=last_floor_step,
                        state=recurrent_state,
                        previous_state=previous_recurrence_state,
                        adaptive=adaptive_floor,
                    ):
                        state = self.floor_attention(state, runtime=floor_runtime)
                        last_floor_step = step_idx + 1
                        floor_steps.append(last_floor_step)
                elif self.cfg.cross_token_mode == "fused":
                    state, accumulator = self.fused_mixer(state, accumulator, runtime=fused_runtime)
                else:
                    raise ValueError(f"unsupported cross_token_mode: {self.cfg.cross_token_mode}")
                if self.cfg.norm_interval_k > 0 and ((step_idx + 1) % self.cfg.norm_interval_k == 0):
                    state = self._normalize_recurrent_state(state)
                    state_norm_count += 1
                    if accumulator is not None:
                        accumulator = self._normalize_accumulator(accumulator)
                        accumulator_norm_count += 1
                previous_recurrence_state = recurrent_state
                if self.training and tbptt_chunk > 0 and ((step_idx + 1) % tbptt_chunk == 0) and (step_idx + 1) < recurrence_steps:
                    state = state.detach()
                    previous_recurrence_state = previous_recurrence_state.detach()
                    if accumulator is not None:
                        accumulator = accumulator.detach()
                    tbptt_detach_count += 1
            view_states.append(state)
            metadata["floor_steps"].append(floor_steps)
            metadata["floor_counts"].append(len(floor_steps))
            metadata["state_norm_counts"].append(state_norm_count)
            metadata["accumulator_norm_counts"].append(accumulator_norm_count)
            metadata["tbptt_detach_counts"].append(tbptt_detach_count)
        logits = self._compute_logits(view_states)
        if return_metadata:
            return logits, metadata
        return logits

    def jacobian_proxy_loss(self, inputs: torch.Tensor, *, recurrence_steps: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if recurrence_steps <= 0:
            zero = torch.zeros((), device=inputs.device)
            return zero, zero, 0
        probe_step = int(torch.randint(0, recurrence_steps, (1,), device=inputs.device).item())
        block_runtimes = [block.prepare_runtime() for block in self.blocks]
        floor_runtime = self.floor_attention.prepare_runtime() if self.cfg.cross_token_mode == "floor" else None
        fused_runtime = self.fused_mixer.prepare_runtime() if self.cfg.cross_token_mode == "fused" else None
        state = self.view_embeddings[0](inputs).detach()
        previous_recurrence_state = state
        accumulator: torch.Tensor | None = None
        last_floor_step = 0
        with torch.no_grad():
            for step_idx in range(probe_step):
                block = self.blocks[step_idx % len(self.blocks)]
                block_runtime = block_runtimes[step_idx % len(self.blocks)]
                state = block(state, runtime=block_runtime, apply_quant_noise=False)
                recurrent_state = state
                if self.cfg.cross_token_mode == "floor":
                    if self._should_fire_floor(
                        step_idx=step_idx,
                        last_floor_step=last_floor_step,
                        state=recurrent_state,
                        previous_state=previous_recurrence_state,
                        adaptive=False,
                    ):
                        state = self.floor_attention(state, runtime=floor_runtime)
                        last_floor_step = step_idx + 1
                elif self.cfg.cross_token_mode == "fused":
                    state, accumulator = self.fused_mixer(
                        state,
                        accumulator,
                        runtime=fused_runtime,
                        apply_quant_noise=False,
                    )
                previous_recurrence_state = recurrent_state
                if self.cfg.norm_interval_k > 0 and ((step_idx + 1) % self.cfg.norm_interval_k == 0):
                    state = self._normalize_recurrent_state(state)
                    if accumulator is not None:
                        accumulator = self._normalize_accumulator(accumulator)
        block_idx = probe_step % len(self.blocks)
        penalty, gain = self.blocks[block_idx].jacobian_proxy_penalty(
            state,
            runtime=block_runtimes[block_idx],
        )
        return penalty, gain, probe_step

    def _active_modules(self) -> list[nn.Module]:
        active: list[nn.Module] = list(self.view_embeddings) + list(self.blocks)
        active.append(self.final_norm)
        if self.cfg.cross_token_mode == "floor":
            active.append(self.floor_attention)
        elif self.cfg.cross_token_mode == "fused":
            active.append(self.fused_mixer)
        if self.cfg.view_combination == "project" and self.view_project is not None:
            active.append(self.view_project)
        else:
            active.append(self.lm_head)
        if self.view_weights is not None:
            active.append(self.view_weights)  # type: ignore[arg-type]
        return active

    def estimate_artifact_bytes(self) -> int:
        bits_per_value = {
            "ternary": 2,
            "int4": 4,
            "int6": 6,
            "fp16": 16,
        }.get(self.cfg.quantization)
        if bits_per_value is None:
            raise ValueError(f"unsupported quantization: {self.cfg.quantization}")
        seen: set[int] = set()
        params = 0
        for module in self._active_modules():
            if isinstance(module, nn.Parameter):
                if id(module) not in seen:
                    seen.add(id(module))
                    params += module.numel()
            else:
                for p in module.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        params += p.numel()
        return math.ceil(params * bits_per_value / 8)


def prepare_byte_enwik8_splits(
    enwik8_path: Path,
    *,
    device: torch.device,
    cache_on_device: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw = np.frombuffer(enwik8_path.read_bytes(), dtype=np.uint8)
    if raw.size < 1024:
        raise ValueError("enwik8 fixture is too small")
    train_end = max(int(raw.size * 0.95), 1)
    val_end = max(train_end + int(raw.size * 0.025), train_end + 1)
    val_end = min(val_end, raw.size - 1)
    train_tokens = torch.tensor(raw[:train_end].astype(np.int64), dtype=torch.long)
    val_tokens = torch.tensor(raw[train_end:val_end].astype(np.int64), dtype=torch.long)
    test_tokens = torch.tensor(raw[val_end:].astype(np.int64), dtype=torch.long)
    train_tokens = maybe_cache_tokens_on_device(train_tokens, device=device, enabled=cache_on_device)
    val_tokens = maybe_cache_tokens_on_device(val_tokens, device=device, enabled=cache_on_device)
    test_tokens = maybe_cache_tokens_on_device(test_tokens, device=device, enabled=cache_on_device)
    return train_tokens, val_tokens, test_tokens, {
        "raw_bytes": int(raw.size),
        "tokenization_mode": "bytes",
        "tokenizer_name": "bytes",
        "tokenizer_vocab_size": 256,
        "residency": "cuda" if cache_on_device and device.type == "cuda" else "cpu",
    }


def choose_eval_starts(starts: list[int], *, batch_size: int, eval_batches: int, seed: int) -> list[int]:
    if not starts:
        return []
    rng = random.Random(seed)
    needed = min(len(starts), batch_size * eval_batches)
    if needed >= len(starts):
        return starts[:needed]
    return rng.sample(starts, needed)


def evaluate_model(
    model: DeepFloorModel,
    tokens: torch.Tensor,
    *,
    starts: list[int],
    cfg: V3Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_tokens = 0
    total_loss = 0.0
    with torch.no_grad():
        with maybe_autocast(device):
            for offset in range(0, len(starts), cfg.batch_size):
                batch_starts = starts[offset : offset + cfg.batch_size]
                inputs, targets = batch_from_starts(tokens, batch_starts, cfg.seq_len, device)
                logits, _ = model(
                    inputs,
                    adaptive_floor=True,
                    recurrence_steps=cfg.eval_recurrence_steps,
                    tbptt_chunk=0,
                    return_metadata=True,
                )
                total_loss += float(
                    F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), reduction="sum").detach().cpu()
                )
                total_tokens += int(targets.numel())
    mean_loss = float(total_loss / max(total_tokens, 1))
    return {
        "loss": mean_loss,
        "bpb": mean_loss / math.log(2.0),
        "tokens": float(total_tokens),
    }


def train_and_evaluate(cfg: V3Config) -> dict[str, Any]:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    enwik8_path = Path(cfg.enwik8_path)
    train_tokens, val_tokens, test_tokens, dataset_meta = prepare_byte_enwik8_splits(
        enwik8_path,
        device=device,
        cache_on_device=cfg.cache_dataset_on_device,
    )
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    test_starts = build_lm_starts(int(test_tokens.numel()), cfg.seq_len, cfg.stride)
    if not train_starts or not val_starts or not test_starts:
        raise ValueError("dataset split is too small for the requested seq_len/stride")

    model = DeepFloorModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return max(float(step + 1) / max(cfg.warmup_steps, 1), 1e-6)
        progress = (step - cfg.warmup_steps) / max(cfg.train_steps - cfg.warmup_steps, 1)
        return cfg.min_lr_scale + 0.5 * (1.0 - cfg.min_lr_scale) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: list[dict[str, Any]] = []
    maybe_reset_cuda_peak_memory(device)
    train_rng = random.Random(cfg.seed + 17)
    train_start_time = time.perf_counter()

    for step in range(cfg.train_steps):
        batch_starts = train_rng.sample(train_starts, min(cfg.batch_size, len(train_starts)))
        inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device):
            logits, metadata = model(
                inputs,
                adaptive_floor=False,
                recurrence_steps=cfg.train_recurrence_steps,
                tbptt_chunk=cfg.tbptt_chunk,
                return_metadata=True,
            )
            ce_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            if cfg.jacobian_lambda > 0.0:
                jacobian_proxy_loss, jacobian_proxy_gain, jacobian_probe_step = model.jacobian_proxy_loss(
                    inputs,
                    recurrence_steps=cfg.train_recurrence_steps,
                )
            else:
                jacobian_proxy_loss = ce_loss.new_zeros(())
                jacobian_proxy_gain = ce_loss.new_zeros(())
                jacobian_probe_step = 0
            loss = ce_loss + cfg.jacobian_lambda * jacobian_proxy_loss
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm))
        optimizer.step()
        scheduler.step()
        record = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "bpb": float(ce_loss.detach().cpu()) / math.log(2.0),
            "grad_norm": grad_norm,
            "jacobian_proxy_loss": float(jacobian_proxy_loss.detach().cpu()),
            "jacobian_proxy_gain": float(jacobian_proxy_gain.detach().cpu()),
            "jacobian_probe_step": int(jacobian_probe_step),
            "floor_counts": metadata["floor_counts"],
            "state_norm_counts": metadata["state_norm_counts"],
            "accumulator_norm_counts": metadata["accumulator_norm_counts"],
            "tbptt_detach_counts": metadata["tbptt_detach_counts"],
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        if cfg.report_every > 0 and ((step + 1) % cfg.report_every == 0 or step == cfg.train_steps - 1):
            maybe_sync_cuda(device)

    val_eval = evaluate_model(
        model,
        val_tokens,
        starts=choose_eval_starts(val_starts, batch_size=cfg.batch_size, eval_batches=cfg.eval_batches, seed=cfg.seed + 101),
        cfg=cfg,
        device=device,
    )
    test_eval = evaluate_model(
        model,
        test_tokens,
        starts=choose_eval_starts(test_starts, batch_size=cfg.batch_size, eval_batches=cfg.eval_batches, seed=cfg.seed + 202),
        cfg=cfg,
        device=device,
    )
    elapsed = time.perf_counter() - train_start_time
    result = {
        "config": asdict(cfg),
        "dataset": {
            "enwik8_path": str(enwik8_path),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "test_tokens": int(test_tokens.numel()),
            **dataset_meta,
        },
        "artifact": {
            "estimated_bytes": int(model.estimate_artifact_bytes()),
            "estimated_mb": float(model.estimate_artifact_bytes() / (1024.0 * 1024.0)),
            "quantization": cfg.quantization,
        },
        "train": {
            "steps": int(cfg.train_steps),
            "elapsed_seconds": float(elapsed),
            "history": history,
            "final_loss": float(history[-1]["loss"]) if history else float("nan"),
        },
        "val": val_eval,
        "test": test_eval,
    }
    if cfg.output_json is not None:
        path = Path(cfg.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepFloor recurrent multi-view language model")
    parser.add_argument("--enwik8-path", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--device", default=V3Config.device)
    parser.add_argument("--seed", type=int, default=V3Config.seed)
    parser.add_argument("--seq-len", type=int, default=V3Config.seq_len)
    parser.add_argument("--stride", type=int, default=V3Config.stride)
    parser.add_argument("--batch-size", type=int, default=V3Config.batch_size)
    parser.add_argument("--train-steps", type=int, default=V3Config.train_steps)
    parser.add_argument("--eval-batches", type=int, default=V3Config.eval_batches)
    parser.add_argument("--report-every", type=int, default=V3Config.report_every)
    parser.add_argument("--recurrent-dim", type=int, default=V3Config.recurrent_dim)
    parser.add_argument("--recurrent-heads", type=int, default=V3Config.recurrent_heads)
    parser.add_argument("--num-distinct-blocks", type=int, default=V3Config.num_distinct_blocks)
    parser.add_argument("--view-count", type=int, default=V3Config.view_count)
    parser.add_argument("--view-combination", choices=("average", "weighted", "project"), default=V3Config.view_combination)
    parser.add_argument("--cross-token-mode", choices=("floor", "fused"), default=V3Config.cross_token_mode)
    parser.add_argument("--block-has-residual", action=argparse.BooleanOptionalAction, default=V3Config.block_has_residual)
    parser.add_argument("--block-nonlinearity", choices=("relu", "gelu", "swish"), default=V3Config.block_nonlinearity)
    parser.add_argument("--recurrence-step-size", type=float, default=V3Config.recurrence_step_size)
    parser.add_argument("--state-decay", type=float, default=V3Config.state_decay)
    parser.add_argument("--contraction-target", type=float, default=V3Config.contraction_target)
    parser.add_argument("--train-recurrence-steps", type=int, default=V3Config.train_recurrence_steps)
    parser.add_argument("--eval-recurrence-steps", type=int, default=V3Config.eval_recurrence_steps)
    parser.add_argument("--tbptt-chunk", type=int, default=V3Config.tbptt_chunk)
    parser.add_argument("--norm-interval-k", type=int, default=V3Config.norm_interval_k)
    parser.add_argument("--train-floor-interval", type=int, default=V3Config.train_floor_interval)
    parser.add_argument("--floor-min-interval", type=int, default=V3Config.floor_min_interval)
    parser.add_argument("--floor-max-interval", type=int, default=V3Config.floor_max_interval)
    parser.add_argument("--floor-threshold", type=float, default=V3Config.floor_threshold)
    parser.add_argument("--kernel-feature-map", choices=("elu_plus_1", "identity"), default=V3Config.kernel_feature_map)
    parser.add_argument("--accumulator-decay", type=float, default=V3Config.accumulator_decay)
    parser.add_argument("--state-core", choices=("scalar_decay", "hippo", "hippo_plus_lowrank"), default=V3Config.state_core)
    parser.add_argument("--hippo-delta-scale", type=float, default=V3Config.hippo_delta_scale)
    parser.add_argument("--hippo-rank", type=int, default=V3Config.hippo_rank)
    parser.add_argument("--quantization", choices=("ternary", "int4", "int6", "fp16"), default=V3Config.quantization)
    parser.add_argument("--jacobian-lambda", type=float, default=V3Config.jacobian_lambda)
    parser.add_argument("--stochastic-round-p", type=float, default=V3Config.stochastic_round_p)
    parser.add_argument("--base-lr", type=float, default=V3Config.base_lr)
    parser.add_argument("--weight-decay", type=float, default=V3Config.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=V3Config.grad_clip_norm)
    parser.add_argument("--warmup-steps", type=int, default=V3Config.warmup_steps)
    parser.add_argument("--min-lr-scale", type=float, default=V3Config.min_lr_scale)
    parser.add_argument(
        "--cache-dataset-on-device",
        action=argparse.BooleanOptionalAction,
        default=V3Config.cache_dataset_on_device,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> V3Config:
    return V3Config(
        enwik8_path=args.enwik8_path,
        output_json=args.output_json,
        device=args.device,
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        report_every=args.report_every,
        recurrent_dim=args.recurrent_dim,
        recurrent_heads=args.recurrent_heads,
        num_distinct_blocks=args.num_distinct_blocks,
        view_count=args.view_count,
        view_combination=args.view_combination,
        cross_token_mode=args.cross_token_mode,
        block_has_residual=args.block_has_residual,
        block_nonlinearity=args.block_nonlinearity,
        recurrence_step_size=args.recurrence_step_size,
        state_decay=args.state_decay,
        contraction_target=args.contraction_target,
        train_recurrence_steps=args.train_recurrence_steps,
        eval_recurrence_steps=args.eval_recurrence_steps,
        tbptt_chunk=args.tbptt_chunk,
        norm_interval_k=args.norm_interval_k,
        train_floor_interval=args.train_floor_interval,
        floor_min_interval=args.floor_min_interval,
        floor_max_interval=args.floor_max_interval,
        floor_threshold=args.floor_threshold,
        kernel_feature_map=args.kernel_feature_map,
        accumulator_decay=args.accumulator_decay,
        state_core=args.state_core,
        hippo_delta_scale=args.hippo_delta_scale,
        hippo_rank=args.hippo_rank,
        quantization=args.quantization,
        jacobian_lambda=args.jacobian_lambda,
        stochastic_round_p=args.stochastic_round_p,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        warmup_steps=args.warmup_steps,
        min_lr_scale=args.min_lr_scale,
        cache_dataset_on_device=args.cache_dataset_on_device,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = train_and_evaluate(config_from_args(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
