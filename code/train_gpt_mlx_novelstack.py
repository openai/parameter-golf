#!/usr/bin/env python3
from __future__ import annotations

"""
Novel MLX proxy stack for OpenAI Parameter Golf.

This file is intentionally designed as a thin overlay on top of the local
`train_gpt_mlx.py` in the repository root. The goal is to let you keep the
existing data pipeline, tokenizer validation, training loop, logging, local
sliding-window evaluation, and int8+zlib roundtrip path, while swapping in a
finalized local model/optimizer/export stack.

Core components:
- TriShift input mixer (3-tap causal depthwise token mixing)
- HarmonicSpectralInit for tied embeddings
- PhaseMix residual-anchor initialization
- Flexible MLP hidden dimension (supports 2.75x / 1408 hidden)
- Decoupled weight decay for Muon + Adam groups
- Optional fp16 passthrough for sensitive tensors
"""

import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local compatibility shim:
# Some train_gpt_mlx.py versions parse MLP_MULT as int at import time.
# Accept fractional MLP_MULT in this overlay by temporarily normalizing env.
_MLP_MULT_RAW = os.environ.get("MLP_MULT")
if _MLP_MULT_RAW is not None:
    try:
        int(_MLP_MULT_RAW)
    except ValueError:
        try:
            _mlp_mult_float = float(_MLP_MULT_RAW)
        except ValueError:
            _mlp_mult_float = None
        if _mlp_mult_float is not None and math.isfinite(_mlp_mult_float):
            os.environ["MLP_MULT"] = str(max(1, int(round(_mlp_mult_float))))

import train_gpt_mlx as base

if _MLP_MULT_RAW is not None:
    os.environ["MLP_MULT"] = _MLP_MULT_RAW

mx = base.mx
nn = base.nn
optim = base.optim

tree_flatten = base.tree_flatten
tree_unflatten = base.tree_unflatten
COMPUTE_DTYPE = base.COMPUTE_DTYPE


def _env_bool(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, "1" if default else "0")))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(x.strip() for x in os.environ.get(name, default).split(",") if x.strip())


class Hyperparameters(base.Hyperparameters):
    # Architecture knobs.
    mlp_mult: float = _env_float("MLP_MULT", getattr(base.Hyperparameters, "mlp_mult", 2))
    mlp_hidden_dim: int = _env_int("MLP_HIDDEN_DIM", 0)
    mlp_hidden_align: int = _env_int("MLP_HIDDEN_ALIGN", 128)

    # Novel lightweight token mixer.
    input_mix_enabled: bool = _env_bool("INPUT_MIX_ENABLED", True)
    input_mix_repeat_boundary: bool = _env_bool("INPUT_MIX_REPEAT_BOUNDARY", True)
    input_mix_curr_init: float = _env_float("INPUT_MIX_CURR_INIT", 2.20)
    input_mix_prev1_init: float = _env_float("INPUT_MIX_PREV1_INIT", 0.15)
    input_mix_prev2_init: float = _env_float("INPUT_MIX_PREV2_INIT", -1.10)
    input_delta_enabled: bool = _env_bool("INPUT_DELTA_ENABLED", False)
    input_delta_scale: float = _env_float("INPUT_DELTA_SCALE", 0.08)
    input_delta_init: float = _env_float("INPUT_DELTA_INIT", -2.20)

    # Distinct init package.
    harmonic_embed_init: bool = _env_bool("HARMONIC_EMBED_INIT", True)
    harmonic_embed_alpha: float = _env_float("HARMONIC_EMBED_ALPHA", 0.18)
    harmonic_embed_power: float = _env_float("HARMONIC_EMBED_POWER", 0.50)
    harmonic_embed_jitter: float = _env_float("HARMONIC_EMBED_JITTER", 0.82)
    phase_resid_mix_init: bool = _env_bool("PHASE_RESID_MIX_INIT", True)
    phase_resid_anchor_max: float = _env_float("PHASE_RESID_ANCHOR_MAX", 0.12)
    phase_resid_anchor_midpoint: float = _env_float("PHASE_RESID_ANCHOR_MIDPOINT", 0.58)
    phase_resid_anchor_sharpness: float = _env_float("PHASE_RESID_ANCHOR_SHARPNESS", 10.0)

    # Decoupled WD.
    muon_weight_decay: float = _env_float("MUON_WEIGHT_DECAY", _env_float("WEIGHT_DECAY", 0.018))
    embed_weight_decay: float = _env_float("EMBED_WEIGHT_DECAY", 0.012)
    scalar_weight_decay: float = _env_float("SCALAR_WEIGHT_DECAY", 0.0)
    scalar_no_decay_patterns: tuple[str, ...] = _env_csv(
        "SCALAR_NO_DECAY_PATTERNS",
        "attn_scale,mlp_scale,resid_mix,q_gain,skip_weights,mix_,delta_gate",
    )

    # Export-time protection. Off by default for the fast/bytes-first mode.
    force_fp16_embed: bool = _env_bool("FORCE_FP16_EMBED", False)
    force_fp16_last_k_proj: bool = _env_bool("FORCE_FP16_LAST_K_PROJ", False)
    force_fp16_patterns: tuple[str, ...] = _env_csv("FORCE_FP16_PATTERNS", "")

    # Optional convenience log marker.
    novelstack_name: str = os.environ.get("NOVELSTACK_NAME", "TriShift-PhaseMix")


CONTROL_PATTERNS_NOVEL = tuple(dict.fromkeys(base.CONTROL_TENSOR_NAME_PATTERNS + ("mix_", "delta_gate")))
base.CONTROL_TENSOR_NAME_PATTERNS = CONTROL_PATTERNS_NOVEL


def _aligned_hidden_dim(dim: int, mlp_mult: float, explicit_hidden: int, align: int) -> int:
    hidden = explicit_hidden if explicit_hidden > 0 else int(round(dim * float(mlp_mult)))
    hidden = max(hidden, dim)
    if align > 1:
        hidden = int(math.ceil(hidden / align) * align)
    return hidden



def _harmonic_spectral_init(
    vocab_size: int,
    dim: int,
    std: float,
    power: float,
    alpha: float,
    jitter: float,
    seed: int,
) -> mx.array:
    rng = np.random.default_rng(seed)
    rnd = rng.standard_normal((vocab_size, dim), dtype=np.float32)
    spectrum = np.power(np.arange(1, dim + 1, dtype=np.float32), -power)
    rnd = rnd * spectrum[None, :]

    phases = np.linspace(0.0, 2.0 * np.pi, vocab_size, endpoint=False, dtype=np.float32)[:, None]
    freqs = np.arange(1, dim + 1, dtype=np.float32)[None, :]
    harmonic = np.sin(phases * freqs) * spectrum[None, :]

    mix = jitter * rnd + alpha * harmonic
    rms = float(np.sqrt(np.mean(np.square(mix), dtype=np.float64)) + 1e-8)
    mix = (mix / rms) * std
    return mx.array(mix, dtype=COMPUTE_DTYPE)



def _phase_anchor(depth_frac: float, args: Hyperparameters) -> float:
    if not args.phase_resid_mix_init:
        return 0.0
    z = args.phase_resid_anchor_sharpness * (depth_frac - args.phase_resid_anchor_midpoint)
    sig = 1.0 / (1.0 + math.exp(-z))
    return float(args.phase_resid_anchor_max * sig)



def _phase_resid_init(dim: int, layer_idx: int, num_layers: int, args: Hyperparameters) -> mx.array:
    frac = 0.0 if num_layers <= 1 else float(layer_idx) / float(num_layers - 1)
    anchor = _phase_anchor(frac, args)
    curr = np.full((dim,), 1.0 - anchor, dtype=np.float32)
    base_mix = np.full((dim,), anchor, dtype=np.float32)
    return mx.array(np.stack((curr, base_mix), axis=0))



def _causal_shift(x: mx.array, shift: int, repeat_boundary: bool) -> mx.array:
    if shift <= 0:
        return x
    seq_len = int(x.shape[1])
    if seq_len <= 0:
        return x
    if seq_len <= shift:
        if repeat_boundary:
            return mx.concatenate([x[:, :1, :] for _ in range(seq_len)], axis=1)
        return mx.zeros_like(x)
    if repeat_boundary:
        pad = mx.concatenate([x[:, :1, :] for _ in range(shift)], axis=1)
    else:
        pad = mx.zeros_like(x[:, :shift, :])
    return mx.concatenate([pad, x[:, :-shift, :]], axis=1)


class TriShiftMixer(nn.Module):
    """A small causal token mixer.

    It blends the current token with the previous one and the token from two
    steps back using per-channel learned logits normalized with a softmax.
    Overhead is tiny: a few extra elementwise ops, no extra large tables.
    """

    def __init__(self, dim: int, args: Hyperparameters):
        super().__init__()
        self.repeat_boundary = args.input_mix_repeat_boundary
        self.delta_enabled = args.input_delta_enabled
        self.delta_scale = args.input_delta_scale
        self.mix_curr_logit = mx.ones((dim,), dtype=mx.float32) * args.input_mix_curr_init
        self.mix_prev1_logit = mx.ones((dim,), dtype=mx.float32) * args.input_mix_prev1_init
        self.mix_prev2_logit = mx.ones((dim,), dtype=mx.float32) * args.input_mix_prev2_init
        if self.delta_enabled:
            self.delta_gate = mx.ones((dim,), dtype=mx.float32) * args.input_delta_init

    def __call__(self, x: mx.array) -> mx.array:
        prev1 = _causal_shift(x, 1, self.repeat_boundary)
        prev2 = _causal_shift(x, 2, self.repeat_boundary)
        logits = mx.stack(
            [
                self.mix_curr_logit.astype(x.dtype),
                self.mix_prev1_logit.astype(x.dtype),
                self.mix_prev2_logit.astype(x.dtype),
            ],
            axis=0,
        )
        logits = logits - mx.max(logits, axis=0, keepdims=True)
        weights = mx.exp(logits)
        weights = weights / mx.sum(weights, axis=0, keepdims=True)
        y = (
            weights[0][None, None, :] * x
            + weights[1][None, None, :] * prev1
            + weights[2][None, None, :] * prev2
        )
        if self.delta_enabled:
            gate = 1.0 / (1.0 + mx.exp(-self.delta_gate.astype(x.dtype)))
            y = y + (gate[None, None, :] * (x - prev1)) * self.delta_scale
        return y


class FlexMLP(nn.Module):
    def __init__(self, dim: int, args: Hyperparameters):
        super().__init__()
        hidden = _aligned_hidden_dim(dim, args.mlp_mult, args.mlp_hidden_dim, args.mlp_hidden_align)
        self.hidden_dim = hidden
        self.fc = base.CastedLinear(dim, hidden)
        self.proj = base.CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.fc(x))
        return self.proj(h * h)


class NovelBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int,
        num_layers: int,
        args: Hyperparameters,
    ):
        super().__init__()
        self.attn_norm = base.RMSNormNoWeight()
        self.mlp_norm = base.RMSNormNoWeight()
        self.attn = base.CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = FlexMLP(dim, args)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = _phase_resid_init(dim, layer_idx, num_layers, args)

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class NovelGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tied_embed_init_std: float,
        qk_gain_init: float,
    ):
        super().__init__()
        args = Hyperparameters()
        self.args = args
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")

        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.input_mixer = TriShiftMixer(dim, args) if args.input_mix_enabled else None

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            NovelBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rope_base=rope_base,
                qk_gain_init=qk_gain_init,
                layer_idx=i,
                num_layers=num_layers,
                args=args,
            )
            for i in range(num_layers)
        ]
        self.final_norm = base.RMSNormNoWeight()

        for block in self.blocks:
            block.attn.proj.weight = mx.zeros_like(block.attn.proj.weight)
            block.mlp.proj.weight = mx.zeros_like(block.mlp.proj.weight)

        if args.harmonic_embed_init:
            self.tok_emb.weight = _harmonic_spectral_init(
                vocab_size=vocab_size,
                dim=dim,
                std=tied_embed_init_std,
                power=args.harmonic_embed_power,
                alpha=args.harmonic_embed_alpha,
                jitter=args.harmonic_embed_jitter,
                seed=args.seed,
            )
        else:
            self.tok_emb.weight = (
                mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
            ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.input_mixer is not None:
            x = self.input_mixer(x)
        x = base.rms_norm(x)
        x0 = x
        skips: list[mx.array] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)


class MuonWD(base.Muon):
    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        wd = getattr(self.args, "muon_weight_decay", 0.0)
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = base.zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            p_eff = p * (1.0 - lr * wd) if wd > 0.0 else p
            out[k] = p_eff - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizersNovel:
    def __init__(self, model: NovelGPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k != self.embed_key and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_PATTERNS_NOVEL)
        ]
        self.scalar_keys = [k for k in params if k != self.embed_key and k not in self.matrix_keys]
        self.scalar_decay_keys = [
            k for k in self.scalar_keys if not any(pattern in k for pattern in self.args.scalar_no_decay_patterns)
        ]
        self.scalar_no_decay_keys = [k for k in self.scalar_keys if k not in self.scalar_decay_keys]

        self.muon = MuonWD(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def _apply_decay(self, updates: dict[str, mx.array], lr: float, wd: float, keys: Iterable[str]) -> None:
        if wd <= 0.0:
            return
        mul = 1.0 - lr * wd
        for k in keys:
            if k in updates:
                updates[k] = updates[k] * mul

    def step(self, model: NovelGPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        embed_lr = self.args.tied_embed_lr * lr_mul
        self.adam_embed.learning_rate = embed_lr
        embed_updates = self.adam_embed.apply_gradients(
            {self.embed_key: grads[self.embed_key]},
            {self.embed_key: params[self.embed_key]},
        )
        self._apply_decay(embed_updates, embed_lr, self.args.embed_weight_decay, (self.embed_key,))
        updated.update(embed_updates)

        scalar_lr = self.args.scalar_lr * lr_mul
        self.adam_scalar.learning_rate = scalar_lr
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        scalar_updates = self.adam_scalar.apply_gradients(scalar_grads, scalar_params)
        self._apply_decay(scalar_updates, scalar_lr, self.args.scalar_weight_decay, self.scalar_decay_keys)
        updated.update(scalar_updates)
        model.update(tree_unflatten(list(updated.items())))


# -----------------------------------------------------------------------------
# Export override: keep selected tensors in fp16, even if large.
# -----------------------------------------------------------------------------

def _compile_forced_fp16_patterns(flat_state: dict[str, mx.array]) -> tuple[str, ...]:
    args = Hyperparameters()
    patterns: list[str] = list(args.force_fp16_patterns)
    if args.force_fp16_embed:
        patterns.append("tok_emb.weight")
    if args.force_fp16_last_k_proj:
        candidates = []
        regex = re.compile(r"^blocks\.(\d+)\.attn\.c_k\.weight$")
        for name in flat_state:
            match = regex.match(name)
            if match:
                candidates.append((int(match.group(1)), name))
        if candidates:
            candidates.sort()
            patterns.append(candidates[-1][1])
    return tuple(dict.fromkeys(p for p in patterns if p))



def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    forced_fp16_patterns = _compile_forced_fp16_patterns(flat_state)

    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )

    def _force_keep(name: str) -> bool:
        return any(pattern in name for pattern in forced_fp16_patterns)

    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)

        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if _force_keep(name):
            if arr.dtype in {mx.float32, mx.bfloat16}:
                passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
            kept = np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=np.float16, copy=False))
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        if int(arr.size) <= base.INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = base.keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = base.quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


# -----------------------------------------------------------------------------
# Install overlay and delegate the rest to the repository's training harness.
# -----------------------------------------------------------------------------
base.Hyperparameters = Hyperparameters
base.GPT = NovelGPT
base.Muon = MuonWD
base.SplitOptimizers = SplitOptimizersNovel
base.quantize_state_dict_int8 = quantize_state_dict_int8


if __name__ == "__main__":
    base.main()
