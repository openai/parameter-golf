from __future__ import annotations

from dataclasses import dataclass, replace
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .mlp import MLP


def _logspace_half_lives(start: float, end: float, count: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(start), np.log(end), count, dtype=np.float32))


def _decays_from_half_lives(half_lives: np.ndarray) -> np.ndarray:
    return np.exp(np.log(0.5, dtype=np.float32) / half_lives.astype(np.float32, copy=False))


def _osc_pair_count(config: "ConkerThreeConfig") -> int:
    osc_pairs = int((config.linear_modes * config.oscillatory_frac) // 2)
    return max(min(osc_pairs, config.linear_modes // 2), 0)


def _random_in_proj(rng: np.random.Generator, embedding_dim: int, mode_count: int) -> np.ndarray:
    scale = 1.0 / math.sqrt(embedding_dim)
    return rng.standard_normal((embedding_dim, mode_count), dtype=np.float32) * scale


def _orthogonal_rows_in_proj(rng: np.random.Generator, embedding_dim: int, mode_count: int) -> np.ndarray:
    if mode_count <= 0:
        return np.zeros((embedding_dim, 0), dtype=np.float32)
    mat = rng.standard_normal((mode_count, embedding_dim), dtype=np.float32)
    q, _ = np.linalg.qr(mat, mode="reduced")
    proj = q.T.astype(np.float32, copy=False)
    proj *= np.float32(math.sqrt(mode_count / embedding_dim))
    return proj


def _split_bank_in_proj(
    rng: np.random.Generator,
    embedding_dim: int,
    non_osc_modes: int,
    osc_mode_count: int,
) -> np.ndarray:
    in_proj = np.zeros((embedding_dim, non_osc_modes + osc_mode_count), dtype=np.float32)
    if non_osc_modes <= 0:
        in_proj[:, :] = _orthogonal_rows_in_proj(rng, embedding_dim, osc_mode_count)
        return in_proj
    if osc_mode_count <= 0:
        in_proj[:, :] = _orthogonal_rows_in_proj(rng, embedding_dim, non_osc_modes)
        return in_proj
    non_osc_dim = max(embedding_dim // 2, 1)
    osc_dim = max(embedding_dim - non_osc_dim, 1)
    in_proj[:non_osc_dim, :non_osc_modes] = _orthogonal_rows_in_proj(rng, non_osc_dim, non_osc_modes)
    osc_block = np.zeros((osc_dim, osc_mode_count), dtype=np.float32)
    osc_pairs = osc_mode_count // 2
    for idx in range(osc_pairs):
        base = _orthogonal_rows_in_proj(rng, osc_dim, 1).reshape(-1)
        start = 2 * idx
        osc_block[:, start] = base
        osc_block[:, start + 1] = base
    if osc_mode_count % 2 == 1:
        osc_block[:, -1] = _orthogonal_rows_in_proj(rng, osc_dim, 1).reshape(-1)
    in_proj[embedding_dim - osc_dim :, non_osc_modes:] = osc_block
    return in_proj


def _kernel_from_decays(decays: np.ndarray, max_seq_len: int) -> np.ndarray:
    time_idx = np.arange(max_seq_len, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    mask = delta >= 0
    safe_delta = np.where(mask, delta, 0).astype(np.float32)
    kernel = np.power(decays[None, None, :], safe_delta[..., None], dtype=np.float32)
    kernel = np.where(mask[..., None], kernel, 0.0).astype(np.float32)
    return np.transpose(kernel, (2, 0, 1))


def _kernel_from_damped_oscillators(
    decays: np.ndarray,
    periods: np.ndarray,
    max_seq_len: int,
) -> np.ndarray:
    time_idx = np.arange(max_seq_len, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    mask = delta >= 0
    lags = np.where(mask, delta, 0).astype(np.float32)
    envelope = np.power(decays[:, None, None], lags[None, :, :], dtype=np.float32)
    omega = (2.0 * np.pi / periods.astype(np.float32, copy=False))[:, None, None]
    cos_kernel = np.where(mask[None, :, :], envelope * np.cos(omega * lags[None, :, :]), 0.0).astype(np.float32)
    sin_kernel = np.where(mask[None, :, :], envelope * np.sin(omega * lags[None, :, :]), 0.0).astype(np.float32)
    kernel = np.empty((decays.shape[0] * 2, max_seq_len, max_seq_len), dtype=np.float32)
    kernel[0::2] = cos_kernel
    kernel[1::2] = sin_kernel
    return kernel


def _build_linear_bank(config: "ConkerThreeConfig") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    osc_pairs = _osc_pair_count(config)
    non_osc_modes = config.linear_modes - 2 * osc_pairs
    osc_mode_count = 2 * osc_pairs

    if config.input_proj_scheme == "random":
        in_proj = _random_in_proj(rng, config.embedding_dim, config.linear_modes)
    elif config.input_proj_scheme == "orthogonal_rows":
        in_proj = _orthogonal_rows_in_proj(rng, config.embedding_dim, config.linear_modes)
    elif config.input_proj_scheme == "split_banks":
        in_proj = _split_bank_in_proj(rng, config.embedding_dim, non_osc_modes, osc_mode_count)
    elif config.input_proj_scheme == "kernel_energy":
        in_proj = _random_in_proj(rng, config.embedding_dim, config.linear_modes)
    else:
        raise ValueError(f"Unknown Conker-3 input_proj_scheme: {config.input_proj_scheme}")

    kernels: list[np.ndarray] = []
    decay_parts: list[np.ndarray] = []

    if non_osc_modes > 0:
        half_lives = _logspace_half_lives(config.linear_half_life_min, config.linear_half_life_max, non_osc_modes)
        decays = _decays_from_half_lives(half_lives)
        kernels.append(_kernel_from_decays(decays, config.max_seq_len))
        decay_parts.append(decays)

    if osc_pairs > 0:
        periods = _logspace_half_lives(config.oscillatory_period_min, config.oscillatory_period_max, osc_pairs)
        half_lives = _logspace_half_lives(
            max(config.linear_half_life_min, 2.0),
            config.linear_half_life_max,
            osc_pairs,
        )
        decays = _decays_from_half_lives(half_lives)
        if config.input_proj_scheme not in {"split_banks"}:
            for idx in range(osc_pairs):
                start = non_osc_modes + 2 * idx
                if config.input_proj_scheme == "random":
                    base = _random_in_proj(rng, config.embedding_dim, 1).reshape(-1)
                    in_proj[:, start] = base
                    in_proj[:, start + 1] = base
                elif config.input_proj_scheme == "orthogonal_rows":
                    base = _orthogonal_rows_in_proj(rng, config.embedding_dim, 1).reshape(-1)
                    in_proj[:, start] = base
                    in_proj[:, start + 1] = base
                elif config.input_proj_scheme == "kernel_energy":
                    base = _random_in_proj(rng, config.embedding_dim, 1).reshape(-1)
                    in_proj[:, start] = base
                    in_proj[:, start + 1] = base
        kernels.append(_kernel_from_damped_oscillators(decays, periods, config.max_seq_len))
        decay_parts.append(np.repeat(decays, 2))

    if not kernels:
        raise ValueError("Conker-3 linear bank must contain at least one mode.")

    kernel = np.concatenate(kernels, axis=0).astype(np.float32, copy=False)
    decays_full = np.concatenate(decay_parts, axis=0).astype(np.float32, copy=False)
    if config.input_proj_scheme == "kernel_energy":
        mode_energy = np.sqrt(np.mean(kernel * kernel, axis=(1, 2), dtype=np.float32)).astype(np.float32, copy=False)
        mode_energy = mode_energy / max(float(np.mean(mode_energy)), 1e-6)
        in_proj = in_proj * mode_energy[None, :]
    return in_proj.astype(np.float32, copy=False), decays_full, kernel


@dataclass(frozen=True)
class ConkerThreeConfig:
    embedding_dim: int = 32
    linear_modes: int = 256
    max_seq_len: int = 256
    linear_half_life_min: float = 1.5
    linear_half_life_max: float = 512.0
    linear_hidden: tuple[int, ...] = (128,)
    local_window: int = 8
    local_hidden: tuple[int, ...] = (128,)
    local_scale: float = 0.25
    mix_mode: str = "additive"
    share_embedding: bool = False
    linear_impl: str = "kernel"
    enable_linear: bool = True
    enable_local: bool = True
    oscillatory_frac: float = 0.0
    oscillatory_period_min: float = 4.0
    oscillatory_period_max: float = 64.0
    static_bank_gate: bool = False
    bank_gate_span: float = 0.5
    input_proj_scheme: str = "random"


class ConkerThreeModel(nn.Module):
    """Frozen linear substrate plus a parallel local residual coder."""

    def __init__(self, vocab_size: int, config: ConkerThreeConfig = ConkerThreeConfig()):
        super().__init__()
        if not config.enable_linear and not config.enable_local:
            raise ValueError("Conker-3 must enable at least one path.")
        if config.mix_mode not in {"additive", "gated"}:
            raise ValueError(f"Unknown Conker-3 mix_mode: {config.mix_mode}")
        if config.linear_impl not in {"kernel", "fft"}:
            raise ValueError(f"Unknown Conker-3 linear_impl: {config.linear_impl}")
        if config.oscillatory_frac < 0.0 or config.oscillatory_frac >= 1.0:
            raise ValueError("Conker-3 oscillatory_frac must be in [0, 1).")
        if config.oscillatory_frac > 0.0 and config.linear_impl != "kernel":
            raise ValueError("Conker-3 oscillatory modes currently require linear_impl='kernel'.")
        if config.local_window < 1:
            raise ValueError("Conker-3 local_window must be >= 1.")

        self.vocab_size = vocab_size
        self.config = config

        self.shared_embedding = None
        self.linear_embedding = None
        self.local_embedding = None
        self.linear_in_proj = None
        self.linear_decays = None
        self.linear_kernel = None
        self.linear_readout = None
        self.local_readout = None
        self.gate_proj = None
        self.bank_gate_logits = None
        self.non_osc_modes = 0
        self.osc_mode_count = 0

        if config.share_embedding and config.enable_linear and config.enable_local:
            self.shared_embedding = nn.Embedding(vocab_size, config.embedding_dim)

        if config.enable_linear:
            if self.shared_embedding is None:
                self.linear_embedding = nn.Embedding(vocab_size, config.embedding_dim)

            in_proj, decays, kernel = _build_linear_bank(config)
            self.linear_in_proj = mx.array(in_proj)
            self.linear_decays = mx.array(decays.astype(np.float32))

            if config.linear_impl == "kernel":
                self.linear_kernel = mx.array(kernel)

            self.linear_readout = MLP(
                config.linear_modes + config.embedding_dim,
                config.linear_hidden,
                vocab_size,
            )
            osc_pairs = _osc_pair_count(config)
            self.non_osc_modes = config.linear_modes - 2 * osc_pairs
            self.osc_mode_count = 2 * osc_pairs
            if config.static_bank_gate and self.osc_mode_count > 0:
                self.bank_gate_logits = mx.zeros((2,), dtype=mx.float32)

        if config.enable_local:
            if self.shared_embedding is None:
                self.local_embedding = nn.Embedding(vocab_size, config.embedding_dim)
            self.local_readout = MLP(
                config.local_window * config.embedding_dim,
                config.local_hidden,
                vocab_size,
            )

        if config.enable_linear and config.enable_local and config.mix_mode == "gated":
            self.gate_proj = nn.Linear(6, 1)

        freeze_keys = [key for key in ("linear_in_proj", "linear_decays", "linear_kernel") if getattr(self, key) is not None]
        if freeze_keys:
            self.freeze(keys=freeze_keys, strict=False)

    def set_linear_decays(self, decays: np.ndarray) -> None:
        if not self.config.enable_linear:
            raise RuntimeError("Conker-3 linear path is disabled.")
        decays = np.asarray(decays, dtype=np.float32)
        if decays.shape != (self.config.linear_modes,):
            raise ValueError(
                f"Conker-3 expected {self.config.linear_modes} decays, got shape {decays.shape}"
            )
        self.linear_decays = mx.array(decays.astype(np.float32, copy=False))
        if self.config.linear_impl == "kernel":
            self.linear_kernel = mx.array(_kernel_from_decays(decays, self.config.max_seq_len))
        self.freeze(keys=("linear_decays", "linear_kernel"), strict=False)

    @staticmethod
    def _logit_features(logits: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        probs = mx.exp(log_probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        max_logit = mx.max(logits, axis=-1)
        centered = logits - mx.mean(logits, axis=-1, keepdims=True)
        variance = mx.mean(centered * centered, axis=-1)
        return entropy, max_logit, variance

    def _embed_linear(self, chars: mx.array) -> mx.array:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.linear_embedding is None:
            raise RuntimeError("Conker-3 linear path has no embedding table.")
        return self.linear_embedding(chars)

    def _embed_local(self, chars: mx.array) -> mx.array:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.local_embedding is None:
            raise RuntimeError("Conker-3 local path has no embedding table.")
        return self.local_embedding(chars)

    def _linear_states_fft(self, drive: mx.array, timesteps: int) -> mx.array:
        if self.linear_decays is None:
            raise RuntimeError("Conker-3 FFT path is missing linear decays.")
        drive_mb = mx.transpose(drive, (0, 2, 1))
        n_fft = 1 << int(math.ceil(math.log2(max(2 * timesteps - 1, 1))))
        time = mx.arange(timesteps, dtype=drive.dtype)
        kernel = mx.power(self.linear_decays[:, None], time[None, :])
        drive_f = mx.fft.rfft(drive_mb, n=n_fft, axis=-1)
        kernel_f = mx.fft.rfft(kernel[None, :, :], n=n_fft, axis=-1)
        states_mb = mx.fft.irfft(drive_f * kernel_f, n=n_fft, axis=-1)[..., :timesteps]
        return mx.transpose(states_mb, (0, 2, 1))

    def _apply_mode_gate(self, states: mx.array, mode_gate: mx.array | None) -> mx.array:
        if mode_gate is None:
            return states
        if mode_gate.ndim == 1:
            return states * mode_gate[None, None, :]
        if mode_gate.ndim == 2:
            return states * mode_gate[:, None, :]
        raise ValueError(f"Conker-3 mode_gate must be rank-1 or rank-2, got shape {mode_gate.shape}")

    def _static_bank_mode_gate(self) -> mx.array | None:
        if self.bank_gate_logits is None or self.osc_mode_count <= 0:
            return None
        values = 1.0 + self.config.bank_gate_span * mx.tanh(self.bank_gate_logits)
        pieces = []
        if self.non_osc_modes > 0:
            pieces.append(mx.broadcast_to(values[0:1], (self.non_osc_modes,)))
        if self.osc_mode_count > 0:
            pieces.append(mx.broadcast_to(values[1:2], (self.osc_mode_count,)))
        return mx.concatenate(pieces, axis=0) if pieces else None

    def _linear_states(self, chars: mx.array, mode_gate: mx.array | None = None) -> tuple[mx.array, mx.array]:
        _, timesteps = chars.shape
        if timesteps > self.config.max_seq_len:
            raise ValueError(
                f"Conker-3 max_seq_len={self.config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.linear_in_proj is None or self.linear_readout is None:
            raise RuntimeError("Conker-3 linear path is disabled.")
        x = self._embed_linear(chars)
        drive = mx.matmul(x, self.linear_in_proj)
        if self.config.linear_impl == "kernel":
            if self.linear_kernel is None:
                raise RuntimeError("Conker-3 kernel path is missing its materialized kernel.")
            kernels = self.linear_kernel[:, :timesteps, :timesteps]
            drive_mb = mx.transpose(drive, (2, 0, 1))
            states_mb = mx.matmul(drive_mb, mx.transpose(kernels, (0, 2, 1)))
            states = mx.transpose(states_mb, (1, 2, 0))
        else:
            states = self._linear_states_fft(drive, timesteps)
        states = self._apply_mode_gate(states, self._static_bank_mode_gate())
        return self._apply_mode_gate(states, mode_gate), x

    def _linear_logits(self, chars: mx.array, mode_gate: mx.array | None = None) -> mx.array:
        states, x = self._linear_states(chars, mode_gate=mode_gate)
        return self.linear_readout(mx.concatenate([states, x], axis=-1))

    def _local_window_stack(self, x: mx.array) -> mx.array:
        batch, timesteps, dim = x.shape
        window = self.config.local_window
        if window == 1:
            return x
        pad = mx.zeros((batch, window - 1, dim), dtype=x.dtype)
        padded = mx.concatenate([pad, x], axis=1)
        views = []
        for offset in range(window):
            start = window - 1 - offset
            views.append(padded[:, start : start + timesteps, :])
        return mx.concatenate(views, axis=-1)

    def _local_logits(self, chars: mx.array) -> mx.array:
        if self.local_readout is None:
            raise RuntimeError("Conker-3 local path is disabled.")
        x = self._embed_local(chars)
        stacked = self._local_window_stack(x)
        return self.local_readout(stacked)

    def __call__(self, chars: mx.array) -> mx.array:
        logits_linear = self._linear_logits(chars) if self.config.enable_linear else None
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            features = mx.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], axis=-1)
            gate = mx.sigmoid(self.gate_proj(features)) * self.config.local_scale

        return logits_linear + gate * logits_local

    def forward_with_mode_gate(self, chars: mx.array, mode_gate: mx.array | None) -> mx.array:
        logits_linear = self._linear_logits(chars, mode_gate=mode_gate) if self.config.enable_linear else None
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            features = mx.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], axis=-1)
            gate = mx.sigmoid(self.gate_proj(features)) * self.config.local_scale

        return logits_linear + gate * logits_local


def scale_config(config: ConkerThreeConfig, scale: float) -> ConkerThreeConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        embedding_dim=max(int(round(config.embedding_dim * scale)), 1),
        linear_modes=max(int(round(config.linear_modes * scale)), 1),
        linear_hidden=tuple(max(int(round(width * scale)), 1) for width in config.linear_hidden),
        local_hidden=tuple(max(int(round(width * scale)), 1) for width in config.local_hidden),
    )
