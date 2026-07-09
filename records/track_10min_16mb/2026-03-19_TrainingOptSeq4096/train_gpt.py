"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from toricgt.wandb_organization import configure_wandb_metrics, organize_wandb_payload
except Exception:  # pragma: no cover - keep this record copy standalone.
    def organize_wandb_payload(payload, **_kwargs):
        return payload

    def configure_wandb_metrics(wandb_module, *, step_metric: str = "trainer/step") -> None:
        wandb_module.define_metric(step_metric)
        wandb_module.define_metric("*", step_metric=step_metric)

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Training Opt Seq4096 v1 run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 4096, tied embeddings
# - 393,216 train tokens per step (3/4 batch) for 20,000 iterations with a ~10 minute cap
# - tuned Muon optimizer: momentum 0.99, lower LR (0.020/0.020/0.030), warmdown 3000

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", "training_opt_seq4096_v1")
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_max_sequences = int(os.environ.get("VAL_MAX_SEQUENCES", 0))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", 0))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints/training_opt_seq4096")
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")
    reset_optimizer_on_resume = os.environ.get(
        "RESET_OPTIMIZER_ON_RESUME",
        os.environ.get("RESET_OPTIMIZER", "0"),
    ).lower() in {"1", "true", "yes", "on"}
    reset_rng_on_resume = os.environ.get("RESET_RNG_ON_RESUME", "0").lower() in {"1", "true", "yes", "on"}
    reset_loader_on_resume = os.environ.get("RESET_LOADER_ON_RESUME", "0").lower() in {"1", "true", "yes", "on"}
    wandb_enabled = os.environ.get("WANDB", "1").lower() not in {"0", "false", "no", "off", "disabled"}
    wandb_project = os.environ.get("WANDB_PROJECT", "toricgt-parameter-golf")
    wandb_entity = os.environ.get("WANDB_ENTITY", "amelie-iska-math")
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", run_id)
    wandb_run_id = os.environ.get("WANDB_RUN_ID", run_id)
    wandb_resume = os.environ.get("WANDB_RESUME", "allow")
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb_key_path = os.environ.get("WANDB_KEY_PATH", "")
    wandb_log_every = int(os.environ.get("WANDB_LOG_EVERY", 1))
    target_bpb = float(os.environ.get("TARGET_BPB", 1.2))
    artifact_size_limit_bytes = int(os.environ.get("ARTIFACT_SIZE_LIMIT_BYTES", 16_000_000))
    export_prune_fraction = float(os.environ.get("EXPORT_PRUNE_FRACTION", 0.0))
    checkpoint_on_train_bpb_below = float(os.environ.get("CHECKPOINT_ON_TRAIN_BPB_BELOW", 0.0))
    checkpoint_on_train_bpb_cooldown_steps = int(os.environ.get("CHECKPOINT_ON_TRAIN_BPB_COOLDOWN_STEPS", 25))
    checkpoint_on_train_bpb_max = int(os.environ.get("CHECKPOINT_ON_TRAIN_BPB_MAX", 4))
    val_on_train_bpb_checkpoint = os.environ.get(
        "VAL_ON_TRAIN_BPB_CHECKPOINT",
        "1",
    ).lower() in {"1", "true", "yes", "on"}

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 393_216))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    polarquant_kv_bits = int(os.environ.get("POLARQUANT_KV_BITS", 8))
    polarquant_train = os.environ.get("POLARQUANT_TRAIN", "1").lower() in {"1", "true", "yes", "on"}
    polarquant_train_sample_tokens = int(os.environ.get("POLARQUANT_TRAIN_SAMPLE_TOKENS", 16))
    polarquant_eval_sample_tokens = int(os.environ.get("POLARQUANT_EVAL_SAMPLE_TOKENS", 256))
    polarquant_seed = int(os.environ.get("POLARQUANT_SEED", 271828))
    polarquant_train_start_step = int(os.environ.get("POLARQUANT_TRAIN_START_STEP", 0))
    polarquant_train_warmup_steps = int(os.environ.get("POLARQUANT_TRAIN_WARMUP_STEPS", 512))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.030))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.020))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.020))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    nonfinite_audit_every = int(os.environ.get("NONFINITE_AUDIT_EVERY", 25))
    bigram_bias = os.environ.get("BIGRAM_BIAS", "0").lower() in {"1", "true", "yes", "on"}
    bigram_bias_lr = float(os.environ.get("BIGRAM_BIAS_LR", 0.05))
    bigram_bias_init_std = float(os.environ.get("BIGRAM_BIAS_INIT_STD", 0.0))
    bigram_bias_scale = float(os.environ.get("BIGRAM_BIAS_SCALE", 1.0))
    bigram_bias_init_from_data = os.environ.get("BIGRAM_BIAS_INIT_FROM_DATA", "0").lower() in {"1", "true", "yes", "on"}
    bigram_bias_init_tokens = int(os.environ.get("BIGRAM_BIAS_INIT_TOKENS", 100_000_000))
    bigram_bias_init_alpha = float(os.environ.get("BIGRAM_BIAS_INIT_ALPHA", 0.1))
    bigram_bias_init_strength = float(os.environ.get("BIGRAM_BIAS_INIT_STRENGTH", 0.35))
    hash_ngram_bias = os.environ.get("HASH_NGRAM_BIAS", "0").lower() in {"1", "true", "yes", "on"}
    hash_ngram_bias_buckets = int(os.environ.get("HASH_NGRAM_BIAS_BUCKETS", 2048))
    hash_ngram_bias_order = int(os.environ.get("HASH_NGRAM_BIAS_ORDER", 3))
    hash_ngram_bias_lr = float(os.environ.get("HASH_NGRAM_BIAS_LR", 0.02))
    hash_ngram_bias_scale = float(os.environ.get("HASH_NGRAM_BIAS_SCALE", 1.0))
    hash_ngram_bias_init_std = float(os.environ.get("HASH_NGRAM_BIAS_INIT_STD", 0.0))
    hash_ngram_bias_init_from_data = os.environ.get("HASH_NGRAM_BIAS_INIT_FROM_DATA", "0").lower() in {"1", "true", "yes", "on"}
    hash_ngram_bias_init_tokens = int(os.environ.get("HASH_NGRAM_BIAS_INIT_TOKENS", 100_000_000))
    hash_ngram_bias_init_alpha = float(os.environ.get("HASH_NGRAM_BIAS_INIT_ALPHA", 0.05))
    hash_ngram_bias_init_strength = float(os.environ.get("HASH_NGRAM_BIAS_INIT_STRENGTH", 0.35))
    advanced_loss_scale = float(os.environ.get("ADVANCED_LOSS_SCALE", 0.0))
    graphcg_loss_weight = float(os.environ.get("GRAPHCG_LOSS_WEIGHT", 0.0))
    toric_tropical_loss_weight = float(os.environ.get("TORIC_TROPICAL_LOSS_WEIGHT", 0.0))
    slepian_loss_weight = float(os.environ.get("SLEPIAN_LOSS_WEIGHT", 0.0))
    koszul_bgg_loss_weight = float(os.environ.get("KOSZUL_BGG_LOSS_WEIGHT", 0.0))
    analogy_loss_weight = float(os.environ.get("ANALOGY_LOSS_WEIGHT", 0.0))
    advanced_loss_sample_tokens = int(os.environ.get("ADVANCED_LOSS_SAMPLE_TOKENS", 256))
    toric_tropical_fan_bins = int(os.environ.get("TORIC_TROPICAL_FAN_BINS", 8))
    advanced_loss_log_only = os.environ.get("ADVANCED_LOSS_LOG_ONLY", "0").lower() in {"1", "true", "yes", "on"}
    advanced_loss_start_step = int(os.environ.get("ADVANCED_LOSS_START_STEP", 0))
    advanced_loss_end_step = int(os.environ.get("ADVANCED_LOSS_END_STEP", 0))
    advanced_loss_every = int(os.environ.get("ADVANCED_LOSS_EVERY", 1))
    advanced_loss_warmup_steps = int(os.environ.get("ADVANCED_LOSS_WARMUP_STEPS", 0))
    advanced_loss_min_best_val_bpb = float(os.environ.get("ADVANCED_LOSS_MIN_BEST_VAL_BPB", 0.0))
    advanced_loss_max_ce_ratio = float(os.environ.get("ADVANCED_LOSS_MAX_CE_RATIO", 0.0))


SECRET_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")


def find_wandb_api_key(explicit_path: str = "") -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]

    candidate_paths: list[Path] = []
    if explicit_path:
        candidate_paths.append(Path(explicit_path))
    candidate_paths.append(Path("keys.txt"))
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        candidate_paths.extend(root / "keys.txt" for root in (base, *base.parents))

    seen: set[Path] = set()
    for path in candidate_paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            text = resolved.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if "wandb" not in line.lower():
                continue
            tokens = [
                token
                for token in SECRET_TOKEN_RE.findall(line)
                if token.lower() not in {"wandb", "api", "key", "apikey", "token"}
            ]
            if tokens:
                return tokens[-1]
    return None


def _sample_current_embeddings(model: nn.Module, input_ids: Tensor, max_tokens: int) -> tuple[Tensor, Tensor]:
    flat = input_ids.reshape(-1)
    limit = max(4, int(max_tokens))
    stride = max(flat.numel() // limit, 1)
    seq_ids = flat[::stride][:limit]
    vocab_ids = torch.unique(seq_ids)
    if vocab_ids.numel() < 4:
        vocab_ids = flat[: min(flat.numel(), limit)]
        seq_ids = vocab_ids
    emb = model.tok_emb.weight.index_select(0, vocab_ids).float()
    seq = model.tok_emb(seq_ids).float()
    return F.rms_norm(emb, (emb.size(-1),)), F.rms_norm(seq, (seq.size(-1),))


def advanced_embedding_losses(
    model: nn.Module,
    args: object,
    input_ids: Tensor,
    runtime_scale: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    scale = float(getattr(args, "advanced_loss_scale", 0.0) if runtime_scale is None else runtime_scale)
    weights = {
        "graphcg": float(getattr(args, "graphcg_loss_weight", 0.0)),
        "toric": float(getattr(args, "toric_tropical_loss_weight", 0.0)),
        "slepian": float(getattr(args, "slepian_loss_weight", 0.0)),
        "koszul": float(getattr(args, "koszul_bgg_loss_weight", 0.0)),
        "analogy": float(getattr(args, "analogy_loss_weight", 0.0)),
    }
    zero = model.tok_emb.weight.float().sum() * 0.0
    log_only = bool(getattr(args, "advanced_loss_log_only", False))
    if (scale <= 0.0 and not log_only) or max(weights.values()) <= 0.0:
        return zero, {}
    emb, seq = _sample_current_embeddings(model, input_ids, int(getattr(args, "advanced_loss_sample_tokens", 256)))
    n = max(emb.size(0), 1)
    metrics: dict[str, Tensor] = {}
    weighted = zero

    def finite_aux_scalar(name: str, value: Tensor, max_value: float = 1.0e4) -> Tensor:
        finite = torch.isfinite(value.detach())
        metrics[f"advanced/{name}_nonfinite"] = (~finite).to(dtype=torch.float32)
        safe = torch.where(torch.isfinite(value), value, zero)
        safe = torch.nan_to_num(safe, nan=0.0, posinf=max_value, neginf=0.0)
        return safe.clamp_min(0.0)

    if weights["graphcg"] > 0.0:
        cov = emb.T @ emb / float(n)
        diag = torch.diagonal(cov)
        off = cov - torch.diag(diag)
        graphcg = off.square().mean() / (diag.square().mean().detach() + 1e-6)
        graphcg = finite_aux_scalar("graphcg_loss", graphcg)
        probs = (diag.abs() + 1e-6) / (diag.abs().sum() + 1e-6)
        entropy = torch.nan_to_num(-(probs * probs.log()).sum() / math.log(max(probs.numel(), 2)), nan=0.0, posinf=1.0, neginf=0.0)
        weighted = weighted + weights["graphcg"] * graphcg
        metrics.update({
            "advanced/graphcg_loss": graphcg.detach(),
            "advanced/graphcg_spectral_entropy": entropy.detach(),
            "advanced/graphcg_offdiag_coherence": off.detach().abs().mean(),
        })

    if weights["toric"] > 0.0:
        k = max(4, int(getattr(args, "toric_tropical_fan_bins", 8)))
        theta = torch.linspace(0.0, 2.0 * math.pi, k + 1, device=emb.device, dtype=emb.dtype)[:-1]
        rays = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        phase = emb[:, :2] if emb.size(1) >= 2 else F.pad(emb, (0, 2 - emb.size(1)))
        logits = phase @ rays.T
        probs = torch.softmax(logits, dim=-1)
        chamber = probs.mean(dim=0)
        entropy = -(chamber * (chamber + 1e-6).log()).sum() / math.log(k)
        top2 = torch.topk(logits, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        toric = (1.0 - entropy) + F.relu(0.15 - margin).mean()
        toric = finite_aux_scalar("toric_tropical_loss", toric)
        weighted = weighted + weights["toric"] * toric
        metrics.update({
            "advanced/toric_tropical_loss": toric.detach(),
            "advanced/toric_fan_entropy": entropy.detach(),
            "advanced/tropical_active_face_margin": margin.detach().mean(),
        })

    if weights["slepian"] > 0.0 and seq.size(0) >= 8:
        centered = seq - seq.mean(dim=0, keepdim=True)
        spec = torch.fft.rfft(centered, dim=0)
        power = spec.real.square() + spec.imag.square()
        low = max(2, power.size(0) // 5)
        concentration = power[:low].sum() / (power.sum() + 1e-6)
        slepian = 1.0 - concentration
        slepian = finite_aux_scalar("slepian_pollak_loss", slepian)
        weighted = weighted + weights["slepian"] * slepian
        metrics.update({
            "advanced/slepian_pollak_loss": slepian.detach(),
            "advanced/slepian_concentration": concentration.detach(),
            "advanced/slepian_leakage": (1.0 - concentration).detach(),
        })

    if weights["koszul"] > 0.0:
        r = max(2, min(16, emb.size(1) // 3))
        a, b, c = emb[:, :r], emb[:, r : 2 * r], emb[:, 2 * r : 3 * r]
        d1 = a.T @ b / float(n)
        d2 = b.T @ c / float(n)
        d2_resid = (d1 @ d2).square().mean() / ((d1.square().mean() * d2.square().mean()).detach() + 1e-6)
        leakage = (a.T @ c / float(n)).square().mean() / ((a.square().mean() * c.square().mean()).detach() + 1e-6)
        koszul = d2_resid + 0.25 * leakage
        koszul = finite_aux_scalar("koszul_bgg_loss", koszul)
        weighted = weighted + weights["koszul"] * koszul
        metrics.update({
            "advanced/koszul_bgg_loss": koszul.detach(),
            "advanced/koszul_d2_residual": d2_resid.detach(),
            "advanced/bgg_standard_leakage": leakage.detach(),
        })

    if (weights["toric"] > 0.0 or weights["koszul"] > 0.0) and seq.size(0) >= 8:
        try:
            from toricgt.combinatorial_toric_metrics import (
                CombinatorialToricConfig,
                combinatorial_toric_cca_topology_loss,
            )

            positions = torch.arange(seq.size(0), device=seq.device, dtype=torch.long).view(1, -1)
            cca_metrics = combinatorial_toric_cca_topology_loss(
                seq.unsqueeze(0),
                positions,
                config=CombinatorialToricConfig(
                    max_points=min(12, int(getattr(args, "advanced_loss_sample_tokens", 64))),
                    max_windows=1,
                    window_size=min(32, seq.size(0)),
                    step_stride=16,
                    num_chambers=max(4, int(getattr(args, "toric_tropical_fan_bins", 8))),
                    temperature=0.16,
                    max_loss_value=16.0,
                ),
            )
            cca_loss = finite_aux_scalar(
                "toric_cca_topology_loss",
                cca_metrics["toric_cca_topology_loss"],
                max_value=16.0,
            )
            cca_weight = 0.15 * weights["toric"] + 0.15 * weights["koszul"]
            weighted = weighted + cca_weight * cca_loss
            for cca_key, cca_value in cca_metrics.items():
                if isinstance(cca_value, torch.Tensor):
                    metrics[f"advanced/{cca_key}"] = cca_value.detach()
            metrics["advanced/toric_cca_weight"] = torch.as_tensor(
                cca_weight,
                device=seq.device,
                dtype=torch.float32,
            )
            metrics["advanced/toric_cca_bridge_failed"] = torch.zeros((), device=seq.device, dtype=torch.float32)
        except Exception:
            metrics["advanced/toric_cca_bridge_failed"] = torch.ones((), device=seq.device, dtype=torch.float32)

    if weights["analogy"] > 0.0 and seq.size(0) >= 6:
        deltas = seq[1:] - seq[:-1]
        analogue = (deltas[2:] - deltas[:-2]).square().mean()
        denom = deltas.square().mean().detach() + 1e-6
        analogy = analogue / denom
        analogy = finite_aux_scalar("analogy_loss", analogy)
        weighted = weighted + weights["analogy"] * analogy
        metrics.update({
            "advanced/analogy_loss": analogy.detach(),
            "advanced/analogical_transport_residual": analogy.detach(),
        })

    weighted = finite_aux_scalar("weighted_unscaled_loss", weighted)
    effective_total = weighted.new_zeros(()) if log_only else scale * weighted
    total = finite_aux_scalar("aux_loss_total", effective_total)
    metrics["advanced/weighted_unscaled_loss"] = weighted.detach()
    metrics["advanced/log_only"] = torch.as_tensor(float(log_only), device=weighted.device, dtype=weighted.dtype)
    metrics["advanced/runtime_scale"] = torch.as_tensor(scale, device=weighted.device, dtype=weighted.dtype)
    metrics["advanced/aux_loss"] = total.detach()
    return total, metrics


def polarquant_train_runtime_scale(args: object, step: int) -> float:
    if not bool(getattr(args, "polarquant_train", False)):
        return 0.0
    start_step = int(getattr(args, "polarquant_train_start_step", 0))
    if step < start_step:
        return 0.0
    warmup = int(getattr(args, "polarquant_train_warmup_steps", 0))
    if warmup <= 0:
        return 1.0
    return min(max(step - start_step + 1, 0) / float(warmup), 1.0)


def set_polarquant_train_scale_(model: nn.Module, scale: float) -> None:
    for module in model.modules():
        if hasattr(module, "polarquant_train_scale"):
            module.polarquant_train_scale.fill_(float(scale))


def advanced_loss_runtime_scale(args: object, step: int, best_val_bpb: float) -> float:
    scale = float(getattr(args, "advanced_loss_scale", 0.0))
    if scale <= 0.0:
        return 0.0
    start_step = int(getattr(args, "advanced_loss_start_step", 0))
    if step < start_step:
        return 0.0
    end_step = int(getattr(args, "advanced_loss_end_step", 0))
    if end_step > 0 and step >= end_step:
        return 0.0
    every = max(1, int(getattr(args, "advanced_loss_every", 1)))
    if every > 1 and (step - start_step) % every != 0:
        return 0.0
    min_best_val = float(getattr(args, "advanced_loss_min_best_val_bpb", 0.0))
    if min_best_val > 0.0 and (not math.isfinite(best_val_bpb) or best_val_bpb > min_best_val):
        return 0.0
    warmup = int(getattr(args, "advanced_loss_warmup_steps", 0))
    if warmup > 0:
        scale *= min(max(step - start_step + 1, 0) / float(warmup), 1.0)
    return scale

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if args.val_max_sequences > 0:
        total_seqs = min(total_seqs, int(args.val_max_sequences))
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

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

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def prune_state_dict_for_export(state_dict: dict[str, Tensor], fraction: float) -> tuple[dict[str, Tensor], int, int]:
    frac = max(0.0, float(fraction))
    if frac <= 0.0:
        return {name: tensor.detach().to("cpu").contiguous() for name, tensor in state_dict.items()}, 0, 0
    pruned: dict[str, Tensor] = {}
    zeroed = 0
    considered = 0
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if t.is_floating_point() and t.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            threshold = torch.quantile(t.float().abs().flatten(), min(frac, 1.0))
            mask = t.float().abs() <= threshold
            zeroed += int(mask.sum())
            considered += int(mask.numel())
            t = t.clone()
            t[mask] = 0
        pruned[name] = t
    return pruned, zeroed, considered


def quantized_export_blob_for_limit(
    state_dict: dict[str, Tensor],
    code_bytes: int,
    artifact_size_limit_bytes: int,
    export_prune_fraction: float,
) -> tuple[bytes, dict[str, Tensor], dict[str, object], dict[str, float | int]]:
    fractions = []
    requested = max(0.0, float(export_prune_fraction))
    if requested > 0.0:
        fractions.append(requested)
    fractions.extend([0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30])
    seen = set()
    best: tuple[bytes, dict[str, Tensor], dict[str, object], dict[str, float | int]] | None = None
    for fraction in fractions:
        fraction = round(float(fraction), 6)
        if fraction in seen:
            continue
        seen.add(fraction)
        candidate_state, zeroed, considered = prune_state_dict_for_export(state_dict, fraction)
        quant_obj, quant_stats = quantize_state_dict_int8(candidate_state)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        blob = zlib.compress(quant_raw, level=9)
        metrics: dict[str, float | int] = {
            "export_prune_fraction": fraction,
            "export_pruned_values": zeroed,
            "export_prune_considered_values": considered,
            "int8_zlib_model_bytes": len(blob),
            "int8_zlib_total_bytes": len(blob) + int(code_bytes),
            "int8_raw_torch_bytes": len(quant_raw),
            "int8_payload_bytes": int(quant_stats["int8_payload_bytes"]),
            "int8_payload_ratio": (
                float(quant_stats["baseline_tensor_bytes"]) / max(float(quant_stats["int8_payload_bytes"]), 1.0)
            ),
            "under_artifact_size_limit": int(len(blob) + int(code_bytes) <= int(artifact_size_limit_bytes)),
        }
        best = (blob, candidate_state, quant_obj, metrics)
        if metrics["under_artifact_size_limit"]:
            return best
    assert best is not None
    return best


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_data_shard_prefix(file: Path, max_tokens: int) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    count = max(0, min(int(max_tokens), num_tokens))
    tokens_np = np.fromfile(file, dtype="<u2", count=count, offset=header_bytes)
    if tokens_np.size != count:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def build_bigram_logit_bias_from_tokens(
    tokens: Tensor,
    vocab_size: int,
    alpha: float,
    strength: float,
) -> Tensor:
    if tokens.numel() < 2:
        return torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    ids = tokens.to(dtype=torch.int64).reshape(-1)
    prev = ids[:-1]
    nxt = ids[1:]
    valid = (prev >= 0) & (prev < vocab_size) & (nxt >= 0) & (nxt < vocab_size)
    if not bool(valid.any()):
        return torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    flat = prev[valid] * vocab_size + nxt[valid]
    counts = torch.bincount(flat, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).to(torch.float32)
    counts.add_(float(alpha))
    log_probs = counts.log() - counts.sum(dim=1, keepdim=True).log()
    log_probs = log_probs - log_probs.mean(dim=1, keepdim=True)
    return log_probs.mul(float(strength)).to(dtype=torch.float32)


def build_bigram_logit_bias_from_shards(
    pattern: str,
    vocab_size: int,
    max_tokens: int,
    alpha: float,
    strength: float,
    log0,
) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    remaining: int | None = None if int(max_tokens) <= 0 else int(max_tokens)
    counts = torch.full((vocab_size, vocab_size), float(alpha), dtype=torch.float32)
    total_tokens = 0
    previous_tail: Tensor | None = None
    for file in files:
        if remaining == 0:
            break
        take = remaining if remaining is not None else 2**63 - 1
        tokens = load_data_shard_prefix(file, take)
        if tokens.numel() == 0:
            continue
        read_count = int(tokens.numel())
        if previous_tail is not None:
            tokens = torch.cat((previous_tail, tokens))
        ids = tokens.to(dtype=torch.int64).reshape(-1)
        prev = ids[:-1]
        nxt = ids[1:]
        valid = (prev >= 0) & (prev < vocab_size) & (nxt >= 0) & (nxt < vocab_size)
        if bool(valid.any()):
            flat = prev[valid] * vocab_size + nxt[valid]
            counts.add_(torch.bincount(flat, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).to(torch.float32))
        previous_tail = ids[-1:].to(dtype=torch.uint16)
        total_tokens += read_count
        if remaining is not None:
            remaining = max(0, remaining - read_count)
    log_probs = counts.log() - counts.sum(dim=1, keepdim=True).log()
    log_probs = log_probs - log_probs.mean(dim=1, keepdim=True)
    bias = log_probs.mul(float(strength)).to(dtype=torch.float32)
    log0(
        "bigram_bias_init_from_data:"
        f"tokens:{total_tokens} files:{len(files)} alpha:{alpha} strength:{strength} "
        f"bias_min:{float(bias.min()):.4f} bias_max:{float(bias.max()):.4f}"
    )
    return bias


def hash_ngram_context_ids(input_ids: Tensor, num_buckets: int, order: int) -> Tensor:
    if num_buckets <= 0:
        raise ValueError("HASH_NGRAM_BIAS_BUCKETS must be positive")
    context_order = max(1, int(order) - 1)
    ids = input_ids.to(dtype=torch.int64)
    hashed = torch.zeros_like(ids, dtype=torch.int64)
    for offset in range(context_order):
        shifted = torch.zeros_like(ids, dtype=torch.int64)
        if offset == 0:
            shifted = ids
        elif ids.size(-1) > offset:
            shifted[..., offset:] = ids[..., :-offset]
        hashed = (hashed * 1_000_003 + shifted + 104_729 + offset * 9_176) % int(num_buckets)
    return hashed


def build_hash_ngram_logit_bias_from_shards(
    pattern: str,
    vocab_size: int,
    num_buckets: int,
    order: int,
    max_tokens: int,
    alpha: float,
    strength: float,
    log0,
) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    remaining: int | None = None if int(max_tokens) <= 0 else int(max_tokens)
    counts = torch.full((num_buckets, vocab_size), float(alpha), dtype=torch.float32)
    tail = torch.empty((0,), dtype=torch.int64)
    total_tokens = 0
    for file in files:
        if remaining == 0:
            break
        take = remaining if remaining is not None else 2**63 - 1
        tokens = load_data_shard_prefix(file, take)
        if tokens.numel() == 0:
            continue
        read_count = int(tokens.numel())
        ids = torch.cat((tail, tokens.to(dtype=torch.int64).reshape(-1))) if tail.numel() else tokens.to(dtype=torch.int64).reshape(-1)
        if ids.numel() >= 2:
            context = ids[:-1].unsqueeze(0)
            target = ids[1:]
            buckets = hash_ngram_context_ids(context, num_buckets, order).reshape(-1)
            valid = (target >= 0) & (target < vocab_size)
            flat = buckets[valid] * vocab_size + target[valid]
            counts.add_(torch.bincount(flat, minlength=num_buckets * vocab_size).reshape(num_buckets, vocab_size).to(torch.float32))
        keep = max(0, int(order) - 1)
        tail = ids[-keep:].clone() if keep > 0 else torch.empty((0,), dtype=torch.int64)
        total_tokens += read_count
        if remaining is not None:
            remaining = max(0, remaining - read_count)
    log_probs = counts.log() - counts.sum(dim=1, keepdim=True).log()
    log_probs = log_probs - log_probs.mean(dim=1, keepdim=True)
    bias = log_probs.mul(float(strength)).to(dtype=torch.float32)
    log0(
        "hash_ngram_bias_init_from_data:"
        f"tokens:{total_tokens} files:{len(files)} buckets:{num_buckets} order:{order} "
        f"alpha:{alpha} strength:{strength} bias_min:{float(bias.min()):.4f} bias_max:{float(bias.max()):.4f}"
    )
    return bias


def initialize_bigram_bias_from_data_(model, args, log0, *, reason: str) -> None:
    if model.prev_token_bias is None:
        raise RuntimeError("BIGRAM_BIAS_INIT_FROM_DATA requires BIGRAM_BIAS=1")
    with torch.no_grad():
        bias = build_bigram_logit_bias_from_shards(
            args.train_files,
            args.vocab_size,
            args.bigram_bias_init_tokens,
            args.bigram_bias_init_alpha,
            args.bigram_bias_init_strength,
            log0,
        )
        target = model.prev_token_bias.weight
        target.copy_(bias.to(device=target.device, dtype=target.dtype))
    log0(
        "bigram_bias_init_from_data_applied:"
        f"reason:{reason} tokens:{args.bigram_bias_init_tokens} "
        f"alpha:{args.bigram_bias_init_alpha} strength:{args.bigram_bias_init_strength}"
    )


def initialize_hash_ngram_bias_from_data_(model, args, log0, *, reason: str) -> None:
    if model.hash_ngram_bias is None:
        raise RuntimeError("HASH_NGRAM_BIAS_INIT_FROM_DATA requires HASH_NGRAM_BIAS=1")
    with torch.no_grad():
        bias = build_hash_ngram_logit_bias_from_shards(
            args.train_files,
            args.vocab_size,
            args.hash_ngram_bias_buckets,
            args.hash_ngram_bias_order,
            args.hash_ngram_bias_init_tokens,
            args.hash_ngram_bias_init_alpha,
            args.hash_ngram_bias_init_strength,
            log0,
        )
        target = model.hash_ngram_bias.weight
        target.copy_(bias.to(device=target.device, dtype=target.dtype))
    log0(
        "hash_ngram_bias_init_from_data_applied:"
        f"reason:{reason} tokens:{args.hash_ngram_bias_init_tokens} "
        f"buckets:{args.hash_ngram_bias_buckets} order:{args.hash_ngram_bias_order} "
        f"alpha:{args.hash_ngram_bias_init_alpha} strength:{args.hash_ngram_bias_init_strength}"
    )


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def state_dict(self) -> dict[str, int]:
        return {"file_idx": self.file_idx, "pos": self.pos}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.file_idx = int(state["file_idx"]) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = int(state["pos"])

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def state_dict(self) -> dict[str, object]:
        return {"stream": self.stream.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        stream_state = state.get("stream")
        if not isinstance(stream_state, dict):
            raise ValueError("Loader checkpoint is missing stream state")
        self.stream.load_state_dict(stream_state)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        # Initial validation runs under torch.inference_mode().  In PyTorch
        # 2.8, tensors created there cannot later be saved for backward, so
        # return normal tensors even when the cache was populated during eval.
        return self._cos_cached.to(dtype=dtype).clone(), self._sin_cached.to(dtype=dtype).clone()


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def hadamard_orthogonal_transform(x: Tensor) -> Tensor:
    dim = x.size(-1)
    y = x
    h = 1
    while h < dim:
        y = y.reshape(*y.shape[:-1], dim // (h * 2), 2, h)
        left = y[..., 0, :]
        right = y[..., 1, :]
        y = torch.stack((left + right, left - right), dim=-2).reshape(*x.shape[:-1], dim)
        h *= 2
    return y * (float(dim) ** -0.5)


def recursive_polar_angle_quant_dequant(x: Tensor, bits: int) -> Tensor:
    if bits <= 0 or x.size(-1) < 2 or (x.size(-1) & (x.size(-1) - 1)):
        return x
    signs = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    current = x.abs()
    angles: list[Tensor] = []
    levels = float((1 << int(bits)) - 1)
    half_pi = math.pi / 2.0
    while current.size(-1) > 1:
        pair_dim = current.size(-1)
        pairs = current.reshape(*current.shape[:-1], pair_dim // 2, 2)
        left = pairs[..., 0]
        right = pairs[..., 1]
        radius = torch.sqrt(left.square() + right.square()).clamp_min(1e-12)
        angle = torch.atan2(right, left).clamp(0.0, half_pi)
        angles.append(torch.round(angle / half_pi * levels) / levels * half_pi)
        current = radius
    for angle in reversed(angles):
        left = current * torch.cos(angle)
        right = current * torch.sin(angle)
        current = torch.stack((left, right), dim=-1).flatten(-2)
    return current * signs


def polarquant_kv_perturb(x: Tensor, signs: Tensor, bits: int) -> Tensor:
    if bits <= 0 or x.size(-1) < 2 or (x.size(-1) & (x.size(-1) - 1)):
        return x
    original_dtype = x.dtype
    sign_view = signs.view(*([1] * (x.ndim - 1)), x.size(-1))
    signed = x.float() * sign_view
    preconditioned = hadamard_orthogonal_transform(signed)
    quantized = recursive_polar_angle_quant_dequant(preconditioned, bits)
    restored = hadamard_orthogonal_transform(quantized) * sign_view
    return restored.to(dtype=original_dtype)


def sampled_polarquant_kv_perturb(x: Tensor, signs: Tensor, bits: int, max_tokens: int) -> Tensor:
    if max_tokens <= 0 or x.size(-2) <= max_tokens:
        return polarquant_kv_perturb(x, signs, bits)
    stride = max(1, math.ceil(float(x.size(-2)) / float(max_tokens)))
    index = torch.arange(0, x.size(-2), stride, device=x.device)[:max_tokens]
    sample = x.index_select(-2, index)
    perturbed_sample = polarquant_kv_perturb(sample, signs, bits)
    out = x.clone()
    return out.index_copy(-2, index, perturbed_sample)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        polarquant_kv_bits: int = 0,
        polarquant_train: bool = False,
        polarquant_train_sample_tokens: int = 0,
        polarquant_eval_sample_tokens: int = 0,
        polarquant_seed: int = 271828,
        layer_idx: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.polarquant_kv_bits = int(polarquant_kv_bits)
        self.polarquant_train = bool(polarquant_train)
        self.polarquant_train_sample_tokens = int(polarquant_train_sample_tokens)
        self.polarquant_eval_sample_tokens = int(polarquant_eval_sample_tokens)
        self.register_buffer("polarquant_train_scale", torch.ones((), dtype=torch.float32), persistent=False)
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(polarquant_seed) + 104729 * int(layer_idx))
        signs = torch.where(
            torch.rand(self.head_dim, generator=gen) < 0.5,
            torch.full((self.head_dim,), -1.0),
            torch.ones(self.head_dim),
        )
        self.register_buffer("polarquant_signs", signs, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.polarquant_kv_bits > 0 and (self.polarquant_train or not self.training):
            sample_tokens = self.polarquant_train_sample_tokens if self.training else self.polarquant_eval_sample_tokens
            k = sampled_polarquant_kv_perturb(
                k,
                self.polarquant_signs.to(device=k.device),
                self.polarquant_kv_bits,
                sample_tokens,
            )
            v = sampled_polarquant_kv_perturb(
                v,
                self.polarquant_signs.to(device=v.device),
                self.polarquant_kv_bits,
                sample_tokens,
            )
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        polarquant_kv_bits: int = 0,
        polarquant_train: bool = False,
        polarquant_train_sample_tokens: int = 0,
        polarquant_eval_sample_tokens: int = 0,
        polarquant_seed: int = 271828,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            polarquant_kv_bits=polarquant_kv_bits,
            polarquant_train=polarquant_train,
            polarquant_train_sample_tokens=polarquant_train_sample_tokens,
            polarquant_eval_sample_tokens=polarquant_eval_sample_tokens,
            polarquant_seed=polarquant_seed,
            layer_idx=layer_idx,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
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
        bigram_bias: bool = False,
        bigram_bias_init_std: float = 0.0,
        bigram_bias_scale: float = 1.0,
        hash_ngram_bias: bool = False,
        hash_ngram_bias_buckets: int = 2048,
        hash_ngram_bias_order: int = 3,
        hash_ngram_bias_init_std: float = 0.0,
        hash_ngram_bias_scale: float = 1.0,
        polarquant_kv_bits: int = 0,
        polarquant_train: bool = False,
        polarquant_train_sample_tokens: int = 0,
        polarquant_eval_sample_tokens: int = 0,
        polarquant_seed: int = 271828,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if hash_ngram_bias and hash_ngram_bias_order < 2:
            raise ValueError("HASH_NGRAM_BIAS_ORDER must be at least 2")
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.bigram_bias_init_std = bigram_bias_init_std
        self.bigram_bias_scale = bigram_bias_scale
        self.hash_ngram_bias_buckets = hash_ngram_bias_buckets
        self.hash_ngram_bias_order = hash_ngram_bias_order
        self.hash_ngram_bias_init_std = hash_ngram_bias_init_std
        self.hash_ngram_bias_scale = hash_ngram_bias_scale
        self.polarquant_kv_bits = int(polarquant_kv_bits)
        self.polarquant_train = bool(polarquant_train)
        self.polarquant_train_sample_tokens = int(polarquant_train_sample_tokens)
        self.polarquant_eval_sample_tokens = int(polarquant_eval_sample_tokens)
        self.polarquant_seed = int(polarquant_seed)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.prev_token_bias = nn.Embedding(vocab_size, vocab_size) if bigram_bias else None
        self.hash_ngram_bias = nn.Embedding(hash_ngram_bias_buckets, vocab_size) if hash_ngram_bias else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    polarquant_kv_bits=self.polarquant_kv_bits,
                    polarquant_train=self.polarquant_train,
                    polarquant_train_sample_tokens=self.polarquant_train_sample_tokens,
                    polarquant_eval_sample_tokens=self.polarquant_eval_sample_tokens,
                    polarquant_seed=self.polarquant_seed,
                    layer_idx=i,
                )
                for i in range(num_layers)
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
        if self.prev_token_bias is not None:
            if self.bigram_bias_init_std > 0.0:
                nn.init.normal_(self.prev_token_bias.weight, mean=0.0, std=self.bigram_bias_init_std)
            else:
                nn.init.zeros_(self.prev_token_bias.weight)
        if self.hash_ngram_bias is not None:
            if self.hash_ngram_bias_init_std > 0.0:
                nn.init.normal_(self.hash_ngram_bias.weight, mean=0.0, std=self.hash_ngram_bias_init_std)
            else:
                nn.init.zeros_(self.hash_ngram_bias.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        if self.prev_token_bias is not None:
            transition_bias = self.prev_token_bias(input_ids.reshape(-1)).to(dtype=logits_proj.dtype)
            logits_proj = logits_proj + transition_bias * self.bigram_bias_scale
        if self.hash_ngram_bias is not None:
            bucket_ids = hash_ngram_context_ids(
                input_ids,
                self.hash_ngram_bias_buckets,
                self.hash_ngram_bias_order,
            ).reshape(-1)
            memory_bias = self.hash_ngram_bias(bucket_ids).to(dtype=logits_proj.dtype)
            logits_proj = logits_proj + memory_bias * self.hash_ngram_bias_scale
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(
        f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1} "
        f"max_sequences:{args.val_max_sequences if args.val_max_sequences > 0 else 'full'}"
    )

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_bias=args.bigram_bias,
        bigram_bias_init_std=args.bigram_bias_init_std,
        bigram_bias_scale=args.bigram_bias_scale,
        hash_ngram_bias=args.hash_ngram_bias,
        hash_ngram_bias_buckets=args.hash_ngram_bias_buckets,
        hash_ngram_bias_order=args.hash_ngram_bias_order,
        hash_ngram_bias_init_std=args.hash_ngram_bias_init_std,
        hash_ngram_bias_scale=args.hash_ngram_bias_scale,
        polarquant_kv_bits=args.polarquant_kv_bits,
        polarquant_train=args.polarquant_train,
        polarquant_train_sample_tokens=args.polarquant_train_sample_tokens,
        polarquant_eval_sample_tokens=args.polarquant_eval_sample_tokens,
        polarquant_seed=args.polarquant_seed,
    )
    if args.bigram_bias and args.bigram_bias_init_from_data and not args.resume_checkpoint:
        initialize_bigram_bias_from_data_(base_model, args, log0, reason="fresh_init")
    if args.hash_ngram_bias and args.hash_ngram_bias_init_from_data and not args.resume_checkpoint:
        initialize_hash_ngram_bias_from_data_(base_model, args, log0, reason="fresh_init")
    base_model = base_model.to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.prev_token_bias is not None:
        optimizer_bigram = torch.optim.Adam(
            [{"params": [base_model.prev_token_bias.weight], "lr": args.bigram_bias_lr, "base_lr": args.bigram_bias_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_bigram)
    if base_model.hash_ngram_bias is not None:
        optimizer_hash_ngram = torch.optim.Adam(
            [{"params": [base_model.hash_ngram_bias.weight], "lr": args.hash_ngram_bias_lr, "base_lr": args.hash_ngram_bias_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_hash_ngram)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"polarquant_kv:bits:{args.polarquant_kv_bits} train:{int(args.polarquant_train)} "
        f"train_sample_tokens:{args.polarquant_train_sample_tokens} "
        f"eval_sample_tokens:{args.polarquant_eval_sample_tokens} seed:{args.polarquant_seed}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"bigram_bias:{int(args.bigram_bias)} bigram_bias_lr:{args.bigram_bias_lr} "
        f"bigram_bias_scale:{args.bigram_bias_scale}"
    )
    log0(
        f"hash_ngram_bias:{int(args.hash_ngram_bias)} buckets:{args.hash_ngram_bias_buckets} "
        f"order:{args.hash_ngram_bias_order} hash_ngram_bias_lr:{args.hash_ngram_bias_lr} "
        f"hash_ngram_bias_scale:{args.hash_ngram_bias_scale}"
    )
    log0(
        "advanced_losses:"
        f"scale:{args.advanced_loss_scale} graphcg:{args.graphcg_loss_weight} "
        f"toric_tropical:{args.toric_tropical_loss_weight} slepian:{args.slepian_loss_weight} "
        f"koszul_bgg:{args.koszul_bgg_loss_weight} analogy:{args.analogy_loss_weight} "
        f"sample_tokens:{args.advanced_loss_sample_tokens} fan_bins:{args.toric_tropical_fan_bins} "
        f"log_only:{int(args.advanced_loss_log_only)} start_step:{args.advanced_loss_start_step} "
        f"end_step:{args.advanced_loss_end_step} every:{args.advanced_loss_every} "
        f"warmup_steps:{args.advanced_loss_warmup_steps} "
        f"min_best_val_bpb:{args.advanced_loss_min_best_val_bpb} "
        f"max_ce_ratio:{args.advanced_loss_max_ce_ratio}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    if args.checkpoint_every > 0:
        log0(f"checkpointing:every:{args.checkpoint_every} dir:{args.checkpoint_dir}")
    if args.resume_checkpoint:
        log0(f"resume_checkpoint:{args.resume_checkpoint}")
        log0(
            "resume_controls:"
            f"reset_optimizer:{int(args.reset_optimizer_on_resume)} "
            f"reset_rng:{int(args.reset_rng_on_resume)} "
            f"reset_loader:{int(args.reset_loader_on_resume)}"
        )
    log0(f"seed:{args.seed}")

    wandb_run = None

    def wandb_log(metrics: dict[str, float | int], step_for_wandb: int) -> None:
        nonlocal wandb_run
        if wandb_run is None:
            return
        try:
            metrics.setdefault("trainer/step", step_for_wandb)
            wandb_run.log(organize_wandb_payload(metrics))
        except Exception as exc:
            log0(f"wandb:log_disabled_after_error:{type(exc).__name__}")
            wandb_run = None

    if master_process and args.wandb_enabled:
        wandb_api_key = find_wandb_api_key(args.wandb_key_path)
        if wandb_api_key:
            os.environ.setdefault("WANDB_API_KEY", wandb_api_key)
        elif args.wandb_mode == "online":
            log0("wandb:disabled reason:missing_api_key")
        if wandb_api_key or args.wandb_mode != "online":
            try:
                import wandb

                if wandb_api_key:
                    wandb.login(key=wandb_api_key, relogin=False)
                wandb_config = {
                    name: getattr(args, name)
                    for name in dir(args)
                    if not name.startswith("_")
                    and isinstance(getattr(args, name), (str, int, float, bool))
                    and "key" not in name.lower()
                }
                wandb_config.update(
                    {
                        "model_params": n_params,
                        "world_size": world_size,
                        "grad_accum_steps": grad_accum_steps,
                        "parameter_golf_record": "track_10min_16mb/2026-03-19_TrainingOptSeq4096",
                    }
                )
                wandb_run = wandb.init(
                    entity=args.wandb_entity or None,
                    project=args.wandb_project,
                    id=args.wandb_run_id,
                    name=args.wandb_run_name,
                    resume=args.wandb_resume,
                    mode=args.wandb_mode,
                    config=wandb_config,
                )
                configure_wandb_metrics(wandb)
                log0(
                    f"wandb:enabled entity:{args.wandb_entity} "
                    f"project:{args.wandb_project} run:{args.wandb_run_id}"
                )
            except Exception as exc:
                log0(f"wandb:disabled reason:{type(exc).__name__}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def save_training_checkpoint(
        step: int,
        training_time_ms: float,
        *,
        force: bool = False,
        reason: str = "scheduled",
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        if not master_process or (args.checkpoint_every <= 0 and not force):
            return
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{args.run_id}_step_{step:06d}.pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        compact_config = {
            "vocab_size": args.vocab_size,
            "num_layers": args.num_layers,
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "mlp_mult": args.mlp_mult,
            "tie_embeddings": args.tie_embeddings,
            "bigram_bias": args.bigram_bias,
            "hash_ngram_bias": args.hash_ngram_bias,
            "hash_ngram_bias_buckets": args.hash_ngram_bias_buckets,
            "hash_ngram_bias_order": args.hash_ngram_bias_order,
            "logit_softcap": args.logit_softcap,
            "rope_base": args.rope_base,
            "qk_gain_init": args.qk_gain_init,
            "bigram_bias_scale": args.bigram_bias_scale,
            "hash_ngram_bias_scale": args.hash_ngram_bias_scale,
            "polarquant_kv_bits": args.polarquant_kv_bits,
            "polarquant_train": args.polarquant_train,
            "polarquant_train_sample_tokens": args.polarquant_train_sample_tokens,
            "polarquant_eval_sample_tokens": args.polarquant_eval_sample_tokens,
            "polarquant_seed": args.polarquant_seed,
            "polarquant_train_start_step": args.polarquant_train_start_step,
            "polarquant_train_warmup_steps": args.polarquant_train_warmup_steps,
        }
        payload = {
            "step": step,
            "training_time_ms": training_time_ms,
            "model": base_model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "loader": train_loader.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "best_val_bpb": best_val_bpb,
            "initial_val_bpb": initial_val_bpb,
            "config": compact_config,
        }
        config_sidecar = checkpoint_dir / "run_config.json"
        config_tmp = config_sidecar.with_suffix(config_sidecar.suffix + ".tmp")
        config_tmp.write_text(json.dumps(compact_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(config_tmp, config_sidecar)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        checkpoint_bytes = path.stat().st_size
        latest = checkpoint_dir / f"{args.run_id}_latest.pt"
        latest_tmp = latest.with_suffix(latest.suffix + ".tmp")
        if latest_tmp.exists():
            latest_tmp.unlink()
        try:
            os.link(path, latest_tmp)
            os.replace(latest_tmp, latest)
        except OSError:
            torch.save(payload, latest_tmp)
            os.replace(latest_tmp, latest)
        log0(f"checkpoint_saved:{path} bytes:{checkpoint_bytes} reason:{reason}")
        metrics = {
            "trainer/step": step,
            "checkpoint/step": step,
            "checkpoint/bytes": checkpoint_bytes,
            "checkpoint/training_time_ms": training_time_ms,
            "checkpoint/reason_low_train_bpb": float(reason == "low_train_bpb"),
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        wandb_log(metrics, step)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    step = 0
    best_val_bpb = math.inf
    initial_val_bpb: float | None = None
    low_train_bpb_checkpoint_count = 0
    last_low_train_bpb_checkpoint_step = -1_000_000_000
    if args.resume_checkpoint:
        if world_size != 1:
            raise ValueError("RESUME_CHECKPOINT currently supports WORLD_SIZE=1")
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        allow_new_bias_params = args.bigram_bias or args.hash_ngram_bias
        load_result = base_model.load_state_dict(checkpoint["model"], strict=not allow_new_bias_params)
        if allow_new_bias_params:
            missing = sorted(load_result.missing_keys)
            unexpected = sorted(load_result.unexpected_keys)
            log0(f"resume_model_load:missing:{missing} unexpected:{unexpected}")
            allowed_missing = set()
            if args.bigram_bias:
                allowed_missing.add("prev_token_bias.weight")
            if args.hash_ngram_bias:
                allowed_missing.add("hash_ngram_bias.weight")
            disallowed_missing = [key for key in missing if key not in allowed_missing]
            if disallowed_missing or unexpected:
                raise RuntimeError(
                    f"Unexpected checkpoint key mismatch with optional bias params: "
                    f"missing={missing} unexpected={unexpected}"
                )
        if args.bigram_bias and args.bigram_bias_init_from_data:
            initialize_bigram_bias_from_data_(base_model, args, log0, reason="after_resume_load")
        if args.hash_ngram_bias and args.hash_ngram_bias_init_from_data:
            initialize_hash_ngram_bias_from_data_(base_model, args, log0, reason="after_resume_load")
        optimizer_states = checkpoint["optimizers"]
        if args.reset_optimizer_on_resume:
            log0("resume_optimizer_reset:1")
        elif len(optimizer_states) != len(optimizers):
            log0(
                "resume_optimizer_reset:1 "
                f"reason:optimizer_count_mismatch checkpoint:{len(optimizer_states)} current:{len(optimizers)}"
            )
        else:
            for opt, state in zip(optimizers, optimizer_states, strict=True):
                opt.load_state_dict(state)
        if args.reset_loader_on_resume:
            log0("resume_loader_reset:1")
        else:
            train_loader.load_state_dict(checkpoint["loader"])
        training_time_ms = float(checkpoint.get("training_time_ms", 0.0))
        step = int(checkpoint["step"])
        best_val_bpb = float(checkpoint.get("best_val_bpb", math.inf))
        checkpoint_initial_val_bpb = checkpoint.get("initial_val_bpb")
        initial_val_bpb = None if checkpoint_initial_val_bpb is None else float(checkpoint_initial_val_bpb)
        if args.reset_rng_on_resume:
            log0("resume_rng_reset:1")
        elif "rng_cpu" in checkpoint:
            torch.set_rng_state(checkpoint["rng_cpu"])
        if not args.reset_rng_on_resume and "rng_cuda" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        log0(f"checkpoint_resumed:{args.resume_checkpoint} step:{step} train_time:{training_time_ms:.0f}ms")
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if initial_val_bpb is None:
                initial_val_bpb = val_bpb
            best_val_bpb = min(best_val_bpb, val_bpb)
            target_gap = val_bpb - args.target_bpb
            step_avg_ms = training_time_ms / max(step, 1)
            wandb_log(
                {
                    "trainer/step": step,
                    "trainer/total_steps": args.iterations,
                    "trainer/progress": step / max(args.iterations, 1),
                    "progress/step": step,
                    "progress/total_steps": args.iterations,
                    "progress/fraction": step / max(args.iterations, 1),
                    "progress/remaining_steps": max(args.iterations - step, 0),
                    "val/loss": val_loss,
                    "val/bpb": val_bpb,
                    "val/perplexity": math.exp(min(val_loss, 20.0)),
                    "val_bpb": val_bpb,
                    "bpb": val_bpb,
                    "openai_parameter_golf/bpb": val_bpb,
                    "fineweb/val_loss": val_loss,
                    "fineweb/val_bpb": val_bpb,
                    "fineweb/best_val_bpb": best_val_bpb,
                    "fineweb/target_bpb": args.target_bpb,
                    "fineweb/target_gap_bpb": target_gap,
                    "fineweb/target_reached": float(best_val_bpb <= args.target_bpb),
                    "bpb/val": val_bpb,
                    "bpb/best": best_val_bpb,
                    "bpb/target": args.target_bpb,
                    "bpb/gap_to_target": target_gap,
                    "bpb/improvement_from_initial": (
                        0.0 if initial_val_bpb is None else initial_val_bpb - val_bpb
                    ),
                    "bpb/target_reached": float(best_val_bpb <= args.target_bpb),
                    "time/train_ms": training_time_ms,
                    "time/train_seconds": training_time_ms / 1000.0,
                    "time/step_avg_ms": step_avg_ms,
                    "time/steps_per_second": 1000.0 / max(step_avg_ms, 1e-9),
                },
                step,
            )
            checkpoint_due = args.checkpoint_every > 0 and step > 0 and (last_step or step % args.checkpoint_every == 0)
            if checkpoint_due:
                save_training_checkpoint(step, training_time_ms)
                if distributed:
                    dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        polarquant_train_scale = polarquant_train_runtime_scale(args, step)
        set_polarquant_train_scale_(base_model, polarquant_train_scale)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_aux_loss = torch.zeros((), device=device)
        advanced_metric_sums: dict[str, Tensor] = {}
        advanced_runtime_scale_sum = torch.zeros((), device=device)
        train_token_count = torch.zeros((), device=device, dtype=torch.float64)
        train_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        nonfinite_microbatch_skip_count = torch.zeros((), device=device, dtype=torch.float32)
        nonfinite_ce_loss_count = torch.zeros((), device=device, dtype=torch.float32)
        nonfinite_aux_loss_count = torch.zeros((), device=device, dtype=torch.float32)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                ce_loss = model(x, y)
            runtime_scale = advanced_loss_runtime_scale(args, step, best_val_bpb)
            aux_loss, aux_metrics = advanced_embedding_losses(base_model, args, x, runtime_scale=runtime_scale)
            ce_loss_is_finite = torch.isfinite(ce_loss.detach())
            if not bool(ce_loss_is_finite.item()):
                nonfinite_ce_loss_count += 1.0
                aux_metrics["advanced/ce_loss_nonfinite"] = torch.ones((), device=device, dtype=torch.float32)
            else:
                aux_metrics["advanced/ce_loss_nonfinite"] = torch.zeros((), device=device, dtype=torch.float32)
            if not torch.isfinite(aux_loss.detach()):
                nonfinite_aux_loss_count += 1.0
                aux_metrics["advanced/aux_loss_nonfinite_suppressed"] = torch.ones((), device=device, dtype=torch.float32)
                aux_loss = torch.zeros_like(ce_loss)
            else:
                aux_metrics["advanced/aux_loss_nonfinite_suppressed"] = torch.zeros((), device=device, dtype=torch.float32)
            if runtime_scale > 0.0 and args.advanced_loss_max_ce_ratio > 0.0:
                max_aux_loss = ce_loss.detach() * args.advanced_loss_max_ce_ratio
                aux_loss = torch.minimum(aux_loss, max_aux_loss)
                aux_metrics["advanced/aux_loss_clamp_ce_ratio"] = torch.as_tensor(
                    args.advanced_loss_max_ce_ratio,
                    device=device,
                    dtype=torch.float32,
                )
                aux_metrics["advanced/aux_loss_clamped"] = (
                    aux_loss.detach() < aux_metrics.get("advanced/aux_loss", aux_loss.detach())
                ).to(dtype=torch.float32)
                aux_metrics["advanced/aux_loss"] = aux_loss.detach()
            loss = ce_loss + aux_loss
            loss_is_finite = torch.isfinite(loss.detach())
            train_loss += torch.nan_to_num(ce_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            train_aux_loss += torch.nan_to_num(aux_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            advanced_runtime_scale_sum += torch.as_tensor(runtime_scale, device=device, dtype=torch.float32)
            for key, value in aux_metrics.items():
                advanced_metric_sums[key] = advanced_metric_sums.get(key, torch.zeros((), device=device)) + value.detach()
            if not bool(loss_is_finite.item()):
                nonfinite_microbatch_skip_count += 1.0
                continue
            (loss * grad_scale).backward()
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            train_token_count += float(tgt_ids.numel())
            train_byte_count += token_bytes.to(torch.float64).sum()
        train_loss /= grad_accum_steps
        train_aux_loss /= grad_accum_steps
        advanced_metric_avgs = {key: value / grad_accum_steps for key, value in advanced_metric_sums.items()}
        advanced_runtime_scale_avg = advanced_runtime_scale_sum / grad_accum_steps
        if distributed:
            dist.all_reduce(train_token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_byte_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_aux_loss, op=dist.ReduceOp.SUM)
            train_aux_loss /= world_size
            dist.all_reduce(advanced_runtime_scale_avg, op=dist.ReduceOp.SUM)
            advanced_runtime_scale_avg /= world_size
            for key in list(advanced_metric_avgs):
                dist.all_reduce(advanced_metric_avgs[key], op=dist.ReduceOp.SUM)
                advanced_metric_avgs[key] /= world_size

        train_bits_per_token = float(train_loss.item()) / math.log(2.0)
        train_tokens_per_byte = float(train_token_count.item()) / max(float(train_byte_count.item()), 1.0)
        train_bpb = train_bits_per_token * train_tokens_per_byte

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        grad_norm_value = 0.0
        nonfinite_update_skip = 0.0
        if (
            not torch.isfinite(train_loss.detach())
            or not torch.isfinite(train_aux_loss.detach())
            or not math.isfinite(train_bpb)
            or float(nonfinite_microbatch_skip_count.item()) > 0.0
        ):
            nonfinite_update_skip = 1.0
        else:
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum

            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["base_lr"] * scale

            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
                grad_norm_value = float(
                    torch.nan_to_num(
                        grad_norm.detach().float(),
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=float("inf"),
                    ).item()
                )
                if not math.isfinite(grad_norm_value):
                    nonfinite_update_skip = 1.0
            if nonfinite_update_skip < 0.5:
                for opt in optimizers:
                    opt.step()
        audit_params = args.nonfinite_audit_every > 0 and (
            step <= 10 or step % args.nonfinite_audit_every == 0 or nonfinite_update_skip > 0.5
        )
        nonfinite_param_count = 0.0
        nonfinite_grad_count = 0.0
        max_abs_param = 0.0
        max_abs_grad = 0.0
        if audit_params:
            with torch.no_grad():
                for param in base_model.parameters():
                    data = param.detach()
                    finite_data = torch.isfinite(data)
                    nonfinite_param_count += float((~finite_data).sum().item())
                    if finite_data.any():
                        max_abs_param = max(max_abs_param, float(data[finite_data].abs().max().item()))
                    if param.grad is not None:
                        grad = param.grad.detach()
                        finite_grad = torch.isfinite(grad)
                        nonfinite_grad_count += float((~finite_grad).sum().item())
                        if finite_grad.any():
                            max_abs_grad = max(max_abs_grad, float(grad[finite_grad].abs().max().item()))
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        step_avg_ms = approx_training_time_ms / step
        tokens_per_second = args.train_batch_tokens * 1000.0 / max(step_avg_ms, 1e-9)
        if args.wandb_log_every > 0 and (step <= 10 or step % args.wandb_log_every == 0 or stop_after_step is not None):
            train_metrics = {
                "trainer/step": step,
                "trainer/total_steps": args.iterations,
                "trainer/progress": step / max(args.iterations, 1),
                "progress/step": step,
                "progress/total_steps": args.iterations,
                "progress/fraction": step / max(args.iterations, 1),
                "progress/remaining_steps": max(args.iterations - step, 0),
                "train/loss": float(train_loss.item()),
                "train/total_loss": float((train_loss + train_aux_loss).item()),
                "train/advanced_aux_loss": float(train_aux_loss.item()),
                "advanced/runtime_scale_applied": float(advanced_runtime_scale_avg.item()),
                "advanced/backprop_enabled": float(advanced_runtime_scale_avg.item() > 0.0),
                "polarquant/train_scale": polarquant_train_scale,
                "polarquant/train_start_step": args.polarquant_train_start_step,
                "polarquant/train_warmup_steps": args.polarquant_train_warmup_steps,
                "train/perplexity": math.exp(min(float(train_loss.item()), 20.0)),
                "train/bpb": train_bpb,
                "train_bpb": train_bpb,
                "train/bits_per_token": train_bits_per_token,
                "train/tokens_per_byte": train_tokens_per_byte,
                "train/bytes_per_token": 1.0 / max(train_tokens_per_byte, 1.0e-12),
                "train/batch_tokens": float(train_token_count.item()),
                "train/batch_bytes": float(train_byte_count.item()),
                "fineweb/train_loss": float(train_loss.item()),
                "fineweb/train_bpb": train_bpb,
                "bpb/train": train_bpb,
                "openai_parameter_golf/train_bpb": train_bpb,
                "fineweb/train_time_ms": approx_training_time_ms,
                "time/train_ms": approx_training_time_ms,
                "time/train_seconds": approx_training_time_ms / 1000.0,
                "time/step_avg_ms": step_avg_ms,
                "time/steps_per_second": 1000.0 / max(step_avg_ms, 1e-9),
                "perf/train_tokens_per_second": tokens_per_second,
                "perf/train_tokens_per_step": args.train_batch_tokens,
                "optim/lr_scale": scale,
                "optim/muon_momentum": muon_momentum,
                "optim/token_lr": token_lr * scale,
                "optim/matrix_lr": args.matrix_lr * scale,
                "optim/scalar_lr": args.scalar_lr * scale,
                "optim/grad_norm": grad_norm_value,
                "train/nonfinite_update_skip": nonfinite_update_skip,
                "train/nonfinite_microbatch_skip_count": float(nonfinite_microbatch_skip_count.item()),
                "train/nonfinite_ce_loss_count": float(nonfinite_ce_loss_count.item()),
                "train/nonfinite_aux_loss_count": float(nonfinite_aux_loss_count.item()),
                "train/nonfinite_param_count": nonfinite_param_count,
                "train/nonfinite_grad_count": nonfinite_grad_count,
                "optim/max_abs_param": max_abs_param,
                "optim/max_abs_grad": max_abs_grad,
            }
            if base_model.prev_token_bias is not None:
                train_metrics["optim/bigram_bias_lr"] = args.bigram_bias_lr * scale
            if base_model.hash_ngram_bias is not None:
                train_metrics["optim/hash_ngram_bias_lr"] = args.hash_ngram_bias_lr * scale
                train_metrics["memory/hash_ngram_bias_buckets"] = args.hash_ngram_bias_buckets
                train_metrics["memory/hash_ngram_bias_order"] = args.hash_ngram_bias_order
            if advanced_metric_avgs:
                train_metrics.update({key: float(value.item()) for key, value in advanced_metric_avgs.items()})
            if device.type == "cuda":
                train_metrics.update(
                    {
                        "system/vram_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
                        "system/vram_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
                    }
                )
            wandb_log(train_metrics, step)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"train_bpb:{train_bpb:.4f}"
            )

        low_train_bpb_triggered = (
            args.checkpoint_on_train_bpb_below > 0.0
            and train_bpb <= args.checkpoint_on_train_bpb_below
            and low_train_bpb_checkpoint_count < max(args.checkpoint_on_train_bpb_max, 0)
            and step - last_low_train_bpb_checkpoint_step
            >= max(args.checkpoint_on_train_bpb_cooldown_steps, 1)
        )
        if low_train_bpb_triggered:
            torch.cuda.synchronize()
            training_time_ms = approx_training_time_ms
            trigger_metrics = {
                "trigger/low_train_bpb": 1.0,
                "trigger/low_train_bpb_threshold": args.checkpoint_on_train_bpb_below,
                "trigger/train_bpb": train_bpb,
                "trigger/train_loss": float(train_loss.item()),
                "trigger/advanced_aux_loss": float(train_aux_loss.item()),
            }
            if args.val_on_train_bpb_checkpoint:
                val_loss, val_bpb = eval_val(
                    args,
                    model,
                    rank,
                    world_size,
                    device,
                    grad_accum_steps,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                if initial_val_bpb is None:
                    initial_val_bpb = val_bpb
                best_val_bpb = min(best_val_bpb, val_bpb)
                log0(
                    f"low_train_bpb_trigger_val step:{step}/{args.iterations} "
                    f"train_bpb:{train_bpb:.4f} threshold:{args.checkpoint_on_train_bpb_below:.4f} "
                    f"val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{training_time_ms:.0f}ms"
                )
                trigger_metrics.update(
                    {
                        "trigger/val_loss": val_loss,
                        "trigger/val_bpb": val_bpb,
                        "trigger/best_val_bpb": best_val_bpb,
                        "val/loss": val_loss,
                        "val/bpb": val_bpb,
                        "val_bpb": val_bpb,
                        "bpb": val_bpb,
                        "openai_parameter_golf/bpb": val_bpb,
                        "fineweb/val_bpb": val_bpb,
                        "fineweb/best_val_bpb": best_val_bpb,
                        "bpb/best": best_val_bpb,
                    }
                )
            log0(
                f"low_train_bpb_trigger_checkpoint step:{step}/{args.iterations} "
                f"train_bpb:{train_bpb:.4f} threshold:{args.checkpoint_on_train_bpb_below:.4f} "
                f"count:{low_train_bpb_checkpoint_count + 1}/{args.checkpoint_on_train_bpb_max}"
            )
            save_training_checkpoint(
                step,
                training_time_ms,
                force=True,
                reason="low_train_bpb",
                extra_metrics=trigger_metrics,
            )
            low_train_bpb_checkpoint_count += 1
            last_low_train_bpb_checkpoint_step = step
            if distributed:
                dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
        wandb_log(
            {
                "artifact/raw_model_bytes": model_bytes,
                "artifact/code_bytes": code_bytes,
                "artifact/raw_total_bytes": model_bytes + code_bytes,
            },
            step,
        )

    quant_blob, export_state, quant_obj, export_metrics = quantized_export_blob_for_limit(
        base_model.state_dict(),
        code_bytes=code_bytes,
        artifact_size_limit_bytes=args.artifact_size_limit_bytes,
        export_prune_fraction=args.export_prune_fraction,
    )
    quant_raw_bytes = int(export_metrics["int8_raw_torch_bytes"])
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        ratio = float(export_metrics["int8_payload_ratio"])
        total_bytes = quant_file_bytes + code_bytes
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{int(export_metrics['int8_payload_bytes'])} raw_torch:{quant_raw_bytes} "
            f"payload_ratio:{ratio:.2f}x export_prune:{float(export_metrics['export_prune_fraction']):.3f})"
        )
        log0(
            f"Total submission size int8+zlib: {total_bytes} bytes "
            f"limit:{args.artifact_size_limit_bytes} under_limit:{int(total_bytes <= args.artifact_size_limit_bytes)}"
        )
        wandb_log(
            {
                "artifact/int8_zlib_model_bytes": quant_file_bytes,
                "artifact/int8_zlib_total_bytes": total_bytes,
                "artifact/int8_payload_bytes": int(export_metrics["int8_payload_bytes"]),
                "artifact/int8_raw_torch_bytes": quant_raw_bytes,
                "artifact/int8_payload_ratio": ratio,
                "artifact/export_prune_fraction": float(export_metrics["export_prune_fraction"]),
                "artifact/export_pruned_values": int(export_metrics["export_pruned_values"]),
                "artifact/export_prune_considered_values": int(export_metrics["export_prune_considered_values"]),
                "artifact/size_limit_bytes": args.artifact_size_limit_bytes,
                "artifact/under_size_limit": float(total_bytes <= args.artifact_size_limit_bytes),
            },
            step,
        )

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    wandb_log(
        {
            "final/int8_zlib_roundtrip_loss": q_val_loss,
            "final/int8_zlib_roundtrip_bpb": q_val_bpb,
            "final/int8_zlib_roundtrip_eval_ms": 1000.0 * (time.perf_counter() - t_qeval),
        },
        step,
    )

    if distributed:
        dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
