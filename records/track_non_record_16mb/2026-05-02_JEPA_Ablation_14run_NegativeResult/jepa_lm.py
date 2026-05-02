"""JEPA-on-LM architecture for parameter-golf non-record / unlimited-compute track.

A standard parameter-golf BaselineGPT backbone (encoder-decoder skip, GQA,
augmentations, tied embeddings) drives the cross-entropy LM head and val_bpb.
On top of that, JEPA paths share a small predictor MLP:

  Path A — Hidden-state aux JEPA:
      For each non-final position t, predict the model's own final hidden
      state at position t + chunk (stop-grad target). Loss = MSE +
      VICReg variance regularization (+ optional off-diagonal covariance).

  Path B — Token-decoder JEPA:
      Project the predicted embedding through the tied LM head and apply CE
      against the actual token at position t + chunk.

  Injection (optional, JEPA_INJECTION=1):
      Project predicted latent through a zero-init linear and ADD to the
      hidden stream at chunk-positions before CE compute. JEPA actively
      contributes a feature, not just a regularizer. Inspired by jfprincz
      PR #832 (val_bpb 1.1903, beats baseline 1.2244 by 0.034).

Combined loss returned to the trainer:

      total = ce_main + alpha * (mse_aux + var_w * vicreg + covar_w * covar) + beta * ce_jepa

v2 changes vs v1 (informed by parameter-golf community PRs):

  - Defaults dropped 40x: alpha=0.005, beta=0.005 (was 0.2 / 0.05). Successful
    JEPA submissions in parameter-golf use lambda ~= 0.001-0.005, not 0.1+.
    "JEPA contributes ~0.1% of peak gradient signal" (PR #832).
  - Off-diagonal covariance penalty (V-JEPA style) opt-in via
    JEPA_COVAR_WEIGHT > 0. Prevents low-rank predictor collapse beyond what
    pure variance regularization catches (PR #1581 finding).
  - Predictor injection mode opt-in via JEPA_INJECTION=1. Predicted latents
    flow into the LM head as features (zero-init), not just as a side loss.

Setting JEPA_ALPHA=0 disables path A, JEPA_BETA=0 disables path B,
JEPA_INJECTION=0 disables injection. All three at default-off recovers
plain BaselineGPT numerics.

Env vars (read in the builder, not via Hyperparameters):

    JEPA_ALPHA           default 0.005   weight for hidden-state aux loss
    JEPA_BETA            default 0.005   weight for token-decoder loss
    JEPA_VAR_WEIGHT      default 0.1     VICReg variance-reg weight
    JEPA_COVAR_WEIGHT    default 0.0     off-diagonal covariance penalty (V-JEPA)
    JEPA_CHUNK           default 8       positions ahead to predict
    JEPA_PREDICTOR_DIM   default 64      bottleneck dim of predictor MLP
    JEPA_INJECTION       default 0       1 = inject predicted latent into hidden stream

The predictor and injection projection are zero-initialized on their output
layers, so JEPA paths start as a no-op and the trainer sees pure baseline
gradients at step 0.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.architectures.baseline import BaselineGPT
from crucible.models.registry import register_model, register_schema


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return default if val is None or val == "" else float(val)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return default if val is None or val == "" else int(val)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")


def _covariance_off_diag(z: Tensor) -> Tensor:
    """V-JEPA-style off-diagonal covariance penalty.

    Decorrelates feature dimensions by penalizing off-diagonal entries of the
    feature covariance matrix. Sums squared off-diagonals, normalized by D.
    Input z: [N, D]. Returns scalar.
    """
    z = z.float()
    n = max(z.shape[0] - 1, 1)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / n  # [D, D]
    d = cov.shape[0]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum() / d).clamp_min(0.0)


class JepaLM(BaselineGPT):
    """BaselineGPT backbone + JEPA aux head + optional injection."""

    def __init__(
        self,
        *,
        jepa_alpha: float = 0.005,
        jepa_beta: float = 0.005,
        jepa_var_weight: float = 0.1,
        jepa_covar_weight: float = 0.0,
        jepa_chunk: int = 8,
        jepa_predictor_dim: int = 64,
        jepa_injection: bool = False,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        if jepa_chunk < 1:
            raise ValueError(f"JEPA_CHUNK must be >= 1, got {jepa_chunk}")
        if jepa_predictor_dim < 1:
            raise ValueError(f"JEPA_PREDICTOR_DIM must be >= 1, got {jepa_predictor_dim}")
        self.jepa_alpha = float(jepa_alpha)
        self.jepa_beta = float(jepa_beta)
        self.jepa_var_weight = float(jepa_var_weight)
        self.jepa_covar_weight = float(jepa_covar_weight)
        self.jepa_chunk = int(jepa_chunk)
        self.jepa_injection = bool(jepa_injection)
        d = base_kwargs["model_dim"]
        self.jepa_predictor = nn.Sequential(
            nn.Linear(d, jepa_predictor_dim, bias=False),
            nn.GELU(),
            nn.Linear(jepa_predictor_dim, d, bias=False),
        )
        # Zero-init the output projection so JEPA contributes nothing at step 0.
        nn.init.zeros_(self.jepa_predictor[2].weight)
        nn.init.normal_(
            self.jepa_predictor[0].weight,
            std=1.0 / math.sqrt(d),
        )
        # Optional injection projection: predicted latent -> residual stream
        # contribution at chunk-aligned positions. Zero-init keeps step-0
        # behavior identical to baseline.
        if self.jepa_injection:
            self.jepa_inject_proj = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.jepa_inject_proj.weight)
        else:
            self.jepa_inject_proj = None

    def _maybe_inject(self, h: Tensor, h_pred: Tensor) -> Tensor:
        """Add zero-init projected predicted latents into the hidden stream.

        h: [B, T, D] full hidden. h_pred: [B, T-chunk, D] predictions made at
        positions 0..T-chunk-1 of what positions chunk..T-1 will look like.
        We add the prediction at position t-chunk INTO h[t] for t >= chunk.
        Positions 0..chunk-1 receive no injection (no prediction available).
        """
        if self.jepa_inject_proj is None:
            return h
        chunk = self.jepa_chunk
        inject = self.jepa_inject_proj(h_pred)            # [B, T-chunk, D]
        # Pad zero on the left for positions 0..chunk-1
        b, _, d = h.shape
        zero_head = torch.zeros(b, chunk, d, dtype=h.dtype, device=h.device)
        full_inject = torch.cat([zero_head, inject], dim=1)  # [B, T, D]
        return h + full_inject

    def _components(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        lora: Any = None,
    ) -> dict[str, Tensor]:
        """Forward + per-component losses."""
        h = self.hidden(input_ids, lora=lora)
        chunk = self.jepa_chunk
        seq_len = h.size(1)
        do_jepa = (self.jepa_alpha > 0.0 or self.jepa_beta > 0.0 or self.jepa_injection) and seq_len > chunk

        if not do_jepa:
            ce_main = self.compute_loss(h, target_ids, lora=lora)
            return {"ce_loss": ce_main, "loss": ce_main}

        h_curr = h[:, :-chunk, :]                    # [B, T-chunk, D]
        h_target = h[:, chunk:, :].detach()          # stop-grad target
        h_pred = self.jepa_predictor(h_curr)         # [B, T-chunk, D]

        # Inject BEFORE computing main CE so injection helps the LM head.
        h_for_ce = self._maybe_inject(h, h_pred)
        ce_main = self.compute_loss(h_for_ce, target_ids, lora=lora)
        out: dict[str, Tensor] = {"ce_loss": ce_main}
        total = ce_main

        if self.jepa_alpha > 0.0:
            # Normalize before MSE so un-RMSNormed magnitudes don't dominate.
            h_pred_n = self.final_norm(h_pred)
            h_target_n = self.final_norm(h_target)
            mse_aux = F.mse_loss(h_pred_n, h_target_n)
            # VICReg variance hinge over the predictor's feature dimension.
            z_std = torch.sqrt(h_pred_n.float().var(dim=(0, 1)) + 1e-4)
            vicreg = torch.relu(1.0 - z_std).mean()
            jepa_aux = mse_aux + self.jepa_var_weight * vicreg
            # V-JEPA off-diagonal covariance penalty (anti-collapse beyond
            # variance reg). Opt-in via JEPA_COVAR_WEIGHT > 0.
            if self.jepa_covar_weight > 0.0:
                flat = h_pred_n.reshape(-1, h_pred_n.size(-1))
                covar = _covariance_off_diag(flat)
                jepa_aux = jepa_aux + self.jepa_covar_weight * covar
                out["jepa_covar"] = covar.detach()
            total = total + self.jepa_alpha * jepa_aux
            out["jepa_mse"] = mse_aux.detach()
            out["jepa_vicreg"] = vicreg.detach()

        if self.jepa_beta > 0.0:
            # Token-decoder JEPA: decode predicted embedding through tied LM head.
            target_chunk_ids = input_ids[:, chunk:]
            x = self.final_norm(h_pred)
            flat = x.reshape(-1, x.size(-1))
            logits_proj = (
                self.tied_logits(flat) if self.tie_embeddings else self.lm_head(flat)
            )
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            ce_jepa = F.cross_entropy(
                logits.float(),
                target_chunk_ids.reshape(-1),
                reduction="mean",
                ignore_index=-100,
            )
            total = total + self.jepa_beta * ce_jepa
            out["jepa_token_ce"] = ce_jepa.detach()

        out["loss"] = total
        return out

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        lora: Any = None,
    ) -> Tensor:  # type: ignore[override]
        return self._components(input_ids, target_ids, lora=lora)["loss"]

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        return self._components(
            batch["input_ids"],
            batch["target_ids"],
            lora=batch.get("lora"),
        )

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        # Validation reports val_bpb based on ce_loss only — JEPA aux is
        # training-time regularization. With injection enabled the predicted
        # latent IS part of the LM head input, so we keep that path live;
        # variance/MSE losses are skipped.
        h = self.hidden(batch["input_ids"], lora=batch.get("lora"))
        if self.jepa_injection and h.size(1) > self.jepa_chunk:
            h_curr = h[:, :-self.jepa_chunk, :]
            h_pred = self.jepa_predictor(h_curr)
            h = self._maybe_inject(h, h_pred)
        ce = self.compute_loss(h, batch["target_ids"], lora=batch.get("lora"))
        return {"loss": ce, "ce_loss": ce}


def _build_jepa_lm(args: Any) -> JepaLM:
    base_kwargs = dict(
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
        attention_variant=args.attention_variant,
        residual_variant=args.residual_variant,
        embed_bottleneck_dim=getattr(args, "embed_bottleneck_dim", 0),
        use_smear_gate=getattr(args, "smear_gate", False),
        use_bigram_hash=getattr(args, "bigram_hash", False),
        bigram_hash_buckets=getattr(args, "bigram_hash_buckets", 2048),
        bigram_hash_embed_dim=getattr(args, "bigram_hash_embed_dim", 128),
        ortho_init=getattr(args, "ortho_init", False),
        spectral_embed_init=getattr(args, "spectral_embed_init", False),
        use_conv_block=getattr(args, "conv_block", False),
        conv_kernel=getattr(args, "conv_kernel", 3),
        multiscale_window=getattr(args, "multiscale_window", 0),
        token_merge_layer=getattr(args, "token_merge_layer", 0),
        token_merge_threshold=getattr(args, "token_merge_threshold", 0.9),
        block_pattern=getattr(args, "block_pattern", ""),
        use_trigram_hash=getattr(args, "trigram_hash", False),
        trigram_hash_buckets=getattr(args, "trigram_hash_buckets", 4096),
        activation=getattr(args, "activation", "relu_sq"),
        use_moe=getattr(args, "use_moe", False),
        moe_num_experts=getattr(args, "moe_num_experts", 4),
        moe_top_k=getattr(args, "moe_top_k", 2),
    )
    return JepaLM(
        jepa_alpha=_env_float("JEPA_ALPHA", 0.005),
        jepa_beta=_env_float("JEPA_BETA", 0.005),
        jepa_var_weight=_env_float("JEPA_VAR_WEIGHT", 0.1),
        jepa_covar_weight=_env_float("JEPA_COVAR_WEIGHT", 0.0),
        jepa_chunk=_env_int("JEPA_CHUNK", 8),
        jepa_predictor_dim=_env_int("JEPA_PREDICTOR_DIM", 64),
        jepa_injection=_env_bool("JEPA_INJECTION", False),
        **base_kwargs,
    )


register_model("jepa_lm", _build_jepa_lm)
register_schema("jepa_lm", {
    # Inherits all baseline knobs (MODEL_DIM, NUM_LAYERS, ...) — those are
    # honored via the BaselineGPT constructor. Schema below documents the
    # JEPA-specific env vars introduced by this plugin.
    "JEPA_ALPHA": {"type": "float", "default": 0.005, "description": "Weight for hidden-state aux JEPA loss (MSE + VICReg + covar)"},
    "JEPA_BETA": {"type": "float", "default": 0.005, "description": "Weight for token-decoder JEPA cross-entropy loss"},
    "JEPA_VAR_WEIGHT": {"type": "float", "default": 0.1, "description": "VICReg variance-regularization weight"},
    "JEPA_COVAR_WEIGHT": {"type": "float", "default": 0.0, "description": "V-JEPA off-diagonal covariance penalty (0 = off)"},
    "JEPA_CHUNK": {"type": "int", "default": 8, "description": "Lookahead distance (positions) for JEPA prediction"},
    "JEPA_PREDICTOR_DIM": {"type": "int", "default": 64, "description": "Bottleneck dim of the JEPA predictor MLP"},
    "JEPA_INJECTION": {"type": "bool", "default": False, "description": "Inject predicted latent (zero-init) into hidden stream"},
})
