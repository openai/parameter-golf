"""
train_gpt_mod_act.py — Mixture of Depths + Adaptive Computation Time

Built on top of the parameter-golf baseline (train_gpt.py).
Key additions:
  1. MoDRouter: per-layer top-k token routing (straight-through gradient)
  2. ACTHalting: per-token learned halting probability over recurrence steps
  3. MoDBlock: wraps Block with routing + ACT halting
  4. MoDGPT: drop-in replacement for GPT with MoD+ACT

Environment variables (all optional, have defaults):
  MOD_ENABLED=1          — enable MoD routing (default: 1)
  MOD_CAPACITY=0.75      — fraction of tokens to route through each layer
  ACT_ENABLED=1          — enable ACT halting (default: 1)
  ACT_MAX_STEPS=3        — max recurrence steps
  ACT_PONDER_COST=0.01   — regularization weight for ponder cost
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
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

# ---- Import everything from the baseline ----
# We reuse all hyperparameters, quantization, data loading, and optimizer code.
# Only the model architecture changes.

from train_gpt import (
    Hyperparameters,
    Muon,
    zeropower_via_newtonschulz5,
    build_sentencepiece_luts,
    load_validation_tokens,
    eval_val,
    load_data_shard,
    TokenStream,
    DistributedTokenLoader,
    RMSNorm,
    CastedLinear,
    restore_low_dim_params_to_fp32,
    Rotary,
    apply_rotary_emb,
    CausalSelfAttention,
    MLP,
    Block,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    CONTROL_TENSOR_NAME_PATTERNS,
)

# -----------------------------
# MoD + ACT HYPERPARAMETERS
# -----------------------------

MOD_ENABLED = bool(int(os.environ.get("MOD_ENABLED", "1")))
MOD_CAPACITY = float(os.environ.get("MOD_CAPACITY", "0.75"))  # fraction of tokens routed through layer
ACT_ENABLED = bool(int(os.environ.get("ACT_ENABLED", "1")))
ACT_MAX_STEPS = int(os.environ.get("ACT_MAX_STEPS", "3"))
ACT_PONDER_COST = float(os.environ.get("ACT_PONDER_COST", "0.01"))

# Which layers to apply MoD routing to (skip first and last for stability)
# By default: all middle layers
MOD_SKIP_FIRST_N = int(os.environ.get("MOD_SKIP_FIRST_N", "1"))
MOD_SKIP_LAST_N = int(os.environ.get("MOD_SKIP_LAST_N", "1"))

# -----------------------------
# MIXTURE OF DEPTHS ROUTER
# -----------------------------

class MoDRouter(nn.Module):
    """
    Lightweight per-layer token router for Mixture of Depths.

    For each token position, computes a scalar routing score.
    Top-k tokens (by score) are routed through the transformer block.
    Remaining tokens bypass the block entirely (identity pass-through).

    Uses straight-through gradient estimation: scores are used for
    selection (discrete), but gradients flow through the scores
    to encourage the router to learn meaningful routing decisions.

    Reference: Raposo et al. (2024) "Mixture of Depths"
    arXiv:2404.02258
    """

    def __init__(self, dim: int, capacity: float = 0.75):
        super().__init__()
        self.capacity = capacity
        # Single linear projection: dim -> 1 scalar score per token
        # No bias — we want the router to learn relative importance
        self.router = nn.Linear(dim, 1, bias=False)
        # Init to small values so routing starts near-uniform
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            routed_x:   (batch, k, dim) — tokens selected for processing
            indices:    (batch, k) — which positions were selected
            scores:     (batch, seq_len) — routing scores for all tokens
        """
        bsz, seq_len, dim = x.shape
        k = max(1, int(seq_len * self.capacity))

        # Compute routing scores: (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.router(x).squeeze(-1)  # (batch, seq_len)

        # Select top-k token positions by score
        topk_scores, indices = torch.topk(scores, k, dim=1, sorted=True)  # (batch, k)

        # Gather selected tokens
        # indices: (batch, k) -> expand to (batch, k, dim) for gather
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, dim)
        routed_x = torch.gather(x, 1, idx_expanded)  # (batch, k, dim)

        return routed_x, indices, scores

    def scatter_back(self, x: Tensor, processed: Tensor, indices: Tensor, scores: Tensor) -> Tensor:
        """
        Scatter processed tokens back to their original positions.
        Unrouted tokens keep their original values (identity).

        Uses straight-through: multiply by softmax of scores so
        gradients flow to the router.

        Args:
            x:          (batch, seq_len, dim) — original input
            processed:  (batch, k, dim) — block output for routed tokens
            indices:    (batch, k) — positions of routed tokens
            scores:     (batch, seq_len) — all routing scores

        Returns:
            output: (batch, seq_len, dim)
        """
        bsz, seq_len, dim = x.shape
        output = x.clone()

        # Weight processed tokens by their normalized routing scores
        # This gives the router a gradient signal
        topk_scores = torch.gather(scores, 1, indices)  # (batch, k)
        # Softmax over selected scores for weighting
        weights = torch.softmax(topk_scores, dim=1).unsqueeze(-1)  # (batch, k, 1)
        weighted_processed = processed * weights

        # Scatter back: add delta (processed - original) at selected positions
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, dim)
        original_at_indices = torch.gather(x, 1, idx_expanded)

        # output = x everywhere, but at routed positions: use weighted processed output
        output.scatter_(1, idx_expanded, weighted_processed + (1 - weights) * original_at_indices)

        return output


# -----------------------------
# ADAPTIVE COMPUTATION TIME
# -----------------------------

class ACTHaltingUnit(nn.Module):
    """
    Per-token adaptive halting unit (Graves 2016, PonderNet 2021).

    At each recurrence step, computes a halting probability h_t in [0,1].
    Tokens accumulate probability mass until sum >= 1 - epsilon, at which
    point they halt. The final step absorbs remaining probability mass.

    Loss includes a ponder cost: mean number of steps taken, weighted
    by ponder_cost hyperparameter.

    References:
        Graves (2016): arXiv:1603.08983
        Banino et al. (2021): arXiv:2107.05407
    """

    def __init__(self, dim: int):
        super().__init__()
        # Halting probability: scalar per token
        self.halt_linear = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.halt_linear.weight)
        nn.init.constant_(self.halt_linear.bias, -1.0)  # start with low halting prob

    def forward(self, states: list[Tensor], ponder_cost_weight: float = 0.01) -> tuple[Tensor, Tensor]:
        """
        Combine per-step states using ACT halting probabilities.

        Args:
            states: list of (batch, seq_len, dim) tensors, one per recurrence step
            ponder_cost_weight: lambda for ponder regularization

        Returns:
            output: (batch, seq_len, dim) — weighted combination of states
            ponder_loss: scalar tensor — regularization term
        """
        n_steps = len(states)
        bsz, seq_len, dim = states[0].shape
        device = states[0].device

        # Accumulate halting probabilities and weighted state
        halting_acc = torch.zeros(bsz, seq_len, device=device)
        output = torch.zeros(bsz, seq_len, dim, device=device, dtype=states[0].dtype)
        remainders = torch.ones(bsz, seq_len, device=device)
        n_updates = torch.zeros(bsz, seq_len, device=device)

        epsilon = 0.01

        for step, state in enumerate(states):
            # Compute halting probability for this step
            h = torch.sigmoid(self.halt_linear(state).squeeze(-1))  # (batch, seq_len)

            # On last step, always halt (absorb remainder)
            is_last = (step == n_steps - 1)

            # Tokens that haven't halted yet
            still_running = (halting_acc < 1.0 - epsilon).float()

            # Would halting this step exceed 1.0?
            new_acc = halting_acc + h * still_running

            # Clamp at 1.0 for final absorption
            use_remainder = (new_acc > 1.0 - epsilon).float() * still_running
            use_halting = (1.0 - use_remainder) * still_running

            # Probability mass assigned to this step
            p = use_halting * h + use_remainder * remainders  # (batch, seq_len)

            # Weighted accumulation
            output = output + p.unsqueeze(-1) * state

            # Update tracking
            halting_acc = halting_acc + p * still_running
            remainders = remainders - p * still_running
            n_updates = n_updates + still_running

            if is_last:
                break

        # Ponder cost: penalize taking more steps
        ponder_loss = ponder_cost_weight * n_updates.mean()

        return output, ponder_loss


# -----------------------------
# MoD BLOCK (wraps Block)
# -----------------------------

class MoDBlock(nn.Module):
    """
    Wraps the baseline Block with optional MoD token routing.

    When routing is enabled:
    - Top-k tokens pass through the transformer block
    - Remaining tokens skip the block (identity)
    - Straight-through gradient flows to the router

    When routing is disabled: identical to baseline Block.
    """

    def __init__(
        self,
        block: Block,
        dim: int,
        use_routing: bool = True,
        capacity: float = 0.75,
    ):
        super().__init__()
        self.block = block
        self.use_routing = use_routing
        if use_routing:
            self.router = MoDRouter(dim, capacity)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        if not self.use_routing:
            return self.block(x, x0)

        # Route top-k tokens through the block
        routed_x, indices, scores = self.router(x)

        # Process routed tokens through full block
        # Need to pass x0 sliced to the same positions
        bsz, seq_len, dim = x.shape
        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, dim)
        routed_x0 = torch.gather(x0, 1, idx_expanded)

        # Block expects (batch, seq, dim) — routed_x has shape (batch, k, dim) ✓
        processed = self.block(routed_x, routed_x0)

        # Scatter processed tokens back, identity for unrouted
        output = self.router.scatter_back(x, processed, indices, scores)
        return output


# -----------------------------
# MoD + ACT GPT
# -----------------------------

class MoDGPT(nn.Module):
    """
    Drop-in replacement for GPT with Mixture of Depths + ACT.

    Architecture:
    - Same encoder/decoder U-Net structure as baseline GPT
    - MoD routing on middle layers (skip first N and last N)
    - ACT halting over recurrence steps in the decoder half
    - All other components identical to baseline

    The ponder_loss is accumulated and returned as part of forward()
    when training=True, added to the cross-entropy loss with weight
    ACT_PONDER_COST.
    """

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
        # MoD + ACT params
        mod_enabled: bool = True,
        mod_capacity: float = 0.75,
        mod_skip_first_n: int = 1,
        mod_skip_last_n: int = 1,
        act_enabled: bool = True,
        act_max_steps: int = 3,
        act_ponder_cost: float = 0.01,
    ):
        super().__init__()

        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")

        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.act_enabled = act_enabled
        self.act_max_steps = act_max_steps
        self.act_ponder_cost = act_ponder_cost
        self.model_dim = model_dim

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Build blocks with MoD routing
        blocks = []
        for i in range(num_layers):
            base_block = Block(
                model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init
            )
            # Apply routing to middle layers only
            use_routing = (
                mod_enabled
                and i >= mod_skip_first_n
                and i < (num_layers - mod_skip_last_n)
            )
            blocks.append(MoDBlock(base_block, model_dim, use_routing, mod_capacity))

        self.blocks = nn.ModuleList(blocks)

        # ACT halting unit (applied in decoder half)
        if act_enabled:
            self.act_unit = ACTHaltingUnit(model_dim)

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

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        skips: list[Tensor] = []
        ponder_loss = torch.tensor(0.0, device=x.device)

        # Encoder half
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)

        # Decoder half — with optional ACT recurrence
        if self.act_enabled:
            # Run decoder act_max_steps times, collecting states
            decoder_states = []
            x_dec = x

            for step in range(self.act_max_steps):
                x_step = x_dec
                local_skips = list(skips)  # copy so we can pop repeatedly

                for i in range(self.num_decoder_layers):
                    if local_skips:
                        skip_w = self.skip_weights[i].to(dtype=x_step.dtype)
                        x_step = x_step + skip_w[None, None, :] * local_skips.pop()
                    x_step = self.blocks[self.num_encoder_layers + i](x_step, x0)

                decoder_states.append(x_step)

            # Combine states with ACT halting
            x, ponder_loss = self.act_unit(decoder_states, self.act_ponder_cost)

        else:
            # Standard decoder (no ACT)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        lm_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        return lm_loss + ponder_loss


# -----------------------------
# MAIN — same as baseline but uses MoDGPT
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp,
        enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Build MoDGPT
    model = MoDGPT(
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
        mod_enabled=MOD_ENABLED,
        mod_capacity=MOD_CAPACITY,
        mod_skip_first_n=MOD_SKIP_FIRST_N,
        mod_skip_last_n=MOD_SKIP_LAST_N,
        act_enabled=ACT_ENABLED,
        act_max_steps=ACT_MAX_STEPS,
        act_ponder_cost=ACT_PONDER_COST,
    ).to(device)

    restore_low_dim_params_to_fp32(model)

    if master_process:
        total_params = sum(p.numel() for p in model.parameters())
        router_params = sum(
            p.numel() for name, p in model.named_parameters() if "router" in name
        )
        act_params = sum(
            p.numel() for name, p in model.named_parameters() if "act_unit" in name
        )
        print(f"MoDGPT | total params: {total_params:,}")
        print(f"  Router params: {router_params:,} | ACT params: {act_params:,}")
        print(f"  MoD enabled: {MOD_ENABLED}, capacity: {MOD_CAPACITY}")
        print(f"  ACT enabled: {ACT_ENABLED}, max_steps: {ACT_MAX_STEPS}, ponder_cost: {ACT_PONDER_COST}")

    model = torch.compile(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if distributed else model

    # Separate param groups: matrices -> Muon, everything else -> AdamW
    matrix_params = [
        p for name, p in raw_model.named_parameters()
        if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name
    ]
    scalar_params = [
        p for name, p in raw_model.named_parameters()
        if not (p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name)
        and "tok_emb" not in name and "lm_head" not in name
    ]
    embed_params = [p for name, p in raw_model.named_parameters() if "tok_emb" in name]
    head_params = [p for name, p in raw_model.named_parameters() if "lm_head" in name]

    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    optimizer_adam = torch.optim.AdamW(
        [
            {"params": scalar_params, "lr": args.scalar_lr},
            {"params": embed_params, "lr": args.embed_lr if not args.tie_embeddings else args.tied_embed_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    t0 = time.time()
    step = 0
    train_loss_accum = 0.0

    def lr_schedule(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.iterations - args.warmup_steps - args.warmdown_iters)
        if step >= args.iterations - args.warmdown_iters:
            decay = (args.iterations - step) / max(1, args.warmdown_iters)
            return max(0.0, decay)
        return 1.0

    model.train()

    for step in range(args.iterations):
        elapsed = time.time() - t0
        if elapsed > args.max_wallclock_seconds:
            if master_process:
                print(f"Wallclock limit {args.max_wallclock_seconds}s reached at step {step}")
            break

        # LR schedule
        lr_scale = lr_schedule(step)
        for group in optimizer_muon.param_groups:
            group["lr"] = args.matrix_lr * lr_scale
        for i, group in enumerate(optimizer_adam.param_groups):
            base_lrs = [args.scalar_lr, args.embed_lr if not args.tie_embeddings else args.tied_embed_lr, args.head_lr]
            group["lr"] = base_lrs[i] * lr_scale

        # Muon momentum warmup
        if step < args.muon_momentum_warmup_steps:
            m = args.muon_momentum_warmup_start + (args.muon_momentum - args.muon_momentum_warmup_start) * (
                step / args.muon_momentum_warmup_steps
            )
            for group in optimizer_muon.param_groups:
                group["momentum"] = m

        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adam.zero_grad(set_to_none=True)

        step_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
            step_loss += loss.item() * grad_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer_muon.step()
        optimizer_adam.step()

        train_loss_accum += step_loss

        if step % args.train_log_every == 0 and master_process:
            avg_loss = train_loss_accum / args.train_log_every if step > 0 else step_loss
            train_loss_accum = 0.0
            elapsed = time.time() - t0
            print(f"step={step:5d} | loss={avg_loss:.4f} | lr={lr_scale:.4f} | t={elapsed:.1f}s")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0:
            val_loss, val_bpb = eval_val(
                args, raw_model, rank, world_size, device,
                grad_accum_steps, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            if master_process:
                print(f"  val_loss={val_loss:.4f} | val_bpb={val_bpb:.4f}")
            model.train()

    # Final validation
    val_loss, val_bpb = eval_val(
        args, raw_model, rank, world_size, device,
        grad_accum_steps, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )

    if master_process:
        print(f"\n{'='*50}")
        print(f"FINAL | val_loss={val_loss:.4f} | val_bpb={val_bpb:.4f}")
        print(f"{'='*50}")

        # Quantize and measure artifact size
        state_dict = raw_model.state_dict()
        quant_obj, quant_stats = quantize_state_dict_int8(state_dict)
        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        compressed = zlib.compress(buf.getvalue(), level=9)
        code_bytes = len(code.encode("utf-8"))
        model_bytes = len(compressed)
        total_bytes = code_bytes + model_bytes
        print(f"Artifact: code={code_bytes:,}B | model={model_bytes:,}B | total={total_bytes:,}B")
        print(f"Under 16MB: {total_bytes < 16_000_000}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
