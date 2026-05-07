"""
Modded-NanoGPT Tricks Training Script
======================================

Integrates proven techniques from the NanoGPT speedrun world record
into the Parameter Golf baseline architecture:

  1. Value Embeddings — Per-token V lookup added to first/last attention layers.
     Shared pairwise (layer 0 <-> layer N-1, layer 1 <-> layer N-2) so gradients
     flow from both early and late layers. Pure embedding lookup = zero compute cost.

  2. MQA (1 KV head) — Reduces K,V projection from 4 KV heads to 1, freeing
     ~1.8M params and reducing attention compute. The freed budget remains
     unallocated for now (keeps model smaller / compresses better).

The baseline already has x0-mixin via resid_mix and UNet skip connections
via skip_weights, so those are not reimplemented here.

Architecture: ModdedGPT subclasses GPT, adds value embeddings and overrides
CausalSelfAttention to accept optional VE tensors. Fully torch.compile compatible.

Usage:
    # Baseline with modded tricks (no JEPA losses)
    EXPERIMENT=exp5_modded RUN_ID=exp5_run1 \
        torchrun --standalone --nproc_per_node=1 jepa_ntp/train_modded.py

    # MQA only (no value embeddings)
    EXPERIMENT=exp5_mqa_only RUN_ID=mqa_run1 \
        torchrun --standalone --nproc_per_node=1 jepa_ntp/train_modded.py
"""

from __future__ import annotations

import copy
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# --- Add parent directory to path so we can import train_gpt ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_gpt import (
    GPT,
    Block,
    CausalSelfAttention,
    CastedLinear,
    MLP,
    RMSNorm,
    Rotary,
    apply_rotary_emb,
    Hyperparameters,
    Muon,
    DistributedTokenLoader,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
    zeropower_via_newtonschulz5,
    CONTROL_TENSOR_NAME_PATTERNS,
)

# ============================================================
# EXPERIMENT CONFIGS
# ============================================================

MODDED_CONFIGS = {
    # Full modded tricks: MQA + value embeddings
    "exp5_modded": {
        "num_kv_heads": 1,
        "use_value_embeds": True,
        "ve_layers_each_side": 2,  # VE on first 2 and last 2 layers
    },
    # MQA only (ablation)
    "exp5_mqa_only": {
        "num_kv_heads": 1,
        "use_value_embeds": False,
        "ve_layers_each_side": 0,
    },
    # VE only, keep 4 KV heads (ablation)
    "exp5_ve_only": {
        "num_kv_heads": 4,
        "use_value_embeds": True,
        "ve_layers_each_side": 2,
    },
    # Full modded + 3x MLP (uses freed MQA params)
    "exp5_modded_3x_mlp": {
        "num_kv_heads": 1,
        "use_value_embeds": True,
        "ve_layers_each_side": 2,
        "mlp_mult": 3,
    },
    # Baseline (pure, no changes — for fair comparison in this script)
    "baseline": {
        "num_kv_heads": 4,
        "use_value_embeds": False,
        "ve_layers_each_side": 0,
    },
}

# ============================================================
# ATTENTION WITH VALUE EMBEDDINGS
# ============================================================

class ModdedAttention(nn.Module):
    """
    CausalSelfAttention with optional value embedding injection.

    When ve_table is provided during forward(), the looked-up value embeddings
    are mixed into V via learned scalars: v = v_lambda * v + ve_lambda * ve
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        has_value_embed: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
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

        # Value embedding mixing scalars
        self.has_value_embed = has_value_embed
        if has_value_embed:
            # v_lambda init=1 (keep projected V), ve_lambda init=0 (VE starts off)
            self.v_lambda = nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.ve_lambda = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: Tensor, ve: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # Inject value embeddings if present
        if self.has_value_embed and ve is not None:
            # ve shape: (bsz, seqlen, kv_dim) -> (bsz, num_kv_heads, seqlen, head_dim)
            ve = ve.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_lambda.to(dtype=v.dtype) * v + self.ve_lambda.to(dtype=v.dtype) * ve

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


# ============================================================
# MODDED BLOCK (uses ModdedAttention)
# ============================================================

class ModdedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        has_value_embed: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = ModdedAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, has_value_embed)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, ve: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), ve=ve)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# ============================================================
# MODDED GPT
# ============================================================

class ModdedGPT(nn.Module):
    """
    GPT with value embeddings and configurable KV heads.

    Value embeddings are shared pairwise between first/last layers:
      Layer 0 <-> Layer N-1  (share VE table 0)
      Layer 1 <-> Layer N-2  (share VE table 1)
      ...

    This ensures short gradient paths from both directions.
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
        # Modded tricks
        use_value_embeds: bool = False,
        ve_layers_each_side: int = 2,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.use_value_embeds = use_value_embeds
        self.ve_layers_each_side = ve_layers_each_side
        self.num_layers = num_layers

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # UNet structure (same as baseline)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Determine which layers get value embeddings
        # ve_layer_map[i] = index into ve_tables, or -1 if no VE
        self.ve_layer_map: list[int] = [-1] * num_layers
        if use_value_embeds and ve_layers_each_side > 0:
            n = min(ve_layers_each_side, num_layers // 2)
            for k in range(n):
                self.ve_layer_map[k] = k                    # first N layers
                self.ve_layer_map[num_layers - 1 - k] = k   # last N layers (shared)

        # Value embedding tables
        kv_dim = num_kv_heads * (model_dim // num_heads)
        num_ve_tables = max(0, max(self.ve_layer_map) + 1) if use_value_embeds else 0
        if num_ve_tables > 0:
            # Spherical Gaussian init (small scale, following modded-nanogpt)
            self.ve_tables = nn.Parameter(
                0.01 * torch.randn(num_ve_tables, vocab_size, kv_dim)
            )
        else:
            self.register_parameter("ve_tables", None)

        # Build blocks with ModdedAttention where VE is needed
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            has_ve = self.ve_layer_map[i] >= 0
            self.blocks.append(
                ModdedBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    has_value_embed=has_ve,
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

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # Look up value embeddings once (pure embedding lookup, no matmul)
        # ve_lookups[k] has shape (bsz, seq, kv_dim) for table k
        ve_lookups: dict[int, Tensor] | None = None
        if self.use_value_embeds and self.ve_tables is not None:
            ve_lookups = {}
            for k in range(self.ve_tables.size(0)):
                # input_ids: (bsz, seq) -> index into ve_tables[k]: (vocab, kv_dim)
                ve_lookups[k] = self.ve_tables[k][input_ids]  # (bsz, seq, kv_dim)

        # Encoder half (stores skips)
        for i in range(self.num_encoder_layers):
            ve_k = self.ve_layer_map[i]
            ve = ve_lookups[ve_k] if (ve_lookups is not None and ve_k >= 0) else None
            x = self.blocks[i](x, x0, ve=ve)
            skips.append(x)

        # Decoder half (consumes skips in reverse)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            layer_idx = self.num_encoder_layers + i
            ve_k = self.ve_layer_map[layer_idx]
            ve = ve_lookups[ve_k] if (ve_lookups is not None and ve_k >= 0) else None
            x = self.blocks[layer_idx](x, x0, ve=ve)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main() -> None:
    global zeropower_via_newtonschulz5

    import sentencepiece as spm
    import subprocess
    import glob as glob_mod
    from torch.nn.parallel import DistributedDataParallel as DDP

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    _compile_disabled = (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"
        or os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1"
    )
    if not _compile_disabled:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # --- Load experiment config ---
    exp_name = os.environ.get("EXPERIMENT", "exp5_modded")
    if exp_name not in MODDED_CONFIGS:
        raise ValueError(
            f"Unknown experiment '{exp_name}'. Choose from: {list(MODDED_CONFIGS.keys())}"
        )
    modded_cfg = MODDED_CONFIGS[exp_name]
    num_kv_heads = modded_cfg.get("num_kv_heads", args.num_kv_heads)
    use_value_embeds = modded_cfg.get("use_value_embeds", False)
    ve_layers = modded_cfg.get("ve_layers_each_side", 2)
    mlp_mult = modded_cfg.get("mlp_mult", args.mlp_mult)

    print(f"[MODDED] Experiment: {exp_name}")
    print(f"[MODDED] KV heads: {num_kv_heads}, Value Embeds: {use_value_embeds} (layers_each_side={ve_layers}), MLP mult: {mlp_mult}")

    # --- Distributed + CUDA setup (same as baseline) ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
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

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Tokenizer + Validation ---
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # --- Model (ModdedGPT) ---
    base_model = ModdedGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_value_embeds=use_value_embeds,
        ve_layers_each_side=ve_layers,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Log VE layer map
    if use_value_embeds:
        ve_map_str = ", ".join(
            f"L{i}->VE{k}" for i, k in enumerate(base_model.ve_layer_map) if k >= 0
        )
        log0(f"[MODDED] Value embedding map: {ve_map_str}")

    # --- Compile + DDP ---
    # torch.compile hangs on some GPU architectures (e.g. Blackwell).
    # Respect TORCH_COMPILE_DISABLE / TORCHDYNAMO_DISABLE env vars.
    use_compile = not (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"
        or os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1"
    )
    # DDP must wrap the model BEFORE torch.compile (PyTorch DDP+compile guidance).
    if distributed:
        model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        model = base_model

    if use_compile:
        log0("[MODDED] torch.compile enabled (fullgraph=False for DDP compatibility)")
        model = torch.compile(model, dynamic=False, fullgraph=False)
    else:
        log0("[MODDED] torch.compile DISABLED")

    # --- Optimizers (same split as baseline) ---
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    # VE tables go into scalar optimizer (they're embeddings, not matrices for Muon)
    if base_model.ve_tables is not None:
        scalar_params.append(base_model.ve_tables)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")

    # --- WandB ---
    wandb_run = None
    if master_process:
        try:
            import wandb
            wandb_run = wandb.init(
                project="jepa-ntp-parameter-golf",
                entity=os.environ.get("WANDB_ENTITY", None),  # auto-detect if not set
                group=exp_name,
                tags=[exp_name, f"kv{num_kv_heads}", f"mlp{mlp_mult}x"]
                      + (["value-embeds"] if use_value_embeds else []),
                name=f"{exp_name}_{args.run_id}",
                config={
                    "experiment": exp_name,
                    "model_dim": args.model_dim,
                    "num_layers": args.num_layers,
                    "vocab_size": args.vocab_size,
                    "num_kv_heads": num_kv_heads,
                    "mlp_mult": mlp_mult,
                    "use_value_embeds": use_value_embeds,
                    "ve_layers_each_side": ve_layers,
                    "total_params": n_params,
                },
            )
            log0(f"[MODDED] WandB run: {wandb_run.url}")
        except ImportError:
            log0("[MODDED] wandb not installed, logging to stdout only")
        except Exception as e:
            log0(f"[MODDED] WandB init failed: {e}, continuing without WandB")

    # --- Data loader ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
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

    # --- Warmup (same as baseline, resets state after) ---
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        # Reset everything
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        log0(f"[MODDED] Warmup complete ({args.warmup_steps} steps)")

    # --- Main training loop ---
    training_time_ms = 0.0
    total_tokens_seen: int = 0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.train()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        # --- Validation ---
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"tokens:{total_tokens_seen:,} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if wandb_run:
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/bpb": val_bpb,
                    "throughput/tokens_seen": total_tokens_seen,
                }, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        # --- Training step ---
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()

        train_loss /= grad_accum_steps
        total_tokens_seen += args.train_batch_tokens

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        # LR schedule
        for opt in optimizers:
            for group in opt.param_groups:
                if "base_lr" in group:
                    group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1

        # --- Logging ---
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            tokens_per_sec = total_tokens_seen / (approx_training_time_ms / 1000.0) if approx_training_time_ms > 0 else 0.0
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"tokens:{total_tokens_seen:,} tok/s:{tokens_per_sec:,.0f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
            if wandb_run:
                wandb_run.log({
                    "loss/train": train_loss.item(),
                    "throughput/tokens_seen": total_tokens_seen,
                    "throughput/tokens_per_sec": tokens_per_sec,
                }, step=step)

        # --- Wallclock cap ---
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

    # --- Serialization (same as baseline) ---
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_size = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes (code: {code_size})")
        log0(f"Total submission size: {model_bytes + code_size} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_size = len(code.encode("utf-8"))
        log0(f"Int8+zlib model: {quant_file_bytes} bytes, total: {quant_file_bytes + code_size} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )

    if wandb_run:
        wandb_run.log({
            "final/val_loss": q_val_loss,
            "final/val_bpb": q_val_bpb,
            "final/model_size_bytes": os.path.getsize("final_model.int8.ptz") if master_process else 0,
        })
        wandb_run.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
