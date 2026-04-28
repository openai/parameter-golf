"""
Local pilot for Activation-Space Compressed-Sensing Adapters (ACSA).

This script is intentionally derived from the LoRA TTT reference run instead of
starting from the current SOTA stack. The goal is to validate the adapter
mechanism and the score-before-update evaluation loop on a smaller, cheaper
pilot before porting the idea to stronger SP4096 / SP8192 stacks.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import importlib.util
import io
import json
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


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = REPO_ROOT / "records" / "track_10min_16mb" / "2026-03-17_LoRA_TTT" / "train_gpt.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(f"Base script not found: {BASE_SCRIPT}")
_spec = importlib.util.spec_from_file_location("lora_ttt_base", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(_spec)
# TorchDynamo may later try to re-import this module by name during compilation.
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)


class Hyperparameters(base.Hyperparameters):
    compile_model = bool(int(os.environ.get("COMPILE_MODEL", "0")))
    autocast_dtype = os.environ.get("AUTOCAST_DTYPE", "bf16").strip().lower()
    debug_val_docs = int(os.environ.get("DEBUG_VAL_DOCS", "0"))
    output_dir = os.environ.get("OUTPUT_DIR", ".").strip() or "."
    log_filename = os.environ.get("LOG_FILENAME", "").strip()
    write_submission_json = bool(int(os.environ.get("WRITE_SUBMISSION_JSON", "0")))
    submission_filename = os.environ.get("SUBMISSION_FILENAME", "submission.json").strip() or "submission.json"
    submission_author = os.environ.get("SUBMISSION_AUTHOR", "").strip()
    submission_github_id = os.environ.get("SUBMISSION_GITHUB_ID", "").strip()
    submission_name = os.environ.get("SUBMISSION_NAME", "Activation-Space Compressed-Sensing Adapters").strip()
    submission_blurb = os.environ.get("SUBMISSION_BLURB", "").strip()
    submission_track = os.environ.get("SUBMISSION_TRACK", "non-record-16mb").strip()
    submission_hardware = os.environ.get("SUBMISSION_HARDWARE", "").strip()
    enforce_artifact_limit = bool(int(os.environ.get("ENFORCE_ARTIFACT_LIMIT", "0")))
    submission_max_bytes = int(os.environ.get("SUBMISSION_MAX_BYTES", "16000000"))
    enforce_eval_limit = bool(int(os.environ.get("ENFORCE_EVAL_LIMIT", "0")))
    eval_max_seconds = float(os.environ.get("EVAL_MAX_SECONDS", "600"))
    enable_acsa_eval = bool(int(os.environ.get("ENABLE_ACSA_EVAL", "1")))
    acsa_progress_every_seconds = float(os.environ.get("ACSA_PROGRESS_EVERY_SECONDS", "15"))
    acsa_dim = int(os.environ.get("ACSA_DIM", "64"))
    acsa_topk = int(os.environ.get("ACSA_TOPK", "16"))
    acsa_alpha_init = float(os.environ.get("ACSA_ALPHA_INIT", "0.05"))
    acsa_tau_init = float(os.environ.get("ACSA_TAU_INIT", "0.01"))
    acsa_lr = float(os.environ.get("ACSA_LR", os.environ.get("TTT_LORA_LR", "0.01")))
    acsa_shrink_mode = os.environ.get("ACSA_SHRINK_MODE", "topk").strip().lower()
    acsa_targets = tuple(
        target.strip().lower()
        for target in os.environ.get("ACSA_TARGETS", "postblock").split(",")
        if target.strip()
    )


def resolve_model_dtype(args: Hyperparameters) -> torch.dtype:
    if args.autocast_dtype == "bf16":
        return torch.bfloat16
    if args.autocast_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AUTOCAST_DTYPE={args.autocast_dtype!r}; expected bf16 or fp16")


def autocast_context(args: Hyperparameters):
    return torch.autocast(device_type="cuda", dtype=resolve_model_dtype(args), enabled=True)


def fwht(x: Tensor) -> Tensor:
    n = x.size(-1)
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT requires a power-of-two last dimension, got {n}")
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, 2 * h)
        a, b = y[..., :h], y[..., h:]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(*x.shape[:-1], n)
        h *= 2
    return y * (n ** -0.5)


class BatchedActivationCSA(nn.Module):
    def __init__(self, bsz: int, model: "ACSAGPT", args: Hyperparameters):
        super().__init__()
        self.bsz = bsz
        self.dim = model.tok_emb.embedding_dim
        self.num_layers = len(model.blocks)
        self.use_postblock = "postblock" in args.acsa_targets
        self.use_prehead = "prehead" in args.acsa_targets
        self.measure_dim = min(max(args.acsa_dim, 1), self.dim)
        self.topk = min(max(args.acsa_topk, 1), self.measure_dim)
        self.alpha_init = args.acsa_alpha_init
        self.tau_init = args.acsa_tau_init
        self.shrink_mode = args.acsa_shrink_mode
        self.num_targets = (self.num_layers if self.use_postblock else 0) + (1 if self.use_prehead else 0)
        if self.num_targets <= 0:
            raise ValueError("ACSA_TARGETS must include postblock and/or prehead")
        if self.dim & (self.dim - 1):
            raise ValueError(f"ACSA currently requires power-of-two model_dim, got {self.dim}")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1337)
        perm = torch.randperm(self.dim, generator=gen)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(self.dim)
        signs = torch.where(torch.rand(self.dim, generator=gen) > 0.5, 1.0, -1.0)
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv_perm", inv_perm, persistent=False)
        self.register_buffer("signs", signs, persistent=False)
        self.gates = nn.Parameter(torch.empty(bsz, self.num_targets, self.measure_dim))
        self.alpha = nn.Parameter(torch.empty(bsz, self.num_targets, 1))
        self.tau = nn.Parameter(torch.empty(bsz, self.num_targets, 1))
        self.reset()

    def reset(self) -> None:
        with torch.no_grad():
            self.gates.fill_(1.0)
            self.alpha.fill_(self.alpha_init)
            self.tau.fill_(self.tau_init)

    def num_trainable_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def _sense(self, x: Tensor) -> Tensor:
        y = x * self.signs.to(dtype=x.dtype)[None, None, :]
        y = y.index_select(-1, self.perm)
        y = fwht(y)
        return y[..., : self.measure_dim]

    def _reconstruct(self, z: Tensor) -> Tensor:
        full = torch.zeros(*z.shape[:-1], self.dim, dtype=z.dtype, device=z.device)
        full[..., : self.measure_dim] = z
        y = fwht(full)
        y = y.index_select(-1, self.inv_perm)
        return y * self.signs.to(dtype=y.dtype)[None, None, :]

    def _shrink(self, z: Tensor, target_idx: int) -> Tensor:
        gate = self.gates[:, target_idx].to(dtype=z.dtype)[:, None, :]
        alpha = self.alpha[:, target_idx].to(dtype=z.dtype)[:, None, :]
        tau = self.tau[:, target_idx].to(dtype=z.dtype).abs()[:, None, :]
        gated = gate * z
        if self.shrink_mode == "threshold":
            sparse = gated.sign() * F.relu(gated.abs() - tau)
            return alpha * sparse
        topk = min(self.topk, gated.size(-1))
        if topk >= gated.size(-1):
            sparse = gated
        else:
            idx = gated.abs().topk(topk, dim=-1).indices
            mask = torch.zeros_like(gated)
            mask.scatter_(-1, idx, 1.0)
            sparse = torch.where((mask > 0) & (gated.abs() >= tau), gated, torch.zeros_like(gated))
        return alpha * sparse

    def _delta(self, x: Tensor, target_idx: int) -> Tensor:
        return self._reconstruct(self._shrink(self._sense(x), target_idx))

    def apply_postblock(self, layer_idx: int, x: Tensor) -> Tensor:
        if not self.use_postblock:
            return x
        return x + self._delta(x, layer_idx)

    def apply_prehead(self, x: Tensor) -> Tensor:
        if not self.use_prehead:
            return x
        idx = self.num_layers if self.use_postblock else 0
        return x + self._delta(x, idx)


class ACSAGPT(base.GPT):
    def forward(self, input_ids: Tensor, target_ids: Tensor, acsa: BatchedActivationCSA | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            if acsa is not None:
                x = acsa.apply_postblock(i, x)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[bi](x, x0)
            if acsa is not None:
                x = acsa.apply_postblock(bi, x)
        x = self.final_norm(x)
        if acsa is not None:
            x = acsa.apply_prehead(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if acsa is not None:
            bsz, sl, vocab = logits.shape
            return F.cross_entropy(
                logits.float().reshape(-1, vocab),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape(bsz, sl)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")


def _reset_acsa_optimizer(opt: torch.optim.Optimizer) -> None:
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p)
            if not state:
                continue
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            state["step"].fill_(0)


def _build_acsa_optimizer(acsa: BatchedActivationCSA, args: Hyperparameters) -> torch.optim.Optimizer:
    return torch.optim.Adam(acsa.parameters(), lr=args.acsa_lr, betas=(args.beta1, args.beta2), eps=1e-10)


def load_debug_validation_tokens(args: Hyperparameters) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(args.val_files))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {args.val_files}")
    tokens = torch.cat([base.load_data_shard(file) for file in files]).contiguous()
    if args.debug_val_docs > 0:
        docs = base._find_docs(tokens, include_next_bos=False)
        limit = min(args.debug_val_docs, len(docs))
        cutoff = docs[limit - 1][0] + docs[limit - 1][1]
        tokens = tokens[:cutoff].contiguous()
    usable = ((tokens.numel() - 1) // args.train_seq_len) * args.train_seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={args.train_seq_len}")
    return tokens[: usable + 1]


def load_eval_document_tokens(args: Hyperparameters) -> Tensor:
    files = sorted(glob.glob(args.val_files))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {args.val_files}")
    tokens = torch.cat([base.load_data_shard(Path(path)) for path in files]).contiguous()
    if args.debug_val_docs > 0:
        docs = base._find_docs(tokens, include_next_bos=False)
        limit = min(args.debug_val_docs, len(docs))
        cutoff = docs[limit - 1][0] + docs[limit - 1][1]
        tokens = tokens[:cutoff].contiguous()
    return tokens


def eval_val_ttt_acsa(
    args: Hyperparameters,
    base_model: ACSAGPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    log0,
) -> tuple[float, float]:
    def _format_duration(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        minutes, secs = divmod(int(round(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}h{minutes:02d}m{secs:02d}s"
        if minutes > 0:
            return f"{minutes:d}m{secs:02d}s"
        return f"{secs:d}s"

    all_tokens = load_eval_document_tokens(args)
    docs = base._find_docs(all_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    batch_size = args.ttt_batch_size
    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    acsa = BatchedActivationCSA(batch_size, base_model, args).to(device)
    opt = _build_acsa_optimizer(acsa, args)
    log0(
        f"acsa_eval:targets={','.join(args.acsa_targets)} dim:{args.acsa_dim} "
        f"topk:{args.acsa_topk} shrink:{args.acsa_shrink_mode} params:{acsa.num_trainable_params()}"
    )
    eval_start_time = time.perf_counter()
    last_progress_time = eval_start_time
    local_total_docs = len(rank_docs)
    local_total_batches = (local_total_docs + batch_size - 1) // batch_size if local_total_docs > 0 else 0
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi : bi + batch_size]
        bsz = len(batch)
        batch_index = bi // batch_size + 1
        if bsz == batch_size:
            cur_acsa, cur_opt = acsa, opt
            cur_acsa.reset()
            _reset_acsa_optimizer(cur_opt)
        else:
            cur_acsa = BatchedActivationCSA(bsz, base_model, args).to(device)
            cur_opt = _build_acsa_optimizer(cur_acsa, args)
        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pred_len + chunk_size - 1) // chunk_size for pred_len in pred_lens]
        max_nc = max(num_chunks)
        for ci in range(max_nc):
            _, context_size, chunk_offset, _ = base._compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)
            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info: list[tuple[int, int]] = []
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0))
                    continue
                ds, dl = batch[b]
                ws, wl, co, cl = base._compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                chunk = all_tokens[ds + ws : ds + ws + wl + 1]
                toks = chunk.to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]
                doc_info.append((co, cl))
            if needs_train:
                with autocast_context(args):
                    ptl = base_model(x, y, acsa=cur_acsa)
            else:
                with torch.no_grad(), autocast_context(args):
                    ptl = base_model(x, y, acsa=cur_acsa)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    base._accumulate_bpb(
                        ptl,
                        x,
                        y,
                        b,
                        co,
                        cl,
                        base_bytes_lut,
                        has_leading_space_lut,
                        is_boundary_token_lut,
                        loss_sum,
                        byte_sum,
                        token_count,
                    )
            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset : chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad()
                (per_doc * mask).sum().backward()
                cur_opt.step()
        if rank == 0 and local_total_docs > 0:
            now = time.perf_counter()
            processed_docs = min(bi + bsz, local_total_docs)
            should_log_progress = args.acsa_progress_every_seconds <= 0 or processed_docs == local_total_docs
            if not should_log_progress:
                should_log_progress = now - last_progress_time >= args.acsa_progress_every_seconds
            if should_log_progress:
                elapsed = now - eval_start_time
                docs_per_second = processed_docs / max(elapsed, 1e-9)
                remaining_docs = max(local_total_docs - processed_docs, 0)
                eta_seconds = remaining_docs / max(docs_per_second, 1e-9)
                log0(
                    f"acsa_eval_progress rank0_local_docs:{processed_docs}/{local_total_docs} "
                    f"batch:{batch_index}/{local_total_batches} "
                    f"elapsed:{_format_duration(elapsed)} eta:{_format_duration(eta_seconds)} "
                    f"docs_per_s:{docs_per_second:.2f}"
                )
                last_progress_time = now
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb


def maybe_compile(module: nn.Module, args: Hyperparameters) -> nn.Module:
    if args.compile_model:
        return torch.compile(module, dynamic=False, fullgraph=True)
    return module


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = resolve_model_dtype(args)
    if args.compile_model:
        base.zeropower_via_newtonschulz5 = torch.compile(base.zeropower_via_newtonschulz5)
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        if args.log_filename:
            logfile = str(output_dir / args.log_filename)
        else:
            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            logfile = str(logs_dir / f"{args.run_id}.txt")
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for line in str(msg).splitlines() or [""]:
                print(f"{timestamp} {line}")
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only supports SentencePiece .model files: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_debug_validation_tokens(args)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp,
        args.vocab_size,
        device,
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1} debug_val_docs:{args.debug_val_docs}")

    base_model = ACSAGPT(
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
    ).to(device).to(dtype=model_dtype)
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
        if isinstance(module, base.Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    base.restore_low_dim_params_to_fp32(base_model)
    compiled_model = maybe_compile(base_model, args)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
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
    optimizer_muon = base.Muon(
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
    log0(f"compile_model:{int(args.compile_model)} autocast_dtype:{args.autocast_dtype}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

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
                with autocast_context(args):
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
        train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    pre_quant_val_loss: float | None = None
    pre_quant_val_bpb: float | None = None
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = base.eval_val(
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
            pre_quant_val_loss = float(val_loss)
            pre_quant_val_bpb = float(val_bpb)
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
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with autocast_context(args):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
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

    model_path = output_dir / "final_model.pt"
    quant_path = output_dir / "final_model.int8.ptz"
    submission_path = output_dir / args.submission_filename

    model_bytes = None
    code_bytes = len(code.encode("utf-8"))
    total_submission_bytes = None
    quant_file_bytes = None
    total_quant_submission_bytes = None

    if master_process:
        torch.save(base_model.state_dict(), model_path)
        model_bytes = model_path.stat().st_size
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
        total_submission_bytes = model_bytes + code_bytes

    quant_obj, quant_stats = base.quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    quant_file_bytes = len(quant_blob)
    total_quant_submission_bytes = quant_file_bytes + code_bytes
    if master_process:
        with open(quant_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = quant_path.stat().st_size
        total_quant_submission_bytes = quant_file_bytes + code_bytes
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open(quant_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(base.dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
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
    qeval_ms = 1000.0 * (time.perf_counter() - t_qeval)

    acsa_val_loss = None
    acsa_val_bpb = None
    acsa_ms = 0.0
    if args.enable_acsa_eval:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        torch.cuda.synchronize()
        t_acsa = time.perf_counter()
        acsa_val_loss, acsa_val_bpb = eval_val_ttt_acsa(
            args,
            base_model,
            rank,
            world_size,
            device,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log0,
        )
        torch.cuda.synchronize()
        acsa_ms = 1000.0 * (time.perf_counter() - t_acsa)
        log0(
            f"final_int8_ttt_acsa val_loss:{acsa_val_loss:.4f} val_bpb:{acsa_val_bpb:.4f} "
            f"eval_time:{acsa_ms:.0f}ms"
        )

    total_eval_ms = qeval_ms + acsa_ms
    artifact_under_limit = total_quant_submission_bytes is not None and total_quant_submission_bytes <= args.submission_max_bytes
    eval_under_limit = total_eval_ms <= 1000.0 * args.eval_max_seconds
    train_under_limit = args.max_wallclock_seconds <= 0 or training_time_ms <= 1000.0 * args.max_wallclock_seconds
    log0(
        f"compliance artifact_under_limit:{int(bool(artifact_under_limit))} "
        f"eval_under_limit:{int(bool(eval_under_limit))} "
        f"train_under_limit:{int(bool(train_under_limit))} "
        f"artifact_bytes:{total_quant_submission_bytes if total_quant_submission_bytes is not None else -1} "
        f"eval_time_ms:{total_eval_ms:.0f}"
    )

    if master_process and args.write_submission_json:
        final_val_loss = acsa_val_loss if acsa_val_loss is not None else q_val_loss
        final_val_bpb = acsa_val_bpb if acsa_val_bpb is not None else q_val_bpb
        payload = {
            "author": args.submission_author,
            "github_id": args.submission_github_id,
            "name": args.submission_name,
            "blurb": args.submission_blurb,
            "date": time.strftime("%Y-%m-%dT00:00:00Z"),
            "track": args.submission_track,
            "val_loss": round(float(final_val_loss), 8),
            "val_bpb": round(float(final_val_bpb), 8),
            "pre_quant_val_loss": round(float(pre_quant_val_loss), 8) if pre_quant_val_loss is not None else None,
            "pre_quant_val_bpb": round(float(pre_quant_val_bpb), 8) if pre_quant_val_bpb is not None else None,
            "step_stop": int(step),
            "wallclock_seconds": int(round(training_time_ms / 1000.0)),
            "bytes_total": int(total_quant_submission_bytes if total_quant_submission_bytes is not None else -1),
            "artifact_bytes": int(total_quant_submission_bytes if total_quant_submission_bytes is not None else -1),
            "bytes_model_int8_zlib": int(quant_file_bytes) if quant_file_bytes is not None else None,
            "bytes_code": int(code_bytes),
            "artifact_file": quant_path.name,
            "gpu": args.submission_hardware,
            "hardware": args.submission_hardware,
            "compliance": {
                "train_under_limit": bool(train_under_limit),
                "artifact_under_limit": bool(artifact_under_limit),
                "eval_under_limit": bool(eval_under_limit),
                "score_first_ttt": True,
                "debug_val_docs": int(args.debug_val_docs),
                "eval_time_ms": int(round(total_eval_ms)),
            },
        }
        submission_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log0(f"wrote_submission_json:{submission_path}")

    if args.enforce_artifact_limit and not artifact_under_limit:
        raise SystemExit(
            f"artifact limit exceeded: bytes_total={total_quant_submission_bytes} limit={args.submission_max_bytes}"
        )
    if args.enforce_eval_limit and not eval_under_limit:
        raise SystemExit(f"eval time limit exceeded: eval_time_ms={total_eval_ms:.0f} limit_ms={1000.0 * args.eval_max_seconds:.0f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
