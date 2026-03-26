"""Train a Qwen3 backbone with the challenge's tiny 1024-token vocabulary.

This is a thin adapter around Qwen/Qwen3-1.7B-Base:
- the frozen backbone stays in bf16 and never receives gradients
- the challenge tokenizer/vocabulary/sequence settings come from transfer.py
- the new trainable parameters are the tiny token embedding, the input/output
  projection layers, and the tiny-vocab lm_head

The goal is to preserve the repo's validation and serialization flow while
swapping the model body for Qwen3.
"""

from __future__ import annotations

import copy
import io
import math
import os
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel

import transfer as base


class Hyperparameters(base.Hyperparameters):
    qwen_model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-1.7B-FP8")
    tiny_vocab_dim = int(os.environ.get("TINY_VOCAB_DIM", str(base.Hyperparameters.model_dim)))


def infer_hidden_size(config) -> int:
    for name in ("hidden_size", "dim", "model_dim", "hidden_dim", "n_embd"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not infer Qwen hidden size from config")


class Qwen3TinyVocabLM(nn.Module):
    def __init__(
        self,
        model_id: str,
        vocab_size: int,
        tiny_dim: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")

        self.model_id = model_id
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap

        backbone = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        if getattr(backbone.config, "use_cache", None) is not None:
            backbone.config.use_cache = False
        backbone.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone

        hidden_size = infer_hidden_size(backbone.config)
        self.hidden_size = hidden_size
        self.tiny_dim = tiny_dim

        self.tok_emb = nn.Embedding(vocab_size, tiny_dim)
        self.input_proj = base.CastedLinear(tiny_dim, hidden_size, bias=False)
        self.output_proj = base.CastedLinear(hidden_size, tiny_dim, bias=False)
        self.lm_head = base.CastedLinear(tiny_dim, vocab_size, bias=False)
        if self.tie_embeddings:
            if self.lm_head.weight.shape != self.tok_emb.weight.shape:
                raise ValueError("token embedding and lm_head shapes must match when tying embeddings")
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()
        self.freeze_backbone()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        if not self.tie_embeddings:
            nn.init.zeros_(self.lm_head.weight)

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_backbone()
        return self

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not name.startswith("backbone.")
        }

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = self.input_proj(x)
        x = self.backbone(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
        x = self.output_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        x = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def restore_adapter_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.startswith("backbone."):
                continue
            if param.dtype != torch.float32:
                param.data = param.data.float()


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")

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
    val_tokens = base.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = Qwen3TinyVocabLM(
        model_id=args.qwen_model_id,
        vocab_size=args.vocab_size,
        tiny_dim=args.tiny_vocab_dim,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
    ).to(device)
    base_model = base_model.bfloat16()
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    restore_adapter_params_to_fp32(base_model)
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    projection_params: list[nn.Parameter] = [base_model.input_proj.weight, base_model.output_proj.weight]
    if not args.tie_embeddings:
        projection_params.append(base_model.lm_head.weight)
    optimizer = torch.optim.Adam(
        [
            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr},
            {"params": projection_params, "lr": args.matrix_lr, "base_lr": args.matrix_lr},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )

    n_params = sum(p.numel() for p in base_model.parameters())
    n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    log0(f"model_params:{n_params} trainable_params:{n_trainable}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"qwen_model_id:{args.qwen_model_id} hidden_size:{base_model.hidden_size} tiny_dim:{base_model.tiny_dim}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} token_lr:{token_lr} matrix_lr:{args.matrix_lr} "
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        optimizer.zero_grad(set_to_none=True)

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

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
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
            optimizer.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        optimizer.load_state_dict(initial_optimizer_state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = base.DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        # should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        should_validate =False # Disable validation during training to save time, final validation will be done after training completes.
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
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

    if master_process:
        adapter_state = base_model.adapter_state_dict()
        torch.save(adapter_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized adapter checkpoint: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

        quant_obj, quant_stats = base.quantize_state_dict_int8(adapter_state)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zlib.compress(quant_raw, level=9)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized adapter int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{len(quant_raw)} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(base.dequantize_state_dict_int8(quant_state), strict=False)
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

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()