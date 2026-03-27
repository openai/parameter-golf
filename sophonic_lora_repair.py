from __future__ import annotations

import argparse
import contextlib
import glob
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sophonic_eval import (
    EvalConfig,
    GPT,
    SMALL_TENSOR_MAX,
    build_sentencepiece_luts,
    dequant,
    eval_bpb,
    load_validation_tokens,
    quant_per_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimum-viable LoRA repair test on an intN-quantized base model."
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "final_model.pt"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--base-bits", type=int, default=int(os.environ.get("LORA_BASE_BITS", "6")))
    parser.add_argument("--high-bits", type=int, default=int(os.environ.get("LORA_HIGH_BITS", "8")))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("LORA_RANK", "8")))
    parser.add_argument("--alpha", type=float, default=float(os.environ.get("LORA_ALPHA", "16.0")))
    parser.add_argument(
        "--target-regex",
        default=os.environ.get("LORA_TARGET_REGEX", r"^blocks\.[5-8]\.mlp\.(fc|proj)$"),
    )
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("LORA_MAX_STEPS", "300")))
    parser.add_argument(
        "--max-wallclock-seconds",
        type=int,
        default=int(os.environ.get("LORA_MAX_WALLCLOCK_SECONDS", "600")),
    )
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=int(os.environ.get("LORA_TRAIN_BATCH_TOKENS", "8192")),
    )
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LORA_LR", "0.002")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("LORA_WEIGHT_DECAY", "0.0")))
    parser.add_argument("--grad-clip", type=float, default=float(os.environ.get("LORA_GRAD_CLIP", "1.0")))
    parser.add_argument("--eval-every", type=int, default=int(os.environ.get("LORA_EVAL_EVERY", "50")))
    parser.add_argument("--eval-max-seqs", type=int, default=int(os.environ.get("LORA_EVAL_MAX_SEQS", "256")))
    parser.add_argument(
        "--final-eval-max-seqs",
        type=int,
        default=int(os.environ.get("LORA_FINAL_EVAL_MAX_SEQS", "0")),
    )
    parser.add_argument("--val-batch-size", type=int, default=int(os.environ.get("VAL_BATCH_SIZE", "4096")))
    parser.add_argument("--max-train-files", type=int, default=int(os.environ.get("LORA_MAX_TRAIN_FILES", "2")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LORA_SEED", "1337")))
    parser.add_argument(
        "--save-best-path",
        default=os.environ.get("LORA_SAVE_BEST_PATH", "lora_repair_best_merged.pt"),
    )
    return parser


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_state_dict(raw_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    if any(k.startswith("module.") for k in raw_sd):
        return {k.replace("module.", ""): v for k, v in raw_sd.items()}
    return raw_sd


def quantize_large_tensor(t: Tensor, bits: int) -> Tensor:
    q, s = quant_per_row(t.detach().cpu(), bits)
    return dequant(q, s, t.dtype)


def quantized_state_dict(raw_sd: dict[str, Tensor], bits: int) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, tensor in raw_sd.items():
        t = tensor.detach().cpu()
        if not t.is_floating_point() or t.numel() <= SMALL_TENSOR_MAX:
            out[name] = t
        else:
            out[name] = quantize_large_tensor(t, bits)
    return out


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
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


class TokenStream:
    def __init__(self, pattern: str, max_files: int):
        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if max_files > 0:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.files = files
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

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


class LocalTokenLoader:
    def __init__(self, pattern: str, device: torch.device, max_files: int):
        self.device = device
        self.stream = TokenStream(pattern, max_files=max_files)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        local_tokens = (batch_tokens // seq_len) * seq_len
        if local_tokens < seq_len:
            raise ValueError(f"batch_tokens={batch_tokens} too small for seq_len={seq_len}")
        chunk = self.stream.take(local_tokens + 1)
        local = chunk.to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device), y.to(self.device)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 0.0

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        base_out = F.linear(x, self.weight.to(x.dtype), bias)
        lora_mid = F.linear(x, self.lora_a.to(x.dtype))
        lora_out = F.linear(lora_mid, self.lora_b.to(x.dtype))
        return base_out + self.scaling * lora_out

    def merged_weight(self) -> Tensor:
        delta = self.lora_b.detach().float() @ self.lora_a.detach().float()
        return self.weight.detach().float() + self.scaling * delta


def get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def install_lora_modules(model: nn.Module, pattern: str, rank: int, alpha: float) -> list[str]:
    regex = re.compile(pattern)
    chosen: list[str] = []
    replacements: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if regex.search(name):
            replacements.append((name, module))
    for name, module in replacements:
        parent, leaf = get_parent_module(model, name)
        replacement = LoRALinear(module, rank=rank, alpha=alpha).to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        setattr(parent, leaf, replacement)
        chosen.append(name)
    return chosen


def freeze_non_lora_params(model: nn.Module) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        is_lora = name.endswith("lora_a") or name.endswith("lora_b")
        param.requires_grad_(is_lora)
        if is_lora:
            trainable.append(param)
    return trainable


def merged_plain_state_dict(model: nn.Module) -> dict[str, Tensor]:
    merged = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not (name.endswith("lora_a") or name.endswith("lora_b"))
    }
    for module_name, module in model.named_modules():
        if not module_name or not isinstance(module, LoRALinear):
            continue
        merged[f"{module_name}.weight"] = module.merged_weight().to(dtype=module.weight.dtype).cpu().contiguous()
        if module.bias is not None:
            merged[f"{module_name}.bias"] = module.bias.detach().cpu().clone()
    return merged


def reset_rotary_caches(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_cache"):
            module._cache = None
        if hasattr(module, "_seq_len_cached"):
            module._seq_len_cached = 0


@contextlib.contextmanager
def temp_eval_limit(max_seqs: int):
    old = os.environ.get("SOPHONIC_MAX_SEQS")
    os.environ["SOPHONIC_MAX_SEQS"] = str(max_seqs)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SOPHONIC_MAX_SEQS", None)
        else:
            os.environ["SOPHONIC_MAX_SEQS"] = old


def eval_model(
    label: str,
    model: nn.Module,
    val_tokens: Tensor,
    cfg: EvalConfig,
    base_bytes_lut: Tensor,
    has_space_lut: Tensor,
    is_boundary_lut: Tensor,
    device: torch.device,
    eval_max_seqs: int,
) -> tuple[float, float]:
    model.eval()
    with temp_eval_limit(eval_max_seqs):
        t0 = time.time()
        val_loss, bpb = eval_bpb(
            model,
            val_tokens,
            cfg.seq_len,
            cfg.val_batch_size,
            base_bytes_lut,
            has_space_lut,
            is_boundary_lut,
            device,
        )
        dt = time.time() - t0
    print(f"{label:<44} val_bpb={bpb:.4f}  val_loss={val_loss:.4f}  ({dt:.0f}s)")
    reset_rotary_caches(model)
    model.train()
    return val_loss, bpb


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    cfg = EvalConfig()
    cfg.model_path = args.model
    cfg.val_batch_size = args.val_batch_size

    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Loading model: {cfg.model_path}")
    raw_sd = normalize_state_dict(torch.load(cfg.model_path, map_location="cpu", weights_only=True))

    print(f"Loading validation tokens: {cfg.val_files}")
    val_tokens = load_validation_tokens(cfg.val_files, cfg.seq_len)
    print(f"Validation tokens: {val_tokens.numel() - 1:,}")

    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    high_sd = quantized_state_dict(raw_sd, args.high_bits)
    low_sd = quantized_state_dict(raw_sd, args.base_bits)

    int8_model = GPT(cfg).to(device).float()
    int8_model.load_state_dict(high_sd, strict=False)

    repair_model = GPT(cfg).to(device).float()
    repair_model.load_state_dict(low_sd, strict=False)

    chosen_modules = install_lora_modules(repair_model, args.target_regex, rank=args.rank, alpha=args.alpha)
    if not chosen_modules:
        raise ValueError(f"No modules matched target regex: {args.target_regex}")
    trainable = freeze_non_lora_params(repair_model)
    trainable_count = sum(int(p.numel()) for p in trainable)

    print("\n" + "=" * 72)
    print(f"LoRA REPAIR — int{args.base_bits} base, target=int{args.high_bits}")
    print("=" * 72)
    print(f"Target regex: {args.target_regex}")
    print(f"Chosen modules ({len(chosen_modules)}):")
    for name in chosen_modules:
        print(f"  - {name}")
    print(f"Trainable LoRA params: {trainable_count:,}")

    _, high_bpb = eval_model(
        f"Uniform int{args.high_bits} baseline",
        int8_model,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
        args.eval_max_seqs,
    )
    _, low_bpb = eval_model(
        f"Uniform int{args.base_bits} base",
        repair_model,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
        args.eval_max_seqs,
    )

    gap = low_bpb - high_bpb
    print(f"Recoverable gap (int{args.base_bits} -> int{args.high_bits}): {gap:+.4f} BPB")
    reset_rotary_caches(repair_model)

    train_loader = LocalTokenLoader(cfg.data_path + "/fineweb_train_*.bin", device=device, max_files=args.max_train_files)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    best_bpb = low_bpb
    best_step = 0
    best_state_dict = merged_plain_state_dict(repair_model)
    start = time.time()
    last_loss = float("nan")

    print("\nStarting LoRA repair training")
    for step in range(1, args.max_steps + 1):
        if time.time() - start >= args.max_wallclock_seconds:
            print(f"Stopping early at step {step - 1}: wallclock cap")
            break

        x, y = train_loader.next_batch(args.train_batch_tokens, cfg.seq_len)
        optimizer.zero_grad(set_to_none=True)
        loss = repair_model(x, y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())

        if step == 1 or step % 10 == 0:
            elapsed = time.time() - start
            print(f"step:{step}/{args.max_steps} train_loss:{last_loss:.4f} elapsed:{elapsed:.1f}s")

        if step == args.max_steps or step % args.eval_every == 0:
            _, current_bpb = eval_model(
                f"LoRA repair step {step}",
                repair_model,
                val_tokens,
                cfg,
                base_bytes_lut,
                has_space_lut,
                is_boundary_lut,
                device,
                args.eval_max_seqs,
            )
            if current_bpb < best_bpb:
                best_bpb = current_bpb
                best_step = step
                best_state_dict = merged_plain_state_dict(repair_model)
                torch.save(best_state_dict, args.save_best_path)
                print(f"  saved best merged checkpoint to {args.save_best_path}")
            improvement = low_bpb - current_bpb
            recovered = improvement / gap if gap > 0 else 0.0
            print(
                f"  repair gain vs int{args.base_bits}: {improvement:+.4f} BPB "
                f"({100.0 * recovered:.1f}% of int{args.base_bits}->int{args.high_bits} gap)"
            )

    if best_step == 0:
        torch.save(best_state_dict, args.save_best_path)
        print(f"Saved baseline merged checkpoint to {args.save_best_path}")

    if args.final_eval_max_seqs > 0:
        final_int8_model = GPT(cfg).to(device).float()
        final_int8_model.load_state_dict(high_sd, strict=False)
        _, final_high_bpb = eval_model(
            f"Uniform int{args.high_bits} final baseline",
            final_int8_model,
            val_tokens,
            cfg,
            base_bytes_lut,
            has_space_lut,
            is_boundary_lut,
            device,
            args.final_eval_max_seqs,
        )
        final_low_model = GPT(cfg).to(device).float()
        final_low_model.load_state_dict(low_sd, strict=False)
        _, final_low_bpb = eval_model(
            f"Uniform int{args.base_bits} final base",
            final_low_model,
            val_tokens,
            cfg,
            base_bytes_lut,
            has_space_lut,
            is_boundary_lut,
            device,
            args.final_eval_max_seqs,
        )
        final_model = GPT(cfg).to(device).float()
        final_model.load_state_dict(best_state_dict, strict=False)
        _, final_bpb = eval_model(
            "Best merged repair final eval",
            final_model,
            val_tokens,
            cfg,
            base_bytes_lut,
            has_space_lut,
            is_boundary_lut,
            device,
            args.final_eval_max_seqs,
        )
        final_gap = final_low_bpb - final_high_bpb
        final_gain = final_low_bpb - final_bpb
        final_recovered = final_gain / final_gap if final_gap > 0 else 0.0
        print(
            f"Final eval recovered: {final_gain:+.4f} BPB "
            f"({100.0 * final_recovered:.1f}% of final-eval gap)"
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Uniform int{args.high_bits}: {high_bpb:.4f} BPB")
    print(f"Uniform int{args.base_bits}: {low_bpb:.4f} BPB")
    print(f"Best LoRA repair: {best_bpb:.4f} BPB at step {best_step}")
    gain = low_bpb - best_bpb
    recovered = gain / gap if gap > 0 else 0.0
    print(f"Gap to recover: {gap:+.4f} BPB")
    print(f"Recovered by LoRA: {gain:+.4f} BPB ({100.0 * recovered:.1f}% of gap)")
    print(f"Final train loss: {last_loss:.4f}")


if __name__ == "__main__":
    main()
