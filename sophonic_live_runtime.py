from __future__ import annotations

import argparse
import io
import os
import re
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sophonic_eval import EvalConfig, GPT, build_sentencepiece_luts, dequant, eval_bpb, load_validation_tokens, quant_per_row
from sophonic_lora_repair import get_parent_module, normalize_state_dict, quantized_state_dict
from train_gpt import INT8_KEEP_FLOAT_MAX_NUMEL, keep_float_tensor, tensor_nbytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a live intN base + separate Sophonic deltas runtime artifact."
    )
    parser.add_argument("--base-model", required=True, help="Original trained checkpoint before LoRA repair.")
    parser.add_argument("--repaired-model", required=True, help="Merged repaired checkpoint from sophonic_lora_repair.py.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--base-bits", type=int, default=int(os.environ.get("SOPHONIC_BASE_BITS", "5")))
    parser.add_argument("--high-bits", type=int, default=int(os.environ.get("SOPHONIC_HIGH_BITS", "8")))
    parser.add_argument(
        "--target-regex",
        default=os.environ.get("SOPHONIC_TARGET_REGEX", r"^blocks\.[7-8]\.mlp\.(fc|proj)$"),
    )
    parser.add_argument("--delta-rank", type=int, default=int(os.environ.get("SOPHONIC_DELTA_RANK", "16")))
    parser.add_argument(
        "--delta-dtype",
        choices=("float16", "float32"),
        default=os.environ.get("SOPHONIC_DELTA_DTYPE", "float16"),
    )
    parser.add_argument("--val-batch-size", type=int, default=int(os.environ.get("VAL_BATCH_SIZE", "131072")))
    parser.add_argument("--eval-max-seqs", type=int, default=int(os.environ.get("SOPHONIC_MAX_SEQS", "0")))
    parser.add_argument(
        "--save-artifact-path",
        default=os.environ.get("SOPHONIC_SAVE_ARTIFACT_PATH", ""),
        help="Optional path to save the compressed live artifact.",
    )
    parser.add_argument(
        "--code-files",
        default="train_gpt.py,sophonic_eval.py,sophonic_live_runtime.py",
        help="Comma-separated list of code files to count toward artifact size.",
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


def code_bytes(paths: list[str]) -> int:
    total = 0
    for raw in paths:
        path = Path(raw.strip())
        if not raw.strip():
            continue
        if not path.exists():
            raise FileNotFoundError(f"Code file not found: {path}")
        total += path.stat().st_size
    return total


def quantize_state_dict_nbits(state_dict: dict[str, Tensor], bits: int) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "payload_bytes"),
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
            stats["payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quant_per_row(t, bits)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q.contiguous()
        scales[name] = s.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": f"int{bits}_clean_per_row_v1",
        "bits": bits,
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


def dequantize_state_dict_nbits(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        out[name] = dequant(q, s, dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def choose_target_modules(cfg: EvalConfig, pattern: str) -> list[str]:
    regex = re.compile(pattern)
    probe = GPT(cfg)
    chosen: list[str] = []
    for name, module in probe.named_modules():
        if name and isinstance(module, nn.Linear) and regex.search(name):
            chosen.append(name)
    return chosen


def factorize_delta(delta: Tensor, rank: int, store_dtype: torch.dtype) -> tuple[Tensor, Tensor, float]:
    delta_f = delta.float()
    u, s, vh = torch.linalg.svd(delta_f, full_matrices=False)
    r = min(rank, int(s.numel()))
    if r <= 0:
        raise ValueError("delta rank must be positive")
    energy = float((s[:r].square().sum() / s.square().sum()).item()) if s.numel() else 1.0
    a = vh[:r, :].to(dtype=store_dtype).contiguous()
    b = (u[:, :r] * s[:r].unsqueeze(0)).to(dtype=store_dtype).contiguous()
    return a.cpu(), b.cpu(), energy


class SophonicLinear(nn.Module):
    def __init__(self, base: nn.Linear, delta_a: Tensor, delta_b: Tensor):
        super().__init__()
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.register_buffer("delta_a", delta_a.detach().clone())
        self.register_buffer("delta_b", delta_b.detach().clone())

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        base_out = F.linear(x, self.weight.to(x.dtype), bias)
        delta_mid = F.linear(x, self.delta_a.to(x.dtype))
        delta_out = F.linear(delta_mid, self.delta_b.to(x.dtype))
        return base_out + delta_out


def install_sophonic_modules(model: nn.Module, entries: dict[str, dict[str, Tensor]]) -> None:
    for module_name, payload in entries.items():
        parent, leaf = get_parent_module(model, module_name)
        base = getattr(parent, leaf)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {module_name}, found {type(base).__name__}")
        replacement = SophonicLinear(base, payload["a"], payload["b"]).to(
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        setattr(parent, leaf, replacement)


def eval_model(
    label: str,
    model: nn.Module,
    val_tokens: Tensor,
    cfg: EvalConfig,
    base_bytes_lut: Tensor,
    has_space_lut: Tensor,
    is_boundary_lut: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
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
    print(f"{label:<40} val_bpb={bpb:.4f}  val_loss={val_loss:.4f}  ({dt:.0f}s)")
    return val_loss, bpb


def maybe_limit_eval_tokens(val_tokens: Tensor, seq_len: int, max_seqs: int) -> Tensor:
    if max_seqs <= 0:
        return val_tokens
    usable = min(max_seqs * seq_len, val_tokens.numel() - 1)
    usable = (usable // seq_len) * seq_len
    return val_tokens[: usable + 1].contiguous()


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    delta_store_dtype = torch.float16 if args.delta_dtype == "float16" else torch.float32

    cfg = EvalConfig()
    cfg.model_path = args.base_model
    cfg.val_batch_size = args.val_batch_size

    print(f"Device: {device}")
    print(f"Loading base model: {args.base_model}")
    raw_sd = normalize_state_dict(torch.load(args.base_model, map_location="cpu", weights_only=True))
    print(f"Loading repaired model: {args.repaired_model}")
    repaired_sd = normalize_state_dict(torch.load(args.repaired_model, map_location="cpu", weights_only=True))

    print(f"Loading validation tokens: {cfg.val_files}")
    full_val_tokens = load_validation_tokens(cfg.val_files, cfg.seq_len)
    val_tokens = maybe_limit_eval_tokens(full_val_tokens, cfg.seq_len, args.eval_max_seqs)
    print(f"Validation tokens: {val_tokens.numel() - 1:,}")

    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    high_sd = quantized_state_dict(raw_sd, args.high_bits)
    base_obj, base_stats = quantize_state_dict_nbits(raw_sd, args.base_bits)
    low_sd = dequantize_state_dict_nbits(base_obj)

    chosen_modules = choose_target_modules(cfg, args.target_regex)
    if not chosen_modules:
        raise ValueError(f"No modules matched target regex: {args.target_regex}")

    delta_entries: dict[str, dict[str, Tensor]] = {}
    energy_vals: list[float] = []
    delta_payload_bytes = 0
    for module_name in chosen_modules:
        weight_key = f"{module_name}.weight"
        if weight_key not in repaired_sd or weight_key not in low_sd:
            raise KeyError(f"Missing weight for target module: {weight_key}")
        delta = repaired_sd[weight_key].float() - low_sd[weight_key].float()
        a, b, energy = factorize_delta(delta, args.delta_rank, delta_store_dtype)
        delta_entries[module_name] = {"a": a, "b": b}
        energy_vals.append(energy)
        delta_payload_bytes += tensor_nbytes(a) + tensor_nbytes(b)

    artifact_obj: dict[str, object] = {
        "__format__": "sophonic_live_delta_v1",
        "base_bits": args.base_bits,
        "delta_rank": args.delta_rank,
        "delta_dtype": args.delta_dtype,
        "target_modules": chosen_modules,
        "base": base_obj,
        "deltas": delta_entries,
    }
    artifact_buf = io.BytesIO()
    torch.save(artifact_obj, artifact_buf)
    artifact_raw = artifact_buf.getvalue()
    artifact_blob = zlib.compress(artifact_raw, level=9)
    if args.save_artifact_path:
        out_path = Path(args.save_artifact_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(artifact_blob)
        print(f"Saved live Sophonic artifact to {out_path}")

    print("\n" + "=" * 72)
    print(f"LIVE SOPHONIC RUNTIME — int{args.base_bits} base + separate deltas")
    print("=" * 72)
    print(f"Target regex: {args.target_regex}")
    print(f"Chosen modules ({len(chosen_modules)}):")
    for name, energy in zip(chosen_modules, energy_vals, strict=True):
        print(f"  - {name}  energy@rank{args.delta_rank}={100.0 * energy:.1f}%")
    print(f"Mean delta energy captured: {100.0 * sum(energy_vals) / len(energy_vals):.1f}%")

    int8_model = GPT(cfg).to(device).float()
    int8_model.load_state_dict(high_sd, strict=False)
    _, high_bpb = eval_model(
        f"Uniform int{args.high_bits} baseline",
        int8_model,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )

    base_model = GPT(cfg).to(device).float()
    base_model.load_state_dict(low_sd, strict=False)
    _, low_bpb = eval_model(
        f"Uniform int{args.base_bits} base",
        base_model,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )

    merged_model = GPT(cfg).to(device).float()
    merged_model.load_state_dict(repaired_sd, strict=False)
    _, merged_bpb = eval_model(
        "Merged repaired checkpoint",
        merged_model,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )

    live_base_sd = dequantize_state_dict_nbits(artifact_obj["base"])
    live_model = GPT(cfg).to(device).float()
    live_model.load_state_dict(live_base_sd, strict=False)
    install_sophonic_modules(live_model, artifact_obj["deltas"])
    _, live_bpb = eval_model(
        "Live intN + Sophonic deltas",
        live_model,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )

    code_list = [part for part in args.code_files.split(",") if part.strip()]
    bytes_code = code_bytes(code_list)
    bytes_model = len(artifact_blob)
    bytes_total = bytes_model + bytes_code

    gap = low_bpb - high_bpb
    live_gain = low_bpb - live_bpb
    live_recovered = live_gain / gap if gap > 0 else 0.0
    merged_gap = live_bpb - merged_bpb

    print("\n" + "=" * 72)
    print("ARTIFACT SIZE")
    print("=" * 72)
    print(f"bytes_base_payload_estimate: {base_stats['payload_bytes']}")
    print(f"bytes_delta_payload_estimate: {delta_payload_bytes}")
    print(f"bytes_model_live_zlib: {bytes_model}")
    print(f"bytes_code: {bytes_code}")
    print(f"bytes_total: {bytes_total}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Uniform int{args.high_bits}: {high_bpb:.4f} BPB")
    print(f"Uniform int{args.base_bits}: {low_bpb:.4f} BPB")
    print(f"Merged repaired: {merged_bpb:.4f} BPB")
    print(f"Live int{args.base_bits}+deltas: {live_bpb:.4f} BPB")
    print(f"Gap to recover: {gap:+.4f} BPB")
    print(f"Recovered live: {live_gain:+.4f} BPB ({100.0 * live_recovered:.1f}% of gap)")
    print(f"Live vs merged delta: {merged_gap:+.4f} BPB")


if __name__ == "__main__":
    main()
