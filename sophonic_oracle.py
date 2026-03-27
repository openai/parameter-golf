from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch

from sophonic_eval import (
    EvalConfig,
    GPT,
    SMALL_TENSOR_MAX,
    build_sentencepiece_luts,
    dequant,
    eval_bpb,
    find_model_file,
    load_validation_tokens,
    quant_per_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Oracle sensitivity scan for static mixed-precision upgrades."
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--high-bits", type=int, default=int(os.environ.get("ORACLE_HIGH_BITS", "8")))
    parser.add_argument("--low-bits", type=int, default=int(os.environ.get("ORACLE_LOW_BITS", "6")))
    parser.add_argument("--max-seqs", type=int, default=int(os.environ.get("SOPHONIC_MAX_SEQS", "1024")))
    parser.add_argument("--val-batch-size", type=int, default=int(os.environ.get("VAL_BATCH_SIZE", "8192")))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--oracle-ks", default="1,2,4")
    parser.add_argument("--limit-candidates", type=int, default=0)
    parser.add_argument("--output", default="oracle_matrix_scan.tsv")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
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


def normalize_state_dict(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in raw_sd):
        return {k.replace("module.", ""): v for k, v in raw_sd.items()}
    return raw_sd


def quantize_large_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    q, s = quant_per_row(t.detach().cpu(), bits)
    return dequant(q, s, t.dtype)


def quantized_state_dict(raw_sd: dict[str, torch.Tensor], bits: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in raw_sd.items():
        t = tensor.detach().cpu()
        if not t.is_floating_point() or t.numel() <= SMALL_TENSOR_MAX:
            out[name] = t
        else:
            out[name] = quantize_large_tensor(t, bits)
    return out


def candidate_matrix_names(raw_sd: dict[str, torch.Tensor]) -> list[str]:
    names = []
    for name, tensor in raw_sd.items():
        if tensor.is_floating_point() and tensor.ndim == 2 and tensor.numel() > SMALL_TENSOR_MAX:
            names.append(name)
    return sorted(names)


def layer_bucket(name: str) -> str:
    if name == "tok_emb.weight":
        return "tok_emb"
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    return name.split(".")[0]


def parse_ks(raw: str) -> list[int]:
    ks = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"oracle k values must be positive, got {value}")
        ks.append(value)
    if not ks:
        raise ValueError("oracle k list must not be empty")
    return ks


def eval_state_dict(
    label: str,
    model: GPT,
    state_dict: dict[str, torch.Tensor],
    val_tokens: torch.Tensor,
    cfg: EvalConfig,
    base_bytes_lut: torch.Tensor,
    has_space_lut: torch.Tensor,
    is_boundary_lut: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()}, strict=False)
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
    return val_loss, bpb


def main() -> None:
    args = build_parser().parse_args()
    ks = parse_ks(args.oracle_ks)
    os.environ["SOPHONIC_MAX_SEQS"] = str(args.max_seqs)

    cfg = EvalConfig()
    cfg.val_batch_size = args.val_batch_size
    if args.model:
        cfg.model_path = args.model

    device = choose_device(args.device)
    print(f"Device: {device}")

    model_path = cfg.model_path or find_model_file()
    print(f"Loading model: {model_path}")
    raw_sd = normalize_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    model = GPT(cfg).to(device).float()
    model.load_state_dict(raw_sd, strict=False)

    print(f"Loading validation tokens: {cfg.val_files}")
    val_tokens = load_validation_tokens(cfg.val_files, cfg.seq_len)
    print(f"Validation tokens: {val_tokens.numel() - 1:,}")
    print(f"SOPHONIC_MAX_SEQS={args.max_seqs}")

    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    high_sd = quantized_state_dict(raw_sd, args.high_bits)
    low_sd = quantized_state_dict(raw_sd, args.low_bits)

    print("\n" + "=" * 72)
    print(f"ORACLE SCAN — high=int{args.high_bits}, low=int{args.low_bits}")
    print("=" * 72)
    _, high_bpb = eval_state_dict(
        f"Uniform int{args.high_bits} baseline",
        model,
        high_sd,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )
    _, low_bpb = eval_state_dict(
        f"Uniform int{args.low_bits} base",
        model,
        low_sd,
        val_tokens,
        cfg,
        base_bytes_lut,
        has_space_lut,
        is_boundary_lut,
        device,
    )

    total_gap = low_bpb - high_bpb
    print(f"Recoverable gap (int{args.low_bits} -> int{args.high_bits}): {total_gap:+.4f} BPB")

    candidates = candidate_matrix_names(raw_sd)
    if args.limit_candidates > 0:
        candidates = candidates[: args.limit_candidates]
    print(f"Scanning {len(candidates)} candidate matrices")

    results: list[dict[str, object]] = []
    high_tensors = {name: high_sd[name] for name in candidates}

    for idx, name in enumerate(candidates, start=1):
        probe_sd = dict(low_sd)
        probe_sd[name] = high_tensors[name]
        _, bpb = eval_state_dict(
            f"[{idx:02d}/{len(candidates):02d}] upgrade {name}",
            model,
            probe_sd,
            val_tokens,
            cfg,
            base_bytes_lut,
            has_space_lut,
            is_boundary_lut,
            device,
        )
        improvement = low_bpb - bpb
        frac = improvement / total_gap if total_gap > 0 else 0.0
        shape = tuple(int(x) for x in raw_sd[name].shape)
        bucket = layer_bucket(name)
        results.append(
            {
                "name": name,
                "bucket": bucket,
                "shape": shape,
                "numel": int(raw_sd[name].numel()),
                "bpb": float(bpb),
                "improvement": float(improvement),
                "recoverable_frac": float(frac),
            }
        )

    results.sort(key=lambda row: float(row["improvement"]), reverse=True)

    print("\nTop matrices by oracle upgrade value")
    print(f"{'rank':>4}  {'matrix':<42} {'Δrecover':>9}  {'%gap':>6}")
    for rank, row in enumerate(results[: args.top_n], start=1):
        print(
            f"{rank:>4}  {row['name']:<42} "
            f"{row['improvement']:+9.4f}  {100.0 * row['recoverable_frac']:>5.1f}%"
        )

    bucket_rows: list[tuple[str, float, list[str]]] = []
    bucket_improvement: dict[str, float] = defaultdict(float)
    bucket_names: dict[str, list[str]] = defaultdict(list)
    for row in results:
        bucket_improvement[str(row["bucket"])] += float(row["improvement"])
        bucket_names[str(row["bucket"])].append(str(row["name"]))
    for bucket, improvement in bucket_improvement.items():
        bucket_rows.append((bucket, improvement, bucket_names[bucket]))
    bucket_rows.sort(key=lambda item: item[1], reverse=True)

    print("\nTop layers by summed matrix oracle value")
    print(f"{'rank':>4}  {'layer':<16} {'Δrecover':>9}")
    for rank, (bucket, improvement, _) in enumerate(bucket_rows[: args.top_n], start=1):
        print(f"{rank:>4}  {bucket:<16} {improvement:+9.4f}")

    print("\nOracle top-k matrix upgrades")
    print(f"{'k':>4}  {'val_bpb':>8}  {'Δvs low':>9}  {'%gap':>6}")
    for k in ks:
        chosen = {str(row["name"]) for row in results[:k]}
        probe_sd = dict(low_sd)
        for name in chosen:
            probe_sd[name] = high_tensors[name]
        _, bpb = eval_state_dict(
            f"oracle top-{k} matrices",
            model,
            probe_sd,
            val_tokens,
            cfg,
            base_bytes_lut,
            has_space_lut,
            is_boundary_lut,
            device,
        )
        improvement = low_bpb - bpb
        frac = improvement / total_gap if total_gap > 0 else 0.0
        print(f"{k:>4}  {bpb:>8.4f}  {improvement:+9.4f}  {100.0 * frac:>5.1f}%")

    print("\nOracle top-k layer upgrades")
    print(f"{'k':>4}  {'val_bpb':>8}  {'Δvs low':>9}  {'%gap':>6}")
    for k in ks:
        chosen_buckets = {bucket for bucket, _, _ in bucket_rows[:k]}
        chosen_names = {
            str(row["name"])
            for row in results
            if str(row["bucket"]) in chosen_buckets
        }
        probe_sd = dict(low_sd)
        for name in chosen_names:
            probe_sd[name] = high_tensors[name]
        _, bpb = eval_state_dict(
            f"oracle top-{k} layers",
            model,
            probe_sd,
            val_tokens,
            cfg,
            base_bytes_lut,
            has_space_lut,
            is_boundary_lut,
            device,
        )
        improvement = low_bpb - bpb
        frac = improvement / total_gap if total_gap > 0 else 0.0
        print(f"{k:>4}  {bpb:>8.4f}  {improvement:+9.4f}  {100.0 * frac:>5.1f}%")

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("rank\tmatrix\tlayer_bucket\tshape\tnumel\tbpb\timprovement\trecoverable_frac\n")
        for rank, row in enumerate(results, start=1):
            f.write(
                f"{rank}\t{row['name']}\t{row['bucket']}\t{row['shape']}\t{row['numel']}\t"
                f"{row['bpb']:.6f}\t{row['improvement']:.6f}\t{row['recoverable_frac']:.6f}\n"
            )
    print(f"\nSaved matrix ranking to {output_path}")


if __name__ == "__main__":
    main()
