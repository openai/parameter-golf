#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import inspect
import io
import math
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import train_gpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint-only eval/export frontier sweep.")
    parser.add_argument("--checkpoint", required=True, help="Path to a raw model checkpoint (.pt)")
    parser.add_argument("--summary-out", default="", help="Optional CSV output path")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--num-unique-layers", type=int, default=0)
    parser.add_argument("--num-recurrence", type=int, default=1)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--smeargate-enabled", action="store_true")
    parser.add_argument("--bigram-vocab-size", type=int, default=0)
    parser.add_argument("--bigram-dim", type=int, default=0)
    parser.add_argument("--prevlogit-rank", type=int, default=0)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=524288)
    parser.add_argument("--eval-seq-lens", default="", help="Comma-separated eval seq lens; default=train_seq_len")
    parser.add_argument("--strides", default="1024,512,256,128,64")
    parser.add_argument(
        "--modes",
        default="stream_flat,stream_sliding,doc_flat,doc_sliding",
        help="Comma-separated modes: stream_flat,stream_sliding,doc_flat,doc_sliding",
    )
    parser.add_argument(
        "--variant-names",
        default="",
        help="Optional comma-separated subset of variant names to run (e.g. prequant,int8_zlib)",
    )
    parser.add_argument("--late-k-patterns", default="", help="Comma-separated additional fp16 keep-float patterns")
    parser.add_argument(
        "--extra-lowbit-patterns",
        default="",
        help="Comma-separated extra lowbit patterns to append to the core int6 pattern set",
    )
    parser.add_argument("--k-group-size", type=int, default=64, help="Grouped int8 group size for non-passthrough c_k weights")
    parser.add_argument("--max-docs", type=int, default=0, help="If >0, only evaluate the first N documents")
    parser.add_argument("--max-val-tokens", type=int, default=0, help="If >0, only evaluate the first N tokens")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> train_gpt.GPT:
    kwargs = dict(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        num_recurrence=args.num_recurrence,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        compression_aware_kl_weight=0.0,
        smeargate_enabled=args.smeargate_enabled,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    )
    if "prevlogit_rank" in inspect.signature(train_gpt.GPT).parameters:
        kwargs["prevlogit_rank"] = args.prevlogit_rank
    model = train_gpt.GPT(**kwargs).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, train_gpt.CastedLinear):
            module.float()
    train_gpt.restore_low_dim_params_to_fp32(model)
    return model


def configure_cuda_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)


def compile_eval_functions(base_model: train_gpt.GPT) -> tuple[torch.nn.Module, callable]:
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    return compiled_model, compiled_logits


def warm_eval_functions(
    compiled_model: torch.nn.Module,
    compiled_logits: callable,
    val_tokens: torch.Tensor,
    device: torch.device,
    seq_len: int,
    val_batch_size: int,
) -> None:
    warm_batch_seqs = max(1, val_batch_size // seq_len)
    desired = warm_batch_seqs * seq_len + 1
    warm_tokens = val_tokens[:desired]
    usable = ((warm_tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Need at least {seq_len + 1} validation tokens for warmup; got {int(warm_tokens.numel())}")
    warm_tokens = warm_tokens[: usable + 1].to(device=device, dtype=torch.int64, non_blocking=True)
    x = warm_tokens[:-1].reshape(-1, seq_len)
    y = warm_tokens[1:].reshape(-1, seq_len)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            _ = compiled_model(x, y)
            _ = compiled_logits(x)
    torch.cuda.synchronize()


def parse_patterns(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_int_list(text: str, fallback: int) -> list[int]:
    if not text.strip():
        return [fallback]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


@contextlib.contextmanager
def serializer_overrides(
    *,
    keep_fp16_patterns: tuple[str, ...],
    lowbit_patterns: tuple[str, ...],
    lowbit_bits: int,
    group_overrides: tuple[tuple[str, int], ...],
    compressor: str,
):
    saved = {
        "INT8_KEEP_FLOAT_NAME_PATTERNS": train_gpt.INT8_KEEP_FLOAT_NAME_PATTERNS,
        "LOWBIT_NAME_PATTERNS": train_gpt.LOWBIT_NAME_PATTERNS,
        "LOWBIT_BITS": train_gpt.LOWBIT_BITS,
        "INT8_GROUP_OVERRIDES": train_gpt.INT8_GROUP_OVERRIDES,
        "SERIAL_COMPRESSOR": train_gpt.SERIAL_COMPRESSOR,
    }
    try:
        train_gpt.INT8_KEEP_FLOAT_NAME_PATTERNS = tuple(keep_fp16_patterns)
        train_gpt.LOWBIT_NAME_PATTERNS = tuple(lowbit_patterns)
        train_gpt.LOWBIT_BITS = int(lowbit_bits)
        train_gpt.INT8_GROUP_OVERRIDES = tuple(group_overrides)
        train_gpt.SERIAL_COMPRESSOR = compressor
        yield
    finally:
        for key, value in saved.items():
            setattr(train_gpt, key, value)


def serialize_variant_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    keep_fp16_patterns: tuple[str, ...],
    lowbit_patterns: tuple[str, ...],
    lowbit_bits: int,
    group_overrides: tuple[tuple[str, int], ...],
    compressor: str,
) -> tuple[bytes, int, int, dict[str, torch.Tensor]]:
    with serializer_overrides(
        keep_fp16_patterns=keep_fp16_patterns,
        lowbit_patterns=lowbit_patterns,
        lowbit_bits=lowbit_bits,
        group_overrides=group_overrides,
        compressor=compressor,
    ):
        qobj, stats = train_gpt.quantize_state_dict_serial(state_dict)
        buf = io.BytesIO()
        torch.save(qobj, buf)
        raw = buf.getvalue()
        blob = train_gpt.compress_serialized_model(raw)
        restored = torch.load(io.BytesIO(train_gpt.decompress_serialized_model(blob)), map_location="cpu")
        dequantized_state = train_gpt.dequantize_state_dict_int8(restored)
    return blob, len(raw), stats["payload_bytes"], dequantized_state


def load_validation_tokens_full(pattern: str) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return torch.cat([train_gpt.load_data_shard(file) for file in files]).contiguous()


def split_doc_ranges(val_tokens: torch.Tensor, bos_id: int) -> list[tuple[int, int]]:
    bos_positions = (val_tokens == bos_id).nonzero(as_tuple=False).flatten().tolist()
    if not bos_positions:
        return [(0, int(val_tokens.numel()))]
    if bos_positions[0] != 0:
        bos_positions.insert(0, 0)
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(bos_positions):
        end = bos_positions[i + 1] if i + 1 < len(bos_positions) else int(val_tokens.numel())
        if end - start >= 2:
            ranges.append((start, end))
    return ranges


def build_windows(
    ranges: list[tuple[int, int]],
    seq_len: int,
    stride: int | None,
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    for start, end in ranges:
        doc_len = end - start
        if doc_len < 2:
            continue
        if stride is None:
            local_starts = range(0, doc_len - 1, seq_len)
            next_target = 1
        else:
            local_starts = range(0, doc_len - 1, stride)
            next_target = 1
        for local_start in local_starts:
            actual_len = min(seq_len, doc_len - local_start - 1)
            if actual_len <= 0:
                break
            if stride is None:
                score_from = 0
                score_len = actual_len
            else:
                window_target_start = local_start + 1
                window_target_end = local_start + actual_len
                score_target_start = max(next_target, window_target_start)
                if score_target_start > window_target_end:
                    continue
                score_from = score_target_start - window_target_start
                score_len = window_target_end - score_target_start + 1
                next_target = window_target_end + 1
            windows.append((start + local_start, actual_len, score_from, score_len))
    return windows


def eval_windows(
    model: train_gpt.GPT,
    logits_fn: callable,
    seq_len: int,
    val_batch_size: int,
    val_tokens: torch.Tensor,
    windows: list[tuple[int, int, int, int]],
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    vocab_size: int,
    pad_id: int,
) -> tuple[float, float]:
    eval_batch_windows = max(1, val_batch_size // seq_len)
    val_nll_sum = 0.0
    val_token_count = 0
    val_byte_count = 0.0
    model.eval()

    with torch.inference_mode():
        for batch_off in range(0, len(windows), eval_batch_windows):
            batch = windows[batch_off:batch_off + eval_batch_windows]
            bs = len(batch)
            fixed_bs = eval_batch_windows
            x_cpu = torch.full((fixed_bs, seq_len), fill_value=pad_id, dtype=torch.int64)
            y_cpu = torch.full((fixed_bs, seq_len), fill_value=pad_id, dtype=torch.int64)
            score_mask_cpu = torch.zeros((fixed_bs, seq_len), dtype=torch.bool)

            for row_idx, (token_start, actual_len, score_from, score_len) in enumerate(batch):
                chunk = val_tokens[token_start: token_start + actual_len + 1].to(dtype=torch.int64)
                x_cpu[row_idx, :actual_len] = chunk[:-1]
                y_cpu[row_idx, :actual_len] = chunk[1:]
                score_mask_cpu[row_idx, score_from: score_from + score_len] = True

            x_batch = x_cpu.to(device=device, non_blocking=True)
            y_batch = y_cpu.to(device=device, non_blocking=True)
            score_mask = score_mask_cpu.to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = logits_fn(x_batch).view(fixed_bs, seq_len, vocab_size)

            per_token_nll = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y_batch.reshape(-1),
                reduction="none",
            ).view(fixed_bs, seq_len)
            val_nll_sum += float(per_token_nll.masked_select(score_mask).sum().item())
            val_token_count += int(score_mask.sum().item())

            prev_ids = x_batch.masked_select(score_mask)
            tgt_ids = y_batch.masked_select(score_mask)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += float(token_bytes.float().sum().item())

    val_loss = val_nll_sum / val_token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / val_byte_count
    model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def run_eval_mode(
    mode_name: str,
    seq_len: int,
    stride: int | None,
    model: train_gpt.GPT,
    logits_fn: callable,
    val_batch_size: int,
    val_tokens: torch.Tensor,
    stream_ranges: list[tuple[int, int]],
    doc_ranges: list[tuple[int, int]],
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    vocab_size: int,
    pad_id: int,
) -> tuple[float, float]:
    if mode_name == "stream_flat":
        windows = build_windows(stream_ranges, seq_len=seq_len, stride=None)
    elif mode_name == "stream_sliding":
        if stride is None:
            raise ValueError("stream_sliding requires stride")
        windows = build_windows(stream_ranges, seq_len=seq_len, stride=stride)
    elif mode_name == "doc_flat":
        windows = build_windows(doc_ranges, seq_len=seq_len, stride=None)
    elif mode_name == "doc_sliding":
        if stride is None:
            raise ValueError("doc_sliding requires stride")
        windows = build_windows(doc_ranges, seq_len=seq_len, stride=stride)
    else:
        raise ValueError(f"Unknown mode: {mode_name}")

    return eval_windows(
        model=model,
        logits_fn=logits_fn,
        seq_len=seq_len,
        val_batch_size=val_batch_size,
        val_tokens=val_tokens,
        windows=windows,
        device=device,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        vocab_size=vocab_size,
        pad_id=pad_id,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_cuda_runtime()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    summary_path = Path(args.summary_out) if args.summary_out else Path("logs") / f"{checkpoint_path.stem}_frontier.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", 0)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab size {int(sp.vocab_size())}")

    val_tokens = load_validation_tokens_full(os.path.join(args.data_path, "fineweb_val_*.bin"))
    if args.max_val_tokens > 0:
        val_tokens = val_tokens[: args.max_val_tokens].contiguous()
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = train_gpt.build_sentencepiece_luts(sp, args.vocab_size, device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    bos_id = int(sp.bos_id())
    pad_id = int(sp.pad_id())
    doc_ranges = split_doc_ranges(val_tokens, bos_id=bos_id)
    if args.max_docs > 0:
        doc_ranges = doc_ranges[: args.max_docs]
        if doc_ranges:
            val_tokens = val_tokens[: doc_ranges[-1][1]].contiguous()
        doc_ranges = split_doc_ranges(val_tokens, bos_id=bos_id)
    stream_ranges = [(0, int(val_tokens.numel()))]
    print(f"Loaded validation tokens={int(val_tokens.numel())} docs={len(doc_ranges)}")

    base_model = build_model(args, device)
    base_model.load_state_dict(state_dict, strict=True)
    model, compiled_logits = compile_eval_functions(base_model)
    eval_seq_lens = parse_int_list(args.eval_seq_lens, args.train_seq_len)
    modes = parse_patterns(args.modes)
    variant_name_filter = set(parse_patterns(args.variant_names))
    rows: list[dict[str, object]] = []
    strides = [int(piece) for piece in args.strides.split(",") if piece.strip()]
    code_bytes = Path("train_gpt.py").stat().st_size

    late_k_patterns = parse_patterns(args.late_k_patterns)
    extra_lowbit_patterns = parse_patterns(args.extra_lowbit_patterns)
    core_lowbit_patterns = (".mlp.", ".attn.c_q.", ".attn.c_v.", ".attn.proj.") + extra_lowbit_patterns
    k_group_overrides = ((".attn.c_k.", args.k_group_size),) if args.k_group_size > 0 else tuple()
    variants = [
        {
            "name": "int8_zlib",
            "compressor": "zlib",
            "keep_fp16": tuple(),
            "lowbit": tuple(),
            "lowbit_bits": 8,
            "group_overrides": tuple(),
        },
        {
            "name": "int8_zlib_fp16_embed",
            "compressor": "zlib",
            "keep_fp16": ("tok_emb.weight",),
            "lowbit": tuple(),
            "lowbit_bits": 8,
            "group_overrides": tuple(),
        },
        {
            "name": "int6_zstd_core",
            "compressor": "zstd",
            "keep_fp16": tuple(),
            "lowbit": core_lowbit_patterns,
            "lowbit_bits": 6,
            "group_overrides": tuple(),
        },
        {
            "name": "int6_zstd_core_fp16_embed",
            "compressor": "zstd",
            "keep_fp16": ("tok_emb.weight",),
            "lowbit": core_lowbit_patterns,
            "lowbit_bits": 6,
            "group_overrides": tuple(),
        },
        {
            "name": "int6_zstd_core_fp16_embed_groupk",
            "compressor": "zstd",
            "keep_fp16": ("tok_emb.weight",),
            "lowbit": core_lowbit_patterns,
            "lowbit_bits": 6,
            "group_overrides": k_group_overrides,
        },
    ]
    if late_k_patterns:
        variants.append(
            {
                "name": "int6_zstd_core_fp16_embed_latek",
                "compressor": "zstd",
                "keep_fp16": ("tok_emb.weight",) + late_k_patterns,
                "lowbit": core_lowbit_patterns,
                "lowbit_bits": 6,
                "group_overrides": k_group_overrides,
            }
        )

    all_variants = [
        {
            "name": "prequant",
            "compressor": "none",
            "keep_fp16": tuple(),
            "lowbit": tuple(),
            "lowbit_bits": 8,
            "group_overrides": tuple(),
        }
    ] + variants
    if variant_name_filter:
        all_variants = [variant for variant in all_variants if variant["name"] in variant_name_filter]

    for variant in all_variants:
        if variant["name"] == "prequant":
            model_bytes = "NA"
            artifact_bytes = "NA"
        else:
            blob, _raw_bytes, _payload_bytes, dequantized_state = serialize_variant_state_dict(
                state_dict,
                keep_fp16_patterns=variant["keep_fp16"],
                lowbit_patterns=variant["lowbit"],
                lowbit_bits=variant["lowbit_bits"],
                group_overrides=variant["group_overrides"],
                compressor=variant["compressor"],
            )
            model_bytes = len(blob)
            artifact_bytes = model_bytes + code_bytes
            base_model.load_state_dict(dequantized_state, strict=True)

        for eval_seq_len in eval_seq_lens:
            warm_eval_functions(model, compiled_logits, val_tokens, device, eval_seq_len, args.val_batch_size)

            for mode_name in modes:
                if mode_name.endswith("_flat"):
                    t_eval = time.perf_counter()
                    eval_loss, eval_bpb = run_eval_mode(
                        mode_name=mode_name,
                        seq_len=eval_seq_len,
                        stride=None,
                        model=model,
                        logits_fn=compiled_logits,
                        val_batch_size=args.val_batch_size,
                        val_tokens=val_tokens,
                        stream_ranges=stream_ranges,
                        doc_ranges=doc_ranges,
                        device=device,
                        base_bytes_lut=base_bytes_lut,
                        has_leading_space_lut=has_leading_space_lut,
                        is_boundary_token_lut=is_boundary_token_lut,
                        vocab_size=args.vocab_size,
                        pad_id=pad_id,
                    )
                    rows.append(
                        {
                            "variant": variant["name"],
                            "mode": mode_name,
                            "eval_seq_len": eval_seq_len,
                            "stride": eval_seq_len,
                            "val_loss": f"{eval_loss:.8f}",
                            "val_bpb": f"{eval_bpb:.8f}",
                            "model_bytes": model_bytes,
                            "artifact_bytes": artifact_bytes,
                            "compressor": variant["compressor"],
                            "eval_ms": f"{1000.0 * (time.perf_counter() - t_eval):.0f}",
                        }
                    )
                    print(
                        f"{variant['name']:>28} {mode_name:>14} eval_seq={eval_seq_len:<4} stride={eval_seq_len:<4} "
                        f"val_bpb={rows[-1]['val_bpb']} artifact_bytes={artifact_bytes}"
                    )
                    continue

                for stride in strides:
                    if stride >= eval_seq_len:
                        continue
                    t_eval = time.perf_counter()
                    eval_loss, eval_bpb = run_eval_mode(
                        mode_name=mode_name,
                        seq_len=eval_seq_len,
                        stride=stride,
                        model=model,
                        logits_fn=compiled_logits,
                        val_batch_size=args.val_batch_size,
                        val_tokens=val_tokens,
                        stream_ranges=stream_ranges,
                        doc_ranges=doc_ranges,
                        device=device,
                        base_bytes_lut=base_bytes_lut,
                        has_leading_space_lut=has_leading_space_lut,
                        is_boundary_token_lut=is_boundary_token_lut,
                        vocab_size=args.vocab_size,
                        pad_id=pad_id,
                    )
                    rows.append(
                        {
                            "variant": variant["name"],
                            "mode": mode_name,
                            "eval_seq_len": eval_seq_len,
                            "stride": stride,
                            "val_loss": f"{eval_loss:.8f}",
                            "val_bpb": f"{eval_bpb:.8f}",
                            "model_bytes": model_bytes,
                            "artifact_bytes": artifact_bytes,
                            "compressor": variant["compressor"],
                            "eval_ms": f"{1000.0 * (time.perf_counter() - t_eval):.0f}",
                        }
                    )
                    print(
                        f"{variant['name']:>28} {mode_name:>14} eval_seq={eval_seq_len:<4} stride={stride:<4} "
                        f"val_bpb={rows[-1]['val_bpb']} artifact_bytes={artifact_bytes}"
                    )

        base_model.load_state_dict(state_dict, strict=True)

    fieldnames = ["variant", "mode", "eval_seq_len", "stride", "val_loss", "val_bpb", "model_bytes", "artifact_bytes", "compressor", "eval_ms"]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']:>28}  {row['mode']:>14}  eval_seq={row['eval_seq_len']:<4}  stride={row['stride']:<4}  "
            f"val_bpb={row['val_bpb']}  model_bytes={row['model_bytes']}  artifact_bytes={row['artifact_bytes']}"
        )
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
