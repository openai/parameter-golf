#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import time
import zlib
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run eval profiler for Parameter Golf.")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to profile.",
    )
    parser.add_argument(
        "--val-batch-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Validation batch token counts to profile. Defaults to train_gpt.py's VAL_BATCH_SIZE.",
    )
    parser.add_argument(
        "--profile-world-size",
        type=int,
        default=1,
        help="World size to simulate for eval batching. Per-rank memory follows this layout.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override grad accumulation steps used in eval batching. Defaults to 8 // profile_world_size.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Untimed warmup batches to discard after the first timed batch.",
    )
    parser.add_argument(
        "--timed-batches",
        type=int,
        default=3,
        help="Number of steady-state batches to time after warmup.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to load (.pt, .pth, or final_model.int8.ptz).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional dataset root override. Defaults to train_gpt.py DATA_PATH.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use. The profiler is intended for CUDA devices.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.compile to match the scored eval path.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write JSON results to.",
    )
    return parser


def load_modules() -> tuple[object, object]:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import torch  # noqa: WPS433
    import train_gpt as tg  # noqa: WPS433

    return torch, tg


def configure_cuda(torch: object) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)


def maybe_reset_dynamo(torch: object) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()


def build_args(args: argparse.Namespace, tg: object) -> object:
    hp = tg.Hyperparameters()
    if args.data_path is not None:
        hp.data_path = str(args.data_path)
        hp.train_files = str(args.data_path / "fineweb_train_*.bin")
        hp.val_files = str(args.data_path / "fineweb_val_*.bin")
    return hp


def load_checkpoint(model: object, checkpoint: Path, torch: object, tg: object) -> None:
    if checkpoint.suffix == ".ptz":
        blob = checkpoint.read_bytes()
        quant_state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
        state_dict = tg.dequantize_state_dict_int8(quant_state)
    else:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "__quant_format__" in state:
            state_dict = tg.dequantize_state_dict_int8(state)
        else:
            state_dict = state
    model.load_state_dict(state_dict, strict=True)


def build_model(hp: object, checkpoint: Path | None, device: object, compile_model: bool, torch: object, tg: object):
    base_model = tg.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, tg.CastedLinear):
            module.float()
    tg.restore_low_dim_params_to_fp32(base_model)
    if checkpoint is not None:
        load_checkpoint(base_model, checkpoint, torch, tg)
    return torch.compile(base_model, dynamic=False, fullgraph=True) if compile_model else base_model


def profile_case(
    hp: object,
    seq_len: int,
    val_batch_tokens: int,
    profile_world_size: int,
    grad_accum_steps: int,
    warmup_batches: int,
    timed_batches: int,
    checkpoint: Path | None,
    device: object,
    compile_model: bool,
    torch: object,
    tg: object,
) -> dict[str, object]:
    requested_batches = 1 + warmup_batches + timed_batches
    case = {
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "compile": compile_model,
        "device": str(device),
        "grad_accum_steps": grad_accum_steps,
        "profile_world_size": profile_world_size,
        "requested_batches": requested_batches,
        "seq_len": seq_len,
        "status": "ok",
        "timed_batches_requested": timed_batches,
        "val_batch_tokens": val_batch_tokens,
        "warmup_batches": warmup_batches,
    }

    hp.train_seq_len = seq_len
    hp.val_batch_size = val_batch_tokens
    val_tokens = tg.load_validation_tokens(hp.val_files, seq_len)
    layout = tg.get_eval_layout(hp, rank=0, world_size=profile_world_size, grad_accum_steps=grad_accum_steps, val_tokens=val_tokens)
    case.update(layout)
    batch_records: list[dict[str, object]] = []

    torch.cuda.empty_cache()
    gc.collect()
    maybe_reset_dynamo(torch)
    model = None
    try:
        model = build_model(hp, checkpoint, device, compile_model, torch, tg)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for batch_index, (x, y) in enumerate(
                tg.iter_eval_batches(hp, rank=0, world_size=profile_world_size, device=device, grad_accum_steps=grad_accum_steps, val_tokens=val_tokens)
            ):
                if batch_index >= requested_batches:
                    break
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(x, y)
                torch.cuda.synchronize(device)
                elapsed_ms = 1000.0 * (time.perf_counter() - t0)
                batch_records.append(
                    {
                        "batch_index": batch_index,
                        "loss": float(loss.detach().item()),
                        "tokens": int(y.numel()),
                        "wallclock_ms": elapsed_ms,
                    }
                )
    except torch.OutOfMemoryError as exc:
        case["status"] = "oom"
        case["error"] = str(exc).splitlines()[0]
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            case["status"] = "oom"
            case["error"] = str(exc).splitlines()[0]
        else:
            raise
    finally:
        case["peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated(device) // 1024 // 1024
        case["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved(device) // 1024 // 1024
        del model
        gc.collect()
        torch.cuda.empty_cache()

    case["measured_batches"] = len(batch_records)
    case["batch_records"] = batch_records
    if not batch_records:
        case["status"] = case["status"] if case["status"] != "ok" else "no_batches"
        return case

    first_batch = batch_records[0]
    warmup_records = batch_records[1 : 1 + warmup_batches]
    steady_records = batch_records[1 + warmup_batches :]
    if not steady_records and len(batch_records) > 1:
        steady_records = batch_records[1:]

    case["first_batch_ms"] = first_batch["wallclock_ms"]
    case["first_batch_tokens"] = first_batch["tokens"]
    case["warmup_batch_ms"] = [record["wallclock_ms"] for record in warmup_records]
    if steady_records:
        steady_ms = [record["wallclock_ms"] for record in steady_records]
        steady_tokens = sum(int(record["tokens"]) for record in steady_records)
        steady_total_ms = sum(float(record["wallclock_ms"]) for record in steady_records)
        steady_avg_ms = steady_total_ms / len(steady_records)
        case["steady_avg_batch_ms"] = steady_avg_ms
        case["steady_tokens_per_second"] = steady_tokens / (steady_total_ms / 1000.0)
        case["steady_batches_measured"] = len(steady_records)
        case["projected_full_eval_time_ms_first_pass"] = first_batch["wallclock_ms"] + steady_avg_ms * max(layout["num_batches"] - 1, 0)
        case["projected_full_eval_time_ms_steady_state"] = steady_avg_ms * layout["num_batches"]
    else:
        case["steady_avg_batch_ms"] = None
        case["steady_tokens_per_second"] = None
        case["steady_batches_measured"] = 0
        case["projected_full_eval_time_ms_first_pass"] = first_batch["wallclock_ms"] * layout["num_batches"]
        case["projected_full_eval_time_ms_steady_state"] = None

    case["last_loss"] = batch_records[-1]["loss"]
    return case


def render_text(results: dict[str, object]) -> str:
    lines = [
        f"device: {results['device']}",
        f"profile_world_size: {results['profile_world_size']}",
        f"grad_accum_steps: {results['grad_accum_steps']}",
        f"compile: {results['compile']}",
    ]
    for case in results["profiles"]:
        lines.append("")
        lines.append(f"seq_len: {case['seq_len']} val_batch_tokens: {case['val_batch_tokens']} status: {case['status']}")
        lines.append(
            f"local_batch_tokens: {case['local_batch_tokens']} local_batch_seqs: {case['local_batch_seqs']} num_batches: {case['num_batches']}"
        )
        lines.append(
            f"peak_memory_allocated_mib: {case['peak_memory_allocated_mib']} peak_memory_reserved_mib: {case['peak_memory_reserved_mib']}"
        )
        if case.get("first_batch_ms") is not None:
            lines.append(f"first_batch_ms: {case['first_batch_ms']:.2f}")
        if case.get("steady_avg_batch_ms") is not None:
            lines.append(f"steady_avg_batch_ms: {case['steady_avg_batch_ms']:.2f}")
        if case.get("steady_tokens_per_second") is not None:
            lines.append(f"steady_tokens_per_second: {case['steady_tokens_per_second']:.2f}")
        if case.get("projected_full_eval_time_ms_first_pass") is not None:
            lines.append(f"projected_full_eval_time_ms_first_pass: {case['projected_full_eval_time_ms_first_pass']:.2f}")
        if case.get("projected_full_eval_time_ms_steady_state") is not None:
            lines.append(f"projected_full_eval_time_ms_steady_state: {case['projected_full_eval_time_ms_steady_state']:.2f}")
        if case.get("error"):
            lines.append(f"error: {case['error']}")
    return "\n".join(lines)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch, tg = load_modules()
    if args.device != "cuda":
        raise ValueError(f"This profiler is intended for CUDA eval paths, got --device={args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for eval profiling")
    if args.profile_world_size <= 0:
        raise ValueError(f"--profile-world-size must be positive, got {args.profile_world_size}")
    grad_accum_steps = args.grad_accum_steps
    if grad_accum_steps is None:
        if 8 % args.profile_world_size != 0:
            raise ValueError(
                f"--profile-world-size={args.profile_world_size} must divide 8 when --grad-accum-steps is omitted"
            )
        grad_accum_steps = 8 // args.profile_world_size
    if grad_accum_steps <= 0:
        raise ValueError(f"--grad-accum-steps must be positive, got {grad_accum_steps}")
    if args.warmup_batches < 0:
        raise ValueError(f"--warmup-batches must be non-negative, got {args.warmup_batches}")
    if args.timed_batches <= 0:
        raise ValueError(f"--timed-batches must be positive, got {args.timed_batches}")

    configure_cuda(torch)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    hp = build_args(args, tg)
    val_batch_tokens_list = args.val_batch_tokens or [hp.val_batch_size]

    results = {
        "compile": args.compile,
        "device": str(device),
        "grad_accum_steps": grad_accum_steps,
        "profile_world_size": args.profile_world_size,
        "profiles": [],
    }
    for seq_len in args.seq_lens:
        for val_batch_tokens in val_batch_tokens_list:
            results["profiles"].append(
                profile_case(
                    hp=hp,
                    seq_len=seq_len,
                    val_batch_tokens=val_batch_tokens,
                    profile_world_size=args.profile_world_size,
                    grad_accum_steps=grad_accum_steps,
                    warmup_batches=args.warmup_batches,
                    timed_batches=args.timed_batches,
                    checkpoint=args.checkpoint,
                    device=device,
                    compile_model=args.compile,
                    torch=torch,
                    tg=tg,
                )
            )

    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        json.dump(results, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    print(render_text(results))


if __name__ == "__main__":
    main()
