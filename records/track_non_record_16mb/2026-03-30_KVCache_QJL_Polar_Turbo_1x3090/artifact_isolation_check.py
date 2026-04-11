from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch


def parse_cli() -> argparse.Namespace:
    record_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Load the compressed PolarQuant artifact in a fresh process, generate a few tokens, and profile VRAM.",
    )
    parser.add_argument("--artifact", default=str(record_dir / "final_model.polar.ptz"))
    parser.add_argument("--backend", default="qjl_triton")
    parser.add_argument("--prompt", default="La cuantizacion polar")
    parser.add_argument("--generate-tokens", type=int, default=12)
    parser.add_argument("--profile-tokens", type=int, default=2048)
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=128)
    parser.add_argument("--report", default=str(record_dir / "logs" / "artifact_isolation_report.json"))
    return parser.parse_args()


def load_record_module(record_dir: Path):
    if str(record_dir) not in sys.path:
        sys.path.insert(0, str(record_dir))
    spec = importlib.util.spec_from_file_location("record_train_gpt", record_dir / "train_gpt.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load train_gpt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_existing_path(candidates: list[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def configure_env(repo_root: Path, context_len: int, backend: str) -> None:
    sibling_root = repo_root.parent / repo_root.name.removesuffix("-pr")
    data_path = first_existing_path(
        [
            Path(os.environ["DATA_PATH"]) if "DATA_PATH" in os.environ else None,
            repo_root / "data" / "datasets" / "fineweb10B_sp1024",
            sibling_root / "data" / "datasets" / "fineweb10B_sp1024",
        ]
    )
    tokenizer_path = first_existing_path(
        [
            Path(os.environ["TOKENIZER_PATH"]) if "TOKENIZER_PATH" in os.environ else None,
            repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model",
            sibling_root / "data" / "tokenizers" / "fineweb_1024_bpe.model",
        ]
    )
    if data_path is None:
        raise FileNotFoundError("Unable to locate fineweb10B_sp1024 dataset directory")
    if tokenizer_path is None:
        raise FileNotFoundError("Unable to locate fineweb_1024_bpe.model tokenizer")
    os.environ["DATA_PATH"] = str(data_path)
    os.environ["TOKENIZER_PATH"] = str(tokenizer_path)
    os.environ.setdefault("KV_QUANT_BACKEND", backend)
    os.environ.setdefault("KV_EVAL_CONTEXT_LEN", str(context_len))
    os.environ.setdefault("QAT_SCHEME", "polar")
    os.environ.setdefault("WEIGHT_QUANT_SCHEME", "polar")


def make_backend(record, cfg, model, device: torch.device, backend_name: str):
    return record.make_named_kv_backend(
        backend_name,
        cfg,
        head_dim=model.blocks[0].attn.head_dim,
        num_heads=model.blocks[0].attn.num_heads,
        num_kv_heads=model.blocks[0].attn.num_kv_heads,
        device=device,
    )


def greedy_generate(record, cfg, model, sp, device: torch.device, backend_name: str, prompt: str, num_tokens: int, context_len: int) -> dict[str, object]:
    prompt_ids = sp.encode(prompt, out_type=int)
    if not prompt_ids:
        raise ValueError("Prompt encoded to an empty token sequence")
    backend = make_backend(record, cfg, model, device, backend_name)
    record.prewarm_triton_backend(backend, device)
    input_ids = torch.tensor(prompt_ids, device=device, dtype=torch.int64).view(1, -1)
    generated_ids: list[int] = []
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits, kv_cache = model.forward_prefill(input_ids, backend=backend, max_context=context_len)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(num_tokens):
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits, kv_cache = model.forward_decode(
                    next_token,
                    backend=backend,
                    kv_cache=kv_cache,
                    max_context=context_len,
                )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    full_ids = prompt_ids + generated_ids
    return {
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "decoded_text": sp.decode(full_ids),
        "decoded_suffix": sp.decode(generated_ids),
    }


def profile_autoregressive_memory(
    record,
    cfg,
    model,
    device: torch.device,
    backend_name: str,
    profile_tokens: int,
    context_len: int,
    sample_every: int,
) -> dict[str, object]:
    val_tokens = record.load_validation_tokens(cfg.val_files, seq_len=1, max_tokens=profile_tokens)
    total_tokens = min(profile_tokens, int(val_tokens.numel() - 1))
    if total_tokens <= 0:
        raise ValueError("Profile requires at least one token")

    backend = make_backend(record, cfg, model, device, backend_name)
    record.prewarm_triton_backend(backend, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    alloc_history: list[int] = []
    reserved_history: list[int] = []
    peak_history: list[int] = []
    sample_points: list[dict[str, int]] = []
    kv_cache = None

    t0 = time.perf_counter()
    with torch.inference_mode():
        for idx in range(total_tokens):
            token = val_tokens[idx : idx + 1].reshape(1, 1).to(device=device, dtype=torch.int64, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _, kv_cache = model.forward_decode(
                    token,
                    backend=backend,
                    kv_cache=kv_cache,
                    max_context=context_len,
                )
            torch.cuda.synchronize(device)
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
            peak = int(torch.cuda.max_memory_allocated(device))
            alloc_history.append(allocated)
            reserved_history.append(reserved)
            peak_history.append(peak)
            if (
                idx == 0
                or idx + 1 == total_tokens
                or idx + 1 == min(context_len, total_tokens)
                or ((idx + 1) % sample_every == 0)
            ):
                current_logical = sum(backend.cache_nbytes(layer) for layer in kv_cache["layers"] if layer is not None)
                current_actual = sum(backend.actual_cache_nbytes(layer) for layer in kv_cache["layers"] if layer is not None)
                sample_points.append(
                    {
                        "step": idx + 1,
                        "allocated_bytes": allocated,
                        "reserved_bytes": reserved,
                        "peak_allocated_bytes": peak,
                        "cache_logical_bytes": int(current_logical),
                        "cache_tensor_bytes": int(current_actual),
                    }
                )

    elapsed_ms = int(round(1000.0 * (time.perf_counter() - t0)))
    steady_state_start = max(min(context_len, total_tokens) - 1, 0)
    steady_alloc = alloc_history[steady_state_start:]
    steady_reserved = reserved_history[steady_state_start:]
    steady_peaks = peak_history[steady_state_start:]
    peak_events_after_context = 0
    for i in range(steady_state_start + 1, len(peak_history)):
        if peak_history[i] > peak_history[i - 1]:
            peak_events_after_context += 1

    return {
        "backend": backend_name,
        "tokens_profiled": total_tokens,
        "context_len": context_len,
        "elapsed_ms": elapsed_ms,
        "allocated_first_bytes": alloc_history[0],
        "allocated_last_bytes": alloc_history[-1],
        "allocated_peak_bytes": max(alloc_history),
        "peak_memory_allocated_bytes": peak_history[-1],
        "peak_memory_reserved_bytes": max(reserved_history),
        "steady_state_allocated_min_bytes": min(steady_alloc),
        "steady_state_allocated_max_bytes": max(steady_alloc),
        "steady_state_allocated_growth_bytes": steady_alloc[-1] - steady_alloc[0],
        "steady_state_reserved_min_bytes": min(steady_reserved),
        "steady_state_reserved_max_bytes": max(steady_reserved),
        "steady_state_peak_events": peak_events_after_context,
        "steady_state_peak_memory_allocated_bytes": max(steady_peaks),
        "samples": sample_points,
    }


def main() -> None:
    cli = parse_cli()
    record_dir = Path(__file__).resolve().parent
    repo_root = record_dir.parents[2]
    artifact_path = Path(cli.artifact).resolve()
    report_path = Path(cli.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this check")

    configure_env(repo_root, cli.context_len, cli.backend)
    record = load_record_module(record_dir)
    cfg = record.Hyperparameters()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    model = record.build_base_model(cfg, device)
    record.load_quantized_artifact_into_model(model, artifact_path)
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)

    generation = greedy_generate(
        record,
        cfg,
        model,
        sp,
        device,
        backend_name=cli.backend,
        prompt=cli.prompt,
        num_tokens=cli.generate_tokens,
        context_len=cli.context_len,
    )
    profile = profile_autoregressive_memory(
        record,
        cfg,
        model,
        device,
        backend_name=cli.backend,
        profile_tokens=cli.profile_tokens,
        context_len=cli.context_len,
        sample_every=cli.sample_every,
    )

    report = {
        "artifact_path": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "tokenizer_path": cfg.tokenizer_path,
        "backend": cli.backend,
        "generation": generation,
        "profile": profile,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"artifact:{artifact_path}")
    print(f"artifact_bytes:{report['artifact_bytes']}")
    print(f"generated_suffix:{generation['decoded_suffix']}")
    print(f"generated_text:{generation['decoded_text']}")
    print(
        "memory_profile "
        f"tokens:{profile['tokens_profiled']} "
        f"peak_allocated:{profile['peak_memory_allocated_bytes']} "
        f"peak_reserved:{profile['peak_memory_reserved_bytes']} "
        f"steady_growth:{profile['steady_state_allocated_growth_bytes']} "
        f"steady_peak_events:{profile['steady_state_peak_events']}"
    )
    print(f"report:{report_path}")


if __name__ == "__main__":
    main()
