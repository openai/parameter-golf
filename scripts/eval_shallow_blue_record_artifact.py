#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist


def _distributed_init() -> tuple[int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("distributed artifact eval requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the packaged Shallow Blue submission artifact without retraining.")
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--probe-artifact", required=True)
    parser.add_argument("--val-files", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch-windows", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-val-tokens", type=int, default=0)
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir).resolve()
    sys.path.insert(0, str(submission_dir))

    import train_gpt as submission_train
    from shallow_blue_submission_eval import evaluate_shallow_blue_submission

    rank, world_size, _local_rank, device = _distributed_init()
    try:
        hps = submission_train.Hyperparameters()
        model = submission_train.build_gpt_model(hps).to(device)
        model.load_state_dict(
            submission_train.load_state_dict_artifact(Path(args.model_path)),
            strict=True,
        )
        model.eval()

        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        val_tokens = submission_train.load_validation_tokens(
            args.val_files,
            hps.train_seq_len,
        )
        (
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        ) = submission_train.build_sentencepiece_luts(
            sp,
            hps.vocab_size,
            device,
        )

        torch.cuda.synchronize()
        t_qeval = time.perf_counter()
        q_val_loss, q_val_bpb = submission_train.eval_val(
            hps,
            model,
            rank,
            world_size,
            device,
            submission_train.compute_grad_accum_steps(world_size),
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} "
                f"val_bpb:{q_val_bpb:.8f} "
                f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
            )

        torch.cuda.synchronize()
        t_shallow_blue = time.perf_counter()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        shallow_blue_summary = evaluate_shallow_blue_submission(
            model_path=str(Path(args.model_path).resolve()),
            device=device,
            tokenizer_path=args.tokenizer_path,
            val_files=args.val_files,
            probe_artifact_path=args.probe_artifact,
            rank=rank,
            world_size=world_size,
            vocab_size=hps.vocab_size,
            window=args.window,
            stride=args.stride,
            batch_windows=args.batch_windows,
            alpha=args.alpha,
            max_docs=args.max_docs,
            max_val_tokens=args.max_val_tokens,
        )
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"final_shallow_blue_eval docs:{shallow_blue_summary.docs} "
                f"positions:{shallow_blue_summary.scored_positions} "
                f"bytes:{shallow_blue_summary.total_bytes} "
                f"baseline_bpb:{shallow_blue_summary.baseline_bpb:.8f} "
                f"eval_time:{1000.0 * (time.perf_counter() - t_shallow_blue):.0f}ms"
            )
            print(
                f"final_shallow_blue_safe delta_bpb:{shallow_blue_summary.safe_delta_bpb:+.8f} "
                f"mixed_bpb:{shallow_blue_summary.safe_mixed_bpb:.8f} "
                f"bits_saved:{shallow_blue_summary.safe_bits_saved:.2f}"
            )
            print(
                f"final_shallow_blue_probe delta_bpb:{shallow_blue_summary.probe_delta_bpb:+.8f} "
                f"mixed_bpb:{shallow_blue_summary.probe_mixed_bpb:.8f} "
                f"bits_saved:{shallow_blue_summary.probe_bits_saved:.2f} "
                f"mean_alpha:{shallow_blue_summary.probe_mean_alpha:.6f} "
                f"alpha_rows:{shallow_blue_summary.probe_alpha_rows} "
                f"boosted_rows:{shallow_blue_summary.probe_boosted_rows} "
                f"ngram_emitted:{shallow_blue_summary.ngram_emitted} "
                f"repeat_emitted:{shallow_blue_summary.repeat_emitted}"
            )
            print(
                f"final_shallow_blue_probe_exact val_bpb:{shallow_blue_summary.probe_mixed_bpb:.8f} "
                f"delta_bpb:{shallow_blue_summary.probe_delta_bpb:+.8f} "
                f"elapsed_seconds:{shallow_blue_summary.elapsed_seconds:.3f}"
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
