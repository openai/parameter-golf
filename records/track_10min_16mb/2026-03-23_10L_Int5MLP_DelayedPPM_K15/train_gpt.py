"""
Wrapper around the local base_train_gpt.py snapshot that runs delayed PPM eval.

It monkeypatches base_train_gpt.eval_val_sliding so the normal sliding-window
baseline still runs first, then a delayed PPM bank is evaluated in the same
process/model state while sweeping the logit boost factor K.
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_BASE_TRAIN_GPT_PATH = Path(__file__).with_name("base_train_gpt.py")
_BASE_TRAIN_GPT_SPEC = importlib.util.spec_from_file_location("record_base_train_gpt", _BASE_TRAIN_GPT_PATH)
if _BASE_TRAIN_GPT_SPEC is None or _BASE_TRAIN_GPT_SPEC.loader is None:
    raise ImportError(f"Could not load {_BASE_TRAIN_GPT_PATH}")
base_train_gpt = importlib.util.module_from_spec(_BASE_TRAIN_GPT_SPEC)
_BASE_TRAIN_GPT_SPEC.loader.exec_module(base_train_gpt)


def _parse_int_list(name: str, default: str) -> list[int]:
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


def _parse_float_list(name: str, default: str) -> list[float]:
    return [float(x) for x in os.environ.get(name, default).split(",") if x.strip()]


PPM_SWEEP_K_VALUES = _parse_int_list("PPM_SWEEP_K_VALUES", "16,12,8,6")
PPM_SWEEP_MIN_CONFS = _parse_float_list("PPM_SWEEP_MIN_CONFS", "1.0,1.0,1.0,0.95")
PPM_SWEEP_MIN_COUNTS = _parse_int_list("PPM_SWEEP_MIN_COUNTS", "1,1,1,1")
PPM_SWEEP_BOOST_KS = _parse_float_list("PPM_SWEEP_BOOST_KS", "1.2,1.5,2,3,5,7,10,15,20,30,50")
PPM_SWEEP_BOS_ID = int(os.environ.get("PPM_SWEEP_BOS_ID", "1"))
PPM_SWEEP_DELAY = int(os.environ.get("PPM_SWEEP_DELAY", "-1"))

_orig_eval_val_sliding = base_train_gpt.eval_val_sliding


def load_libtrie_ctx(rank: int) -> ctypes.CDLL:
    src_path = Path(__file__).parent / "trie_bench.c"
    lib_path = Path(__file__).parent / "libtrie_ctx.so"
    needs_build = (not lib_path.exists()) or (src_path.stat().st_mtime > lib_path.stat().st_mtime)
    if rank == 0 and needs_build:
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(lib_path), str(src_path)],
            check=True,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    lib = ctypes.CDLL(str(lib_path))
    lib.streaming_ppm_process_delayed.restype = ctypes.c_int64
    lib.streaming_ppm_process_delayed.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    return lib


def build_delayed_ppm_hits(
    lib: ctypes.CDLL,
    val_tokens: torch.Tensor,
    bos_id: int,
    k_values: list[int],
    min_confs: list[float],
    min_counts: list[int],
    delay: int,
) -> tuple[np.ndarray, np.ndarray, int, float, list[tuple[int, int, float]]]:
    if not (len(k_values) == len(min_confs) == len(min_counts)):
        raise ValueError("PPM sweep config lists must have the same length")
    if k_values != sorted(k_values, reverse=True):
        raise ValueError("PPM_SWEEP_K_VALUES must be sorted in descending order")

    tokens_np = val_tokens.numpy().astype(np.int64)
    n_tokens = len(tokens_np)
    n_levels = len(k_values)
    k_arr = np.array(k_values, dtype=np.int32)
    conf_arr = np.array(min_confs, dtype=np.float32)
    count_arr = np.array(min_counts, dtype=np.int32)
    hit_flags = np.zeros(n_tokens, dtype=np.int32)
    pred_tokens = np.zeros(n_tokens, dtype=np.int32)
    pred_confs = np.zeros(n_tokens, dtype=np.float32)
    match_levels = np.full(n_tokens, -1, dtype=np.int32)
    tok_ptr = tokens_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    n_hits = int(
        lib.streaming_ppm_process_delayed(
            tok_ptr,
            ctypes.c_int64(n_tokens),
            ctypes.c_int32(bos_id),
            k_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            conf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            count_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(n_levels),
            ctypes.c_int32(delay),
            hit_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pred_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pred_confs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            match_levels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
    )
    if n_hits < 0:
        raise RuntimeError("streaming_ppm_process_delayed failed")

    hit_mask = hit_flags > 0
    direct_acc = float(np.mean(pred_tokens[hit_mask] == tokens_np[hit_mask])) if hit_mask.any() else 0.0
    level_stats: list[tuple[int, int, float]] = []
    for lv, k_val in enumerate(k_values):
        lv_mask = (match_levels == lv) & hit_mask
        lv_hits = int(lv_mask.sum())
        lv_acc = float(np.mean(pred_tokens[lv_mask] == tokens_np[lv_mask])) if lv_hits > 0 else 0.0
        level_stats.append((k_val, lv_hits, lv_acc))

    del tokens_np, hit_flags, pred_confs, match_levels
    return hit_mask.astype(np.bool_), pred_tokens.astype(np.uint16, copy=False), n_hits, direct_acc, level_stats


def eval_ppm_sweep(
    args: base_train_gpt.Hyperparameters,
    base_model: torch.nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    ppm_hits: np.ndarray,
    ppm_preds: np.ndarray,
    boost_ks: list[float],
    stride: int,
    batch_seqs: int,
) -> tuple[float, list[tuple[float, int, float]], float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    delta_sums = torch.zeros(len(boost_ks), device=device, dtype=torch.float64)
    hit_counts = torch.zeros(len(boost_ks), device=device, dtype=torch.float64)
    boost_tensor = torch.tensor(boost_ks, device=device, dtype=torch.float64)
    log_boosts = boost_tensor.log()

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            hit_rows: list[np.ndarray] = []
            hit_cols: list[np.ndarray] = []
            hit_preds: list[np.ndarray] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

                s = 0 if ws == 0 else max(wlen - stride, 0)
                pos_lo = ws + s + 1
                pos_hi = ws + wlen + 1
                window_hit_mask = ppm_hits[pos_lo:pos_hi]
                if not window_hit_mask.any():
                    continue
                rel = np.flatnonzero(window_hit_mask)
                hit_rows.append(np.full(rel.shape, i, dtype=np.int64))
                hit_cols.append(rel + s)
                hit_preds.append(ppm_preds[pos_lo:pos_hi][window_hit_mask].astype(np.int64))

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)
            logits_f = logits.float()
            lse = torch.logsumexp(logits_f, dim=-1)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if hit_rows:
                rows = torch.from_numpy(np.concatenate(hit_rows)).to(device=device, dtype=torch.long)
                cols = torch.from_numpy(np.concatenate(hit_cols)).to(device=device, dtype=torch.long)
                preds = torch.from_numpy(np.concatenate(hit_preds)).to(device=device, dtype=torch.long)
                pred_logits = logits_f[rows, cols, preds]
                p_ppm = (pred_logits - lse[rows, cols]).exp().clamp(min=1e-12, max=1.0).to(torch.float64)
                correct = (y_batch[rows, cols] == preds)[None, :]
                adj = torch.log1p((boost_tensor[:, None] - 1.0) * p_ppm[None, :])
                delta = torch.where(correct, adj - log_boosts[:, None], adj)
                delta_sums += delta.sum(dim=1)
                hit_counts += float(p_ppm.numel())

            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                pct = done / len(my_windows) * 100
                print(f"  ppm_sweep [{pct:5.1f}%] {done}/{len(my_windows)} windows", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(delta_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(hit_counts, op=dist.ReduceOp.SUM)

    total_bytes = float(byte_count.item())
    results = []
    for i, boost_k in enumerate(boost_ks):
        results.append((boost_k, int(hit_counts[i].item()), float((delta_sums[i] / byte_count / math.log(2.0)).item())))
    return float(token_count.item()), results, total_bytes


def patched_eval_val_sliding(
    args: base_train_gpt.Hyperparameters,
    base_model: torch.nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    stride: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    val_loss, val_bpb = _orig_eval_val_sliding(
        args,
        base_model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        stride,
        batch_seqs,
    )

    delay = PPM_SWEEP_DELAY if PPM_SWEEP_DELAY >= 0 else args.train_seq_len
    lib = load_libtrie_ctx(rank)
    torch.cuda.synchronize()
    t_phase1 = time.perf_counter()
    ppm_hits, ppm_preds, n_hits, direct_acc, level_stats = build_delayed_ppm_hits(
        lib,
        val_tokens,
        PPM_SWEEP_BOS_ID,
        PPM_SWEEP_K_VALUES,
        PPM_SWEEP_MIN_CONFS,
        PPM_SWEEP_MIN_COUNTS,
        delay,
    )
    torch.cuda.synchronize()

    if rank == 0:
        print(
            f"ppm_sweep:enabled k_values:{PPM_SWEEP_K_VALUES} min_confs:{PPM_SWEEP_MIN_CONFS} "
            f"min_counts:{PPM_SWEEP_MIN_COUNTS} delay:{delay} boost_ks:{PPM_SWEEP_BOOST_KS}"
        )
        print(
            f"ppm_phase1 hits:{n_hits} hit_rate:{100.0 * n_hits / val_tokens.numel():.3f}% "
            f"direct_acc:{100.0 * direct_acc:.2f}% time_ms:{1000.0 * (time.perf_counter() - t_phase1):.0f}"
        )
        for k_val, lv_hits, lv_acc in level_stats:
            if lv_hits > 0:
                print(f"ppm_phase1_level k:{k_val} hits:{lv_hits} direct_acc:{100.0 * lv_acc:.2f}%")

    torch.cuda.synchronize()
    t_eval = time.perf_counter()
    total_scored, sweep_results, total_bytes = eval_ppm_sweep(
        args,
        base_model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        ppm_hits,
        ppm_preds,
        PPM_SWEEP_BOOST_KS,
        stride,
        batch_seqs,
    )
    torch.cuda.synchronize()

    if rank == 0:
        print(
            f"ppm_sweep:scored_tokens:{int(total_scored)} bytes:{int(total_bytes)} "
            f"eval_time_ms:{1000.0 * (time.perf_counter() - t_eval):.0f}"
        )
        best_k, best_hits, best_delta = min(sweep_results, key=lambda x: x[2])
        best_bpb = val_bpb + best_delta
        for boost_k, hit_count, delta_bpb in sweep_results:
            hit_pct = 100.0 * hit_count / total_scored if total_scored > 0 else 0.0
            boosted_bpb = val_bpb + delta_bpb
            print(
                f"ppm_sweep_result boost_k:{boost_k:.2f} hits:{hit_count} hit_pct:{hit_pct:.3f}% "
                f"val_bpb:{boosted_bpb:.8f} delta_bpb:{delta_bpb:+.8f}"
            )
        print(
            f"final_int8_zlib_roundtrip_ppm_best boost_k:{best_k:.2f} hits:{best_hits} "
            f"val_bpb:{best_bpb:.8f} delta_bpb:{best_delta:+.8f}"
        )

    return val_loss, val_bpb


base_train_gpt.eval_val_sliding = patched_eval_val_sliding


if __name__ == "__main__":
    base_train_gpt.main()
