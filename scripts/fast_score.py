"""Fast PPM-D scorer for the Path B merged byte records.

Mirrors `score_ppmd_stream` exactly but computes `model.distribution()`
only once per byte (the original `ppmd_prefix_lambda` calls it again
just to read max). Verified equality vs reference on the first N
records before running on the full stream.
"""
import sys, time, math, json, argparse
from pathlib import Path

sys.path.insert(0, "scripts")
import eval_path_b_ppmd as e
from eval_path_b_ppmd import (
    PPMDByteModel, PPMDStreamScoreSummary, _validate_byte_value,
    _log_mixture_probability, merge_shard_records, _coerce_record,
    PathBEvalConfig, build_sliding_eval_result, write_output_json,
    merge_record_npz_shards, build_merge_manifest,
    ACCOUNTING_AUDIT_FILENAME, MERGE_MANIFEST_FILENAME,
    expected_denominator_for_eval, DENOMINATOR_FORMULA,
    SCHEMA_VERSION, PATH_B_VERSION, rank_shard_filename,
    rank_accounting_filename,
)


def fast_score_ppmd_stream(records, *, ppmd_order=5, ppmd_lambda=0.35,
                           ppmd_lambda_hi=0.90, ppmd_lambda_lo=0.05,
                           ppmd_conf_threshold=0.90,
                           ppmd_confidence_gating=True,
                           initial_bytes=b"",
                           progress_every=200000):
    ordered = merge_shard_records([[_coerce_record(r) for r in records]])
    if not ordered:
        raise ValueError("empty stream")
    model = PPMDByteModel(order=ppmd_order)
    model.update_bytes(bytes(initial_bytes))
    mix_nll = ppm_nll = nn_nll = 0.0
    lambdas = []
    base_lam = float(ppmd_lambda)
    lam_hi = float(ppmd_lambda_hi)
    lam_lo = float(ppmd_lambda_lo)
    conf_thr = float(ppmd_conf_threshold)
    gating = bool(ppmd_confidence_gating)
    log = math.log
    log1p = math.log1p
    log2 = math.log(2.0)
    update = model.update
    distribution = model.distribution
    t0 = time.perf_counter()
    n = len(ordered)
    for i, record in enumerate(ordered):
        b = _validate_byte_value(record.byte_value)
        dist = distribution()
        ppm_prob = float(dist[b])
        if ppm_prob <= 0.0:
            raise ValueError(f"PPM-D non-positive probability for byte {b}")
        if gating:
            confidence = max(dist)
            lam = lam_hi if confidence >= conf_thr else lam_lo
        else:
            lam = base_lam
        nn_logprob = float(record.neural_logprob)
        # Inline _log_mixture_probability fast path
        if lam == 0.0:
            mix_logprob = nn_logprob
        elif lam == 1.0:
            mix_logprob = log(ppm_prob)
        else:
            a = log1p(-lam) + nn_logprob
            c = log(lam) + log(ppm_prob)
            if a > c:
                mix_logprob = a + log1p(math.exp(c - a))
            else:
                mix_logprob = c + log1p(math.exp(a - c))
        nn_nll -= nn_logprob
        ppm_nll -= log(ppm_prob)
        mix_nll -= mix_logprob
        lambdas.append(lam)
        update(b)
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta_sec = (n - i - 1) / rate
            print(f"  [{i+1:>9}/{n}] rate={rate:.0f}/s elapsed={elapsed:.0f}s eta={eta_sec/60:.1f}min "
                  f"mix_bpb_so_far={mix_nll/(log2*(i+1)):.5f}", flush=True)
    byte_count = len(ordered)
    denom = log2 * float(byte_count)
    return PPMDStreamScoreSummary(
        byte_count=byte_count,
        mix_nll=float(mix_nll),
        ppm_nll=float(ppm_nll),
        nn_nll=float(nn_nll),
        mix_bpb=float(mix_nll / denom),
        ppm_bpb=float(ppm_nll / denom),
        nn_bpb=float(nn_nll / denom),
        lambdas=lambdas,
        ppmd_history=bytes(model.history),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--subset-tokens", type=int, default=8000000)
    ap.add_argument("--full-eval", action="store_true",
                    help="score the full validation set; expected denominator 151,078,222 bytes")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only score the first N records (debug)")
    ap.add_argument("--verify", action="store_true",
                    help="run reference scorer on first 5000 records and check equality")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    full_eval = bool(args.full_eval)
    cfg_subset_tokens = None if full_eval else int(args.subset_tokens)
    OUT = Path(args.shard_dir)
    shard_paths = [OUT / rank_shard_filename(r) for r in range(args.world_size)]
    print("merging shards from", OUT)
    t0 = time.perf_counter()
    merged = merge_record_npz_shards(shard_paths)
    print(f"merged {len(merged)} records in {time.perf_counter()-t0:.1f}s", flush=True)

    if args.verify:
        ref = e.score_ppmd_stream(merged[:5000])
        fast = fast_score_ppmd_stream(merged[:5000], progress_every=0)
        print("ref  : nn=%.10f ppm=%.10f mix=%.10f" % (ref.nn_bpb, ref.ppm_bpb, ref.mix_bpb))
        print("fast : nn=%.10f ppm=%.10f mix=%.10f" % (fast.nn_bpb, fast.ppm_bpb, fast.mix_bpb))
        assert abs(ref.mix_bpb - fast.mix_bpb) < 1e-12, "mismatch!"
        assert abs(ref.nn_bpb - fast.nn_bpb) < 1e-12
        assert abs(ref.ppm_bpb - fast.ppm_bpb) < 1e-12
        print("VERIFY OK")

    target = merged if args.limit <= 0 else merged[:args.limit]
    print(f"scoring {len(target)} records...", flush=True)
    t0 = time.perf_counter()
    summary = fast_score_ppmd_stream(target)
    runtime = time.perf_counter() - t0
    print(f"scoring took {runtime:.1f}s ({runtime/60:.1f}min)", flush=True)
    print(f"mix_bpb={summary.mix_bpb:.6f} nn_bpb={summary.nn_bpb:.6f} ppm_bpb={summary.ppm_bpb:.6f}")

    # Build full result JSON
    config = PathBEvalConfig(
        source_python_path="results/exp_1876_ppmd/train_gpt_merged.py",
        artifact_path="results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz",
        output_json_path=args.output_json,
        eval_kind="sliding",
        subset_tokens=args.subset_tokens,
        full_eval=full_eval,
    )
    art = Path(config.artifact_path)
    # Load accounting from disk
    acc_paths = [OUT / rank_accounting_filename(r) for r in range(args.world_size)]
    total_tok = total_bytes = total_zero = 0
    for p in acc_paths:
        d = json.loads(p.read_text())
        total_tok += int(d["scored_token_count"])
        total_bytes += int(d["scored_byte_count"])
        total_zero += int(d["zero_byte_token_count"])
    result = build_sliding_eval_result(
        config=config,
        source_module_path=str(Path(config.source_python_path).resolve()),
        artifact_path=str(art.resolve()),
        artifact_size_bytes=art.stat().st_size,
        rank=0, world_size=args.world_size,
        subset_tokens=cfg_subset_tokens, full_eval=full_eval,
        scored_token_count=total_tok,
        scored_byte_count=total_bytes,
        zero_byte_token_count=total_zero,
        runtime_seconds=runtime,
        summary=summary,
        shard_manifest_path=str(OUT / MERGE_MANIFEST_FILENAME),
        accounting_audit_path=str(OUT / ACCOUNTING_AUDIT_FILENAME),
        warnings=[],
        error=None,
    )
    write_output_json(Path(args.output_json), result)
    print("wrote", args.output_json)
    print("claim_ready =", result.get("claim_ready"))


if __name__ == "__main__":
    main()
