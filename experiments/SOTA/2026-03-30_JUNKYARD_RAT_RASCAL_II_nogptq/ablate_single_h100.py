#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AblationCase:
    name: str
    issue: str
    hypothesis: str
    ablation: str
    test: str
    env_overrides: dict[str, str]


def parse_last_float(text: str, pattern: str) -> float | None:
    m = None
    for m in re.finditer(pattern, text):
        pass
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def fmt(v: float | int | None) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    return f"{v:.8f}"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FATAL: {msg}")


def load_train_module(train_script: Path):
    spec = importlib.util.spec_from_file_location("rascal_train_mod", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {train_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def static_bottleneck_model(train_script: Path) -> list[dict[str, str]]:
    mod = load_train_module(train_script)
    args = mod.Hyperparameters()
    model = mod.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, mod.CastedLinear):
            m.float()
    mod.restore_low_dim_params_to_fp32(model)

    matrix_params = [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]
    ws = 8
    padded_rs_bytes = 0
    no_pad_rs_bytes = 0
    for p in matrix_params:
        b = p.shape[0]
        tail = int(math.prod(p.shape[1:]))
        padded_b = ((b + ws - 1) // ws) * ws
        padded_rs_bytes += padded_b * tail * 2
        no_pad_rs_bytes += b * tail * 2
    pad_savings = padded_rs_bytes - no_pad_rs_bytes
    pad_savings_pct = (pad_savings / padded_rs_bytes * 100.0) if padded_rs_bytes > 0 else 0.0

    block_named_params = list(model.blocks.named_parameters())
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in mod.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)
    scalar_params.append(model.smear.gate)
    if model.bigram is not None:
        scalar_params.append(model.bigram.scale)
        if model.bigram.proj is not None:
            scalar_params.append(model.bigram.proj.weight)
    if model.ve_shared is not None:
        scalar_params.append(model.ve_shared.scale)
        if model.ve_shared.proj is not None:
            scalar_params.append(model.ve_shared.proj.weight)
        for s in model.ve_layer_scales:
            scalar_params.append(s)

    tok_params = [model.tok_emb.weight]
    if model.bigram is not None:
        tok_params.append(model.bigram.embed.weight)
    if model.ve_shared is not None:
        tok_params.append(model.ve_shared.embed.weight)
    replicated = tok_params + scalar_params
    if model.lm_head is not None:
        replicated.append(model.lm_head.weight)

    uniq = {}
    for p in replicated:
        uniq[id(p)] = p
    replicated_unique = list(uniq.values())
    grad_bytes = sum(int(p.numel()) * int(p.element_size()) for p in replicated_unique)
    grad_tensors = len(replicated_unique)

    return [
        {
            "name": "comm_padding_model",
            "kind": "static",
            "issue": "Muon bank padding overhead on 8x",
            "hypothesis": "Removing B-axis padding in Muon sharding reduces RS/AG volume and step time on multi-GPU.",
            "ablation": "Model current padded RS input bytes vs no-pad RS bytes from real bank shapes.",
            "test": "Compute bytes/step/rank analytically from model tensor shapes (no training run).",
            "status": "modeled",
            "runtime_s": "0",
            "step_avg_ms": "N/A",
            "post_ema_bpb": "N/A",
            "final_val_bpb": "N/A",
            "final_ngram_bpb": "N/A",
            "notes": (
                f"padded_rs_bytes_per_step_rank={padded_rs_bytes}; "
                f"no_pad_rs_bytes_per_step_rank={no_pad_rs_bytes}; "
                f"savings_bytes={pad_savings}; savings_pct={pad_savings_pct:.2f}"
            ),
            "logfile": "",
        },
        {
            "name": "replicated_allreduce_model",
            "kind": "static",
            "issue": "Many small replicated grad all-reduces",
            "hypothesis": "Bucketing replicated grads into fewer collectives reduces NCCL launch latency.",
            "ablation": "Model current replicated grad payload and tensor count.",
            "test": "Count replicated tensors + bytes from real parameter partition logic (no training run).",
            "status": "modeled",
            "runtime_s": "0",
            "step_avg_ms": "N/A",
            "post_ema_bpb": "N/A",
            "final_val_bpb": "N/A",
            "final_ngram_bpb": "N/A",
            "notes": f"replicated_grad_tensors={grad_tensors}; replicated_grad_bytes={grad_bytes}",
            "logfile": "",
        },
    ]


def build_dynamic_cases() -> list[AblationCase]:
    return [
        AblationCase(
            name="baseline",
            issue="Control",
            hypothesis="Establish RASCAL single-H100 reference for speed and quality.",
            ablation="RASCAL II defaults (coprime loader, Muon NS5, compile on).",
            test="Measure final train step_avg and post-EMA val_bpb.",
            env_overrides={},
        ),
        AblationCase(
            name="loader_cache2",
            issue="Loader reload pressure",
            hypothesis="Keeping two shards hot reduces loader stalls and improves step time.",
            ablation="Increase COPRIME_MAX_LOADED_SHARDS 1 -> 2.",
            test="Compare step_avg vs baseline at same seed/steps.",
            env_overrides={"COPRIME_MAX_LOADED_SHARDS": "2"},
        ),
        AblationCase(
            name="loader_cache4",
            issue="Loader reload pressure",
            hypothesis="Larger shard cache may further reduce shard churn if host memory permits.",
            ablation="Increase COPRIME_MAX_LOADED_SHARDS 1 -> 4.",
            test="Compare step_avg vs baseline and cache2.",
            env_overrides={"COPRIME_MAX_LOADED_SHARDS": "4"},
        ),
        AblationCase(
            name="muon_ns4",
            issue="Muon optimizer compute",
            hypothesis="Reducing NS steps 5 -> 4 speeds optimizer with minimal quality impact.",
            ablation="Set MUON_BACKEND_STEPS=4.",
            test="Compare step_avg and post_ema val_bpb vs baseline.",
            env_overrides={"MUON_BACKEND_STEPS": "4"},
        ),
        AblationCase(
            name="muon_ns3",
            issue="Muon optimizer compute",
            hypothesis="Reducing NS steps 5 -> 3 may improve speed but risks bigger quality drop.",
            ablation="Set MUON_BACKEND_STEPS=3.",
            test="Compare step_avg and post_ema val_bpb vs baseline/ns4.",
            env_overrides={"MUON_BACKEND_STEPS": "3"},
        ),
        AblationCase(
            name="compile_off",
            issue="Compiler overhead / kernel path risk",
            hypothesis="On short signal runs, compile-off can reduce overhead and variance.",
            ablation="Set COMPILE_ENABLED=0 and COMPILE_FULLGRAPH=0.",
            test="Compare step_avg and post_ema val_bpb vs compiled baseline.",
            env_overrides={"COMPILE_ENABLED": "0", "COMPILE_FULLGRAPH": "0"},
        ),
        AblationCase(
            name="sparse_skipgram_ngram",
            issue="Structured-text long-gap context at eval",
            hypothesis=(
                "Sparse skip-gram contexts (e.g., -1,-3,-5) can capture HTML/code/text structure "
                "and improve final sliding+ngram BPB with no extra table memory."
            ),
            ablation=(
                "Enable hashed n-gram eval with sparse gap patterns sharing same hash tables."
            ),
            test="Compare final_sliding_window_ngram*_exact BPB vs baseline (same seed/steps).",
            env_overrides={
                "SKIP_FINAL_EVAL": "0",
                "NGRAM_EVAL_ORDER": "7",
                "NGRAM_EVAL_MIN_ORDER": "2",
                "NGRAM_EVAL_ALPHA": "0.30",
                "NGRAM_EVAL_ADAPTIVE": "1",
                "NGRAM_EVAL_MAX_SECONDS": "180",
                "NGRAM_SPARSE_PATTERNS": (
                    "1,3;1,2;"
                    "1,3,5;1,2,4;"
                    "1,3,5,7;1,2,4,8;"
                    "1,3,5,7,9;1,2,4,8,16;"
                    "1,3,5,7,9,11;1,2,4,8,16,32"
                ),
            },
        ),
    ]


def run_case(
    case: AblationCase,
    train_script: Path,
    repo_root: Path,
    log_dir: Path,
    torchrun_bin: str,
    nproc_per_node: int,
    base_env: dict[str, str],
    dry_run: bool,
) -> dict[str, str]:
    env = base_env.copy()
    env.update(case.env_overrides)
    env["RUN_ID"] = f"ablate_{case.name}_{int(time.time())}"

    log_file = log_dir / f"{case.name}.log"
    cmd = [
        torchrun_bin,
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(train_script),
    ]

    print("\n------------------------------------------------------------")
    print(f"CASE: {case.name}")
    print(f"Issue: {case.issue}")
    print(f"Hypothesis: {case.hypothesis}")
    print(f"Ablation: {case.ablation}")
    print(f"Test: {case.test}")
    print("cmd=" + " ".join(cmd))
    print(f"log={log_file}")
    print("------------------------------------------------------------")

    if dry_run:
        return {
            "name": case.name,
            "kind": "dynamic",
            "issue": case.issue,
            "hypothesis": case.hypothesis,
            "ablation": case.ablation,
            "test": case.test,
            "status": "dry_run",
            "runtime_s": "0",
            "step_avg_ms": "N/A",
            "post_ema_bpb": "N/A",
            "final_val_bpb": "N/A",
            "final_ngram_bpb": "N/A",
            "notes": "",
            "logfile": str(log_file),
        }

    t0 = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        exit_code = proc.wait()
    runtime_s = time.perf_counter() - t0

    text = log_file.read_text(encoding="utf-8", errors="replace")
    step_avg_ms = parse_last_float(text, r"step:\d+/\d+\s+train_loss:[^\n]*step_avg:([0-9.]+)ms")
    post_ema_bpb = parse_last_float(text, r"DIAGNOSTIC post_ema .*?val_bpb:([0-9.]+)")
    final_val_bpb = parse_last_float(text, r"step:\d+/\d+\s+val_loss:[^\n]*val_bpb:([0-9.]+)")
    final_ngram_bpb = parse_last_float(text, r"final_sliding_window_ngram\d+_exact .*?val_bpb:([0-9.]+)")
    peak_mem_mib = parse_last_float(text, r"peak memory allocated:\s*([0-9.]+)\s*MiB")

    status = "ok" if exit_code == 0 else f"failed({exit_code})"
    return {
        "name": case.name,
        "kind": "dynamic",
        "issue": case.issue,
        "hypothesis": case.hypothesis,
        "ablation": case.ablation,
        "test": case.test,
        "status": status,
        "runtime_s": f"{runtime_s:.2f}",
        "step_avg_ms": fmt(step_avg_ms),
        "post_ema_bpb": fmt(post_ema_bpb),
        "final_val_bpb": fmt(final_val_bpb),
        "final_ngram_bpb": fmt(final_ngram_bpb),
        "notes": "" if peak_mem_mib is None else f"peak_mem_mib={peak_mem_mib:.0f}",
        "logfile": str(log_file),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-H100 RASCAL ablation matrix (hypothesis + ablation + test + summary)"
    )
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "444")))
    p.add_argument("--iterations", type=int, default=int(os.environ.get("ITERATIONS", "2000")))
    p.add_argument("--train-batch-tokens", type=int, default=int(os.environ.get("TRAIN_BATCH_TOKENS", "786432")))
    p.add_argument("--train-seq-len", type=int, default=int(os.environ.get("TRAIN_SEQ_LEN", "2048")))
    p.add_argument("--warmdown-iters", type=int, default=int(os.environ.get("WARMDOWN_ITERS", "0")))
    p.add_argument("--max-wallclock-seconds", type=int, default=int(os.environ.get("MAX_WALLCLOCK_SECONDS", "0")))
    p.add_argument("--nproc-per-node", type=int, default=int(os.environ.get("NPROC_PER_NODE", "1")))
    p.add_argument("--torchrun-bin", default=os.environ.get("TORCHRUN_BIN", "torchrun"))
    p.add_argument("--data-path", default=os.environ.get("DATA_PATH", ""))
    p.add_argument("--tokenizer-path", default=os.environ.get("TOKENIZER_PATH", ""))
    p.add_argument("--skip-final-eval", type=int, choices=[0, 1], default=int(os.environ.get("SKIP_FINAL_EVAL", "1")))
    p.add_argument("--post-ema-diagnostic", type=int, choices=[0, 1], default=int(os.environ.get("POST_EMA_DIAGNOSTIC", "1")))
    p.add_argument("--eval-stride", type=int, default=int(os.environ.get("EVAL_STRIDE", "64")))
    p.add_argument("--case", action="append", default=[], help="Run only selected dynamic case names (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    train_script = script_dir / "train_gpt.py"
    parser = build_parser()
    args = parser.parse_args()

    default_data_path = repo_root / "data/datasets/fineweb10B_sp1024"
    default_tokenizer_path = repo_root / "data/tokenizers/fineweb_1024_bpe.model"
    data_path = Path(args.data_path) if args.data_path else default_data_path
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else default_tokenizer_path

    require(train_script.is_file(), f"missing train script: {train_script}")
    require(data_path.is_dir(), f"DATA_PATH does not exist: {data_path}")
    require(tokenizer_path.is_file(), f"TOKENIZER_PATH does not exist: {tokenizer_path}")
    require(shutil.which(args.torchrun_bin) is not None, f"torchrun not found: {args.torchrun_bin}")
    for mod_name in ("numpy", "sentencepiece", "zstandard"):
        require(importlib.util.find_spec(mod_name) is not None, f"missing python module: {mod_name}")

    import torch

    require(torch.cuda.is_available(), "CUDA is not available")
    gpu_count = torch.cuda.device_count()
    require(gpu_count >= 1, "No visible CUDA devices")
    require(args.nproc_per_node >= 1, "nproc_per_node must be >=1")
    require(args.nproc_per_node <= gpu_count, f"nproc_per_node={args.nproc_per_node} exceeds visible_gpus={gpu_count}")

    run_tag = f"ablate_single_h100_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = script_dir / "logs" / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = log_dir / "summary.csv"

    print("============================================================")
    print("RASCAL SINGLE-H100 ABLATION MATRIX")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"visible_gpus={gpu_count} nproc_per_node={args.nproc_per_node}")
    print(f"seed={args.seed} iterations={args.iterations}")
    print(f"train_batch_tokens={args.train_batch_tokens} train_seq_len={args.train_seq_len}")
    print(f"max_wallclock_seconds={args.max_wallclock_seconds}")
    print(f"data_path={data_path}")
    print(f"tokenizer_path={tokenizer_path}")
    print(f"log_dir={log_dir}")
    print("============================================================")

    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONPATH": f"{repo_root / 'flash-attention/hopper'}:{base_env.get('PYTHONPATH', '')}",
            "DATA_PATH": str(data_path),
            "TOKENIZER_PATH": str(tokenizer_path),
            "SEED": str(args.seed),
            "ITERATIONS": str(args.iterations),
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "TRAIN_SEQ_LEN": str(args.train_seq_len),
            "WARMDOWN_ITERS": str(args.warmdown_iters),
            "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
            "LOADER_MODE": "coprime",
            "COPRIME_MAX_LOADED_SHARDS": "1",
            "COPRIME_SHARDS_PER_BATCH": "1",
            "COPRIME_SHARD_HOLD_STEPS": "64",
            "SKIP_GPTQ": "1",
            "SKIP_FINAL_EVAL": str(args.skip_final_eval),
            "POST_EMA_DIAGNOSTIC": str(args.post_ema_diagnostic),
            "EVAL_STRIDE": str(args.eval_stride),
            "TRAIN_LOG_EVERY": "500",
            "VAL_LOSS_EVERY": "4000",
            "COMPILE_ENABLED": "1",
            "COMPILE_FULLGRAPH": "1",
            "MUON_BACKEND_STEPS": "5",
            "COMPLEMENT_ALPHA": "0",
            "XSA_LAST_N": "11",
            "BIGRAM_VOCAB_SIZE": "2048",
            "ROPE_DIMS": "16",
            "SWA_EVERY": "50",
            "MTP_NUM_HEADS": "0",
            "TRIGRAM": "0",
            "NGRAM_EVAL_ORDER": "0",
            "CUBRIC_CADENCE": "0",
            "NGRAM_ENTROPY_SHIFT": "0",
        }
    )

    rows: list[dict[str, str]] = []
    rows.extend(static_bottleneck_model(train_script))

    cases = build_dynamic_cases()
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.name in wanted]
        missing = wanted - {c.name for c in cases}
        require(not missing, f"unknown case(s): {', '.join(sorted(missing))}")

    for case in cases:
        row = run_case(
            case=case,
            train_script=train_script,
            repo_root=repo_root,
            log_dir=log_dir,
            torchrun_bin=args.torchrun_bin,
            nproc_per_node=args.nproc_per_node,
            base_env=base_env,
            dry_run=args.dry_run,
        )
        rows.append(row)
        if row["status"].startswith("failed"):
            print(f"ERROR: case {case.name} failed; continuing to next case for full matrix coverage.")

    fieldnames = [
        "name",
        "kind",
        "issue",
        "hypothesis",
        "ablation",
        "test",
        "status",
        "runtime_s",
        "step_avg_ms",
        "post_ema_bpb",
        "final_val_bpb",
        "final_ngram_bpb",
        "notes",
        "logfile",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n============================================================")
    print("ABLATION SUMMARY")
    print("============================================================")
    for row in rows:
        print(
            f"{row['name']:>22} | {row['kind']:>7} | {row['status']:>10} | "
            f"step_avg_ms={row['step_avg_ms']} | post_ema_bpb={row['post_ema_bpb']} "
            f"| final_ngram_bpb={row['final_ngram_bpb']}"
        )
    print(f"\nCSV={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
