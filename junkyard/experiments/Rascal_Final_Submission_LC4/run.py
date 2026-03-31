#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


RACE_ENV = {
    "ITERATIONS": "20000",
    "WARMDOWN_ITERS": "3500",
    "TRAIN_BATCH_TOKENS": "786432",
    "TRAIN_SEQ_LEN": "2048",
    "EVAL_SEQ_LEN": "2048",
    "MAX_WALLCLOCK_SECONDS": "600",
    "VAL_LOSS_EVERY": "4000",
    "TRAIN_LOG_EVERY": "500",
    "COMPILE_ENABLED": "1",
    "COMPILE_FULLGRAPH": "1",
    "SKIP_GPTQ": "1",
    "LOADER_MODE": "coprime",
    "COPRIME_MAX_LOADED_SHARDS": "4",
    "COPRIME_SHARDS_PER_BATCH": "1",
    "COPRIME_SHARD_HOLD_STEPS": "64",
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

SMOKE_OVERRIDES = {
    "ITERATIONS": "20",
    "WARMDOWN_ITERS": "0",
    "TRAIN_BATCH_TOKENS": "131072",
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "20",
    "TRAIN_LOG_EVERY": "10",
    "SKIP_FINAL_EVAL": "1",
    "POST_EMA_DIAGNOSTIC": "0",
    "SEED": "444",
}


def default_torchrun_bin() -> str:
    override = os.environ.get("TORCHRUN_BIN", "").strip()
    if override:
        return override
    return ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RASCAL FINAL SUBMISSION launcher (LC4)")
    ap.add_argument("--mode", choices=["race", "smoke"], default="race")
    ap.add_argument("--seed", type=int, default=444)
    ap.add_argument("--nproc-per-node", type=int, default=8)
    ap.add_argument(
        "--torchrun-bin",
        default=default_torchrun_bin(),
        help="Optional explicit torchrun binary. If omitted, uses current python -m torch.distributed.run.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def visible_gpu_count() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        try:
            import torch

            return int(torch.cuda.device_count())
        except Exception:
            return 0
    if raw == "-1":
        return 0
    return len([x for x in raw.split(",") if x.strip()])


def _prepend_env_path(env: dict[str, str], key: str, path: str) -> None:
    existing = env.get(key, "").strip()
    if not existing:
        env[key] = path
        return
    parts = [p for p in existing.split(":") if p]
    if path not in parts:
        env[key] = f"{path}:{existing}"


def configure_fa3_env(env: dict[str, str], repo_root: Path) -> None:
    # Hopper FA3 python bindings in repo
    hopper_path = repo_root / "flash-attention" / "hopper"
    if hopper_path.is_dir():
        _prepend_env_path(env, "PYTHONPATH", str(hopper_path))

    # CUDA runtime path needed by flash_attn_interface -> flash_attn_3._C
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cudart_path = (
        Path(sys.prefix) / "lib" / py_ver / "site-packages" / "nvidia" / "cuda_runtime" / "lib"
    )
    if (cudart_path / "libcudart.so.12").is_file():
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(cudart_path))


def check_fa3(env: dict[str, str]) -> tuple[bool, str]:
    code = (
        "import flash_attn_interface as f; "
        "import torch; "
        "print('FA3_OK', f.__file__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = (proc.stdout or "").strip()
    return proc.returncode == 0, output


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    train_script = script_dir / "train_gpt.py"

    data_path = Path(os.environ.get("DATA_PATH", str(repo_root / "data" / "datasets" / "fineweb10B_sp1024")))
    tokenizer_path = Path(os.environ.get("TOKENIZER_PATH", str(repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model")))

    if not train_script.is_file():
        raise SystemExit(f"ERROR: missing trainer: {train_script}")
    if not data_path.is_dir():
        raise SystemExit(f"ERROR: DATA_PATH does not exist: {data_path}")
    if not tokenizer_path.is_file():
        raise SystemExit(f"ERROR: TOKENIZER_PATH does not exist: {tokenizer_path}")

    env = os.environ.copy()
    env.update(RACE_ENV)
    env["DATA_PATH"] = str(data_path)
    env["TOKENIZER_PATH"] = str(tokenizer_path)
    env["SEED"] = str(args.seed)
    configure_fa3_env(env, repo_root)

    if args.mode == "smoke":
        env.update(SMOKE_OVERRIDES)
        env["SEED"] = str(args.seed)

    run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"rascal_final_lc4_{args.mode}_s{args.seed}_{run_tag}"
    env["RUN_ID"] = run_id

    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tee_log = logs_dir / f"{run_id}.log"

    torch_ver = "unknown"
    cuda_ver = "unknown"
    gpu_avail = "unknown"
    try:
        import torch

        torch_ver = str(torch.__version__)
        cuda_ver = str(torch.version.cuda)
        gpu_avail = str(torch.cuda.is_available())
    except Exception:
        pass

    vis = visible_gpu_count()

    print("============================================================")
    print("RASCAL FINAL SUBMISSION (LC4)")
    print(f"mode={args.mode} seed={env['SEED']}")
    print(f"torch={torch_ver} cuda={cuda_ver} gpu_available={gpu_avail}")
    print(f"visible_gpus={vis} nproc_per_node={args.nproc_per_node}")
    launcher = args.torchrun_bin if args.torchrun_bin else f"{sys.executable} -m torch.distributed.run"
    print(f"launcher={launcher}")
    print(f"data_path={data_path}")
    print(f"tokenizer_path={tokenizer_path}")
    print("------------------------------------------------------------")
    print("LOCKED RACE DELTA")
    print(f"COPRIME_MAX_LOADED_SHARDS={env['COPRIME_MAX_LOADED_SHARDS']} (upgraded)")
    print("LOADER_MODE=coprime COPRIME_SHARDS_PER_BATCH=1 COPRIME_SHARD_HOLD_STEPS=64")
    print("TRIGRAM=0 NGRAM_EVAL_ORDER=0 CUBRIC_CADENCE=0 MTP_NUM_HEADS=0")
    print("SKIP_GPTQ=1 (matches submitted Rascal II path)")
    print("============================================================")

    ok_fa3, fa3_msg = check_fa3(env)
    if ok_fa3:
        print(fa3_msg)
    else:
        print("ERROR: FA3 preflight failed.")
        if fa3_msg:
            print(fa3_msg)
        raise SystemExit(
            "Aborting launch: FA3 is required for race mode. "
            "Fix FA3 import/libcudart and rerun."
        )

    if args.nproc_per_node < 1:
        raise SystemExit("ERROR: --nproc-per-node must be >= 1")
    if args.mode == "race" and vis and args.nproc_per_node > vis:
        raise SystemExit(
            f"ERROR: nproc_per_node={args.nproc_per_node} exceeds visible GPUs={vis}. "
            "Set --nproc-per-node to match your pod."
        )

    if args.torchrun_bin:
        cmd = [
            args.torchrun_bin,
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            str(train_script),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            str(train_script),
        ]

    print(f"run_id={run_id}")
    print(f"cmd={' '.join(cmd)}")
    print(f"tee_log={tee_log}")

    if args.dry_run:
        print("dry_run=1 -> not launching")
        return

    with tee_log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
