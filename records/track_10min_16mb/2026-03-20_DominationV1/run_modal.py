"""
Modal deployment script for parameter-golf training on 8xH100 SXM.

Usage:
    Standard run:     modal run run_modal.py --mode standard --profile domv1 --seeds 1337,42,7
    Val-only run:     modal run run_modal.py --mode valonly --profile domv1 --seeds 1337,42,7
    Architecture:     modal run run_modal.py --mode standard --profile domv1 --num-layers 12 --bigram-vocab 2048 --tag 12L_bg2k
    HP sweep:         modal run run_modal.py --mode standard --profile domv1 --muon-wd 0.05 --matrix-lr 0.03 --tag wd05_lr03
    Batch sweep:      modal run run_modal.py --mode standard --profile domv1 --extra-env "TRAIN_BATCH_TOKENS=393216" --tag batch393k

This will:
    - Download the FineWeb dataset inside Modal (cached in a Volume)
    - Run torchrun with 8xH100 GPUs for 10 minutes
    - Download train.log and model artifacts to your local machine
"""

import modal

app = modal.App("parameter-golf")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_vol = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)

TRAIN_SCRIPT_STANDARD_LEGACY = "records/track_10min_16mb/2026-03-19_StandardOptimal/train_gpt.py"
TRAIN_SCRIPT_VALONLY_LEGACY = "records/track_10min_16mb/2026-03-19_CombinedOptimal/train_gpt.py"
TRAIN_SCRIPT_DOMV1 = "records/track_10min_16mb/2026-03-20_DominationV1/train_gpt.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.10",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "tqdm",
        "setuptools",
        "typing-extensions==4.15.0",
        "zstandard",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .add_local_file("data/cached_challenge_fineweb.py", "/root/data/cached_challenge_fineweb.py")
    .add_local_file(TRAIN_SCRIPT_STANDARD_LEGACY, "/root/train_gpt_standard.py")
    .add_local_file(TRAIN_SCRIPT_VALONLY_LEGACY, "/root/train_gpt_valonly.py")
    .add_local_file(TRAIN_SCRIPT_DOMV1, "/root/train_gpt_domv1.py")
)


def _parse_extra_env(raw: str) -> dict[str, str]:
    out = {}
    if not raw:
        return out
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid extra_env entry (expected KEY=VALUE): {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _profile_env(mode: str, profile: str) -> dict[str, str]:
    if profile == "domv1":
        base = {
            "VAL_LOSS_EVERY": "500",
            "TRAIN_LOG_EVERY": "100",
            "TRAIN_SEQ_LEN": "2048",
            "NUM_LAYERS": "11",
            "SWA_ENABLED": "1",
            "SWA_EVERY": "50",
            "SWA_START_FRAC": "0.5",
            "MIXED_QUANT_INT6_CATS": "mlp,attn,other",
            "INT6_QUANT_RANGE_MLP": "15",
            "INT6_QUANT_RANGE_ATTN": "31",
            "INT6_QUANT_RANGE_OTHER": "31",
            "WEIGHT_DECAY": "0.04",
            "MUON_WD": "0.04",
            "MATRIX_LR": "0.025",
            "SCALAR_LR": "0.025",
            "TIED_EMBED_LR": "0.035",
            "MUON_MOMENTUM": "0.99",
            "MUON_MOMENTUM_WARMUP_START": "0.92",
            "MUON_MOMENTUM_WARMUP_STEPS": "1500",
            "WARMDOWN_ITERS": "3000",
            "BIGRAM_VOCAB_SIZE": "4096",
            "BIGRAM_DIM": "128",
            "EVAL_STRIDE": "64",
            "EVAL_BATCH_SEQS": "32",
        }
        if mode == "standard":
            base.update({
                "TRAIN_BATCH_TOKENS": "524288",
                "STE_QAT_ENABLED": "0",
            })
        else:
            base.update({
                "TRAIN_BATCH_TOKENS": "524288",
                "TRAIN_SEQ_LEN": "1024",
                "STE_QAT_ENABLED": "1",
                "STE_QAT_RANGE": "31",
            })
        return base

    if profile in {"counter", "counter_v7"}:
        if mode == "standard":
            return {
                "VAL_LOSS_EVERY": "500",
                "TRAIN_LOG_EVERY": "100",
                "TRAIN_SEQ_LEN": "2048",
                "TRAIN_BATCH_TOKENS": "786432",
                "NUM_LAYERS": "10",
                "SWA_ENABLED": "1",
                "SWA_EVERY": "50",
                "SWA_START_FRAC": "0.5",
                "STE_QAT_ENABLED": "0",
                "MIXED_QUANT_INT6_CATS": "mlp,attn",
                "INT6_QUANT_RANGE_MLP": "15",
                "INT6_QUANT_RANGE_ATTN": "31",
                "INT6_QUANT_RANGE_OTHER": "31",
                "FP16_PASSTHROUGH_PATTERNS": "tok_emb,blocks.9.attn.c_k",
                "WEIGHT_DECAY": "0.04",
                "MUON_WD": "0.04",
            }
        else:
            return {
                "VAL_LOSS_EVERY": "500",
                "TRAIN_LOG_EVERY": "100",
                "TRAIN_SEQ_LEN": "1024",
                "TRAIN_BATCH_TOKENS": "524288",
                "NUM_LAYERS": "11",
                "MATRIX_LR": "0.025",
                "SCALAR_LR": "0.025",
                "SWA_ENABLED": "0",
                "STE_QAT_ENABLED": "0",
                "MIXED_QUANT_INT6_CATS": "mlp,attn",
                "FP16_PASSTHROUGH_PATTERNS": "tok_emb,blocks.10.attn.c_k",
                "WEIGHT_DECAY": "0.034",
                "MUON_WD": "0.034",
            }

    if profile == "baseline":
        return {
            "VAL_LOSS_EVERY": "500",
            "TRAIN_LOG_EVERY": "100",
            "TRAIN_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "786432",
            "NUM_LAYERS": "9",
            "SWA_ENABLED": "1",
            "SWA_EVERY": "50",
            "SWA_START_FRAC": "0.5",
            "STE_QAT_ENABLED": "0",
            "MIXED_QUANT_INT6_CATS": "mlp,attn",
            "INT6_QUANT_RANGE_MLP": "31",
            "INT6_QUANT_RANGE_ATTN": "31",
            "INT6_QUANT_RANGE_OTHER": "31",
            "FP16_PASSTHROUGH_PATTERNS": "tok_emb,blocks.8.attn.c_k",
            "WEIGHT_DECAY": "0.04",
            "MUON_WD": "0.04",
        }

    raise ValueError(f"Unknown profile '{profile}'")


def _select_train_script(profile: str, mode: str) -> str:
    if profile == "domv1":
        return "/root/train_gpt_domv1.py"
    return "/root/train_gpt_standard.py" if mode == "standard" else "/root/train_gpt_valonly.py"


@app.function(
    image=image,
    gpu="H100:8",
    timeout=45 * 60,
    volumes={
        "/data": data_vol,
        "/output": output_vol,
    },
)
def train(
    mode: str = "standard",
    seed: int = 1337,
    tag: str = "",
    profile: str = "domv1",
    num_layers: int = 0,
    quant_mode: str = "auto",
    muon_wd: float = -1.0,
    weight_decay: float = -1.0,
    matrix_lr: float = -1.0,
    scalar_lr: float = -1.0,
    swa_every: int = 0,
    bigram_vocab: int = 0,
    batch_tokens: int = 0,
    rope_base: float = -1.0,
    seq_len: int = 0,
    extra_env: str = "",
):
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/root")

    data_base = "/data/datasets/fineweb10B_sp1024"
    tokenizer_dir = "/data/tokenizers"
    val_shard = f"{data_base}/fineweb_val_000000.bin"
    is_standard = mode == "standard"
    train_shards = "80" if is_standard else "1"
    mode_prefix = "standard" if is_standard else "valonly"
    run_id_base = f"{mode_prefix}_{profile}_s{seed}"
    run_id = f"{run_id_base}_{tag}" if tag else run_id_base

    need_download = not os.path.exists(val_shard)
    if is_standard:
        need_download = need_download or not os.path.exists(f"{data_base}/fineweb_train_000079.bin")

    if need_download:
        print(f"=== Downloading FineWeb data ({train_shards} train shards) ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "/root/data/cached_challenge_fineweb.py",
                "--variant", "sp1024",
                "--train-shards", train_shards,
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": "/root"},
            cwd="/root",
        )
        local_ds = "/root/data/datasets/fineweb10B_sp1024"
        local_tok = "/root/data/tokenizers"
        os.makedirs(data_base, exist_ok=True)
        os.makedirs(tokenizer_dir, exist_ok=True)
        for f in os.listdir(local_ds):
            src = os.path.join(local_ds, f)
            dst = os.path.join(data_base, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        for f in os.listdir(local_tok):
            src = os.path.join(local_tok, f)
            dst = os.path.join(tokenizer_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        data_vol.commit()
        print("=== Data cached to volume ===", flush=True)
    else:
        print("=== Data already cached ===", flush=True)

    if is_standard:
        data_path = data_base
    else:
        valonly_dir = "/data/datasets/fineweb10B_sp1024_valonly"
        os.makedirs(valonly_dir, exist_ok=True)
        valonly_train = f"{valonly_dir}/fineweb_train_000000.bin"
        valonly_val = f"{valonly_dir}/fineweb_val_000000.bin"
        if not os.path.exists(valonly_train):
            shutil.copy2(val_shard, valonly_train)
        if not os.path.exists(valonly_val):
            shutil.copy2(val_shard, valonly_val)
        data_vol.commit()
        data_path = valonly_dir

    print(f"=== Starting training (mode={mode}, profile={profile}, seed={seed}, data={data_path}) ===", flush=True)

    env = {
        **os.environ,
        "RUN_ID": run_id,
        "SEED": str(seed),
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": f"{tokenizer_dir}/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": "600",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    env.update(_profile_env(mode, profile))

    if num_layers > 0:
        env["NUM_LAYERS"] = str(num_layers)
        last_idx = num_layers - 1
        fp16_pats = env.get("FP16_PASSTHROUGH_PATTERNS", "tok_emb")
        if "blocks." in fp16_pats:
            import re
            fp16_pats = re.sub(r"blocks\.\d+", f"blocks.{last_idx}", fp16_pats)
        env["FP16_PASSTHROUGH_PATTERNS"] = fp16_pats
    if muon_wd >= 0:
        env["MUON_WD"] = str(muon_wd)
    if weight_decay >= 0:
        env["WEIGHT_DECAY"] = str(weight_decay)
    if matrix_lr >= 0:
        env["MATRIX_LR"] = str(matrix_lr)
    if scalar_lr >= 0:
        env["SCALAR_LR"] = str(scalar_lr)
    if swa_every > 0:
        env["SWA_ENABLED"] = "1"
        env["SWA_EVERY"] = str(swa_every)
    if bigram_vocab > 0:
        env["BIGRAM_VOCAB_SIZE"] = str(bigram_vocab)
    if batch_tokens > 0:
        env["TRAIN_BATCH_TOKENS"] = str(batch_tokens)
    if rope_base >= 0:
        env["ROPE_BASE"] = str(rope_base)
    if seq_len > 0:
        env["TRAIN_SEQ_LEN"] = str(seq_len)
    if quant_mode == "int5mlp":
        env.update({
            "INT6_QUANT_RANGE_MLP": "15",
            "INT6_QUANT_RANGE_ATTN": "31",
            "STE_QAT_ENABLED": "0",
        })
    elif quant_mode == "int6all":
        env.update({
            "INT6_QUANT_RANGE_MLP": "31",
            "INT6_QUANT_RANGE_ATTN": "31",
            "INT6_QUANT_RANGE_OTHER": "31",
        })
    elif quant_mode != "auto":
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    env.update(_parse_extra_env(extra_env))

    train_script = _select_train_script(profile, mode)

    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            train_script,
        ],
        env=env,
        cwd="/root",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print(f"\n=== Training finished with exit code {result.returncode} ===", flush=True)

    output_base = f"/output/{run_id}"
    os.makedirs(output_base, exist_ok=True)

    for fname in ["final_model.pt", "final_model.int8.ptz"]:
        src = f"/root/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, f"{output_base}/{fname}")
            print(f"  Saved {fname} ({os.path.getsize(src)} bytes)", flush=True)

    log_dir = "/root/logs"
    if os.path.isdir(log_dir):
        for fname in os.listdir(log_dir):
            src = os.path.join(log_dir, fname)
            shutil.copy2(src, f"{output_base}/{fname}")
            print(f"  Saved log: {fname}", flush=True)

    output_vol.commit()
    print(f"\n=== All outputs saved to volume 'parameter-golf-output' at /{run_id}/ ===")


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/output": output_vol},
)
def download_results(run_id: str = "standard_domv1_s1337"):
    import os

    output_base = f"/output/{run_id}"
    if not os.path.isdir(output_base):
        print(f"No results found for run_id={run_id}.")
        return

    for fname in sorted(os.listdir(output_base)):
        fpath = os.path.join(output_base, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}: {size:,} bytes")

        if fname.endswith(".txt"):
            print(f"\n--- {fname} contents (last 40 lines) ---")
            with open(fpath) as f:
                lines = f.readlines()
            for line in lines[-40:]:
                print(line, end="")
            print(f"\n--- end {fname} ---\n")


@app.local_entrypoint()
def main(
    mode: str = "standard",
    profile: str = "domv1",
    seed: int = 1337,
    seeds: str = "",
    tag: str = "",
    num_layers: int = 0,
    quant_mode: str = "auto",
    muon_wd: float = -1.0,
    weight_decay: float = -1.0,
    matrix_lr: float = -1.0,
    scalar_lr: float = -1.0,
    swa_every: int = 0,
    bigram_vocab: int = 0,
    batch_tokens: int = 0,
    rope_base: float = -1.0,
    seq_len: int = 0,
    extra_env: str = "",
):
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()] if seeds else [seed]
    print(
        f"Launching training on Modal 8xH100 SXM (mode={mode}, profile={profile}, "
        f"seeds={seed_list}, tag={tag}, quant_mode={quant_mode})..."
    )
    print("This will take ~15 minutes (10 min train + ~5 min eval + overhead)\n")
    for i, run_seed in enumerate(seed_list):
        run_tag = tag
        if len(seed_list) > 1 and not run_tag:
            run_tag = f"batch{i+1}"
        mode_prefix = "standard" if mode == "standard" else "valonly"
        run_id_base = f"{mode_prefix}_{profile}_s{run_seed}"
        run_id = f"{run_id_base}_{run_tag}" if run_tag else run_id_base
        train.remote(
            mode=mode,
            seed=run_seed,
            tag=run_tag,
            profile=profile,
            num_layers=num_layers,
            quant_mode=quant_mode,
            muon_wd=muon_wd,
            weight_decay=weight_decay,
            matrix_lr=matrix_lr,
            scalar_lr=scalar_lr,
            swa_every=swa_every,
            bigram_vocab=bigram_vocab,
            batch_tokens=batch_tokens,
            rope_base=rope_base,
            seq_len=seq_len,
            extra_env=extra_env,
        )
        print(f"\n=== Fetching results for {run_id} ===\n")
        download_results.remote(run_id=run_id)
        print(f"\nTo download files locally for {run_id}:")
        print(f"  modal volume get parameter-golf-output {run_id}/{run_id}.txt ./train.log")
        print(f"  modal volume get parameter-golf-output {run_id}/final_model.int8.ptz ./final_model.int8.ptz")
