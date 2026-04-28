import modal
import sys
import subprocess
import os
import glob
import shutil
import struct
import json
from collections import deque

app = modal.App("pg-train-detached")

image = (
    modal.Image.from_dockerfile("deploy/Dockerfile", context_dir=".", add_python="3.12")
    .pip_install("huggingface_hub")
)

data_volume = modal.Volume.from_name("pg-data", create_if_missing=True)
output_volume = modal.Volume.from_name("pg-output", create_if_missing=True)


def _pop_result_json(args: list[str]):
    forwarded = list(args)
    result_json = os.environ.get("PG_RESULT_JSON")
    if "--result-json" in forwarded:
        idx = forwarded.index("--result-json")
        if idx + 1 >= len(forwarded):
            raise RuntimeError("--result-json requires a path")
        result_json = forwarded[idx + 1]
        del forwarded[idx : idx + 2]
    return forwarded, result_json


def _write_result_json(path: str | None, result: dict):
    if not path:
        return
    if not path.startswith("/output/"):
        raise RuntimeError("--result-json must write under /output")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
    output_volume.commit()


def _apply_gpu_env_flags(forwarded: list[str]):
    if "--cuda-event-timing" in forwarded:
        forwarded.remove("--cuda-event-timing")
        os.environ["PG_CUDA_EVENT_TIMING"] = "1"
    if "--backward-stage-timing" in forwarded:
        forwarded.remove("--backward-stage-timing")
        os.environ["PG_GPU_BACKWARD_STAGE_TIMING"] = "1"
    if "--save-layer-acts" in forwarded:
        forwarded.remove("--save-layer-acts")
        os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "1"
    if "--save-all-layer-acts" in forwarded:
        forwarded.remove("--save-all-layer-acts")
        os.environ["PG_GPU_SAVE_LAYER_ACTS"] = "all"
    if "--ttt-audit" in forwarded:
        forwarded.remove("--ttt-audit")
        os.environ["PG_TTT_AUDIT"] = "1"
    if "--skip-first-step-timing" in forwarded:
        forwarded.remove("--skip-first-step-timing")
        os.environ["PG_RECORD_TIMING_SKIP_STEPS"] = "1"
    if "--fast-tf32" in forwarded:
        forwarded.remove("--fast-tf32")
        os.environ["PG_CUBLAS_FAST_TF32"] = "1"
        os.environ.setdefault("PG_CUBLAS_FORCE_TENSOR_OP_ALGO", "1")
    if "--disable-bf16-forward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-forward-gemm")
        os.environ["PG_GPU_BF16_FORWARD_GEMM"] = "0"
    if "--enable-bf16-primary-forward-gemm" in forwarded:
        forwarded.remove("--enable-bf16-primary-forward-gemm")
        os.environ["PG_GPU_BF16_PRIMARY_FORWARD_GEMM"] = "1"
    if "--disable-bf16-primary-forward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-primary-forward-gemm")
        os.environ["PG_GPU_BF16_PRIMARY_FORWARD_GEMM"] = "0"
    if "--disable-bf16-backward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-backward-gemm")
        os.environ["PG_GPU_BF16_BACKWARD_GEMM"] = "0"
    if "--disable-bf16-output-gemm" in forwarded:
        forwarded.remove("--disable-bf16-output-gemm")
        os.environ["PG_GPU_BF16_OUTPUT_GEMM"] = "0"
    if "--disable-bf16-output-backward-gemm" in forwarded:
        forwarded.remove("--disable-bf16-output-backward-gemm")
        os.environ["PG_GPU_BF16_OUTPUT_BACKWARD_GEMM"] = "0"
    if "--enable-qkv-dx-beta-accum" in forwarded:
        forwarded.remove("--enable-qkv-dx-beta-accum")
        os.environ["PG_GPU_QKV_DX_BETA_ACCUM"] = "1"
    if "--experimental-fused-qkv-proj" in forwarded:
        forwarded.remove("--experimental-fused-qkv-proj")
        os.environ["PG_GPU_FUSED_QKV_PROJ"] = "1"
    if "--qkv-dx-beta-accum" in forwarded:
        forwarded.remove("--qkv-dx-beta-accum")
        os.environ["PG_GPU_QKV_DX_BETA_ACCUM"] = "1"


def _maybe_seed_data_env():
    if os.environ.get("PG_TRAIN_GLOB") and os.environ.get("PG_VAL_GLOB"):
        return

    candidates = [
        os.environ.get("DATA_DIR"),
        "/data/datasets/fineweb10B_sp8192",
        "/data/datasets/fineweb10B_sp1024",
    ]
    for root in candidates:
        if not root:
            continue
        train_glob = os.path.join(root, "fineweb_train_*.bin")
        val_glob = os.path.join(root, "fineweb_val_[0-9]*.bin")
        if not os.environ.get("PG_TRAIN_GLOB") and glob.glob(train_glob):
            os.environ["PG_TRAIN_GLOB"] = train_glob
        if not os.environ.get("PG_VAL_GLOB") and glob.glob(val_glob):
            os.environ["PG_VAL_GLOB"] = val_glob
        if "sp8192" in root and not os.environ.get("PG_TOKENIZER_VOCAB"):
            for vocab_path in (
                os.path.join(root, "tokenizer.vocab"),
                "/data/tokenizers/fineweb_8192_bpe.vocab",
            ):
                if os.path.exists(vocab_path):
                    os.environ["PG_TOKENIZER_VOCAB"] = vocab_path
                    break
        if os.environ.get("PG_TRAIN_GLOB") and os.environ.get("PG_VAL_GLOB"):
            break

    if not os.environ.get("PG_CASEOPS_BYTE_SIDECAR"):
        sidecar_candidates = [
            os.path.join(root, "fineweb_val_bytes_*.bin")
            for root in candidates
            if root
        ]
        for pattern in sidecar_candidates:
            if glob.glob(pattern):
                os.environ["PG_CASEOPS_BYTE_SIDECAR"] = pattern
                break


def _run_pg_train(args: list[str], label: str):
    os.environ["RUST_LOG"] = "info"
    os.environ.setdefault("DATA_DIR", "/data/datasets/fineweb10B_sp8192")
    _maybe_seed_data_env()
    forwarded, result_json = _pop_result_json(args)
    _apply_gpu_env_flags(forwarded)
    mode = "smoke"
    if "--mode" in forwarded:
        mode_idx = forwarded.index("--mode")
        if mode_idx + 1 < len(forwarded):
            mode = forwarded[mode_idx + 1]
    if mode in {"record", "record-shaped-proxy"}:
        os.environ.setdefault("PG_GPU_HOST_SCALAR_UPDATES", "0")
    if mode == "record-shaped-proxy" and "--allow-unsupported-variants" not in forwarded:
        forwarded.append("--allow-unsupported-variants")
    if os.environ.get("PG_TRAIN_GLOB") and "--train-data" not in forwarded:
        forwarded.extend(["--train-data", os.environ["PG_TRAIN_GLOB"]])
    include_val_data = os.environ.get("PG_INCLUDE_VAL_DATA") == "1"
    if include_val_data and os.environ.get("PG_VAL_GLOB") and "--val-data" not in forwarded:
        forwarded.extend(["--val-data", os.environ["PG_VAL_GLOB"]])
    if os.environ.get("PG_TOKENIZER_VOCAB") and "--tokenizer-vocab" not in forwarded:
        forwarded.extend(["--tokenizer-vocab", os.environ["PG_TOKENIZER_VOCAB"]])
    if (
        os.environ.get("PG_CASEOPS_BYTE_SIDECAR")
        and "--caseops-byte-sidecar" not in forwarded
    ):
        forwarded.extend(["--caseops-byte-sidecar", os.environ["PG_CASEOPS_BYTE_SIDECAR"]])
    if mode != "record" and "--eval-max-tokens" not in forwarded:
        forwarded.extend(["--eval-max-tokens", os.environ.get("PG_EVAL_MAX_TOKENS", "16384")])
    cmd = ["pg-train"] + forwarded
    print(f"Running {label} command:", " ".join(cmd), flush=True)
    print(
        "Data environment:",
        {
            "DATA_DIR": os.environ.get("DATA_DIR"),
            "PG_TRAIN_GLOB": os.environ.get("PG_TRAIN_GLOB"),
            "PG_VAL_GLOB": os.environ.get("PG_VAL_GLOB"),
            "PG_TOKENIZER_VOCAB": os.environ.get("PG_TOKENIZER_VOCAB"),
            "PG_CASEOPS_BYTE_SIDECAR": os.environ.get("PG_CASEOPS_BYTE_SIDECAR"),
        },
        flush=True,
    )

    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)

    proc.wait()
    result = {
        "label": label,
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    _write_result_json(result_json, result)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    output_volume.commit()
    return result


def _run_pg_eval(args: list[str]):
    os.environ["RUST_LOG"] = "info"
    os.environ.setdefault("DATA_DIR", "/data/datasets/fineweb10B_sp8192")
    _maybe_seed_data_env()
    forwarded, result_json = _pop_result_json(args)
    _apply_gpu_env_flags(forwarded)
    if os.environ.get("PG_VAL_GLOB") and "--val-data" not in forwarded:
        forwarded.extend(["--val-data", os.environ["PG_VAL_GLOB"]])
    if os.environ.get("PG_TOKENIZER_VOCAB") and "--tokenizer-vocab" not in forwarded:
        forwarded.extend(["--tokenizer-vocab", os.environ["PG_TOKENIZER_VOCAB"]])
    if (
        os.environ.get("PG_CASEOPS_BYTE_SIDECAR")
        and "--caseops-byte-sidecar" not in forwarded
    ):
        forwarded.extend(["--caseops-byte-sidecar", os.environ["PG_CASEOPS_BYTE_SIDECAR"]])
    leaderboard_eval = "--leaderboard" in forwarded
    if (
        not leaderboard_eval
        and os.environ.get("PG_EVAL_MAX_TOKENS")
        and "--max-tokens" not in forwarded
    ):
        forwarded.extend(["--max-tokens", os.environ["PG_EVAL_MAX_TOKENS"]])
    cmd = ["pg-eval"] + forwarded
    print("Running eval command:", " ".join(cmd), flush=True)

    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)
    proc.wait()
    result = {
        "label": "eval",
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    _write_result_json(result_json, result)
    if proc.returncode != 0:
        raise RuntimeError(
            f"eval command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    return result


def _is_control_piece(piece: str) -> bool:
    return piece in {"<pad>", "<s>", "</s>", "<unk>"} or piece.startswith("<unused")


def _is_byte_piece(piece: str) -> bool:
    return (
        len(piece) == 6
        and piece.startswith("<0x")
        and piece.endswith(">")
        and all(ch in "0123456789abcdefABCDEF" for ch in piece[3:5])
    )


def _build_bpb_luts(vocab_path: str):
    base_bytes: list[int] = []
    has_leading_space: list[bool] = []
    is_boundary: list[bool] = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            piece = line.split("\t", 1)[0]
            boundary = _is_control_piece(piece)
            leading = piece.startswith("▁")
            if boundary:
                nbytes = 0
            elif _is_byte_piece(piece):
                nbytes = 1
            else:
                nbytes = len(piece.lstrip("▁").encode("utf-8"))
            base_bytes.append(min(nbytes, 65535))
            has_leading_space.append(leading)
            is_boundary.append(boundary)
    if not base_bytes:
        raise RuntimeError(f"tokenizer vocab {vocab_path} contained no pieces")
    return base_bytes, has_leading_space, is_boundary


def _token_byte_count(prev: int, target: int, base_bytes, has_leading_space, is_boundary) -> int:
    if target >= len(base_bytes):
        return 1
    nbytes = max(0, base_bytes[target])
    prev_is_boundary = prev >= len(is_boundary) or is_boundary[prev]
    if has_leading_space[target] and not prev_is_boundary:
        nbytes += 1
    return min(nbytes, 65535)


def _read_u16_shard(path: str):
    with open(path, "rb") as f:
        header = f.read(256 * 4)
        if len(header) != 256 * 4:
            raise RuntimeError(f"token shard {path} is missing the 256-int32 header")
        raw = f.read()
    if len(raw) % 2 != 0:
        raise RuntimeError(f"token shard {path} payload has odd byte length")
    tokens = list(struct.unpack(f"<{len(raw) // 2}H", raw))
    return header, tokens


def _write_u16_shard(path: str, header: bytes, values: list[int]):
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack(f"<{len(values)}H", *values))


def _ensure_caseops_byte_sidecars(dataset_dir: str, vocab_path: str) -> str:
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_[0-9]*.bin")))
    if not val_files:
        raise RuntimeError(f"no validation shards found in {dataset_dir}")
    sidecar_pattern = os.path.join(dataset_dir, "fineweb_val_bytes_*.bin")
    sidecars = sorted(glob.glob(sidecar_pattern))
    if len(sidecars) == len(val_files):
        return sidecar_pattern

    print("Generating CaseOps validation byte sidecars", flush=True)
    base_bytes, has_leading_space, is_boundary = _build_bpb_luts(vocab_path)
    prev = 0
    for val_path in val_files:
        header, tokens = _read_u16_shard(val_path)
        byte_counts: list[int] = []
        for tok in tokens:
            byte_counts.append(_token_byte_count(prev, tok, base_bytes, has_leading_space, is_boundary))
            prev = tok
        name = os.path.basename(val_path).replace("fineweb_val_", "fineweb_val_bytes_")
        _write_u16_shard(os.path.join(dataset_dir, name), header, byte_counts)
    return sidecar_pattern

def _run_pg_bench(args: list[str]):
    os.environ["RUST_LOG"] = "info"
    forwarded, result_json = _pop_result_json(args)
    _apply_gpu_env_flags(forwarded)
    if not forwarded:
        raise RuntimeError("bench requires a binary name")
    allowed = {
        "parity-kernels": "pg-parity-kernels",
        "parity-forward": "pg-parity-forward",
        "parity-step": "pg-parity-step",
        "gemm-bench": "pg-gemm-bench",
        "attention-bench": "pg-attention-bench",
        "nccl-bench": "pg-nccl-bench",
        "preliminary": "pg-preliminary",
        "smoke": "pg-smoke",
    }
    binary = allowed.get(forwarded[0])
    if binary is None:
        raise RuntimeError(f"unsupported bench binary {forwarded[0]!r}; allowed={sorted(allowed)}")
    cmd = [binary] + list(forwarded[1:])
    print("Running bench command:", " ".join(cmd), flush=True)

    tail = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    for line in iter(proc.stdout.readline, ""):
        tail.append(line)
        print(line, end="", flush=True)
    proc.wait()
    result = {
        "label": "bench",
        "command": cmd,
        "returncode": proc.returncode,
        "tail": "".join(tail),
    }
    _write_result_json(result_json, result)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench command failed with code {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Last output:\n{result['tail']}"
        )
    return result


def _forwarded_requests_multi_gpu(forwarded: list[str]) -> bool:
    if "--backend" in forwarded:
        idx = forwarded.index("--backend")
        if idx + 1 < len(forwarded) and forwarded[idx + 1] == "cuda-distributed":
            return True
    if "--world-size" in forwarded:
        idx = forwarded.index("--world-size")
        if idx + 1 < len(forwarded):
            try:
                return int(forwarded[idx + 1]) > 1
            except ValueError:
                return False
    return False


@app.function(
    image=image,
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
    },
)
def seed_data():
    from huggingface_hub import snapshot_download

    dataset_dir = "/data/datasets/fineweb10B_sp8192"
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_[0-9]*.bin")))
    vocab_path = "/data/tokenizers/fineweb_8192_bpe.vocab"
    if train_files and val_files and os.path.exists(vocab_path):
        print(
            "SP8192 data already present:",
            {
                "train_files": len(train_files),
                "val_files": len(val_files),
                "vocab_path": vocab_path,
            },
            flush=True,
        )
    else:
        print("Downloading SP8192 shards/tokenizer into pg-data volume", flush=True)
        snapshot_download(
            repo_id="sproos/parameter-golf-tokenizers",
            local_dir="/data",
            allow_patterns=[
                "datasets/fineweb10B_sp8192/*",
                "tokenizers/fineweb_8192_bpe.model",
                "tokenizers/fineweb_8192_bpe.vocab",
            ],
        )

    os.makedirs(dataset_dir, exist_ok=True)
    if os.path.exists(vocab_path):
        shutil.copyfile(vocab_path, os.path.join(dataset_dir, "tokenizer.vocab"))
    sidecar_pattern = None
    if os.path.exists(vocab_path):
        sidecar_pattern = _ensure_caseops_byte_sidecars(dataset_dir, vocab_path)
    train_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_[0-9]*.bin")))
    sidecar_files = sorted(glob.glob(os.path.join(dataset_dir, "fineweb_val_bytes_*.bin")))
    result = {
        "dataset_dir": dataset_dir,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "tokenizer_vocab": vocab_path if os.path.exists(vocab_path) else None,
        "caseops_byte_sidecar": sidecar_pattern,
        "caseops_byte_sidecar_files": len(sidecar_files),
    }
    print("Seed-data result:", result, flush=True)
    if not train_files or not val_files or not result["tokenizer_vocab"]:
        raise RuntimeError(f"SP8192 seed incomplete: {result}")
    if len(sidecar_files) != len(val_files):
        raise RuntimeError(f"CaseOps sidecar generation incomplete: {result}")
    data_volume.commit()
    return result

@app.function(
    image=image,
    gpu="H100:1",
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_command(args: list[str]):
    return _run_pg_train(args, "single-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=3600,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_command_multi(args: list[str]):
    return _run_pg_train(args, "multi-GPU")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1800,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_eval_command(args: list[str]):
    return _run_pg_eval(args)

@app.function(
    image=image,
    gpu="H100:1",
    timeout=1800,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def run_bench_command(args: list[str]):
    return _run_pg_bench(args)

@app.local_entrypoint()
def main(*args: str):
    use_multi = False
    forwarded = list(args)
    wait_for_result = os.environ.get("PG_WAIT") == "1"
    if forwarded and forwarded[0] == "seed-data":
        if wait_for_result:
            result = seed_data.remote()
            print("Seed-data result:", result, flush=True)
            return
        call = seed_data.spawn()
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned seed-data Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "eval":
        if wait_for_result:
            result = run_eval_command.remote(forwarded[1:])
            print("Eval result:", result, flush=True)
            return
        call = run_eval_command.spawn(forwarded[1:])
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned eval Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "bench":
        if wait_for_result:
            result = run_bench_command.remote(forwarded[1:])
            print("Bench result:", result, flush=True)
            return
        call = run_bench_command.spawn(forwarded[1:])
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print("Spawned bench Modal call:", call_id or call, flush=True)
        return
    if forwarded and forwarded[0] == "--multi":
        use_multi = True
        forwarded = forwarded[1:]
    if os.environ.get("PG_MULTI_GPU") == "1":
        use_multi = True
    if _forwarded_requests_multi_gpu(forwarded):
        use_multi = True
    print(
        "Dispatching command to detached runner:",
        {"multi_gpu": use_multi, "args": forwarded},
        flush=True,
    )
    if use_multi:
        if wait_for_result:
            result = run_command_multi.remote(forwarded)
            print("Multi-GPU result:", result, flush=True)
            return
        call = run_command_multi.spawn(forwarded)
    else:
        if wait_for_result:
            result = run_command.remote(forwarded)
            print("Single-GPU result:", result, flush=True)
            return
        call = run_command.spawn(forwarded)
    call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
    print("Spawned Modal call:", call_id or call, flush=True)
