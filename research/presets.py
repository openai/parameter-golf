from __future__ import annotations

from dataclasses import dataclass


DEFAULT_DATA_PATH = "./data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    target: str
    launch_mode: str
    entrypoint: str
    env: dict[str, str]
    min_train_shards: int = 1
    nproc_per_node: int | None = None
    notes: tuple[str, ...] = ()


PRESETS: dict[str, Preset] = {
    "mlx_smoke": Preset(
        name="mlx_smoke",
        description="Official Apple Silicon smoke path from the repository README.",
        target="mlx",
        launch_mode="python",
        entrypoint="train_gpt_mlx.py",
        env={
            "DATA_PATH": DEFAULT_DATA_PATH,
            "TOKENIZER_PATH": DEFAULT_TOKENIZER_PATH,
            "VOCAB_SIZE": "1024",
            "ITERATIONS": "200",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_LOSS_EVERY": "0",
            "VAL_BATCH_SIZE": "8192",
        },
        min_train_shards=1,
        notes=(
            "Matches the README smoke configuration.",
            "Runs final validation only, which keeps the local smoke path short and predictable.",
        ),
    ),
    "local_dev_mlx": Preset(
        name="local_dev_mlx",
        description="Rules-safe local development preset for Apple Silicon iteration.",
        target="mlx",
        launch_mode="python",
        entrypoint="train_gpt_mlx.py",
        env={
            "DATA_PATH": DEFAULT_DATA_PATH,
            "TOKENIZER_PATH": DEFAULT_TOKENIZER_PATH,
            "VOCAB_SIZE": "1024",
            "ITERATIONS": "400",
            "TRAIN_BATCH_TOKENS": "32768",
            "GRAD_ACCUM_STEPS": "4",
            "TRAIN_LOG_EVERY": "25",
            "VAL_LOSS_EVERY": "0",
            "VAL_BATCH_SIZE": "32768",
            "MLX_MAX_MICROBATCH_TOKENS": "8192",
            "MAX_WALLCLOCK_SECONDS": "0",
        },
        min_train_shards=1,
        notes=(
            "Uses a fixed iteration budget instead of the 10-minute cap.",
            "Keeps validation to the final pass so the laptop loop stays lightweight.",
        ),
    ),
    "cuda_remote_baseline": Preset(
        name="cuda_remote_baseline",
        description="Clean remote CUDA baseline matching the default PyTorch path.",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt.py",
        env={
            "DATA_PATH": DEFAULT_DATA_PATH,
            "TOKENIZER_PATH": DEFAULT_TOKENIZER_PATH,
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": "600",
            "TRAIN_LOG_EVERY": "50",
            "VAL_LOSS_EVERY": "200",
        },
        min_train_shards=1,
        nproc_per_node=1,
        notes=(
            "Default is one GPU for cheaper remote iteration.",
            "Override --nproc-per-node 8 on an 8xH100 box for a track-like run.",
        ),
    ),
}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown preset {name!r}. Available presets: {', '.join(sorted(PRESETS))}") from exc
