from __future__ import annotations

from dataclasses import dataclass


DEFAULT_DATA_PATH = "./data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    family: str
    target: str
    launch_mode: str
    entrypoint: str
    env: dict[str, str]
    min_train_shards: int = 1
    nproc_per_node: int | None = None
    diff_base: str | None = None
    notes: tuple[str, ...] = ()
    counted_code_paths: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    legality_summary: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunScale:
    name: str
    description: str
    env: dict[str, str]
    notes: tuple[str, ...] = ()


def with_overrides(base: dict[str, str], /, **overrides: str) -> dict[str, str]:
    merged = dict(base)
    merged.update(overrides)
    return merged


FRONTIER_BASE_ENV = {
    "DATA_PATH": DEFAULT_DATA_PATH,
    "TOKENIZER_PATH": DEFAULT_TOKENIZER_PATH,
    "VOCAB_SIZE": "1024",
    "MAX_WALLCLOCK_SECONDS": "600",
    "TRAIN_LOG_EVERY": "100",
    "VAL_LOSS_EVERY": "500",
    "VAL_BATCH_SIZE": "524288",
    "ITERATIONS": "20000",
    "TRAIN_BATCH_TOKENS": "786432",
    "TRAIN_SEQ_LEN": "2048",
    "WARMUP_STEPS": "20",
    "WARMDOWN_ITERS": "1200",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "3.0",
    "TIE_EMBEDDINGS": "1",
    "QK_GAIN_INIT": "1.5",
    "GRAD_CLIP_NORM": "0.3",
    "EVAL_STRIDE": "64",
    "EVAL_BATCH_SEQS": "32",
    "BIGRAM_DIM": "128",
    "BIGRAM_VOCAB_SIZE": "4096",
    "SMEAR_ENABLED": "1",
    "SKIP_CONNECTIONS_ENABLED": "1",
    "ORTHO_INIT_ENABLED": "1",
    "MATRIX_LR": "0.02",
    "SCALAR_LR": "0.02",
    "TIED_EMBED_LR": "0.03",
    "MUON_MOMENTUM": "0.99",
    "MUON_MOMENTUM_WARMUP_START": "0.92",
    "MUON_MOMENTUM_WARMUP_STEPS": "1500",
    "ADAMW_WEIGHT_DECAY": "0.01",
    "MUON_WEIGHT_DECAY": "0.01",
    "SWA_ENABLED": "0",
    "SWA_START_FRAC": "0.4",
    "SWA_EVERY": "50",
    "MAGNITUDE_PRUNE_PERCENTILE": "0.03",
    "QUANT_MLP_BITS": "6",
    "QUANT_ATTN_BITS": "6",
    "QUANT_BIGRAM_BITS": "6",
    "COMPRESSOR": "zstd",
}


PRESETS: dict[str, Preset] = {
    "mlx_smoke": Preset(
        name="mlx_smoke",
        description="Official Apple Silicon smoke path from the repository README.",
        family="local",
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
        family="local",
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
    "local_frontier_proxy_mlx": Preset(
        name="local_frontier_proxy_mlx",
        description="Apple Silicon proxy for frontier-family local screening: 9L, 3x MLP, seq 2048, frontier-like optimizer settings.",
        family="local",
        target="mlx",
        launch_mode="python",
        entrypoint="train_gpt_mlx.py",
        env={
            "DATA_PATH": DEFAULT_DATA_PATH,
            "TOKENIZER_PATH": DEFAULT_TOKENIZER_PATH,
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "3",
            "TRAIN_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "16384",
            "GRAD_ACCUM_STEPS": "4",
            "MLX_MAX_MICROBATCH_TOKENS": "4096",
            "LOGIT_CHUNK_TOKENS": "8192",
            "MATRIX_LR": "0.02",
            "SCALAR_LR": "0.02",
            "TIED_EMBED_LR": "0.03",
            "MUON_MOMENTUM": "0.99",
            "MUON_MOMENTUM_WARMUP_START": "0.92",
            "MUON_MOMENTUM_WARMUP_STEPS": "1500",
            "GRAD_CLIP_NORM": "0.3",
            "WARMUP_STEPS": "20",
            "WARMDOWN_ITERS": "3000",
            "TRAIN_LOG_EVERY": "25",
            "VAL_LOSS_EVERY": "100",
            "VAL_BATCH_SIZE": "32768",
            "MAX_WALLCLOCK_SECONDS": "0",
        },
        min_train_shards=1,
        notes=(
            "This is a local proxy, not an exact frontier reproduction: it matches shape and schedule more closely than the simple MLX baseline but does not include BigramHash, SWA, or mixed quantization.",
            "Use it to screen architecture and schedule changes before spending remote CUDA time.",
        ),
    ),
    "cuda_remote_baseline": Preset(
        name="cuda_remote_baseline",
        description="Clean remote CUDA baseline matching the default PyTorch path.",
        family="baseline",
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
    "baseline": Preset(
        name="baseline",
        description="Public-frontier baseline rung: 9L int6 family with BigramHash(4096), SmearGate, and sliding eval.",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="9",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        notes=(
            "This is the clean starting point for frontier-family ablations, not the beginner baseline in train_gpt.py.",
            "It keeps the public-best architectural ingredients except the later ladder additions like 10L, int5 MLP, larger BigramHash, and SWA.",
        ),
    ),
    "frontier_partial_quant": Preset(
        name="frontier_partial_quant",
        description="Mixed-quantization rung: int5 MLP with int6 attention and bigram weights.",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="9",
            QUANT_MLP_BITS="5",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        diff_base="baseline",
        notes=(
            "Isolates mixed quantization without changing depth.",
            "Useful for measuring how much byte budget int5 MLP frees before spending it on more capacity.",
        ),
    ),
    "frontier_partial_swa_wd": Preset(
        name="frontier_partial_swa_wd",
        description="Optimization rung: WD=0.04, warmdown=3000, and SWA over the last 40% of training.",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="9",
            WARMDOWN_ITERS="3000",
            ADAMW_WEIGHT_DECAY="0.04",
            MUON_WEIGHT_DECAY="0.04",
            SWA_ENABLED="1",
            SWA_START_FRAC="0.4",
            SWA_EVERY="50",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        diff_base="baseline",
        notes=(
            "Matches the public ladder's optimizer and averaging direction.",
            "SWA is enabled only in this rung and later presets so the effect stays attributable.",
        ),
    ),
    "frontier_partial_bigramhash": Preset(
        name="frontier_partial_bigramhash",
        description="BigramHash rung: increase the hash table from 4096 to 8192 buckets.",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="9",
            BIGRAM_VOCAB_SIZE="8192",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        diff_base="baseline",
        notes=(
            "Uses the first public hash-table expansion step before the full 10240-bucket recipe.",
        ),
    ),
    "frontier_arch_10l_variant": Preset(
        name="frontier_arch_10l_variant",
        description="Architecture rung: add a 10th layer while keeping uniform int6 quantization.",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="10",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        diff_base="baseline",
        notes=(
            "Separates depth from mixed quantization so the extra layer is measured directly.",
        ),
    ),
    "frontier_combined_public_like": Preset(
        name="frontier_combined_public_like",
        description="Public-best-inspired combined recipe: 10L, int5 MLP, WD=0.04, SWA(0.4), BigramHash(10240).",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="10",
            WARMDOWN_ITERS="3000",
            BIGRAM_VOCAB_SIZE="10240",
            ADAMW_WEIGHT_DECAY="0.04",
            MUON_WEIGHT_DECAY="0.04",
            SWA_ENABLED="1",
            SWA_START_FRAC="0.4",
            SWA_EVERY="50",
            QUANT_MLP_BITS="5",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        diff_base="baseline",
        notes=(
            "This is the main public-frontier reproduction preset.",
            "It intentionally stays transparent and self-contained: no seed search, no external checkpoints, no hidden offline machinery.",
        ),
    ),
    "nearby_variant_1": Preset(
        name="nearby_variant_1",
        description="Public-like variant with BigramHash(8192) to trade a small amount of context memory for artifact headroom.",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="10",
            WARMDOWN_ITERS="3000",
            BIGRAM_VOCAB_SIZE="8192",
            ADAMW_WEIGHT_DECAY="0.04",
            MUON_WEIGHT_DECAY="0.04",
            SWA_ENABLED="1",
            SWA_START_FRAC="0.4",
            SWA_EVERY="50",
            QUANT_MLP_BITS="5",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        diff_base="frontier_combined_public_like",
        notes=(
            "Useful when the 10240-bucket table is too expensive once you start spending bytes elsewhere.",
        ),
    ),
    "nearby_variant_2": Preset(
        name="nearby_variant_2",
        description="Public-like variant with slightly later SWA entry to test whether fewer, more-converged checkpoints help quantization.",
        family="frontier",
        target="cuda",
        launch_mode="torchrun",
        entrypoint="train_gpt_frontier.py",
        env=with_overrides(
            FRONTIER_BASE_ENV,
            NUM_LAYERS="10",
            WARMDOWN_ITERS="3000",
            BIGRAM_VOCAB_SIZE="10240",
            ADAMW_WEIGHT_DECAY="0.04",
            MUON_WEIGHT_DECAY="0.04",
            SWA_ENABLED="1",
            SWA_START_FRAC="0.5",
            SWA_EVERY="50",
            QUANT_MLP_BITS="5",
        ),
        min_train_shards=1,
        nproc_per_node=1,
        diff_base="frontier_combined_public_like",
        notes=(
            "This stays close to the public-best path while probing whether the averaging window is too early.",
        ),
    ),
}


RUN_SCALES: dict[str, RunScale] = {
    "smoke": RunScale(
        name="smoke",
        description="Fast local health check with approximate validation and no final quantized roundtrip.",
        env={
            "ITERATIONS": "50",
            "MAX_WALLCLOCK_SECONDS": "0",
            "VAL_LOSS_EVERY": "0",
            "VAL_MAX_SEQS": "32",
            "FINAL_VAL_MAX_SEQS": "64",
            "FINAL_QUANT_EVAL": "0",
            "TRAIN_LOG_EVERY": "10",
            "SUMMARY_EVERY": "10",
            "CHECKPOINT_EVERY": "25",
        },
        notes=(
            "Use this to verify launch, data, logging, and checkpoint wiring.",
        ),
    ),
    "probe_short": RunScale(
        name="probe_short",
        description="Cheap screening run with small approximate validation and resumable checkpoints.",
        env={
            "ITERATIONS": "250",
            "MAX_WALLCLOCK_SECONDS": "0",
            "VAL_LOSS_EVERY": "50",
            "VAL_MAX_SEQS": "64",
            "FINAL_VAL_MAX_SEQS": "128",
            "FINAL_QUANT_EVAL": "0",
            "TRAIN_LOG_EVERY": "25",
            "SUMMARY_EVERY": "25",
            "CHECKPOINT_EVERY": "50",
        },
        notes=(
            "Good first local decision point on an M4.",
        ),
    ),
    "probe_medium": RunScale(
        name="probe_medium",
        description="More stable local probe with larger approximate validation and artifact export enabled.",
        env={
            "ITERATIONS": "1000",
            "MAX_WALLCLOCK_SECONDS": "0",
            "VAL_LOSS_EVERY": "100",
            "VAL_MAX_SEQS": "256",
            "FINAL_VAL_MAX_SEQS": "512",
            "FINAL_QUANT_EVAL": "1",
            "TRAIN_LOG_EVERY": "50",
            "SUMMARY_EVERY": "50",
            "CHECKPOINT_EVERY": "100",
        },
        notes=(
            "Use this once a branch survives probe_short and you want a cleaner local read.",
        ),
    ),
    "long_local_overnight": RunScale(
        name="long_local_overnight",
        description="Overnight local run with resumable checkpoints and moderate-cost approximate validation.",
        env={
            "ITERATIONS": "1000000",
            "MAX_WALLCLOCK_SECONDS": "43200",
            "VAL_LOSS_EVERY": "250",
            "VAL_MAX_SEQS": "256",
            "FINAL_VAL_MAX_SEQS": "1024",
            "FINAL_QUANT_EVAL": "1",
            "TRAIN_LOG_EVERY": "50",
            "SUMMARY_EVERY": "50",
            "CHECKPOINT_EVERY": "250",
        },
        notes=(
            "Tuned for unattended laptop runs where you care more about survivability than exact full-val numbers.",
        ),
    ),
    "long_local_24h": RunScale(
        name="long_local_24h",
        description="Day-scale local run with coarse periodic validation, resumable checkpoints, and final artifact export.",
        env={
            "ITERATIONS": "1000000",
            "MAX_WALLCLOCK_SECONDS": "86400",
            "VAL_LOSS_EVERY": "500",
            "VAL_MAX_SEQS": "512",
            "FINAL_VAL_MAX_SEQS": "2048",
            "FINAL_QUANT_EVAL": "1",
            "TRAIN_LOG_EVERY": "100",
            "SUMMARY_EVERY": "100",
            "CHECKPOINT_EVERY": "500",
        },
        notes=(
            "Use this only for branches that have already earned local trust.",
        ),
    ),
}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown preset {name!r}. Available presets: {', '.join(sorted(PRESETS))}") from exc


def get_scale(name: str) -> RunScale:
    try:
        return RUN_SCALES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scale {name!r}. Available scales: {', '.join(sorted(RUN_SCALES))}") from exc
