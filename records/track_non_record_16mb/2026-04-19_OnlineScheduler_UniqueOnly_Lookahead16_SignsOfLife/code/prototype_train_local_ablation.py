from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path
import glob
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from prototype.model import RoutedTinyLM, TinyRouteConfig
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from model import RoutedTinyLM, TinyRouteConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal routed tiny LM prototype.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--binary-forward", action="store_true")
    parser.add_argument("--binary-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary-lm-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary-router", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary-operators", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary-bridges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--operator-binary-mode", choices=("hard_sign", "tanh_ste"), default="hard_sign")
    parser.add_argument("--operator-binary-scope", choices=("all", "op1", "op2", "op12", "op23"), default="all")
    parser.add_argument("--operator-binary-strength", type=float, default=1.0)
    parser.add_argument("--directional-operators", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detach-operator-feedback-for-router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--toy-mode", choices=("random", "pattern", "bytes", "shards"), default="pattern")
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--router-type", choices=("single", "coarse_to_fine"), default="single")
    parser.add_argument("--router-schedule", choices=("always_on", "delayed_halfway", "delayed_late"), default="always_on")
    parser.add_argument("--coarse-router-temperature", type=float, default=1.5)
    parser.add_argument("--fine-router-temperature", type=float, default=1.5)
    parser.add_argument("--entropy-reg-weight", type=float, default=0.0)
    parser.add_argument("--op3-hidden-dim", type=int, default=256)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--data-files", nargs="*", default=None)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--shared-repeats", type=int, default=3)
    parser.add_argument("--train-pattern", type=str, default="data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
    parser.add_argument("--val-pattern", type=str, default=None)
    parser.add_argument("--max-val-tokens", type=int, default=0)
    parser.add_argument("--tokenizer-model", type=str, default=None)
    parser.add_argument("--variant-name", type=str, default="run")
    parser.add_argument("--summary-out", type=str, default=None)
    parser.add_argument("--curve-out", type=str, default=None)
    parser.add_argument("--max-wallclock-seconds", type=float, default=0.0)
    parser.add_argument(
        "--data-scheduler-policy",
        choices=(
            "none",
            "fifo_lookahead",
            "random_lookahead",
            "novelty_v1",
            "score_unique_only",
            "score_transition_only",
            "score_freq_penalty_only",
            "score_unique_plus_transition",
            "score_unique_minus_freq_penalty",
        ),
        default="none",
    )
    parser.add_argument("--scheduler-lookahead-batches", type=int, default=8)
    parser.add_argument("--scheduler-novelty-weight", type=float, default=1.0)
    parser.add_argument("--scheduler-transition-weight", type=float, default=0.25)
    parser.add_argument("--scheduler-maxfreq-penalty", type=float, default=0.5)
    parser.add_argument("--train-gpus-logical", type=int, default=1)
    parser.add_argument("--scheduler-gpus-logical", type=int, default=0)
    parser.add_argument("--aux-connector-labels", type=str, default=None)
    parser.add_argument("--aux-connector-alpha", type=float, default=0.0)
    parser.add_argument("--aux-connector-num-labels", type=int, default=8)
    parser.add_argument("--bridge-offsets", type=int, nargs="*", default=None)
    parser.add_argument("--bridge-gate-mode", choices=("soft", "binary"), default="soft")
    parser.add_argument("--bridge-hidden-proj", type=int, default=0)
    parser.add_argument("--bridge-entropy-reg-weight", type=float, default=0.0)
    parser.add_argument("--num-regions", type=int, default=0)
    parser.add_argument("--region-seed", type=int, default=0)
    parser.add_argument("--max-region-bridges", type=int, default=0)
    parser.add_argument("--region-bridge-hidden-dim", type=int, default=0)
    parser.add_argument("--region-bridge-gate-mode", choices=("soft", "binary", "fixed"), default="soft")
    parser.add_argument("--region-bridge-fixed-gate-value", type=float, default=0.5)
    parser.add_argument("--region-bridge-no-gate", action="store_true")
    parser.add_argument("--region-bridge-fixed-scale-value", type=float, default=1.0)
    parser.add_argument("--region-bridge-no-learnable-scale", action="store_true")
    parser.add_argument("--region-bridge-message-mode", choices=("plain", "directional"), default="plain")
    parser.add_argument("--region-bridge-target-subset-ratio", type=float, default=1.0)
    return parser


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pattern_batch(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, 256, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len + 1, device=device).unsqueeze(0)
    tokens = (starts + offsets) % 256
    return tokens[:, :-1], tokens[:, 1:]


def make_random_batch(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, 256, (batch_size, seq_len + 1), device=device)
    return tokens[:, :-1], tokens[:, 1:]


class ByteStreamLoader:
    def __init__(self, files: list[str], batch_size: int, seq_len: int, device: torch.device):
        payload = bytearray()
        for file_name in files:
            path = Path(file_name)
            payload.extend(path.read_bytes())
            payload.append(10)
        if len(payload) < batch_size * (seq_len + 1):
            repeats = (batch_size * (seq_len + 1)) // max(len(payload), 1) + 1
            payload = payload * repeats
        self.tokens = torch.tensor(list(payload), dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.cursor = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.batch_size * (self.seq_len + 1)
        if self.cursor + chunk >= self.tokens.numel():
            self.cursor = 0
        flat = self.tokens[self.cursor : self.cursor + chunk]
        self.cursor += self.batch_size
        batch = flat.view(self.batch_size, self.seq_len + 1)
        return batch[:, :-1], batch[:, 1:]


def load_data_shard(file: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.int64, copy=False))


class ShardTokenLoader:
    def __init__(self, pattern: str, batch_size: int, seq_len: int, device: torch.device):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No shard files found for pattern: {pattern}")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.batch_size * (self.seq_len + 1)
        chunks: list[torch.Tensor] = []
        remaining = need
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            take = min(avail, remaining)
            chunks.append(self.tokens[self.pos : self.pos + take])
            self.pos += take
            remaining -= take
        flat = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        batch = flat.view(self.batch_size, self.seq_len + 1).to(self.device)
        return batch[:, :-1], batch[:, 1:]


def score_token_chunk(
    chunk: torch.Tensor,
    *,
    novelty_weight: float,
    transition_weight: float,
    maxfreq_penalty: float,
) -> float:
    local = chunk.to(dtype=torch.int64, device="cpu")
    token_count = int(local.numel())
    if token_count <= 1:
        return 0.0
    uniques = int(torch.unique(local).numel())
    unique_ratio = uniques / token_count
    transitions = float((local[1:] != local[:-1]).to(dtype=torch.float32).mean().item())
    counts = torch.bincount(local, minlength=int(local.max().item()) + 1)
    max_freq_ratio = float(counts.max().item()) / token_count
    return novelty_weight * unique_ratio + transition_weight * transitions - maxfreq_penalty * max_freq_ratio


def scheduler_score_weights(policy: str) -> tuple[float, float, float]:
    if policy == "novelty_v1":
        return (1.0, 0.25, 0.5)
    if policy == "score_unique_only":
        return (1.0, 0.0, 0.0)
    if policy == "score_transition_only":
        return (0.0, 1.0, 0.0)
    if policy == "score_freq_penalty_only":
        return (0.0, 0.0, 1.0)
    if policy == "score_unique_plus_transition":
        return (1.0, 0.25, 0.0)
    if policy == "score_unique_minus_freq_penalty":
        return (1.0, 0.0, 0.5)
    return (1.0, 0.25, 0.5)


class ScheduledShardTokenLoader(ShardTokenLoader):
    def __init__(
        self,
        pattern: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        *,
        policy: str,
        lookahead_batches: int,
        novelty_weight: float,
        transition_weight: float,
        maxfreq_penalty: float,
        seed: int,
    ):
        super().__init__(pattern, batch_size, seq_len, device)
        self.policy = policy
        self.lookahead_batches = max(1, lookahead_batches)
        self.novelty_weight = novelty_weight
        self.transition_weight = transition_weight
        self.maxfreq_penalty = maxfreq_penalty
        self.buffer: list[tuple[float, torch.Tensor]] = []
        self.selected_scores: list[float] = []
        self.total_scored = 0
        self.total_selected = 0
        self.need = self.batch_size * (self.seq_len + 1)
        self.rng = random.Random(seed)

    def _take_flat_chunk(self) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = self.need
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            take = min(avail, remaining)
            chunks.append(self.tokens[self.pos : self.pos + take])
            self.pos += take
            remaining -= take
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def _fill_buffer(self) -> None:
        while len(self.buffer) < self.lookahead_batches:
            chunk = self._take_flat_chunk()
            score = score_token_chunk(
                chunk,
                novelty_weight=self.novelty_weight,
                transition_weight=self.transition_weight,
                maxfreq_penalty=self.maxfreq_penalty,
            )
            self.buffer.append((score, chunk))
            self.total_scored += 1

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._fill_buffer()
        if self.policy == "fifo_lookahead":
            pick_idx = 0
        elif self.policy == "random_lookahead":
            pick_idx = self.rng.randrange(len(self.buffer))
        else:
            pick_idx = max(range(len(self.buffer)), key=lambda idx: self.buffer[idx][0])
        score, flat = self.buffer.pop(pick_idx)
        self.selected_scores.append(score)
        self.total_selected += 1
        batch = flat.view(self.batch_size, self.seq_len + 1).to(self.device)
        return batch[:, :-1], batch[:, 1:]

    def stats(self) -> dict[str, float]:
        avg_score = float(sum(self.selected_scores) / len(self.selected_scores)) if self.selected_scores else float("nan")
        return {
            "docs_scored": float(self.total_scored),
            "avg_selected_score": avg_score,
            "queue_fill_ratio": float(len(self.buffer)) / float(self.lookahead_batches),
        }


class AuxLabelLoader:
    def __init__(self, path: str, batch_size: int, device: torch.device, num_labels: int = 8):
        labels = np.load(path)
        if labels.ndim != 1:
            raise ValueError(f"Expected 1D aux label bitmask array, got shape {labels.shape}")
        self.labels = torch.from_numpy(labels.astype(np.uint8, copy=False))
        self.batch_size = batch_size
        self.device = device
        self.num_labels = num_labels
        self.cursor = 0
        bit_values = np.arange(256, dtype=np.uint16)[:, None]
        bit_masks = (1 << np.arange(num_labels, dtype=np.uint16))[None, :]
        lookup = ((bit_values & bit_masks) > 0).astype(np.float32)
        self.lookup = torch.tensor(lookup, dtype=torch.float32, device=device)

    def next_batch(self) -> torch.Tensor:
        if self.cursor + self.batch_size > self.labels.numel():
            self.cursor = 0
        batch_bits = self.labels[self.cursor : self.cursor + self.batch_size].to(device=self.device, dtype=torch.long)
        self.cursor += self.batch_size
        return self.lookup[batch_bits]


def build_sentencepiece_luts(tokenizer_model: str, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - dependency is optional outside eval runs
        raise RuntimeError("sentencepiece is required when --tokenizer-model is provided") from exc

    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for validation pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def eval_validation(
    args: argparse.Namespace,
    model: RoutedTinyLM,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> tuple[float, float]:
    seq_len = args.seq_len
    batch_size = max(1, args.batch_size)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    val_loss_sum = 0.0
    val_token_count = 0
    val_byte_count = 0.0
    model.eval()
    with torch.inference_mode():
        for seq_start in range(0, total_seqs, batch_size):
            seq_end = min(seq_start + batch_size, total_seqs)
            raw_start = seq_start * seq_len
            raw_end = seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            _, stats = model(
                x,
                y,
                binary_forward=args.binary_forward,
                entropy_reg_weight=0.0,
                bridge_entropy_reg_weight=0.0,
            )
            batch_loss = float(stats["lm_loss"].cpu())
            batch_token_count = int(y.numel())
            val_loss_sum += batch_loss * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += float(token_bytes.to(dtype=torch.float32).sum().cpu())
    model.train()
    val_loss = val_loss_sum / max(val_token_count, 1)
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / max(val_byte_count, 1.0)
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def format_usage(tensor: torch.Tensor) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in tensor.tolist()) + "]"


def active_router_type(args: argparse.Namespace, step: int) -> str:
    if args.router_type != "coarse_to_fine":
        return args.router_type
    if args.router_schedule == "always_on":
        return "coarse_to_fine"
    if args.router_schedule == "delayed_halfway":
        switch_step = max(1, math.ceil(args.steps * 0.5))
        return "single" if step <= switch_step else "coarse_to_fine"
    if args.router_schedule == "delayed_late":
        switch_step = max(1, math.ceil(args.steps * 0.75))
        return "single" if step <= switch_step else "coarse_to_fine"
    raise ValueError(f"Unknown router schedule: {args.router_schedule}")


def run_training(args: argparse.Namespace, *, label: str = "run") -> dict[str, object]:
    device = choose_device(args.device)
    set_seed(args.seed)
    start_time = time.time()

    config = TinyRouteConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        shared_repeats=args.shared_repeats,
        router_temperature=args.router_temperature,
        router_type=args.router_type,
        coarse_router_temperature=args.coarse_router_temperature,
        fine_router_temperature=args.fine_router_temperature,
        op3_hidden_dim=args.op3_hidden_dim,
        binary_embeddings=args.binary_embeddings,
        binary_lm_head=args.binary_lm_head,
        binary_router=args.binary_router,
        binary_operators=args.binary_operators,
        binary_attention=args.binary_attention,
        binary_bridges=args.binary_bridges,
        operator_binary_mode=args.operator_binary_mode,
        operator_binary_scope=args.operator_binary_scope,
        operator_binary_strength=args.operator_binary_strength,
        directional_operators=args.directional_operators,
        detach_operator_feedback_for_router=args.detach_operator_feedback_for_router,
        attention_binary_mode="hard_sign",
        router_binary_mode="hard_sign",
        bridge_binary_mode="hard_sign",
        bridge_offsets=tuple(args.bridge_offsets) if args.bridge_offsets else (),
        bridge_gate_mode=args.bridge_gate_mode,
        bridge_hidden_proj=args.bridge_hidden_proj,
        num_regions=args.num_regions,
        region_seed=args.region_seed,
        max_region_bridges=args.max_region_bridges,
        region_bridge_hidden_dim=args.region_bridge_hidden_dim,
        region_bridge_gate_mode=args.region_bridge_gate_mode,
        region_bridge_fixed_gate_value=args.region_bridge_fixed_gate_value,
        region_bridge_use_gate=not args.region_bridge_no_gate,
        region_bridge_learnable_scale=not args.region_bridge_no_learnable_scale,
        region_bridge_fixed_scale_value=args.region_bridge_fixed_scale_value,
        region_bridge_message_mode=args.region_bridge_message_mode,
        region_bridge_target_subset_ratio=args.region_bridge_target_subset_ratio,
    )
    model = RoutedTinyLM(config).to(device)
    aux_loader = None
    aux_head = None
    if args.aux_connector_labels:
        if args.toy_mode != "shards":
            raise ValueError("--aux-connector-labels currently requires --toy-mode shards")
        aux_loader = AuxLabelLoader(
            args.aux_connector_labels,
            batch_size=args.batch_size,
            device=device,
            num_labels=args.aux_connector_num_labels,
        )
        aux_head = nn.Linear(args.d_model, args.aux_connector_num_labels).to(device)
    parameters = list(model.parameters())
    if aux_head is not None:
        parameters.extend(aux_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)

    print(f"== {label} ==")
    print("config:", asdict(config))
    print("device:", device)
    print("binary_forward:", args.binary_forward)
    print("router_type:", args.router_type)
    print("router_schedule:", args.router_schedule)
    print("entropy_reg_weight:", args.entropy_reg_weight)
    print("bridge_entropy_reg_weight:", args.bridge_entropy_reg_weight)
    print("aux_connector_alpha:", args.aux_connector_alpha)
    print("data_scheduler_policy:", args.data_scheduler_policy)
    print("max_wallclock_seconds:", args.max_wallclock_seconds)
    print("train_gpus_logical:", args.train_gpus_logical)
    print("scheduler_gpus_logical:", args.scheduler_gpus_logical)
    print("parameter_count:", model.parameter_count())

    batch_fn = make_pattern_batch if args.toy_mode == "pattern" else make_random_batch
    byte_loader = None
    shard_loader = None
    if args.toy_mode == "bytes":
        default_files = [
            "README.md",
            "train_gpt.py",
            "../Parameter Golf.md",
            "../目标ICML.md",
        ]
        files = args.data_files if args.data_files else default_files
        byte_loader = ByteStreamLoader(files, args.batch_size, args.seq_len, device)
        print("byte_stream_files:", files)
    elif args.data_files is not None and args.toy_mode != "bytes":
        print("data_files_ignored_for_mode:", args.toy_mode)
    if args.toy_mode == "shards":
        if args.data_scheduler_policy == "none":
            shard_loader = ShardTokenLoader(args.train_pattern, args.batch_size, args.seq_len, device)
        else:
            novelty_weight, transition_weight, maxfreq_penalty = scheduler_score_weights(args.data_scheduler_policy)
            shard_loader = ScheduledShardTokenLoader(
                args.train_pattern,
                args.batch_size,
                args.seq_len,
                device,
                policy=args.data_scheduler_policy,
                lookahead_batches=args.scheduler_lookahead_batches,
                novelty_weight=args.scheduler_novelty_weight if args.data_scheduler_policy == "novelty_v1" else novelty_weight,
                transition_weight=args.scheduler_transition_weight if args.data_scheduler_policy == "novelty_v1" else transition_weight,
                maxfreq_penalty=args.scheduler_maxfreq_penalty if args.data_scheduler_policy == "novelty_v1" else maxfreq_penalty,
                seed=args.seed,
            )
        print("train_pattern:", args.train_pattern)
    final_metrics: dict[str, object] = {}
    curve_rows: list[str] = ["step,loss,lm_loss,aux_loss,router_entropy,coarse_entropy,fine_entropy,collapse_ratio,bridge_gate_collapse,bridge_gate_entropy,bridge_gate_values,region_sizes"]

    for step in range(1, args.steps + 1):
        if args.max_wallclock_seconds > 0 and (time.time() - start_time) >= args.max_wallclock_seconds:
            print(f"stopping_early_wallclock step={step - 1} elapsed={time.time() - start_time:.2f}s")
            break
        if byte_loader is not None:
            inputs, targets = byte_loader.next_batch()
        elif shard_loader is not None:
            inputs, targets = shard_loader.next_batch()
        else:
            inputs, targets = batch_fn(args.batch_size, args.seq_len, device)
        aux_labels = aux_loader.next_batch() if aux_loader is not None else None
        optimizer.zero_grad(set_to_none=True)
        step_router_type = active_router_type(args, step)
        loss, stats = model(
            inputs,
            targets,
            binary_forward=args.binary_forward,
            entropy_reg_weight=args.entropy_reg_weight,
            bridge_entropy_reg_weight=args.bridge_entropy_reg_weight,
            router_type_override=step_router_type,
        )
        if aux_head is not None and aux_labels is not None:
            aux_logits = aux_head(stats["pooled_hidden_live"])
            aux_loss = F.binary_cross_entropy_with_logits(aux_logits, aux_labels, reduction="mean")
            loss = loss + args.aux_connector_alpha * aux_loss
        else:
            aux_loss = torch.tensor(0.0, device=device)
        loss.backward()
        optimizer.step()

        usage = stats["router_usage"].cpu()
        entropy = float(stats["router_entropy"].cpu())
        lm_loss = float(stats["lm_loss"].cpu())
        aux_loss_value = float(aux_loss.detach().cpu())
        coarse_entropy = float(stats["coarse_router_entropy"].cpu()) if "coarse_router_entropy" in stats else float("nan")
        fine_entropy = float(stats["fine_router_entropy"].cpu()) if "fine_router_entropy" in stats else float("nan")
        per_repeat = stats["router_usage_per_repeat"].cpu()
        max_entropy = math.log(usage.numel())
        collapse_ratio = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)
        bridge_gate_collapse = float(stats["bridge_gate_collapse"].cpu()) if "bridge_gate_collapse" in stats else float("nan")
        bridge_gate_entropy = float(stats["bridge_gate_entropy"].cpu()) if "bridge_gate_entropy" in stats else float("nan")
        bridge_gate_values = stats["bridge_gate_values"].cpu().tolist() if "bridge_gate_values" in stats else []
        region_sizes = stats["region_sizes"].cpu().tolist() if "region_sizes" in stats else []

        print(
            f"step={step:03d} "
            f"loss={float(loss.detach().cpu()):.4f} "
            f"lm_loss={lm_loss:.4f} "
            f"aux_loss={aux_loss_value:.4f} "
            f"router_entropy={entropy:.4f} "
            f"coarse_entropy={coarse_entropy:.4f} "
            f"fine_entropy={fine_entropy:.4f} "
            f"collapse_ratio={collapse_ratio:.4f} "
            f"usage={format_usage(usage)}"
        )
        repeat_str = ", ".join(f"r{i}:{format_usage(row)}" for i, row in enumerate(per_repeat))
        print(f"  per_repeat_usage={repeat_str}")
        if bridge_gate_values:
            print(
                f"  bridge_gate_collapse={bridge_gate_collapse:.4f} "
                f"bridge_gate_entropy={bridge_gate_entropy:.4f} "
                f"bridge_gates={format_usage(torch.tensor(bridge_gate_values))}"
            )
        curve_rows.append(
            f"{step},{float(loss.detach().cpu()):.6f},{lm_loss:.6f},{aux_loss_value:.6f},{entropy:.6f},{coarse_entropy:.6f},{fine_entropy:.6f},{collapse_ratio:.6f},"
            f"{bridge_gate_collapse:.6f},{bridge_gate_entropy:.6f},\"{bridge_gate_values}\",\"{region_sizes}\""
        )
        final_metrics = {
            "loss": float(loss.detach().cpu()),
            "lm_loss": lm_loss,
            "aux_loss": aux_loss_value,
            "router_entropy": entropy,
            "coarse_entropy": coarse_entropy,
            "fine_entropy": fine_entropy,
            "collapse_ratio": collapse_ratio,
            "usage": usage.tolist(),
            "bridge_gate_collapse": bridge_gate_collapse,
            "bridge_gate_entropy": bridge_gate_entropy,
            "bridge_gate_values": bridge_gate_values,
            "region_sizes": region_sizes,
        }
    if "lm_loss" not in final_metrics:
        raise RuntimeError("Training stopped before completing any optimization step")
    if args.val_pattern:
        if not args.tokenizer_model:
            raise ValueError("--tokenizer-model is required when --val-pattern is provided")
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            args.tokenizer_model,
            args.vocab_size,
            device,
        )
        val_tokens = load_validation_tokens(args.val_pattern, args.seq_len)
        if args.max_val_tokens > 0:
            usable = min(((args.max_val_tokens - 1) // args.seq_len) * args.seq_len, val_tokens.numel() - 1)
            if usable <= 0:
                raise ValueError(f"--max-val-tokens={args.max_val_tokens} is too small for seq_len={args.seq_len}")
            val_tokens = val_tokens[: usable + 1]
        valid_lm_loss, valid_bpb = eval_validation(
            args,
            model,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        print(f"final_valid_lm_loss={valid_lm_loss:.6f} final_valid_bpb={valid_bpb:.6f}")
        final_metrics["valid_lm_loss"] = valid_lm_loss
        final_metrics["valid_bpb"] = valid_bpb
    wallclock = time.time() - start_time
    final_metrics["wallclock_seconds"] = wallclock
    final_metrics["effective_tokens_per_second"] = (step - 1) * args.batch_size * args.seq_len / max(wallclock, 1e-9)
    final_metrics["scheduler_policy"] = args.data_scheduler_policy
    final_metrics["train_gpus_logical"] = args.train_gpus_logical
    final_metrics["scheduler_gpus_logical"] = args.scheduler_gpus_logical
    if isinstance(shard_loader, ScheduledShardTokenLoader):
        final_metrics.update(shard_loader.stats())
        final_metrics["docs_scored_per_second"] = final_metrics["docs_scored"] / max(wallclock, 1e-9)
    else:
        final_metrics["docs_scored"] = float("nan")
        final_metrics["docs_scored_per_second"] = float("nan")
        final_metrics["avg_selected_score"] = float("nan")
        final_metrics["queue_fill_ratio"] = float("nan")
    if args.curve_out:
        curve_path = Path(args.curve_out)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        curve_path.write_text("\n".join(curve_rows) + "\n", encoding="utf-8")
        print("curve_out:", curve_path)
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "variant_name": args.variant_name,
            "train_pattern": args.train_pattern if args.toy_mode == "shards" else "",
            "val_pattern": args.val_pattern or "",
            "final_train_loss": final_metrics.get("loss", float("nan")),
            "final_train_lm_loss": final_metrics.get("lm_loss", float("nan")),
            "final_train_aux_loss": final_metrics.get("aux_loss", float("nan")),
            "final_valid_lm_loss": final_metrics.get("valid_lm_loss", float("nan")),
            "final_valid_bpb": final_metrics.get("valid_bpb", float("nan")),
            "final_entropy": final_metrics.get("router_entropy", float("nan")),
            "final_coarse_entropy": final_metrics.get("coarse_entropy", float("nan")),
            "final_fine_entropy": final_metrics.get("fine_entropy", float("nan")),
            "final_collapse": final_metrics.get("collapse_ratio", float("nan")),
            "params": model.parameter_count() + (sum(param.numel() for param in aux_head.parameters()) if aux_head is not None else 0),
            "router_type": args.router_type,
            "router_schedule": args.router_schedule,
            "scheduler_policy": final_metrics.get("scheduler_policy", ""),
            "train_gpus": final_metrics.get("train_gpus_logical", 0),
            "scheduler_gpus": final_metrics.get("scheduler_gpus_logical", 0),
            "aux_connector_alpha": args.aux_connector_alpha,
            "wallclock_seconds": wallclock,
            "train_tokens_consumed": (step - 1) * args.batch_size * args.seq_len,
            "effective_tokens_per_second": final_metrics.get("effective_tokens_per_second", float("nan")),
            "docs_scored_per_second": final_metrics.get("docs_scored_per_second", float("nan")),
            "queue_fill_ratio": final_metrics.get("queue_fill_ratio", float("nan")),
            "avg_selected_score": final_metrics.get("avg_selected_score", float("nan")),
        }
        header = list(row.keys())
        lines = [",".join(header)]
        values = []
        for key in header:
            value = row[key]
            values.append(str(value))
        lines.append(",".join(values))
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("summary_out:", summary_path)
    return final_metrics


def run_ablation(args: argparse.Namespace) -> None:
    ablations = [
        ("baseline", dict(binary_forward=False, router_temperature=1.0, entropy_reg_weight=0.0, op3_hidden_dim=256)),
        ("binary", dict(binary_forward=True, router_temperature=1.0, entropy_reg_weight=0.0, op3_hidden_dim=256)),
        ("binary_temp", dict(binary_forward=True, router_temperature=1.5, entropy_reg_weight=0.0, op3_hidden_dim=256)),
        ("binary_entropy", dict(binary_forward=True, router_temperature=1.0, entropy_reg_weight=0.02, op3_hidden_dim=256)),
        ("binary_light_op3", dict(binary_forward=True, router_temperature=1.0, entropy_reg_weight=0.0, op3_hidden_dim=96)),
    ]
    summaries: list[tuple[str, dict[str, object]]] = []
    for idx, (name, overrides) in enumerate(ablations):
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = args.seed + idx
        for key, value in overrides.items():
            setattr(run_args, key.replace("-", "_"), value)
        result = run_training(run_args, label=name)
        summaries.append((name, result))
    print("== ablation_summary ==")
    for name, result in summaries:
        print(
            f"{name:16s} "
            f"loss={result['loss']:.4f} "
            f"lm_loss={result['lm_loss']:.4f} "
            f"entropy={result['router_entropy']:.4f} "
            f"collapse={result['collapse_ratio']:.4f} "
            f"usage={format_usage(torch.tensor(result['usage']))}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.run_ablation:
        run_ablation(args)
        return
    run_training(args)


if __name__ == "__main__":
    main()
