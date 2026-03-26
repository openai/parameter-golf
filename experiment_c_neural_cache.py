from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "datasets" / "fineweb10B_sp1024"
TOKENIZER_PATH = ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "baseline_checkpoint.pt"
RESULTS_JSON_PATH = RESULTS_DIR / "experiment_c.json"

VOCAB_SIZE = 1024
SEQ_LEN = 512
STRIDE = 64
CACHE_MAX_TOKENS = 2048
MIN_VAL_DOCS = 50

MODEL_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
MLP_MULT = 4
ROPE_BASE = 10000.0

TRAIN_STEPS = 500
TRAIN_BATCH_SEQS = 2
LEARNING_RATE = 3e-4
SEED = 1337

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    model_dim: int = MODEL_DIM
    num_layers: int = NUM_LAYERS
    num_heads: int = NUM_HEADS
    mlp_mult: int = MLP_MULT
    rope_base: float = ROPE_BASE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size}, got {file.stat().st_size}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def resolve_shards(pattern: str) -> list[Path]:
    files = sorted(DATASET_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard files found for pattern {DATASET_DIR / pattern}")
    return files


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
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


class TokenStream:
    def __init__(self, files: list[Path]):
        if not files:
            raise ValueError("TokenStream needs at least one shard")
        self.files = files
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def next_train_batch(stream: TokenStream, batch_seqs: int, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
    chunk = stream.take(batch_seqs * (seq_len + 1)).to(dtype=torch.int64)
    chunk = chunk.view(batch_seqs, seq_len + 1)
    x = chunk[:, :-1].to(device=device, non_blocking=True)
    y = chunk[:, 1:].to(device=device, non_blocking=True)
    return x, y


def extract_first_validation_docs(val_files: list[Path], bos_id: int, num_docs: int) -> list[Tensor]:
    docs: list[Tensor] = []
    current: list[int] = []
    for file in val_files:
        shard = load_data_shard(file).numpy()
        for tok in shard:
            token = int(tok)
            if token == bos_id:
                if current:
                    docs.append(torch.tensor(current, dtype=torch.int64))
                    if len(docs) >= num_docs:
                        return docs
                current = [token]
            else:
                if not current:
                    current = [token]
                else:
                    current.append(token)
    if current and len(docs) < num_docs:
        docs.append(torch.tensor(current, dtype=torch.int64))
    if len(docs) < num_docs:
        raise ValueError(f"Only found {len(docs)} validation docs, need at least {num_docs}")
    return docs[:num_docs]


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_base: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, rope_base)

    @staticmethod
    def build_causal_mask(seq_len: int, cache_len: int, device: torch.device) -> Tensor:
        right = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
        if cache_len == 0:
            return right
        left = torch.ones((seq_len, cache_len), device=device, dtype=torch.bool)
        return torch.cat((left, right), dim=1)

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        cache_max_tokens: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary.cos_sin(seqlen, x.device, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if kv_cache is None:
            cache_len = 0
            full_k = k
            full_v = v
        else:
            prev_k, prev_v = kv_cache
            cache_len = prev_k.size(2)
            full_k = torch.cat((prev_k.to(dtype=k.dtype), k), dim=2)
            full_v = torch.cat((prev_v.to(dtype=v.dtype), v), dim=2)

        mask = self.build_causal_mask(seqlen, cache_len, x.device)
        attn_scores = torch.matmul(q, full_k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(~mask[None, None, :, :], torch.finfo(attn_scores.dtype).min)
        attn = torch.softmax(attn_scores, dim=-1)
        y = torch.matmul(attn, full_v)
        out = self.o_proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

        new_cache: tuple[Tensor, Tensor] | None = None
        if cache_max_tokens is not None:
            keep = min(cache_max_tokens, full_k.size(2))
            new_cache = (
                full_k[:, :, -keep:, :].detach(),
                full_v[:, :, -keep:, :].detach(),
            )
        return out, new_cache


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        hidden = dim * mlp_mult
        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, rope_base)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
        cache_max_tokens: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache, cache_max_tokens=cache_max_tokens)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks = nn.ModuleList(
            [Block(cfg.model_dim, cfg.num_heads, cfg.mlp_mult, cfg.rope_base) for _ in range(cfg.num_layers)]
        )
        self.final_norm = RMSNorm()

    def forward_logits(
        self,
        input_ids: Tensor,
        kv_cache: list[tuple[Tensor, Tensor] | None] | None = None,
        cache_max_tokens: int | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None] | None]:
        if kv_cache is not None and len(kv_cache) != len(self.blocks):
            raise ValueError(f"kv_cache length {len(kv_cache)} does not match num_layers {len(self.blocks)}")
        x = self.tok_emb(input_ids)
        new_cache: list[tuple[Tensor, Tensor] | None] = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x, layer_new_cache = block(x, kv_cache=layer_cache, cache_max_tokens=cache_max_tokens)
            if cache_max_tokens is not None:
                new_cache.append(layer_new_cache)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        if cache_max_tokens is not None:
            return logits, new_cache
        return logits, None

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits, _ = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")


def train_baseline(model: TinyGPT, train_stream: TokenStream, device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.0)
    for step in range(1, TRAIN_STEPS + 1):
        x, y = next_train_batch(train_stream, TRAIN_BATCH_SEQS, SEQ_LEN, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == TRAIN_STEPS:
            print(f"train step {step}/{TRAIN_STEPS} loss={loss.item():.6f}", flush=True)


def compute_bpb(loss_sum: float, token_count: float, byte_count: float) -> float:
    if token_count <= 0 or byte_count <= 0:
        raise ValueError(f"Invalid counters: token_count={token_count}, byte_count={byte_count}")
    val_loss = loss_sum / token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count / byte_count
    return bits_per_token * tokens_per_byte


def evaluate_docs(
    model: TinyGPT,
    docs: list[Tensor],
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    *,
    use_cache: bool,
) -> tuple[float, int, int]:
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    total_bytes = 0.0
    total_windows = 0

    with torch.inference_mode():
        for doc in docs:
            if doc.numel() <= SEQ_LEN:
                continue
            x_doc = doc[:-1]
            y_doc = doc[1:]
            max_start = x_doc.numel() - SEQ_LEN
            if max_start < 0:
                continue
            layer_cache: list[tuple[Tensor, Tensor] | None] | None
            layer_cache = [None for _ in range(NUM_LAYERS)] if use_cache else None
            for ws in range(0, max_start + 1, STRIDE):
                x_win = x_doc[ws : ws + SEQ_LEN].to(device=device, dtype=torch.int64).unsqueeze(0)
                y_win = y_doc[ws : ws + SEQ_LEN].to(device=device, dtype=torch.int64).unsqueeze(0)
                if use_cache:
                    logits, layer_cache = model.forward_logits(
                        x_win,
                        kv_cache=layer_cache,
                        cache_max_tokens=CACHE_MAX_TOKENS,
                    )
                else:
                    logits, _ = model.forward_logits(x_win, kv_cache=None, cache_max_tokens=None)

                score_start = SEQ_LEN - STRIDE
                scored_logits = logits[:, score_start:, :].reshape(-1, logits.size(-1)).float()
                scored_targets = y_win[:, score_start:].reshape(-1)
                nll_sum = F.cross_entropy(scored_logits, scored_targets, reduction="sum")
                total_loss_sum += float(nll_sum.item())
                total_tokens += int(scored_targets.numel())

                prev_ids = x_win[:, score_start:].reshape(-1)
                tgt_ids = scored_targets
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                total_bytes += float(token_bytes.to(torch.float64).sum().item())
                total_windows += 1

    val_bpb = compute_bpb(total_loss_sum, float(total_tokens), total_bytes)
    return val_bpb, total_windows, total_tokens


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {TOKENIZER_PATH}")

    train_files = resolve_shards("fineweb_train_*.bin")
    val_files = resolve_shards("fineweb_val_*.bin")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    if int(sp.vocab_size()) != VOCAB_SIZE:
        raise ValueError(f"VOCAB_SIZE={VOCAB_SIZE} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("Tokenizer has no BOS id, cannot recover validation document boundaries")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE, device)

    print("Collecting validation documents...", flush=True)
    val_docs = extract_first_validation_docs(val_files, bos_id=bos_id, num_docs=MIN_VAL_DOCS)
    print(f"validation_docs_loaded={len(val_docs)}", flush=True)

    cfg = ModelConfig()
    model = TinyGPT(cfg).to(device)
    print("Training tiny baseline model (500 steps)...", flush=True)
    train_stream = TokenStream(train_files)
    train_baseline(model, train_stream, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "train_steps": TRAIN_STEPS,
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
            "cache_max_tokens": CACHE_MAX_TOKENS,
            "vocab_size": VOCAB_SIZE,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}", flush=True)

    print("Loading checkpoint into freshly instantiated model...", flush=True)
    fresh_model = TinyGPT(cfg).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    fresh_model.load_state_dict(checkpoint["model_state_dict"])
    fresh_model.eval()

    print("Running Eval A (stride=64, no cache)...", flush=True)
    eval_a_bpb, eval_a_windows, eval_a_tokens = evaluate_docs(
        fresh_model,
        val_docs,
        device,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        use_cache=False,
    )

    print("Running Eval B (stride=64, neural cache, max_cache_tokens=2048)...", flush=True)
    eval_b_bpb, eval_b_windows, eval_b_tokens = evaluate_docs(
        fresh_model,
        val_docs,
        device,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        use_cache=True,
    )

    delta = eval_b_bpb - eval_a_bpb
    passed = eval_b_bpb <= eval_a_bpb
    reasons: list[str] = []
    if not passed:
        reasons = [
            "Cache may amplify overlap duplication because windows are heavily overlapping (stride=64 over seq_len=512).",
            "The tiny model is only trained for 500 steps and may not learn stable long-context usage.",
            "Window-local rotary phases can misalign cached vs current-window token geometry.",
        ]

    print(f"Eval A val_bpb (no cache): {eval_a_bpb:.6f}", flush=True)
    print(f"Eval B val_bpb (with cache): {eval_b_bpb:.6f}", flush=True)
    print(f"Delta (negative = improvement): {delta:+.6f}", flush=True)
    if passed:
        print("PASS: Eval B bpb <= Eval A bpb (cache helps or is neutral)", flush=True)
    else:
        print("FAIL: cache makes things worse", flush=True)
        print("Possible reasons:", flush=True)
        for reason in reasons:
            print(f"- {reason}", flush=True)

    result_payload = {
        "config": asdict(cfg),
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "cache_max_tokens": CACHE_MAX_TOKENS,
        "train_steps": TRAIN_STEPS,
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "docs_evaluated": len(val_docs),
        "eval_a": {
            "name": "standard_sliding_window_no_cache",
            "val_bpb": eval_a_bpb,
            "windows": eval_a_windows,
            "scored_tokens": eval_a_tokens,
        },
        "eval_b": {
            "name": "neural_cache_sliding_window",
            "val_bpb": eval_b_bpb,
            "windows": eval_b_windows,
            "scored_tokens": eval_b_tokens,
        },
        "delta_bpb": delta,
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
        "possible_reasons": reasons,
    }
    RESULTS_JSON_PATH.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {RESULTS_JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()
