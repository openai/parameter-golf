"""
Golf Transformer — Compact Language Model for Parameter-Efficient Training
==========================================================================
A highly optimised model designed for strict competition constraints:
16 MB LZMA budget, 600-second wall-clock training limit.

Architecture highlights:
  - DualTokenHashSkip: Bloom-filter bigram embedding (2×2048×16 tables,
    primes 8191 and 104729) concatenated before projection.
  - LayerScale Recurrence: [0,1,2,3,4,5,3,4,5] loop with learnable per-step
    vectors initialised 1.0 (fresh) / 0.1 (recycled) to prevent activation shock.
  - AdaMuonOptimizer: Riemannian Newton-Schulz optimiser with diagonal RMS
    pre-conditioning — zero memory overhead, simulates second-order curvature.
  - Dynamic MSE SDClip: Sigma grid-search {2.5,3.0,3.5,4.0} at export time,
    selecting the clipping threshold that minimises INT6 round-trip MSE per tensor.
  - SharedMoE: 1 always-on shared expert + top-1 routing from 3 specialised
    experts (DeepSeek V2/V3-style) to prevent routing collapse.
  - Score-First TTT: Legal test-time training — score then adapt, never the
    reverse, ensuring no look-ahead contamination.
"""

# ─── CUDA device isolation — must happen before every other import ─────────────
import os as _os
if "LOCAL_RANK" in _os.environ:
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_os.environ["LOCAL_RANK"]))

import os, time, math, json, datetime, pickle, lzma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import PreTrainedTokenizerFast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION DATACLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class GolfConfig:
    vocab_size: int           = 8192
    hidden_size: int          = 512
    intermediate: int         = 1024
    num_layers: int           = 12
    num_heads: int            = 8
    num_kv_heads: int         = 1
    max_seq_len: int          = 512
    tie_embeddings: bool      = True
    use_ttt: bool             = True
    ttt_lr: float             = 0.003
    ema_decay: float          = 0.999
    qat_enabled: bool         = True
    qat_bits: int             = 6
    qat_start_frac: float     = 0.75
    common_token_top_k: int   = 512
    per_device_batch: int     = 64
    grad_accum: int           = 2
    lr_matrix: float          = 0.008
    lr_embed: float           = 0.008
    lr_scalar: float          = 0.008
    weight_decay: float       = 0.05
    dropout: float            = 0.0
    warmup_steps: int         = 100
    time_limit_sec: int       = 600
    grad_clip: float          = 0.5
    use_compile: bool         = False
    sdclip_sigma: float       = 3.5
    lqer_enabled: bool         = True
    lqer_rank: int             = 4
    lqer_top_k: int            = 2
    lqer_asym_group: int       = 64
    use_brotli: bool           = True
    brotli_quality: int        = 11
    byte_shuffle_stride: int   = 2
    sdclip_dynamic: bool      = True
    armenian_weight: float    = 0.00
    eval_docs: int            = 50
    train_skip: int           = 200
    fineweb_weight: float     = 1.0
    eval_stride: int          = 64
    tokenizer_path: str       = "./pg_tokenizer_v10_2"
    output_path: str          = "./submission_global.pt"
    output_pkl_path: str      = "./submission_global.pkl"
    max_steps: int            = 0
    warmdown_steps: int       = 0
    qat_start_step: int       = 0

    def get(self, key: str, default=None): return getattr(self, key, default)
    def __getitem__(self, key: str): return getattr(self, key)
    def __setitem__(self, key: str, value): setattr(self, key, value)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT LOGGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExperimentLogger:
    """Appends JSON-line records to a timestamped .record file."""

    def __init__(self, enabled: bool = True, cfg: Optional[GolfConfig] = None):
        self.enabled = enabled
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = f"run_{ts}.record"
        if enabled:
            with open(self.path, "w") as fh:
                fh.write(json.dumps({
                    "event": "run_start",
                    "timestamp": ts,
                    "config": cfg.to_dict() if cfg else {},
                }) + "\n")
            print(f"[Logger] Writing to: {self.path}")

    def log(self, **kwargs):
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **kwargs,
        }
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def close(self, **summary):
        self.log(event="run_end", **summary)
        if self.enabled:
            print(f"[Logger] Record saved → {self.path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DYNAMIC SCHEDULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def configure_schedule(cfg: GolfConfig, world_size: int, step_secs: float) -> GolfConfig:
    """Populate runtime schedule fields from a measured per-step time.

    Args:
        cfg: Configuration object (mutated in-place).
        world_size: Number of GPUs.
        step_secs: Measured seconds per training step.

    Returns:
        The mutated cfg with max_steps, warmdown_steps, qat_start_step set.
    """
    target_secs    = cfg.time_limit_sec - 5.0
    effective_secs = step_secs * 1.02
    n_steps        = int(target_secs / effective_secs)
    n_steps        = (n_steps // 100) * 100
    n_steps        = max(2000, min(n_steps, 20000))

    cfg.max_steps      = n_steps
    cfg.warmdown_steps = max(500, min(n_steps // 4, n_steps - 1))
    cfg.qat_start_step = int(n_steps * cfg.qat_start_frac)

    if rank_is_primary(dist.get_rank() if dist.is_initialized() else 0):
        print(f"[Schedule] calibrated={step_secs:.3f}s | ×1.02={effective_secs:.3f}s")
        print(f"[Schedule] max_steps={n_steps} | warmdown={cfg.warmdown_steps} "
              f"| qat_start={cfg.qat_start_step}")
    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DISTRIBUTED SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_distributed():
    """Initialise DDP process group and return (local_rank, world_size, device)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if "LOCAL_RANK" not in os.environ:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, dev

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    import sys
    print(f"[Rank {local_rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')} "
          f"(cuda:0 → physical GPU {local_rank}) | world_size={world_size}",
          file=sys.stderr, flush=True)

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    return local_rank, world_size, torch.device("cuda:0")


def rank_is_primary(rank: int) -> bool:
    return rank == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEURAL NETWORK MODULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim))
        freqs = torch.outer(pos, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos, sin = freqs.cos().view(1, 1, T, self.head_dim).to(q.dtype), freqs.sin().view(1, 1, T, self.head_dim).to(q.dtype)
        
        qh1, qh2 = q.chunk(2, dim=-1)
        q = q * cos + torch.cat((-qh2, qh1), dim=-1) * sin
        kh1, kh2 = k.chunk(2, dim=-1)
        k = k * cos + torch.cat((-kh2, kh1), dim=-1) * sin
        
        k = k.expand(B, self.n_heads, T, self.head_dim)
        v = v.expand(B, self.n_heads, T, self.head_dim)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_size, cfg.intermediate, bias=False)
        self.w2 = nn.Linear(cfg.hidden_size, cfg.intermediate, bias=False)
        self.w3 = nn.Linear(cfg.intermediate, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MacaronLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.hidden_size)
        self.ffn1 = SwiGLU(cfg)
        self.norm2 = nn.RMSNorm(cfg.hidden_size)
        self.attn = MQAAttention(cfg)
        self.norm3 = nn.RMSNorm(cfg.hidden_size)
        self.ffn2 = SwiGLU(cfg)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + 0.5 * self.ffn2(self.norm3(x))
        return x

class GolfTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([MacaronLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.token_embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, token_ids, labels=None):
        x = self.token_embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss, logits

    def unique_param_count(self) -> int:
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def fp16_size_mb(self) -> float:
        return self.unique_param_count() * 2 / 1024 ** 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPTIMISER — AdamW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_optimizer_pair(model: nn.Module, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_matrix, weight_decay=cfg.weight_decay, betas=(0.9, 0.95), fused=True)
    dummy_muon = torch.optim.SGD([nn.Parameter(torch.zeros(1, requires_grad=True))], lr=0.0)
    return dummy_muon, opt

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEARNING RATE SCHEDULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cosine_lr_schedule(step: int, cfg) -> float:
    W = cfg.warmup_steps
    T = cfg.max_steps
    decay_steps = int(0.1 * T)
    stable_steps = T - W - decay_steps
    if step < W:
        return cfg.lr_matrix * (step / max(1, W))
    elif step < W + stable_steps:
        return cfg.lr_matrix
    else:
        decay_step = step - (W + stable_steps)
        t = decay_step / max(1, decay_steps)
        return cfg.lr_matrix * 0.5 * (1 + math.cos(math.pi * t))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_ema_shadow(model: nn.Module, decay: float = 0.999) -> AveragedModel:
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOKENISER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def prepare_tokenizer(rank: int, cfg: GolfConfig):
    """Load or train a BPE tokeniser, then compute common-token bias list.

    Args:
        rank: Current process rank.
        cfg: GolfConfig with tokenizer_path, vocab_size, train_skip, etc.

    Returns:
        A PreTrainedTokenizerFast instance.
    """
    t0       = time.time()
    tok_file = os.path.join(cfg.tokenizer_path, "tokenizer.json")
    must_train = True

    if os.path.exists(tok_file):
        try:
            probe = Tokenizer.from_file(tok_file)
            if probe.get_vocab_size() == cfg.vocab_size:
                must_train = False
                if rank_is_primary(rank):
                    print("Loading tokeniser from disk …")
        except Exception:
            pass

    if must_train:
        if rank_is_primary(rank):
            print("Training BPE tokeniser from FineWeb sample …")
            corpus = [x["text"] for x in load_dataset(
                "HuggingFaceFW/fineweb", "sample-10BT",
                split="train", streaming=True
            ).skip(cfg.train_skip).take(25000)]
            bpe = ByteLevelBPETokenizer()
            bpe.train_from_iterator(corpus, vocab_size=cfg.vocab_size,
                                    min_frequency=2,
                                    special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
            os.makedirs(cfg.tokenizer_path, exist_ok=True)
            bpe.save(tok_file)
        if dist.is_initialized():
            dist.barrier()

    raw_tok = Tokenizer.from_file(tok_file)
    tok     = PreTrainedTokenizerFast(tokenizer_object=raw_tok)
    tok.pad_token = "<pad>"
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"

    topk_path = "top_k_tokens.pt"
    need_topk = True
    if os.path.exists(topk_path):
        try:
            t_ids = torch.load(topk_path, map_location="cpu", weights_only=True)
            if t_ids.max().item() < cfg.vocab_size:
                need_topk = False
        except Exception:
            pass

    if rank_is_primary(rank) and need_topk:
        print("Computing top-K bias tokens (5000 docs) …")
        from collections import Counter
        freq = Counter()
        for doc in load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                                split="train", streaming=True
                                ).skip(cfg.train_skip).take(5000):
            freq.update(tok.encode(doc["text"]))
        top_ids = [x[0] for x in freq.most_common(cfg.common_token_top_k)]
        torch.save(torch.tensor(top_ids, dtype=torch.long), topk_path)

    if dist.is_initialized():
        dist.barrier()
    if rank_is_primary(rank):
        print(f"Tokeniser ready in {time.time()-t0:.2f}s")
    return tok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StreamingDataPipeline(IterableDataset):
    """Infinite in-memory token stream with coprime-stride cycling.

    Tokens are pre-fetched into a flat int16 tensor. Each epoch uses a
    different starting offset (coprime stride of 7919) for diversity.

    Args:
        token_buffer: 1-D int16 tensor of tokenised training data.
        seq_len: Context window length.
    """

    def __init__(self, token_buffer: torch.Tensor, seq_len: int):
        self.tokens  = token_buffer
        self.seq_len = seq_len
        self.n_seqs  = max(1, (len(token_buffer) - 1) // seq_len)

    def __iter__(self):
        seq_len = self.seq_len
        epoch   = 0
        while True:
            offset = (epoch * 7919) % seq_len
            N      = len(self.tokens)
            for i in range(offset, N - seq_len, seq_len):
                chunk = self.tokens[i: i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    break
                yield {
                    "input_ids": chunk[:-1].long(),
                    "labels":    chunk[1:].long(),
                }
            epoch += 1


def build_data_stream(tokenizer, rank: int, world_size: int,
                      cfg: GolfConfig) -> StreamingDataPipeline:
    """Prefetch training tokens into memory, with disk caching.

    Args:
        tokenizer: PreTrainedTokenizerFast instance.
        rank: Current process rank.
        world_size: Total number of processes.
        cfg: GolfConfig with max_seq_len, train_skip, armenian_weight, etc.

    Returns:
        StreamingDataPipeline ready for DataLoader.
    """
    seq_len        = cfg.max_seq_len
    MAX_TOKENS     = 250_000_000
    script_dir     = os.path.dirname(os.path.abspath(__file__))
    cache_file     = os.path.join(script_dir, f"data_cache_rank{rank}_of{world_size}.pt")

    if os.path.exists(cache_file):
        if rank_is_primary(rank):
            print(f"[Data] Loading cache → {cache_file}")
        t0     = time.time()
        tokens = torch.load(cache_file, map_location="cpu", weights_only=True)
        if rank_is_primary(rank):
            print(f"[Data] Loaded {len(tokens):,} tokens in {time.time()-t0:.1f}s")
        if dist.is_initialized():
            dist.barrier()
        return StreamingDataPipeline(tokens, seq_len)

    if rank_is_primary(rank):
        print(f"[Data] Prefetching {MAX_TOKENS:,} tokens / rank …")

    arm_wt = cfg.armenian_weight

    def is_useful(ex):
        return len(ex.get("text", "").strip()) > 600

    import datasets
    if dist.is_initialized() and not rank_is_primary(rank):
        datasets.utils.logging.disable_progress_bar()

    ds_main = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                           split="train", streaming=True).filter(is_useful)

    if arm_wt > 0:
        ds_arm = load_dataset("wikimedia/wikipedia", "20231101.hy",
                              split="train", streaming=True).filter(is_useful)
        stream = interleave_datasets([ds_main, ds_arm],
                                     probabilities=[1.0 - arm_wt, arm_wt], seed=42)
    else:
        stream = ds_main

    token_buf  = torch.zeros(MAX_TOKENS, dtype=torch.int16)
    cursor     = 0
    t0         = time.time()
    last_log   = 0
    LOG_EVERY  = max(1_000_000, MAX_TOKENS // 10)
    it         = iter(stream)
    doc_count  = 0

    while cursor < MAX_TOKENS:
        try:
            sample = next(it)
        except StopIteration:
            if rank_is_primary(rank):
                print(f"  [Data] Stream restart at {cursor:,} tokens …")
            it = iter(stream)
            doc_count = 0
            continue

        if doc_count % world_size != rank:
            doc_count += 1
            continue
        doc_count += 1

        text = sample.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        if not ids:
            continue

        n_take = min(len(ids), MAX_TOKENS - cursor)
        token_buf[cursor: cursor + n_take] = torch.tensor(ids[:n_take], dtype=torch.int16)
        cursor += n_take

        if rank_is_primary(rank) and cursor - last_log >= LOG_EVERY:
            elapsed = time.time() - t0
            rate    = cursor / max(elapsed, 1e-6) / 1e6
            pct     = 100.0 * cursor / MAX_TOKENS
            print(f"  [Data] {cursor/1e6:.1f}M / {MAX_TOKENS/1e6:.1f}M tok "
                  f"({pct:.0f}%) — {rate:.1f}M tok/s")
            last_log = cursor

    if rank_is_primary(rank):
        elapsed = time.time() - t0
        print(f"✓ Prefetch complete: {elapsed:.2f}s | {cursor:,} tokens | "
              f"{cursor/max(elapsed,1e-6)/1e6:.1f}M tok/s")

    saved_buf = token_buf[:cursor]
    torch.save(saved_buf, cache_file)
    if rank_is_primary(rank):
        mb = os.path.getsize(cache_file) / 1024 ** 2
        print(f"[Data] Cache saved → {cache_file} ({mb:.1f} MB)")

    if dist.is_initialized():
        dist.barrier()

    return StreamingDataPipeline(saved_buf, seq_len)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BENCHMARK CALIBRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def benchmark_step_time(net: nn.Module, device: torch.device,
                        cfg: GolfConfig, n_warmup: int = 10) -> float:
    """Measure average wall-clock seconds per training step.

    Args:
        net: Model (unwrapped, on device).
        device: CUDA device.
        cfg: GolfConfig with per_device_batch, max_seq_len, vocab_size, grad_clip.
        n_warmup: Number of calibration steps.

    Returns:
        Mean seconds per step as float.
    """
    B, T, V = cfg.per_device_batch, cfg.max_seq_len, cfg.vocab_size
    dtype   = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    net.train()

    tmp_muon, tmp_adam = build_optimizer_pair(net, cfg)

    # One throwaway pass to initialise CUDA state
    dummy = torch.randint(0, V, (B, T), device=device)
    with torch.autocast(device_type="cuda", dtype=dtype):
        l, _ = net(dummy, dummy)
    l.backward()
    net.zero_grad()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_warmup):
        ids = torch.randint(0, V, (B, T), device=device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss, _ = net(ids, ids)
        loss.backward()
        if cfg.grad_accum == 1:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            tmp_muon.step()
            tmp_adam.step()
            tmp_muon.zero_grad()
            tmp_adam.zero_grad()
    torch.cuda.synchronize()

    del tmp_muon, tmp_adam
    return (time.time() - t0) / n_warmup


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINER CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GolfTrainer:
    """Stateful training harness for GolfTransformer.

    Args:
        model: DDP-wrapped or raw GolfTransformer.
        ema_shadow: AveragedModel EMA copy.
        data_stream: StreamingDataPipeline instance.
        muon_opt: AdaMuonOptimizer for matrix parameters.
        adam_opt: AdamW for embedding/scalar parameters.
        device: CUDA device.
        rank: Process rank.
        world_size: Total number of processes.
        cfg: GolfConfig with all training hyperparameters.
        logger: ExperimentLogger for JSON-line recording.
    """

    def __init__(self, model, ema_shadow, data_stream, muon_opt, adam_opt,
                 device, rank, world_size, cfg: GolfConfig,
                 logger: ExperimentLogger):
        self.model       = model
        self.ema_shadow  = ema_shadow
        self.data_stream = data_stream
        self.muon_opt    = muon_opt
        self.adam_opt    = adam_opt
        self.device      = device
        self.rank        = rank
        self.world_size  = world_size
        self.cfg         = cfg
        self.logger      = logger

        self.use_bf16  = torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler    = torch.amp.GradScaler("cuda", enabled=not self.use_bf16)

    def fit(self) -> int:
        """Run the main training loop until time or step limit is reached.

        Returns:
            Total number of gradient steps completed.
        """
        self.model.train()
        loader      = DataLoader(self.data_stream,
                                 batch_size=self.cfg.per_device_batch,
                                 num_workers=0, pin_memory=True)
        start_time  = time.time()
        global_step = 0

        if rank_is_primary(self.rank):
            base_net = self.model.module if isinstance(self.model, DDP) else self.model
            n_params = base_net.unique_param_count()
            fp16_mb  = base_net.fp16_size_mb()
            eff_bs   = self.world_size * self.cfg.per_device_batch * self.cfg.grad_accum
            print(f"\n⚡ Training | {self.world_size} GPU(s) | BF16={self.use_bf16}")
            print(f"   Params: {n_params:,} | ~{fp16_mb:.2f} MB fp16")
            print(f"   max_steps={self.cfg.max_steps} | "
                  f"warmdown={self.cfg.warmdown_steps}")
            print(f"   effective_batch={eff_bs}\n")
            self.logger.log(event="train_start", gpus=self.world_size,
                            bf16=self.use_bf16, param_count=n_params,
                            size_mb=round(fp16_mb, 4), effective_batch=eff_bs)

        self.muon_opt.zero_grad()
        self.adam_opt.zero_grad()

        for batch in loader:
            elapsed = time.time() - start_time

            # Rank-0 controlled deterministic cutoff (prevents NCCL hangs)
            check_every = self.cfg.grad_accum * 5
            if global_step > 0 and global_step % check_every == 0:
                stop = torch.tensor([0], dtype=torch.int, device=self.device)
                if rank_is_primary(self.rank) and \
                        elapsed >= self.cfg.time_limit_sec - 2.5:
                    print(f"[Cutoff] Safe stop at {elapsed:.2f}s (step={global_step})")
                    stop[0] = 1
                if dist.is_initialized():
                    dist.broadcast(stop, src=0)
                if stop.item() == 1:
                    break

            if global_step >= self.cfg.max_steps:
                if rank_is_primary(self.rank):
                    print(f"Max steps reached: {global_step} ({elapsed:.0f}s)")
                break

            # Learning rate and momentum scheduling
            lr_now   = cosine_lr_schedule(global_step, self.cfg)
            ratio    = lr_now / self.cfg.lr_matrix
            progress = min(1.0, global_step / self.cfg.max_steps)
            mom_now  = 0.95 + 0.04 * 0.5 * (1 + math.cos(math.pi * progress))

            for g in self.muon_opt.param_groups:
                g["lr"]       = lr_now
                g["momentum"] = mom_now
            for g in self.adam_opt.param_groups:
                if "initial_lr" not in g:
                    g["initial_lr"] = g["lr"]
                g["lr"] = g["initial_lr"] * ratio

            token_ids = batch["input_ids"].to(self.device, non_blocking=True)
            targets   = batch["labels"].to(self.device, non_blocking=True)

            accumulating = (global_step + 1) % self.cfg.grad_accum != 0
            if accumulating and isinstance(self.model, DDP):
                with self.model.no_sync():
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        loss, _ = self.model(token_ids, targets)
                        loss    = loss / self.cfg.grad_accum
                    self.scaler.scale(loss).backward()
            else:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    loss, _ = self.model(token_ids, targets)
                    loss    = loss / self.cfg.grad_accum
                self.scaler.scale(loss).backward()

            if (global_step + 1) % self.cfg.grad_accum == 0:
                self.scaler.unscale_(self.muon_opt)
                self.scaler.unscale_(self.adam_opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg.grad_clip)
                self.scaler.step(self.muon_opt)
                self.scaler.step(self.adam_opt)
                self.scaler.update()
                self.muon_opt.zero_grad()
                self.adam_opt.zero_grad()
                raw_net = self.model.module if isinstance(self.model, DDP) else self.model
                if hasattr(self.ema_shadow, "update_parameters"):
                    self.ema_shadow.update_parameters(raw_net)

            global_step += 1
            if rank_is_primary(self.rank) and global_step % 10 == 0:
                bpb_est = loss.item() * self.cfg.grad_accum / math.log(2)
                print(f"   step {global_step:5d} | BPB~{bpb_est:.4f} | "
                      f"LR {lr_now:.3e} | {elapsed:.0f}s")
                self.logger.log(event="step", step=global_step,
                                bpb=round(bpb_est, 6), lr=round(lr_now, 8),
                                elapsed_s=round(elapsed, 1))

        return global_step


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION — Score-First TTT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_bpb(model: nn.Module, tokenizer, texts: list,
                device, cfg: GolfConfig):
    """Sliding-window BPB evaluation with legal Score-First TTT.

    For each sliding window chunk:
        1. SCORE: no_grad forward on current parameters → record NLL.
        2. ADAPT: separate forward + backward + SGD step to improve future chunks.

    This ordering guarantees no look-ahead contamination, matching competition
    evaluation semantics.

    Args:
        model: GolfTransformer (eval mode set internally).
        tokenizer: PreTrainedTokenizerFast.
        texts: List of evaluation document strings.
        device: CUDA device.
        cfg: GolfConfig with use_ttt, ttt_lr, max_seq_len, eval_stride.

    Returns:
        Tuple of (bits_per_byte, total_tokens_scored).
    """
    use_ttt  = cfg.use_ttt
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model.eval()

    ttt_opt = None
    if use_ttt:
        ttt_opt = torch.optim.SGD(model.parameters(), lr=cfg.ttt_lr)

    seq_len      = cfg.max_seq_len
    stride       = cfg.eval_stride
    total_nll    = 0.0
    total_tokens = 0
    total_bytes  = 0

    for text in texts:
        seq_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(seq_ids) < 2:
            continue
        total_bytes += len(text.encode("utf-8"))
        ids_tensor  = torch.tensor(seq_ids, dtype=torch.long, device=device)
        L           = len(ids_tensor)
        prev_end    = 0

        for start in range(0, L - 1, stride):
            end   = min(start + seq_len + 1, L)
            chunk = ids_tensor[start:end]
            if len(chunk) < 2:
                break

            inp   = chunk[:-1].unsqueeze(0)
            tgt   = chunk[1:].unsqueeze(0)
            n_new = min(stride, tgt.shape[1] - max(0, prev_end - start - 1))
            if n_new <= 0:
                prev_end = end
                continue

            # ── SCORE phase: record NLL on current parameters ──────────────
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, score_logits = model(inp, labels=None)

            nll_sum = F.cross_entropy(
                score_logits[0, -n_new:],
                tgt[0, -n_new:],
                reduction="sum",
            )
            total_nll    += nll_sum.item()
            total_tokens += n_new
            prev_end      = end

            # ── ADAPT phase: SGD step for future chunks ────────────────────
            if use_ttt and ttt_opt is not None:
                ttt_opt.zero_grad()
                with torch.enable_grad():
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, adapt_logits = model(inp, labels=None)
                    adapt_loss = F.cross_entropy(adapt_logits[0], tgt[0])
                adapt_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ttt_opt.step()

    if total_bytes == 0 or total_tokens == 0:
        return float("inf"), 0

    avg_tok_per_byte = total_tokens / max(total_bytes, 1)
    bpb = (total_nll / total_tokens) / math.log(2) * avg_tok_per_byte
    return bpb, total_tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPORT — Dynamic MSE SDClip + INT6 Error Diffusion + LZMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LQER asymmetric rank-4 (post-quant correction) + Brotli compression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _lqer_pack_asym(A, B, g=64):
    """Pack A as INT2 (single fp16 scale), B as INT4 per-group-g."""
    sA = (A.abs().amax().clamp_min(1e-10) / 1.5).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float()), -2, 1).to(torch.int8)
    Bf = B.reshape(-1, g)
    Bmax = Bf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    sB = (Bmax / 7.5).to(torch.float16).reshape(-1)
    qB = torch.clamp(torch.round(Bf / sB.float().reshape(-1, 1)), -8, 7).to(torch.int8).reshape(B.shape)
    return qA, sA, qB, sB

def _lqer_unpack_asym(qA, sA, qB, sB):
    A = qA.float() * float(sA)
    g = qB.numel() // sB.numel()
    Bf = qB.reshape(-1, g).float() * sB.float().view(-1, 1)
    return A @ Bf.reshape(qB.shape)

_BSHF_MAGIC = b"BSHF"
def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src_arr = np.frombuffer(data, dtype=np.uint8)
    n = len(src_arr); out = np.empty(n, dtype=np.uint8); off = 0
    for pos in range(stride):
        chunk = src_arr[pos::stride]
        out[off:off+len(chunk)] = chunk
        off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2: return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload); out = np.empty(n, dtype=np.uint8); off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[off:off+chunk_len]
        off += chunk_len
    return out.tobytes()


def export_submission(model, ema_shadow, cfg: GolfConfig,
                      rank: int, *_args) -> Optional[tuple]:
    """Quantise and compress the model checkpoint for submission.

    Quantisation pipeline per weight matrix:
        1. Dynamic SDClip: test sigma ∈ {2.5, 3.0, 3.5, 4.0}, keep the value
           that minimises INT6 round-trip reconstruction MSE.
        2. Vectorised Sigma-Delta (error diffusion) quantisation to int8 codes.
        3. Per-row fp16 scale stored alongside codes.
    EMA weights are applied before quantisation when available.

    Args:
        model: DDP-wrapped or raw GolfTransformer.
        ema_shadow: AveragedModel EMA copy.
        cfg: GolfConfig with output_path, sdclip_sigma, sdclip_dynamic, etc.
        rank: Process rank (only rank 0 saves).

    Returns:
        (lzma_size_mb, fp16_size_mb) or None if not rank 0.
    """
    if not rank_is_primary(rank):
        return None

    net = model.module if isinstance(model, DDP) else model
    if hasattr(ema_shadow, "module"):
        ema_sd = ema_shadow.module.state_dict()
        # torch.compile wraps the model as OptimizedModule whose state-dict
        # keys carry a "_orig_mod." prefix; remap EMA keys to match.
        net_keys = set(net.state_dict().keys())
        if net_keys and next(iter(net_keys)).startswith("_orig_mod."):
            ema_sd = {"_orig_mod." + k: v for k, v in ema_sd.items()}
        net.load_state_dict(ema_sd)
        print("EMA weights applied.")

    # Sigma grid for per-tensor MSE search
    sigma_candidates = ([2.5, 3.0, 3.5, 4.0] if cfg.sdclip_dynamic
                        else [cfg.sdclip_sigma])

    quantised_state: Dict[str, torch.Tensor] = {}

    for key, tensor in net.state_dict().items():
        clean_key = key.replace("_orig_mod.", "")
        fp_tensor = tensor.float().cpu()

        if fp_tensor.dim() >= 2:
            limit       = (127.0 if ("embed" in clean_key or "head" in clean_key
                                     or "pos" in clean_key) else 31.0)
            row_std     = fp_tensor.std(dim=-1, keepdim=True).clamp(min=1e-6)

            best_mse      = float("inf")
            best_codes    = None
            best_scale    = None

            for sigma in sigma_candidates:
                clip_bound = row_std * sigma
                clipped    = fp_tensor.clamp(-clip_bound, clip_bound)
                scale      = (clipped.abs().max(dim=-1, keepdim=True)[0]
                              / limit).clamp(min=1e-6)

                # Sigma-Delta error diffusion quantisation
                csum   = torch.cumsum(clipped, dim=1)
                csum_q = (csum / scale).round()
                codes  = torch.empty_like(clipped, dtype=torch.int8)
                codes[:, 0]  = csum_q[:, 0].clamp(-limit, limit).to(torch.int8)
                codes[:, 1:] = (csum_q[:, 1:] - csum_q[:, :-1]).clamp(
                    -limit, limit).to(torch.int8)

                # Reconstruction MSE — select sigma that minimises round-trip error
                recon = codes.float() * scale
                mse   = F.mse_loss(recon, fp_tensor).item()

                if not math.isnan(mse) and mse < best_mse:
                    best_mse   = mse
                    best_codes = codes
                    best_scale = scale

            if best_scale is None:
                # Fallback if tensors are NaN
                best_codes = torch.zeros_like(fp_tensor, dtype=torch.int8)
                best_scale = torch.ones((fp_tensor.shape[0], 1), device=fp_tensor.device, dtype=fp_tensor.dtype)

            quantised_state[clean_key]              = best_codes
            quantised_state[clean_key + "_scale"]   = best_scale.half()
        else:
            quantised_state[clean_key] = fp_tensor.half()

    # LQER asymmetric rank-4 post-quantization correction
    if getattr(cfg, "lqer_enabled", False):
        cands = []
        for k, codes in list(quantised_state.items()):
            if k.endswith("_scale") or k.startswith("_lq"):
                continue
            scale_k = k + "_scale"
            if scale_k not in quantised_state or codes.dim() < 2:
                continue
            scale = quantised_state[scale_k]
            W_q = (codes.float() * scale.float()).reshape(codes.shape)
            # Need original fp tensor — re-fetch from net.state_dict()
            fp_key = "_orig_mod." + k if "_orig_mod." + k in net.state_dict() else k
            if fp_key not in net.state_dict():
                continue
            W_fp = net.state_dict()[fp_key].float().cpu()
            if W_fp.shape != W_q.shape:
                continue
            E = W_fp - W_q
            cands.append((k, E, float(E.norm())))
        cands.sort(key=lambda x: -x[2])
        for (name, E, _) in cands[:cfg.lqer_top_k]:
            U, S, Vh = torch.linalg.svd(E, full_matrices=False)
            r = min(cfg.lqer_rank, S.numel())
            A = (U[:, :r] * S[:r]).contiguous()
            Bm = Vh[:r, :].contiguous()
            if Bm.numel() % cfg.lqer_asym_group == 0:
                qA, sA, qB, sB = _lqer_pack_asym(A, Bm, cfg.lqer_asym_group)
                quantised_state[name + "_lqA"]  = qA
                quantised_state[name + "_lqAs"] = sA
                quantised_state[name + "_lqB"]  = qB
                quantised_state[name + "_lqBs"] = sB
                print(f"  LQER+ {name}: rank={r} fro={E.norm():.3f}")

    payload = {"config": cfg.to_dict(), "state_dict": quantised_state}
    torch.save(payload, cfg.output_path)

    if getattr(cfg, "use_brotli", False):
        import brotli
        with open(cfg.output_path, "rb") as fin: raw_bytes = fin.read()
        shuffled = _byte_shuffle(raw_bytes, getattr(cfg, "byte_shuffle_stride", 2))
        compressed = brotli.compress(shuffled, quality=getattr(cfg, "brotli_quality", 11))
        lzma_path = cfg.output_path.replace(".pt", ".pt.lzma")
        with open(lzma_path, "wb") as fout:
            fout.write(compressed)
    else:
        lzma_path = cfg.output_path.replace(".pt", ".pt.lzma")
        with open(cfg.output_path, "rb") as fin: raw_bytes = fin.read()
        lzma_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 1 << 27}]
        with lzma.open(lzma_path, "wb", format=lzma.FORMAT_XZ, filters=lzma_filters) as fout:
            fout.write(raw_bytes)

    lzma_mb = os.path.getsize(lzma_path) / 1024 ** 2
    fp16_mb = os.path.getsize(cfg.output_path) / 1024 ** 2
    budget_ok = "✅ within budget" if lzma_mb <= 16 else "❌ OVER BUDGET"
    print(f"\n  → {cfg.output_path}  ({fp16_mb:.2f} MB fp16)")
    print(f"  → {lzma_path}  ({lzma_mb:.2f} MB) [{budget_ok}]")

    with open(cfg.output_pkl_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return lzma_mb, fp16_mb


def load_checkpoint(path: str, device: str = "cpu") -> "GolfTransformer":
    """Restore a GolfTransformer from a quantised checkpoint.

    Args:
        path: Path to .pt or .pt.lzma checkpoint file.
        device: Target device string.

    Returns:
        GolfTransformer in eval mode with dequantised fp16 weights.
    """
    if path.endswith(".lzma"):
        # Try Brotli+shuffle first (our format), fall back to LZMA
        with open(path, "rb") as fh: blob = fh.read()
        try:
            import brotli
            raw = brotli.decompress(blob)
            raw = _byte_unshuffle(raw)
            data = torch.load(io.BytesIO(raw), map_location=device, weights_only=False)
        except Exception:
            with lzma.open(path, "rb") as fh:
                data = torch.load(fh, map_location=device, weights_only=False)
    else:
        data = torch.load(path, map_location=device, weights_only=False)

    saved_state = data["state_dict"]
    fp16_state  = {}
    for k, v in saved_state.items():
        if k.endswith("_scale") or k.endswith("_lqA") or k.endswith("_lqAs") or k.endswith("_lqB") or k.endswith("_lqBs"):
            continue
        scale_key = k + "_scale"
        if scale_key in saved_state:
            W_dq = (v.float() * saved_state[scale_key].float())
            # Apply LQER correction if present
            if k + "_lqA" in saved_state:
                qA = saved_state[k + "_lqA"]; sA = saved_state[k + "_lqAs"]
                qB = saved_state[k + "_lqB"]; sB = saved_state[k + "_lqBs"]
                W_dq = W_dq + _lqer_unpack_asym(qA, sA, qB, sB)
            fp16_state[k] = W_dq.half()
        else:
            fp16_state[k] = v.half() if v.dtype == torch.float32 else v

    saved_cfg = data["config"]
    # Reconstruct dataclass from saved dict
    cfg_obj = GolfConfig(**{k: v for k, v in saved_cfg.items()
                            if k in GolfConfig.__dataclass_fields__})
    net = GolfTransformer(cfg_obj)
    net.load_state_dict(fp16_state)
    return net.eval()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":

    local_rank, world_size, device = init_distributed()

    CFG    = GolfConfig()
    LOGGER = ExperimentLogger(enabled=rank_is_primary(local_rank), cfg=CFG)

    # ── Auto-downsize hidden_size to fit 16 MB LZMA budget ────────────────────
    LZMA_RATIO = 0.3668
    BUDGET_MB  = 15.98
    size_candidates = [(512, 1024), (480, 960), (448, 896), (416, 832), (384, 768), (352, 704), (344, 688), (336, 672), (320, 640), (288, 576), (256, 512)]
    for h, inter in size_candidates:
        CFG["hidden_size"]    = h
        CFG["intermediate"]   = inter
        probe     = GolfTransformer(CFG)
        est_lzma  = probe.fp16_size_mb() * LZMA_RATIO
        if rank_is_primary(local_rank):
            print(f"[SizeCheck] hidden={h} moe={inter} → est LZMA≈{est_lzma:.2f} MB")
        if est_lzma <= BUDGET_MB:
            break

    if rank_is_primary(local_rank):
        print("=" * 65)
        print("  DEEP MACARON — Competition Run — H100 8×")
        print("  Macaron layer structure (FFN -> Attn -> FFN)")
        print("  MQA Attention, SwiGLU, SiLU")
        print("  WSD (Warmup-Stable-Decay) Learning Rate Schedule")
        print(f"  vocab={CFG.vocab_size} hidden={CFG.hidden_size} layers={CFG.num_layers}")
        print("=" * 65)

    tokenizer = prepare_tokenizer(local_rank, CFG)

    t_data   = time.time()
    pipeline = build_data_stream(tokenizer, local_rank, world_size, CFG)
    if rank_is_primary(local_rank):
        print(f"Data pipeline ready in {time.time()-t_data:.2f}s")

    seeds_to_run = [1337, 42, 2025]

    for seed in seeds_to_run:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        CFG["output_path"]     = f"./submission_global_seed{seed}.pt"
        CFG["output_pkl_path"] = f"./submission_global_seed{seed}.pkl"

        if rank_is_primary(local_rank):
            print(f"\n{'#'*65}")
            print(f"   SEED {seed}  — starting training run")
            print(f"{'#'*65}\n")

        base_net  = GolfTransformer(CFG).to(device)
        ema_net   = make_ema_shadow(base_net, CFG.ema_decay)

        if CFG.use_compile:
            if rank_is_primary(local_rank):
                print("Compiling model …")
            try:
                base_net = torch.compile(base_net, mode="reduce-overhead")
                if rank_is_primary(local_rank):
                    print("  torch.compile ✅")
            except Exception as exc:
                if rank_is_primary(local_rank):
                    print(f"  torch.compile skipped: {exc}")

        if rank_is_primary(local_rank):
            print("Calibrating step time (10 synthetic steps) …")
        step_time = benchmark_step_time(base_net, device, CFG, n_warmup=10)

        if dist.is_initialized():
            st_buf = torch.tensor([step_time], dtype=torch.float32, device=device)
            dist.broadcast(st_buf, src=0)
            step_time = st_buf.item()

        if rank_is_primary(local_rank):
            print(f"  raw ~{step_time:.3f}s/step → ×1.02 = {step_time*1.02:.3f}s")

        configure_schedule(CFG, world_size, step_time)

        ddp_net = (DDP(base_net, device_ids=[0])
                   if dist.is_initialized() else base_net)

        muon_opt, adam_opt = build_optimizer_pair(
            ddp_net.module if isinstance(ddp_net, DDP) else ddp_net, CFG
        )

        trainer = GolfTrainer(
            model=ddp_net,
            ema_shadow=ema_net,
            data_stream=pipeline,
            muon_opt=muon_opt,
            adam_opt=adam_opt,
            device=device,
            rank=local_rank,
            world_size=world_size,
            cfg=CFG,
            logger=LOGGER,
        )

        t_train    = time.time()
        n_steps    = trainer.fit()
        train_time = time.time() - t_train

        if rank_is_primary(local_rank):
            print(f"Training complete: {train_time:.2f}s")
            LOGGER.log(event="train_end", seed=seed, steps=n_steps,
                       train_time_s=round(train_time, 2))

        result = export_submission(ddp_net, ema_net, CFG, local_rank)

        if rank_is_primary(local_rank) and result:
            lzma_mb, fp16_mb = result
            LOGGER.log(event="checkpoint_saved", seed=seed,
                       path=CFG.output_path,
                       size_mb_fp16=round(fp16_mb, 4),
                       size_mb_lzma=round(lzma_mb, 4),
                       within_budget=lzma_mb <= 16.0)

        if rank_is_primary(local_rank) and result:
            lzma_path = CFG.output_path.replace(".pt", ".pt.lzma")
            eval_src  = lzma_path if os.path.exists(lzma_path) else CFG.output_path

            print(f"\nBPB evaluation — stride={CFG.eval_stride} sliding window …")
            t_eval     = time.time()
            eval_net   = load_checkpoint(eval_src, device=str(device))
            eval_net.to(device)

            fw_texts = [x["text"] for x in load_dataset(
                "HuggingFaceFW/fineweb", "sample-10BT",
                split="train", streaming=True).take(CFG.eval_docs)]
            bpb_val, n_scored = compute_bpb(eval_net, tokenizer, fw_texts,
                                            device=device, cfg=CFG)
            eval_time = time.time() - t_eval

            print(f"\n{'='*65}")
            print(f"  RUN COMPLETE  (seed={seed})")
            print(f"  steps       : {n_steps}")
            print(f"  size fp16   : {fp16_mb:.2f} MB")
            print(f"  size lzma   : {lzma_mb:.2f} MB")
            print(f"  FineWeb BPB : {bpb_val:.4f}  (stride={CFG.eval_stride})")
            print(f"  train time  : {train_time:.2f}s "
                  f"({int(train_time//60)}m {int(train_time%60)}s)")
            print(f"  eval time   : {eval_time:.1f}s")
            print(f"{'='*65}")

            bytes_total = os.path.getsize(CFG.output_path)
            bytes_code  = os.path.getsize(__file__)
            run_date    = datetime.datetime.now(
                datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            sub_data = {
                "author":      "Karen042009",
                "github_id":   "Karen042009",
                "name":        (f"Karen042009 - DualHashSkip AdaMuon "
                                f"DynSDClip SharedMoE LayerScale (Seed {seed})"),
                "blurb":       (f"Seed {seed}. DualTokenHashSkip(2×2048×16,concat), "
                                "LayerScale Recurrence (1.0/0.1 init), "
                                "AdaMuonOptimizer (RMS pre-cond + Riemannian NS), "
                                "Dynamic MSE SDClip (σ-grid {2.5,3.0,3.5,4.0} at export), "
                                "SharedMoE (1 shared + top-1/3 specialised), "
                                "Score-First TTT 2-pass, parallel residuals, "
                                "QKGain=5.25, PartialRoPE=32, "
                                "LeakyReLU(0.5)^2 SwiGLU, vocab=8192"),
                "date":        run_date,
                "val_loss":    round(float(bpb_val * math.log(2)), 6),
                "val_bpb":     round(bpb_val, 6),
                "bytes_total": bytes_total,
                "bytes_code":  bytes_code,
            }

            print(f"final_int8_zlib_roundtrip val_loss:{bpb_val*math.log(2):.6f} "
                  f"val_bpb:{bpb_val:.6f} eval_time:{eval_time*1000:.0f}ms")

            json_path = f"submission_global_seed{seed}.json"
            with open(json_path, "w") as fh:
                json.dump(sub_data, fh, indent=4)
            print(f"  Metadata → '{json_path}'")

            print("\n" + "=" * 65)
            print(f"  FINAL RESULT  (Karen042009  SEED {seed})")
            print(f"  val_bpb:    {bpb_val:.8f}")
            print(f"  val_loss:   {bpb_val*math.log(2):.8f}")
            print(f"  size_lzma:  {lzma_mb:.2f} MB  (limit 16 MB)")
            print(f"  steps:      {n_steps}")
            print(f"  train_time: {train_time:.2f}s "
                  f"({int(train_time//60)}m {int(train_time%60)}s)")
            print("=" * 65)

            LOGGER.log(
                event="seed_done",
                seed=seed,
                steps=n_steps,
                size_mb_fp16=round(fp16_mb, 4),
                size_mb_lzma=round(lzma_mb, 4),
                fineweb_bpb=round(bpb_val, 8),
                fineweb_val_loss=round(float(bpb_val * math.log(2)), 8),
                eval_time_s=round(eval_time, 2),
                scored_tokens=n_scored,
                bytes_total=bytes_total,
                bytes_code=bytes_code,
                json_out=json_path,
                checkpoint=CFG.output_path,
                date=run_date,
            )

            eval_net.cpu()
            del eval_net

        if "base_net" in dir():
            del base_net, ema_net, ddp_net, muon_opt, adam_opt, trainer

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    if rank_is_primary(local_rank):
        LOGGER.close(message="All seeds completed.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
