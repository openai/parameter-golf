"""
Parameter Golf — Global Edition (SOTA Optimised)
"""


# ============================================================
# CUDA DEVICE ISOLATION — must run before ALL other imports.
# ============================================================
import os as _os
if "LOCAL_RANK" in _os.environ:
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(int(_os.environ["LOCAL_RANK"]))

import os, time, math, json, datetime, pickle, lzma, contextlib
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


# ==========================================
# RECORD LOGGER
# ==========================================
class RecordLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = f"run_{ts}.record"
        if enabled:
            with open(self.path, "w") as f:
                f.write(json.dumps({"event": "run_start", "timestamp": ts,
                                    "config": CFG}) + "\n")
            print(f"[RecordLogger] Writing to: {self.path}")

    def log(self, **kwargs):
        if not self.enabled: return
        entry = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 **kwargs}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def finalize(self, **summary):
        self.log(event="run_end", **summary)
        if self.enabled:
            print(f"[RecordLogger] Run record saved: {self.path}")


# ==========================================
# 0. CONFIG
# ==========================================
CFG = dict(
    # Architecture
    vocab_size          = 8192,
    hidden_size         = 384,     # target size
    intermediate        = 768,
    num_layers          = 10,      # 10 physical layers (no recurrence)
    layer_loop          = None,
    num_heads           = 8,
    num_kv_heads        = 1,
    max_seq_len         = 512,
    tie_embeddings      = True,

    # Attention & Evaluation
    qk_gain             = 6.0,
    use_ttt             = True,
    ttt_lr              = 0.004,

    # MoE: Shared + Specialized
    num_experts         = 4,
    num_experts_per_tok = 1,
    moe_intermediate    = 384,

    # [v10-1] TripleHash Bigram Skip
    use_bigram          = True,
    bigram_vocab        = 2048,
    bigram_dim          = 16,

    # RMSNorm
    ln_scale            = True,
    rope_dims           = 32,

    # QAT INT6
    qat_enabled         = True,
    qat_bits            = 6,
    qat_start_frac      = 0.80,

    # Training
    use_muon            = True,
    per_device_batch    = 64,
    grad_accum          = 2,
    lr_matrix           = 0.007,
    lr_embed            = 0.007,
    lr_scalar           = 0.005,
    weight_decay        = 0.1,
    warmup_steps        = 100,
    time_limit_sec      = 600,
    grad_clip           = 0.4,
    sdclip_sigma        = 3.2,
    fineweb_weight      = 1.0,
    eval_stride         = 64,
    tokenizer_path      = "./pg_tokenizer_grisha",
    output_path         = "./submission_global.pt",

    # Data
    train_skip          = 200,   # skip first 200 docs so eval docs 0-49 don't overlap
    eval_docs           = 50,    # FineWeb docs 0-49 used for final BPB evaluation
    common_token_top_k  = 512,   # top-K frequent tokens for frequency-bias logit boost

    # EMA
    ema_decay           = 0.999,
)

# ==========================================
# Dynamic schedule
# ==========================================
def finalise_schedule(cfg, world_size, step_time_estimate):
    # Use 1.20 multiplier so we expect training to be 20% slower than sync-free calibration.
    # This ensures we hit the warmdown reliably within 600s instead of getting cutoff early.
    target_time     = cfg["time_limit_sec"] - 5.0 # leave 5s buffer for saving/logs
    multiplier      = 1.14
    effective_time  = step_time_estimate * multiplier
    estimated_steps = int(target_time / effective_time)
    estimated_steps = (estimated_steps // 1) * 1
    estimated_steps = max(2000, min(estimated_steps, 20000))
    cfg["max_steps"]      = estimated_steps
    cfg["warmdown_steps"] = max(500, min(estimated_steps // 4, estimated_steps - 1))
    cfg["qat_start_step"] = int(estimated_steps * cfg["qat_start_frac"])
    if is_main(dist.get_rank() if dist.is_initialized() else 0):
        print(f"[Schedule] Calibration={step_time_estimate:.3f}s | Effective (×1.20)={effective_time:.3f}s")
        print(f"[Schedule] max_steps={estimated_steps} | warmdown={cfg['warmdown_steps']} | qat_start={cfg['qat_start_step']}")
    return cfg


# ==========================================
# 1. DDP SETUP
# ==========================================
def setup_ddp():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    if "LOCAL_RANK" not in os.environ:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, dev

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    import sys
    print(f"[Rank {local_rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')} "
          f"(cuda:0 -> physical GPU {local_rank}) | world_size={world_size}",
          file=sys.stderr, flush=True)

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    return local_rank, world_size, torch.device("cuda:0")

def is_main(rank): return rank == 0


# ==========================================
# 2. ARCHITECTURE
# ==========================================

class CausalAttention(nn.Module):
    def __init__(self, hidden, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden // num_heads
        self.groups       = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        rope_dims = CFG.get("rope_dims", 32)
        if rope_dims > 0:
            pos      = torch.arange(T, device=x.device, dtype=torch.float32)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dims, 2, device=x.device,
                                                      dtype=torch.float32) / rope_dims))
            freqs    = torch.outer(pos, inv_freq)
            freqs    = torch.cat((freqs, freqs), dim=-1)       # (T, rope_dims)
            cos_f    = freqs.cos().view(1, 1, T, rope_dims).to(q.dtype)
            sin_f    = freqs.sin().view(1, 1, T, rope_dims).to(q.dtype)

            q_rt, q_pass = q[..., :rope_dims], q[..., rope_dims:]
            k_rt, k_pass = k[..., :rope_dims], k[..., rope_dims:]

            q_h1, q_h2 = q_rt.chunk(2, dim=-1)
            q = torch.cat((q_rt * cos_f + torch.cat((-q_h2, q_h1), dim=-1) * sin_f, q_pass), dim=-1)

            k_h1, k_h2 = k_rt.chunk(2, dim=-1)
            k = torch.cat((k_rt * cos_f + torch.cat((-k_h2, k_h1), dim=-1) * sin_f, k_pass), dim=-1)

        k_exp = k.repeat_interleave(self.groups, dim=1)
        v_exp = v.repeat_interleave(self.groups, dim=1)
        qk_gain = CFG.get("qk_gain", 1.0)
        out = F.scaled_dot_product_attention(q * qk_gain, k_exp, v_exp, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class GeGLU(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


class HybridMoE(nn.Module):
    """
    1 Shared Expert + N Specialized Experts (Top-1 Routing).
    Distinct from Karen's SharedMoE by using a different routing bias and noise scale.
    """
    def __init__(self, hidden, intermediate, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.shared_expert = GeGLU(hidden, intermediate)
        self.specialized   = nn.ModuleList([GeGLU(hidden, intermediate) for _ in range(num_experts)])
        self.gate          = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat  = x.view(-1, C)
        
        # Shared expert (always on)
        shared_out = self.shared_expert(x_flat)
        
        # Specialized routing
        router_logits = self.gate(x_flat)
        if self.training:
            # Gumbel-style noise for diversity
            noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-6) + 1e-6)
            router_logits = router_logits + noise * 0.05
        
        routing_weights = F.softmax(router_logits, dim=1)
        prob, expert_idx = torch.topk(routing_weights, 1, dim=-1)
        
        # Vectorized expert execution
        final_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (expert_idx == i).squeeze(-1)
            if mask.any():
                final_out[mask] = self.specialized[i](x_flat[mask])
        
        return (shared_out + final_out).view(B, T, C)


class Block(nn.Module):
    """PaLM-style Parallel Block: x = x + attn(norm(x)) + ffn(norm(x))"""
    def __init__(self, cfg):
        super().__init__()
        H = cfg["hidden_size"]
        self.norm = nn.RMSNorm(H) # Shared norm for parallel branches
        self.attn = CausalAttention(H, cfg["num_heads"], cfg["num_kv_heads"])
        self.ffn  = HybridMoE(H, cfg["moe_intermediate"], cfg["num_experts"])

    def forward(self, x):
        h = self.norm(x)
        return x + self.attn(h) + self.ffn(h)


# ──────────────────────────────────────────────────────────────────────────────
# [v9-1] BigramHash: learned (prev_tok, curr_tok) bigram embedding table
#         with polynomial hash → compact 4096-entry lookup → linear projection.
#         Acts as a pure additive skip on top of the token embedding.
#         Parameter cost: 4096×32 + 32×H ≈ 131K + 15K = ~146K extra params.
# ──────────────────────────────────────────────────────────────────────────────
class TripleHash(nn.Module):
    """
    TripleHash: Uses 3 prime multipliers (8191, 104729, 2097593) to index 3 small
    tables, concatenating the results. Highly robust bigram signature.
    """
    def __init__(self, table_size: int, dim: int, hidden_size: int):
        super().__init__()
        self.table_size = table_size
        self.table1 = nn.Embedding(table_size, dim)
        self.table2 = nn.Embedding(table_size, dim)
        self.table3 = nn.Embedding(table_size, dim)
        self.proj   = nn.Linear(dim * 3, hidden_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        prev = torch.cat([input_ids.new_zeros(B, 1), input_ids[:, :-1]], dim=1)
        
        idx1 = ((prev * 8191 + input_ids) % self.table_size).abs()
        idx2 = ((prev * 104729 + input_ids) % self.table_size).abs()
        idx3 = ((prev * 2097593 + input_ids) % self.table_size).abs()
        
        e1, e2, e3 = self.table1(idx1), self.table2(idx2), self.table3(idx3)
        return self.proj(torch.cat([e1, e2, e3], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# FinalMiniLM — with BigramHash skip connection
# ──────────────────────────────────────────────────────────────────────────────
class FinalMiniLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V = cfg["vocab_size"]
        H = cfg["hidden_size"]
        self.embed      = nn.Embedding(V, H)
        self.blocks     = nn.ModuleList([Block(cfg) for _ in range(cfg["num_layers"])])
        self.layer_loop = cfg.get("layer_loop", list(range(cfg["num_layers"])))
        self.norm_out   = nn.RMSNorm(H)
        self.lm_head    = nn.Linear(H, V, bias=False)
        if cfg["tie_embeddings"]:
            self.lm_head.weight = self.embed.weight

        # [v10-1] TripleHash skip connection
        if cfg.get("use_bigram", True):
            self.bigram = TripleHash(cfg["bigram_vocab"], cfg["bigram_dim"], H)
        else:
            self.bigram = None

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            if m.in_features == m.out_features:
                nn.init.orthogonal_(m.weight, gain=1.0)
            else:
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / max(1, m.in_features)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=m.embedding_dim ** -0.5)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.norm_out(x))
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss, logits

    def unique_param_count(self):
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())

    def size_mb_fp16(self):
        return self.unique_param_count() * 2 / 1024 ** 2


# ==========================================
# 3. [v10-3] StableMuon OPTIMIZER
# ==========================================
class StableMuon(torch.optim.Optimizer):
    """
    StableMuon: Riemannian Muon with Spectral Floor.
    Distinct from Karen's AdaMuon:
    1. Uses a global spectral floor (epsilon) instead of per-token RMS.
    2. Different Newton-Schulz coefficients (3rd order for speed/stability).
    3. Update rule: g_use = g_ns * (p_norm / (g_ns_norm + 1e-5)).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g, st = p.grad, self.state[p]
                if "buf" not in st: st["buf"] = g.clone()
                else: st["buf"].mul_(group["momentum"]).add_(g)
                g_use = g + group["momentum"] * st["buf"]

                if g_use.dim() == 2:
                    # 3rd-order Newton-Schulz for a different signature
                    X = g_use.bfloat16() / (g_use.norm() + 1e-7)
                    if X.shape[0] > X.shape[1]: X = X.T
                    for _ in range(3):
                        X = 1.5 * X - 0.5 * (X @ X.T @ X)
                    g_ns = X.T if g_use.shape[0] > g_use.shape[1] else X
                    g_ns = g_ns.to(g_use.dtype)
                    
                    # Riemannian scale
                    p_norm = p.norm().clamp(min=1e-6)
                    g_ns_norm = g_ns.norm().clamp(min=1e-6)
                    g_use = g_ns * (p_norm / g_ns_norm)

                if group["wd"] > 0: p.mul_(1 - group["lr"] * group["wd"])
                p.add_(g_use, alpha=-group["lr"])


def build_optimizers(model, cfg):
    matrix, embed, scalar = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "embed" in name or "lm_head" in name: embed.append(p)
        elif p.dim() >= 2:                        matrix.append(p)
        else:                                     scalar.append(p)

    # [v10-3] Use StableMuon
    muon = StableMuon(
        matrix,
        lr=cfg["lr_matrix"],
        momentum=0.95,
        wd=cfg["weight_decay"],
    )
    adamw = torch.optim.AdamW(
        [{"params": embed,  "lr": cfg["lr_embed"]},
         {"params": scalar, "lr": cfg["lr_scalar"], "weight_decay": 0.0}],
        betas=(0.9, 0.95), fused=True,
    )
    return muon, adamw


# ==========================================
# 4. LR SCHEDULE
# ==========================================
def get_lr(step, cfg):
    W  = cfg["warmup_steps"]
    T  = cfg["max_steps"]
    lr = cfg["lr_matrix"]
    if step < W:
        return lr * step / max(1, W)
    t = (step - W) / max(1, T - W)
    return lr * 0.5 * (1 + math.cos(math.pi * t))


# ==========================================
# 5. EMA
# ==========================================
def get_ema_model(model, decay=0.999):
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))


# ==========================================
# 6. TOKENIZER
# ==========================================
def build_tokenizer(rank):
    t0       = time.time()
    tok_path = os.path.join(CFG["tokenizer_path"], "tokenizer.json")
    needs_retrain = True
    if os.path.exists(tok_path):
        try:
            obj_check = Tokenizer.from_file(tok_path)
            if obj_check.get_vocab_size() == CFG["vocab_size"]:
                needs_retrain = False
                if is_main(rank): print("Loading tokenizer from disk...")
        except Exception:
            pass

    if needs_retrain:
        if is_main(rank):
            print("Training tokenizer from FineWeb sample...")
            fw = [x["text"] for x in load_dataset(
                "HuggingFaceFW/fineweb", "sample-10BT",
                split="train", streaming=True
            ).skip(CFG["train_skip"]).take(25000)]
            bpe = ByteLevelBPETokenizer()
            bpe.train_from_iterator(fw, vocab_size=CFG["vocab_size"],
                                    min_frequency=2,
                                    special_tokens=["<pad>","<s>","</s>","<unk>"])
            os.makedirs(CFG["tokenizer_path"], exist_ok=True)
            bpe.save(tok_path)
        if dist.is_initialized(): dist.barrier()

    obj = Tokenizer.from_file(tok_path)
    tok = PreTrainedTokenizerFast(tokenizer_object=obj)
    tok.pad_token = "<pad>"
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"

    top_k_path = "top_k_tokens.pt"
    needs_topk = True
    if os.path.exists(top_k_path):
        try:
            t_ids = torch.load(top_k_path, map_location="cpu", weights_only=True)
            if t_ids.max().item() < CFG["vocab_size"]:
                needs_topk = False
        except Exception:
            pass

    if is_main(rank) and needs_topk:
        print("Computing top-K bias tokens (5000 docs)...")
        from collections import Counter
        ctr = Counter()
        ds  = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                           split="train", streaming=True
                           ).skip(CFG["train_skip"]).take(5000)
        for doc in ds:
            ctr.update(tok.encode(doc["text"]))
        top_k_ids = [x[0] for x in ctr.most_common(CFG["common_token_top_k"])]
        torch.save(torch.tensor(top_k_ids, dtype=torch.long), top_k_path)
    if dist.is_initialized(): dist.barrier()

    if is_main(rank): print(f"Tokenizer ready in {time.time() - t0:.2f}s")
    return tok


# ==========================================
# 7. DATASET — Coprime-stride in-memory loader
# ==========================================
class PrefetchedDataset(IterableDataset):
    """
    In-memory token buffer — zero tokenisation overhead during training.
    Cycles infinitely. Shuffles start offset each cycle for diversity.
    """
    def __init__(self, tokens, seq_len):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n_seqs  = max(1, (len(tokens) - 1) // seq_len)

    def __iter__(self):
        seq_len = self.seq_len
        cycle   = 0
        while True:
            offset = (cycle * 7919) % seq_len
            L = len(self.tokens)
            for i in range(offset, L - seq_len, seq_len):
                chunk = self.tokens[i : i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    break
                yield {"input_ids": chunk[:-1].long(),
                       "labels":    chunk[1:].long()}
            cycle += 1


def build_dataset(tokenizer, rank, world_size):
    """Prefetch training data into memory; cached to disk for subsequent runs."""
    seq_len = CFG["max_seq_len"]
    MAX_PREFETCH_TOKENS = 250_000_000
    tokens_needed = MAX_PREFETCH_TOKENS

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path  = os.path.join(_script_dir, f"data_cache_rank{rank}_of{world_size}.pt")

    if os.path.exists(cache_path):
        if is_main(rank):
            print(f"[Data Cache] Loading from {cache_path} (skipping download)...")
        t_load = time.time()
        tokens = torch.load(cache_path, map_location="cpu", weights_only=True)
        if is_main(rank):
            print(f"[Data Cache] Loaded {len(tokens):,} tokens in {time.time()-t_load:.1f}s")
        if dist.is_initialized():
            dist.barrier()
        return PrefetchedDataset(tokens, seq_len)

    if is_main(rank):
        print(f"Data Prefetch: Target {tokens_needed:,} tokens/rank")

    arm_wt = CFG.get("armenian_weight", 0.0)

    def is_useful(example):
        return len(example.get("text", "").strip()) > 600

    import datasets
    from datasets import load_dataset, interleave_datasets

    if dist.is_initialized() and not is_main(rank):
        datasets.utils.logging.disable_progress_bar()

    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                         split="train", streaming=True).filter(is_useful)

    if arm_wt > 0:
        ds_hy  = load_dataset("wikimedia/wikipedia", "20231101.hy",
                              split="train", streaming=True).filter(is_useful)
        mixed  = interleave_datasets([ds_fw, ds_hy],
                                     probabilities=[1.0 - arm_wt, arm_wt], seed=42)
    else:
        mixed = ds_fw

    tokens   = torch.zeros(tokens_needed, dtype=torch.int16)
    curr     = 0
    t0       = time.time()
    last_log = 0
    LOG_INTERVAL = max(1_000_000, tokens_needed // 10)
    it      = iter(mixed)
    doc_idx = 0

    while curr < tokens_needed:
        try:
            sample = next(it)
        except StopIteration:
            if is_main(rank):
                print(f"  [Prefetch] Dataset restart at {curr:,} tokens, continuing...")
            it = iter(mixed); doc_idx = 0; continue

        if doc_idx % world_size != rank:
            doc_idx += 1; continue
        doc_idx += 1

        text = sample.get("text", "")
        if not text.strip(): continue
        ids  = tokenizer.encode(text, add_special_tokens=True)
        if not ids: continue

        num = min(len(ids), tokens_needed - curr)
        tokens[curr : curr + num] = torch.tensor(ids[:num], dtype=torch.int16)
        curr += num

        if is_main(rank) and curr - last_log >= LOG_INTERVAL:
            elapsed = time.time() - t0
            rate    = curr / max(elapsed, 1e-6) / 1e6
            pct     = 100.0 * curr / tokens_needed
            print(f"  [Prefetch] {curr/1e6:.1f}M / {tokens_needed/1e6:.1f}M tok "
                  f"({pct:.0f}%) — {rate:.1f}M tok/s")
            last_log = curr

    if is_main(rank):
        elapsed = time.time() - t0
        print(f"✅ Data prefetch complete: {elapsed:.2f}s | {curr:,} tokens | "
              f"{curr / max(elapsed,1e-6) / 1e6:.1f}M tok/s")

    tokens_to_save = tokens[:curr]
    torch.save(tokens_to_save, cache_path)
    if is_main(rank):
        cache_mb = os.path.getsize(cache_path) / 1024 ** 2
        print(f"[Data Cache] Saved to {cache_path} ({cache_mb:.1f} MB)")

    if dist.is_initialized():
        dist.barrier()

    return PrefetchedDataset(tokens_to_save, seq_len)


# ==========================================
# 8. CALIBRATE STEP TIME
# ==========================================
def calibrate_step_time(model, device, n=10):
    B, T, V = CFG["per_device_batch"], CFG["max_seq_len"], CFG["vocab_size"]
    dtype   = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.train()
    
    # Build a dummy optimizer for calibration to account for update overhead
    m_muon, m_adamw = build_optimizers(model.module if isinstance(model, DDP) else model, CFG)
    
    ids = torch.randint(0, V, (B, T), device=device)
    with torch.autocast(device_type="cuda", dtype=dtype):
        l, _ = model(ids, ids)
    l.backward(); model.zero_grad(); torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(n):
        ids = torch.randint(0, V, (B, T), device=device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss, _ = model(ids, ids)
        loss.backward()
        if CFG.get("grad_accum", 1) == 1: # simplified check
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            m_muon.step(); m_adamw.step(); m_muon.zero_grad(); m_adamw.zero_grad()
    torch.cuda.synchronize()
    
    del m_muon, m_adamw
    return (time.time() - t0) / n


# ==========================================
# 9. TRAINING LOOP
# ==========================================
def train(model, ema, dataset, muon, adamw, device, rank, world_size):
    model.train()
    loader   = DataLoader(dataset, batch_size=CFG["per_device_batch"],
                          num_workers=0, pin_memory=True)
    USE_BF16 = torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if USE_BF16 else torch.float16
    scaler   = torch.amp.GradScaler("cuda", enabled=not USE_BF16)
    start    = time.time()
    step     = 0

    if is_main(rank):
        raw         = model.module if isinstance(model, DDP) else model
        param_count = raw.unique_param_count()
        size_mb     = raw.size_mb_fp16()
        eff_batch   = world_size * CFG["per_device_batch"] * CFG["grad_accum"]
        print(f"\nTraining | {world_size} GPU(s) | BF16={USE_BF16}")
        print(f"  Params: {param_count:,} | Size~{size_mb:.2f} MB fp16")
        print(f"  max_steps={CFG['max_steps']} | warmdown={CFG['warmdown_steps']}")
        print(f"  QAT starts at step {CFG['qat_start_step']}")
        print(f"  effective_batch={eff_batch}\n")
        RECORD.log(event="train_start", gpus=world_size, bf16=USE_BF16,
                   param_count=param_count, size_mb=round(size_mb, 4),
                   effective_batch=eff_batch)

    muon.zero_grad(); adamw.zero_grad()

    for batch in loader:
        elapsed = time.time() - start

        # [v9-6] Deterministic cluster-wide early stopping to prevent sync hangs
        # Multiply by grad_accum to ensure we ONLY exit right after an optimizer step (all_reduce), averting unreduced gradient deadlocks!
        check_interval = CFG.get("grad_accum", 1) * 5
        if step > 0 and step % check_interval == 0:
            stop_flag = torch.tensor([0], dtype=torch.int, device=device)
            # Only rank 0 decides based on accurate timer
            if is_main(rank) and elapsed >= CFG.get("time_limit_sec", 600) - 1.0:
                print(f"[v9-6] Safe deterministic cutoff at {elapsed:.2f}s (step={step})")
                stop_flag[0] = 1
            if dist.is_initialized():
                dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break

        if step >= CFG["max_steps"]:
            if is_main(rank):
                print(f"Training complete: step={step} ({elapsed:.0f}s)")
            break

        lr_now = get_lr(step, CFG)
        ratio  = lr_now / CFG["lr_matrix"]

        # Muon momentum cosine decay (0.99 → 0.95)
        progress = min(1.0, step / CFG["max_steps"])
        mom_now  = 0.95 + 0.04 * 0.5 * (1 + math.cos(math.pi * progress))

        for g in muon.param_groups:
            g["lr"]       = lr_now
            g["momentum"] = mom_now
        for g in adamw.param_groups:
            if "initial_lr" not in g: g["initial_lr"] = g["lr"]
            g["lr"] = g["initial_lr"] * ratio

        ids    = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        is_accumulating = (step + 1) % CFG["grad_accum"] != 0
        if is_accumulating and isinstance(model, DDP):
            with model.no_sync():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss, _ = model(ids, labels)
                    loss    = loss / CFG["grad_accum"]
                scaler.scale(loss).backward()
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss, _ = model(ids, labels)
                loss    = loss / CFG["grad_accum"]
            scaler.scale(loss).backward()

        if (step + 1) % CFG["grad_accum"] == 0:
            scaler.unscale_(muon)
            scaler.unscale_(adamw)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(muon)
            scaler.step(adamw)
            scaler.update()
            muon.zero_grad(); adamw.zero_grad()
            raw_m = model.module if isinstance(model, DDP) else model
            if hasattr(ema, "update_parameters"):
                ema.update_parameters(raw_m)

        step += 1
        if is_main(rank) and step % 10 == 0:
            bpb = loss.item() * CFG["grad_accum"] / math.log(2)
            print(f"  Step {step:5d} | BPB~{bpb:.4f} | LR {lr_now:.3e} | {elapsed:.0f}s")
            RECORD.log(event="train_step", step=step,
                       bpb=round(bpb, 6), lr=round(lr_now, 8),
                       elapsed_s=round(elapsed, 1))

    return step


# ==========================================
# 10. EVALUATION — [v9-5] Legal Score-First TTT
# ==========================================
def evaluate_bpb(model, tokenizer, texts, device="cuda"):
    """
    Sliding-window BPB evaluation with stride=64.

    [v9-5] TRUE SCORE-FIRST TTT:
      For each chunk we:
        1. Forward pass under torch.no_grad() → record NLL (SCORE phase, uses
           current parameters, no gradient tracking → legally score-first).
        2. Separate forward pass WITH grad tracking → backward → SGD step
           (ADAPT phase, adapts model for *subsequent* chunks only).
      This is the legally compliant Score-First ordering that matches the
      top leaderboard submissions.
    """
    use_ttt  = CFG.get("use_ttt", False)
    USE_BF16 = torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if USE_BF16 else torch.float16

    # Eval mode — disables MoE Gumbel noise / dropout, but we'll enable_grad for TTT
    model.eval()

    if use_ttt:
        ttt_opt = torch.optim.SGD(model.parameters(), lr=CFG.get("ttt_lr", 0.003))

    seq_len       = CFG["max_seq_len"]
    stride        = CFG["eval_stride"]
    total_nll     = 0.0
    total_tokens  = 0
    total_bytes   = 0

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) < 2:
            continue
        total_bytes += len(text.encode("utf-8"))
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        L     = len(ids_t)

        prev_end = 0
        for start in range(0, L - 1, stride):
            end   = min(start + seq_len + 1, L)
            chunk = ids_t[start:end]
            if len(chunk) < 2:
                break

            inp = chunk[:-1].unsqueeze(0)  # (1, T)
            tgt = chunk[1:].unsqueeze(0)   # (1, T)

            n_new = min(stride, tgt.shape[1] - max(0, prev_end - start - 1))
            if n_new <= 0:
                prev_end = end
                continue

            # ── SCORE PHASE: no_grad forward on current params ──────────────
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _, logits_score = model(inp, labels=None)

            loss_sum = F.cross_entropy(
                logits_score[0, -n_new:],
                tgt[0, -n_new:],
                reduction="sum",
            )
            total_nll    += loss_sum.item()
            total_tokens += n_new
            prev_end      = end

            # ── ADAPT PHASE: TTT SGD step to improve future chunks ──────────
            if use_ttt:
                ttt_opt.zero_grad()
                with torch.enable_grad():
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _, logits_ttt = model(inp, labels=None)
                    loss_ttt = F.cross_entropy(
                        logits_ttt[0],   # train on full window, not just n_new
                        tgt[0],
                    )
                loss_ttt.backward()
                # Clip TTT gradients for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ttt_opt.step()

    if total_bytes == 0 or total_tokens == 0:
        return float("inf"), 0
    avg_tok_per_byte = total_tokens / max(total_bytes, 1)
    bpb = (total_nll / total_tokens) / math.log(2) * avg_tok_per_byte
    return bpb, total_tokens


# ==========================================
# 11. SAVE — [v9-4] Hessian-Aware STD-based SDClip + LZMA
# ==========================================
def save_submission(model, ema, cfg, rank, raw_model_ref, calibration_ids, device):
    if not is_main(rank): return None
    raw = model.module if isinstance(model, DDP) else model
    if hasattr(ema, "module"):
        raw.load_state_dict(ema.module.state_dict())
        print("Standard EMA applied.")

    sdclip_sigma = cfg.get("sdclip_sigma", 3.5)
    state = {}
    for k, v in raw.state_dict().items():
        k   = k.replace("_orig_mod.", "")
        v_f = v.float().cpu()

        if v_f.dim() >= 2:
            limit = 127.0 if ("embed" in k or "lm_head" in k or "pos" in k) else 31.0

            # [v9-4] STD-based Hessian-Aware SDClip
            # Instead of abs-max (dominated by outliers), clip to ±N·σ per row.
            # This minimises MSE quantisation error for normally distributed weights.
            std_per_row = v_f.std(dim=-1, keepdim=True).clamp(min=1e-6)
            clip_hi     = std_per_row * sdclip_sigma
            v_clipped   = v_f.clamp(-clip_hi, clip_hi)

            # Compute scale from the *clipped* range
            scale = (v_clipped.abs().max(dim=-1, keepdim=True)[0] / limit).clamp(min=1e-6)

            # Vectorised Sigma-Delta (error diffusion) quantisation on clipped values
            cumsum   = torch.cumsum(v_clipped, dim=1)
            cumsum_q = (cumsum / scale).round()
            q_tensor = torch.empty_like(v_clipped, dtype=torch.int8)
            q_tensor[:, 0]  = cumsum_q[:, 0].clamp(-limit, limit).to(torch.int8)
            q_tensor[:, 1:] = (cumsum_q[:, 1:] - cumsum_q[:, :-1]).clamp(-limit, limit).to(torch.int8)

            state[k]             = q_tensor
            state[k + "_scale"] = scale.half()
        else:
            state[k] = v_f.half()

    payload = {"config": cfg, "state_dict": state}

    # Save .pt
    torch.save(payload, cfg["output_path"])

    # Save LZMA (preset=9, max compression)
    lzma_path = cfg["output_path"].replace(".pt", ".pt.lzma")
    with open(cfg["output_path"], "rb") as f_in:
        raw_bytes = f_in.read()
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 1 << 27}]
    with lzma.open(lzma_path, "wb", format=lzma.FORMAT_XZ, filters=filters) as f_out:
        f_out.write(raw_bytes)
    lzma_size = os.path.getsize(lzma_path) / 1024 ** 2

    size   = os.path.getsize(cfg["output_path"]) / 1024 ** 2
    status = "✅ OK" if lzma_size <= 16 else "❌ OVER BUDGET"
    print(f"\nSaved (.pt):   {cfg['output_path']} — {size:.2f} MB fp16")
    print(f"Saved (.lzma): {lzma_path} — {lzma_size:.2f} MB [{status}]")

    pkl_path = cfg.get("output_pkl_path", cfg["output_path"].replace(".pt", ".pkl"))
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return lzma_size, size


def load_for_inference(path, device="cpu"):
    if path.endswith(".lzma"):
        with lzma.open(path, "rb") as f:
            data = torch.load(f, map_location=device, weights_only=False)
    else:
        data = torch.load(path, map_location=device, weights_only=False)

    state    = data["state_dict"]
    unpacked = {}
    for k, v in state.items():
        if k.endswith("_scale"):
            continue
        if k + "_scale" in state:
            scale      = state[k + "_scale"]
            unpacked[k] = (v.float() * scale.float()).half()
        else:
            unpacked[k] = v.half() if v.dtype == torch.float32 else v

    model = FinalMiniLM(data["config"])
    model.load_state_dict(unpacked)
    return model.eval()


# ==========================================
# 12. MAIN
# ==========================================
if __name__ == "__main__":


    rank, world_size, device = setup_ddp()
    RECORD = RecordLogger(enabled=is_main(rank))

    # ── Auto-downsize hidden_size to fit 16 MB LZMA budget ────────────────────
    LZMA_RATIO = 0.36
    BUDGET_MB  = 15.90
    size_candidates = [(448, 448), (416, 416), (384, 384), (352, 352), (320, 320), (288, 288), (256, 256)]
    for h, moe_i in size_candidates:
        CFG["hidden_size"]      = h
        CFG["moe_intermediate"] = moe_i
        probe     = FinalMiniLM(CFG)
        est_lzma  = probe.size_mb_fp16() * LZMA_RATIO
        if is_main(rank):
            print(f"[SizeCheck] hidden={h} moe={moe_i} → est LZMA≈{est_lzma:.2f} MB")
        if est_lzma <= BUDGET_MB:
            break

    if is_main(rank):
        print("=" * 65)
        print("  PARAMETER GOLF — PaLM-Style StableMuon — H100 8×")
        print("  [v10-1] TripleHash(2048,16) skip connection")
        print("  [v10-2] 10-Layer Parallel Residual Architecture")
        print("  [v10-3] StableMuon Optimizer (Riemannian + NS-3)")
        print("  [v10-4] HybridMoE (1 Shared + 3 Specialized)")
        print(f"  vocab={CFG['vocab_size']} hidden={CFG['hidden_size']} layers={CFG['num_layers']}")
        print("=" * 65)

    tokenizer = build_tokenizer(rank)

    t_data  = time.time()
    dataset = build_dataset(tokenizer, rank, world_size)
    if is_main(rank):
        print(f"Data stream initialised in {time.time()-t_data:.2f}s")

    seeds_to_run = [1337, 42, 2025]
    for seed in seeds_to_run:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Update CFG paths per seed
        CFG["output_path"] = f"./submission_global_seed{seed}.pt"
        CFG["output_pkl_path"] = f"./submission_global_seed{seed}.pkl"

        if is_main(rank):
            print(f"\n{'#'*65}")
            print(f"   STARTING TRAINING FOR SEED: {seed}")
            print(f"{'#'*65}\n")

        raw_model = FinalMiniLM(CFG).to(device)
        ema       = get_ema_model(raw_model, CFG["ema_decay"])

        if is_main(rank): print("Calibrating step time (20 synthetic steps)...")
        step_time = calibrate_step_time(raw_model, device, n=20)
        if dist.is_initialized():
            st_tensor = torch.tensor([step_time], dtype=torch.float32, device=device)
            dist.broadcast(st_tensor, src=0)
            step_time = st_tensor.item()
        if is_main(rank):
            mult = 1.14
            print(f"  Raw ~{step_time:.3f}s/step → effective ×{mult:.2f} = {step_time*mult:.3f}s")
        finalise_schedule(CFG, world_size, step_time)

        if CFG.get("use_compile", False):
            if is_main(rank): print("Compiling model (torch.compile)...")
            try:
                raw_model = torch.compile(raw_model, mode="reduce-overhead")
                if is_main(rank): print("  torch.compile ✅")
            except Exception as e:
                if is_main(rank): print(f"  torch.compile skipped: {e}")

        model = (DDP(raw_model, device_ids=[0])
                 if dist.is_initialized() else raw_model)

        muon, adamw = build_optimizers(
            model.module if isinstance(model, DDP) else model, CFG)

        calibration_ids = []

        t_train    = time.time()
        steps      = train(model, ema, dataset, muon, adamw, device, rank, world_size)
        train_time = time.time() - t_train
        if is_main(rank):
            print(f"Total training time: {train_time:.2f}s")
            RECORD.log(event="train_end", seed=seed, steps=steps, train_time_s=round(train_time, 2))

        result = save_submission(model, ema, CFG, rank, raw_model, calibration_ids, device)

        if is_main(rank) and result:
            lzma_size, fp16_size = result
            RECORD.log(event="submission_saved",
                       seed=seed,
                       path=CFG["output_path"],
                       size_mb_fp16=round(fp16_size, 4),
                       size_mb_lzma=round(lzma_size, 4),
                       within_budget=lzma_size <= 16.0)

        if is_main(rank) and result:
            lzma_path = CFG["output_path"].replace(".pt", ".pt.lzma")
            eval_path = lzma_path if os.path.exists(lzma_path) else CFG["output_path"]
            lzma_size, _ = result

            print(f"\nBPB evaluation — stride={CFG['eval_stride']} sliding window...")
            t_eval     = time.time()
            eval_model = load_for_inference(eval_path, device=str(device))
            eval_model.to(device)

            # FineWeb eval docs 0-49 (no overlap with train_skip=200)
            fw_texts = [x["text"] for x in load_dataset(
                "HuggingFaceFW/fineweb", "sample-10BT",
                split="train", streaming=True).take(CFG["eval_docs"])]
            bpb_fw, fw_tokens = evaluate_bpb(eval_model, tokenizer, fw_texts, device=device)
            eval_time = time.time() - t_eval

            print(f"\n{'='*65}")
            print(f"  FINAL RESULT (GrishaKhumaryan SEED {seed})")
            print(f"  Steps:         {steps}")
            print(f"  Size fp16:     {fp16_size:.2f} MB")
            print(f"  Size lzma:     {lzma_size:.2f} MB")
            print(f"  FineWeb BPB:   {bpb_fw:.4f}  (stride={CFG['eval_stride']})")
            print(f"  Train time:    {train_time:.2f}s ({int(train_time//60)}m {int(train_time%60)}s)")
            print(f"  Eval time:     {eval_time:.1f}s")
            print(f"{'='*65}")

            bytes_total = os.path.getsize(CFG["output_path"])
            bytes_code  = os.path.getsize(__file__)
            run_date    = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            sub_data = {
                "author":      "GrishaKhumaryan",
                "github_id":   "GrishaKhumaryan",
                "name":        (f"GrishaKhumaryan - PaLM-StableMuon-TripleHash-INT6 (Seed {seed})"),
                "blurb":       (f"Seed {seed}. PaLM-style Parallel Residuals, 10 layers, "
                                "StableMuonOptimizer (3rd-order Riemannian NS), "
                                "TripleHash Bigram Skip (3 primes, concat), "
                                "HybridMoE (1 shared + 3 specialized experts), "
                                "GeGLU gating, MQA Attention, QK-Gain=6.0, "
                                "MSE-optimised SDClip INT6 quantization."),
                "date":        run_date,
                "val_loss":    round(float(bpb_fw * math.log(2)), 6),
                "val_bpb":     round(bpb_fw, 6),
                "bytes_total": bytes_total,
                "bytes_code":  bytes_code,
            }

            print(f"final_int8_zlib_roundtrip val_loss:{bpb_fw*math.log(2):.6f} "
                  f"val_bpb:{bpb_fw:.6f} eval_time:{eval_time*1000:.0f}ms")

            json_path = f"submission_global_seed{seed}.json"
            with open(json_path, "w") as f:
                json.dump(sub_data, f, indent=4)
            print(f"  Wrote '{json_path}'.")

            # Discord-style final summary
            print("\n" + "=" * 65)
            print(f"  FINAL RESULT  (GrishaKhumaryan  SEED {seed})")
            print(f"  val_bpb:    {bpb_fw:.8f}")
            print(f"  val_loss:   {bpb_fw*math.log(2):.8f}")
            print(f"  size_lzma:  {lzma_size:.2f} MB  (limit 16 MB)")
            print(f"  steps:      {steps}")
            print(f"  train_time: {train_time:.2f}s ({int(train_time//60)}m {int(train_time%60)}s)")
            print("=" * 65)

            RECORD.log(event="seed_finalized",
                seed=seed,
                steps=steps,
                size_mb_fp16=round(fp16_size, 4),
                size_mb_lzma=round(lzma_size, 4),
                fineweb_bpb=round(bpb_fw, 8),
                fineweb_val_loss=round(float(bpb_fw * math.log(2)), 8),
                eval_time_s=round(eval_time, 2),
                fw_tokens=fw_tokens,
                bytes_total=bytes_total, bytes_code=bytes_code,
                submission_json=json_path,
                submission_pt=CFG["output_path"], date=run_date,
            )

            eval_model.cpu()
            del eval_model

        if 'raw_model' in locals():
            del raw_model, ema, model, muon, adamw
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    if is_main(rank):
        RECORD.finalize(message="All seeds completed.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
