from __future__ import annotations

import pathlib
import textwrap
import urllib.request

RAW_URL = (
    "https://raw.githubusercontent.com/openai/parameter-golf/main/records/track_10min_16mb/"
    "2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
)
BASE_LOCAL_FILE = pathlib.Path("_base_record_train_gpt.py")
OUT_DIR = pathlib.Path("records/track_10min_16mb/2026-03-25_CAGE5_NGramStarter")
OUT_FILE = OUT_DIR / "train_gpt.py"
README_FILE = OUT_DIR / "README.md"
REQ_FILE = OUT_DIR / "requirements.txt"
SUBMISSION_FILE = OUT_DIR / "submission.json"

HP_INSERT = textwrap.dedent(
    '''
    ngram_eval_enabled = bool(int(os.environ.get("NGRAM_EVAL_ENABLED", "0")))
    ngram_eval_order = int(os.environ.get("NGRAM_EVAL_ORDER", 5))
    ngram_eval_alpha = float(os.environ.get("NGRAM_EVAL_ALPHA", 0.20))
    ngram_eval_min_count = int(os.environ.get("NGRAM_EVAL_MIN_COUNT", 2))
    ngram_eval_backoff = float(os.environ.get("NGRAM_EVAL_BACKOFF", 8.0))
    ngram_eval_smoothing = float(os.environ.get("NGRAM_EVAL_SMOOTHING", 0.02))
    '''
).strip("\n")

HELPERS = textwrap.dedent(
    '''

# --- CAGE-5 hashed causal n-gram helpers ---
_HASH_MASK = (1 << 64) - 1
_HASH_OFFSET = 1469598103934665603
_HASH_PRIME = 1099511628211
_HASH_MIX = 0x9E3779B185EBCA87


def _hash_token_sequence(tokens: list[int]) -> int:
    h = _HASH_OFFSET
    for tok in tokens:
        h ^= int(tok) + 1
        h = (h * _HASH_PRIME) & _HASH_MASK
    return h


def _hash_pair(ctx_hash: int, token_id: int) -> int:
    x = ((int(token_id) + 1) * _HASH_MIX) & _HASH_MASK
    return ((ctx_hash ^ x) * _HASH_PRIME) & _HASH_MASK


class HashedNGramCache:
    def __init__(self, order: int, vocab_size: int, min_count: int, backoff: float, smoothing: float):
        self.order = max(1, int(order))
        self.vocab_size = max(1, int(vocab_size))
        self.min_count = max(1, int(min_count))
        self.backoff = float(backoff)
        self.smoothing = float(smoothing)
        self.total_seen = 0
        self.unigram_counts: dict[int, int] = {}
        self.context_totals: dict[int, int] = {}
        self.context_target_counts: dict[int, int] = {}
        self.history: list[int] = []

    def prob(self, target: int) -> float:
        # Base unigram estimate with additive smoothing.
        denom = self.total_seen + self.smoothing * self.vocab_size
        prob = (self.unigram_counts.get(int(target), 0) + self.smoothing) / max(denom, 1e-12)
        max_ctx = min(self.order - 1, len(self.history))
        for ctx_len in range(1, max_ctx + 1):
            ctx = self.history[-ctx_len:]
            ctx_hash = _hash_token_sequence(ctx)
            total = self.context_totals.get(ctx_hash, 0)
            if total < self.min_count:
                continue
            hit = self.context_target_counts.get(_hash_pair(ctx_hash, int(target)), 0)
            lam = total / (total + self.backoff)
            mle = hit / total if total > 0 else 0.0
            prob = lam * mle + (1.0 - lam) * prob
        return float(min(max(prob, 1e-12), 1.0))

    def update(self, target: int) -> None:
        target = int(target)
        self.unigram_counts[target] = self.unigram_counts.get(target, 0) + 1
        max_ctx = min(self.order - 1, len(self.history))
        for ctx_len in range(1, max_ctx + 1):
            ctx = self.history[-ctx_len:]
            ctx_hash = _hash_token_sequence(ctx)
            self.context_totals[ctx_hash] = self.context_totals.get(ctx_hash, 0) + 1
            joint_key = _hash_pair(ctx_hash, target)
            self.context_target_counts[joint_key] = self.context_target_counts.get(joint_key, 0) + 1
        self.history.append(target)
        self.total_seen += 1
'''
)

EVAL_REPLACEMENT = textwrap.dedent(
    '''
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context.

    Optional CAGE-5 path: interpolate the neural model with a strictly causal
    hashed n-gram cache that only updates on already-scored validation tokens.
    """
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    use_ngram = bool(args.ngram_eval_enabled)
    cache = (
        HashedNGramCache(
            order=args.ngram_eval_order,
            vocab_size=args.vocab_size,
            min_count=args.ngram_eval_min_count,
            backoff=args.ngram_eval_backoff,
            smoothing=args.ngram_eval_smoothing,
        )
        if use_ngram else None
    )
    alpha = float(args.ngram_eval_alpha)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_targets = y_batch[i, s:wlen]
                scored_prev = x_batch[i, s:wlen]
                scored_lp = log_probs[i, s:wlen]
                if not use_ngram:
                    token_lp = scored_lp.gather(1, scored_targets.unsqueeze(1)).squeeze(1).to(torch.float64)
                    loss_sum -= token_lp.sum()
                else:
                    assert cache is not None
                    for j in range(wlen - s):
                        target = int(scored_targets[j].item())
                        lm_logp = float(scored_lp[j, target].item())
                        p_lm = math.exp(max(lm_logp, -60.0))
                        p_cache = cache.prob(target)
                        p_mix = (1.0 - alpha) * p_lm + alpha * p_cache
                        loss_sum += -math.log(max(p_mix, 1e-12))
                        cache.update(target)
                token_count += float(wlen - s)
                tgt = scored_targets
                prev = scored_prev
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
'''
)

README = textwrap.dedent(
    '''# CAGE5 NGram Starter

This is a starter fork of the accepted LeakyReLU² + Legal TTT + Parallel Muon record,
with one extra idea added for development: a strictly causal hashed n-gram interpolation
inside sliding-window evaluation.

It is intended for local experimentation and first cloud runs, not as a final verified record.
'''
)

SUBMISSION = textwrap.dedent(
    '''{
  "name": "YOUR_NAME",
  "github": "YOUR_GITHUB",
  "val_bpb": 0.0,
  "notes": "starter record folder generated locally; do not submit before filling logs and metadata"
}
'''
)


def main() -> None:
    if BASE_LOCAL_FILE.exists():
        text = BASE_LOCAL_FILE.read_text(encoding="utf-8")
    else:
        text = urllib.request.urlopen(RAW_URL).read().decode("utf-8")

    marker = 'ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))'
    if marker not in text:
        raise RuntimeError("Could not find Hyperparameters insertion marker")
    text = text.replace(marker, marker + "\n    " + HP_INSERT.replace("\n", "\n    "))

    fn_marker = '# --- Sliding window evaluation ---\n'
    if fn_marker not in text:
        raise RuntimeError("Could not find helper insertion marker")
    text = text.replace(fn_marker, HELPERS + "\n" + fn_marker, 1)

    start = text.find('def eval_val_sliding(')
    end = text.find('\ndef eval_val_sliding_ttt(', start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate eval_val_sliding block")
    text = text[:start] + EVAL_REPLACEMENT + text[end + 1:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(text, encoding="utf-8")
    README_FILE.write_text(README, encoding="utf-8")
    REQ_FILE.write_text("zstandard\n", encoding="utf-8")
    SUBMISSION_FILE.write_text(SUBMISSION, encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
