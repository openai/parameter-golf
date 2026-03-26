# PROTEUS+STYX: LeakyReLU(0.9)² + 5-gram Eval Cache

**val_bpb:** 0.8495 (3-seed mean, std 0.0013)
**Improvement over merged SOTA (#549):** -0.270 BPB

## Architecture

PR #549 base stack with two modifications:

1. **LeakyReLU(0.9)²** — `F.leaky_relu(x, 0.9).square()` replacing the standard 0.5 slope. Based on our 7-point monotonic sweep (0.1–0.9) showing higher slope = lower BPB at this model scale.

2. **Backward-looking 5-gram eval cache** — numpy hash table (4M buckets) built from already-scored tokens during sliding window eval. Fixed-alpha blending: `p_final = 0.8 * p_model + 0.2 * p_cache`. No safety gate, no target-aware selection, no training data access.

| Parameter | Value |
|-----------|-------|
| Layers | 11 |
| Dimension | 512 |
| Heads | 8 (4 KV, GQA) |
| MLP | 3x (1536) |
| Activation | LeakyReLU(0.9)² |
| Vocab | 1024 BPE, tied embeddings |
| Quantization | Mixed INT6/INT8 + LZMA |
| Cache | 5-gram, 4M buckets, alpha=0.2 |
| Eval stride | 64, seq_len=2048 |

## Results (8×H100 SXM, RunPod)

### Current Seeds (v1.1 — sliding window fix + script cleanup)

| Seed | val_bpb | Artifact Size | Cache Hit Rate |
|------|---------|---------------|----------------|
| 42 | 0.8494 | 15,921,591 bytes | 98.2% |
| 1337 | 0.8482 | 15,919,103 bytes | 98.2% |
| 2024 | 0.8508 | 15,905,947 bytes | 98.2% |
| **Mean** | **0.8495** | | **std: 0.0013** |

Training loop exit controlled by `MAX_WALLCLOCK_SECONDS=600`. Logged wallclock includes `torch.cuda.synchronize()` overhead (~60-120ms beyond the 600s check).

<details>
<summary>Superseded Seeds (v1.0)</summary>

We're showing the original v1.0 results for full transparency. They had two issues we caught in self-review: a seed 42 artifact that exceeded the 16MB cap, and a sliding window eval that never executed due to a double `torch.compile` invocation. Rather than quietly replace them, we're documenting what went wrong and why.

| Seed | val_bpb | Artifact Size | Note |
|------|---------|---------------|------|
| 42 | 0.8513 | 16,025,731 bytes | Over 16MB cap |
| 1337 | 0.8502 | 15,939,991 bytes | |
| 2024 | 0.8510 | 15,910,119 bytes | |
| **Mean** | **0.8508** | | **std: 0.0006** |

These scores were from the int6 roundtrip eval path (non-sliding). The sliding window + n-gram cache eval path crashed silently under `torchrun`. Fixed in v1.1.
</details>

## Verification: Not an Overlap Artifact

| Stride | BPB | Hit Rate | Overlap |
|--------|-----|----------|---------|
| 64 (standard) | 0.8494 | 98.2% | 97% |
| 2048 (zero overlap) | 0.8709 | 97.9% | 0% |
| No cache | 1.1477 | — | — |

The 0.02 BPB gap between stride=64 and stride=2048 is the overlap contribution. The remaining 0.26 BPB improvement is genuine cache benefit from backward-looking n-gram statistics.

## Rule Compliance Checklist

- [x] **Artifact ≤ 16,000,000 bytes** — All 3 seeds: 15.91–15.92 MB (78–94 KB headroom)
- [x] **Training ≤ 10 min on 8×H100 SXM** — 600s wallclock, ~6800 steps
- [x] **Evaluation ≤ 10 min on 8×H100 SXM** — Sliding window eval completes in ~371s
- [x] **No training data access during evaluation** — Eval paths use `val_tokens` only
- [x] **No training on validation data** — Mid-training val checks are inference-only (`model.eval()` + `torch.no_grad()`)
- [x] **N-gram cache is backward-looking** — Cache updated AFTER scoring each window
- [x] **No oracle/hindsight selection** — Fixed alpha (0.2), no min(NLL) comparison, no target-dependent gating
- [x] **No external downloads or network calls during eval** — Self-contained artifact
- [x] **3 seeds with tight std** — std 0.0013 across seeds 42, 1337, 2024
- [x] **Cross-model peer review** — Independent audit by GPT Codex (gpt-5.4) verified compliance, cache ordering, and artifact sizes against competition rules

### Note on N-gram Cache Legality

The competition [README](https://github.com/openai/parameter-golf/blob/main/README.md) does not address n-gram eval caches. No rule in the official documentation prohibits or permits this technique. The README states: "TTT only on tokens already graded" — our cache satisfies this: it is updated only with already-scored tokens. We note that 15+ concurrent PRs (#779, #797, #795, #786, #796, #798, #800, #806, among others) employ the same backward-looking n-gram cache concept.

## How the Cache Works

```python
ctx_table = np.zeros(4_194_304, dtype=np.uint32)
full_table = np.zeros(4_194_304, dtype=np.uint32)

# Per-token: look up 4-token context, blend if found
if ctx_table[ctx_hash] >= 2:
    p_ngram = min(full_table[full_hash], ctx_table[ctx_hash]) / ctx_table[ctx_hash]
    p_final = 0.8 * p_model + 0.2 * p_ngram

# After scoring window: update tables with scored tokens
```

## Related Work

The n-gram eval cache concept has seen significant community adoption since our [initial analysis on Issue #140](https://github.com/openai/parameter-golf/issues/140#issuecomment-4129882814):

- PR #659 (@deanbrr) — First n-gram cache submission; ruled invalid for oracle min(NLL) gate, not for the cache concept
- PR #779 (@deanbrr) — BackoffNgramMixer + Drift-Free TTT (0.6683 BPB)
- PR #778 (@raahilshah) — Multi-order backoff with fixed and entropy-adaptive alpha
- PR #797 (@armantsaturian) — 7-gram cache (0.8960 BPB)
- PR #795 (@hypery11) — Order-adaptive 11-gram (0.8881 BPB)
- PR #786 (@shinegami-2002) — Classical compression + n-gram backoff (0.8128 BPB)
- PR #796 (@Robby955) — Prefill cache + 7-gram entropy-adaptive (0.6567 BPB)
- PR #798 (@travispchen) — Order-adaptive entropy gating (0.5466 BPB)
- PR #800 (@newjordan) — Shared n-gram tables + Cubric (0.5644 BPB)
- PR #806 (@ibarrajo) — Backoff n-gram + LeakyReLU(0.9)² (0.6678 BPB)

Our LeakyReLU(0.9)² slope sweep was independently cited by PR #764 (@ndokutovich).

## Logs

### v1.1 (current)
- `log_seed42_v1.1.txt`
- `log_seed1337_v1.1.txt`
- `log_seed2024_v1.1.txt`

### v1.0 (superseded)
- `log_seed42_v1.0.txt`
- `log_seed1337_v1.0.txt`
- `log_seed2024_v1.0.txt`
- `verify_stride2048.log`

## Docker

`matotezitanka/proteus-pytorch:2.11.0-cuda12.8`

## Verification

This submission was independently audited by [OpenAI Codex CLI](https://github.com/openai/codex) (gpt-5.4) as a cross-model peer reviewer — verifying rule compliance, cache ordering, artifact sizes, and training logs against competition rules. Both Claude Code (Anthropic) and Codex (OpenAI) were used throughout development: Claude Code for architecture, implementation, and competition analysis; Codex for independent verification and audit.

Built with [PROTEUS+STYX](https://lightspeedup.com) by Light Speed Up
