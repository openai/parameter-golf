# Record: PR #1854 neural stack, budget-compliant 3-seed reproduction — val_bpb 1.06777 (3-seed mean)

**val_bpb: 1.06777** (3-seed mean, std 0.00106) | **15,951,074 bytes** (mean) | 8×H100 SXM, ≤600s train / ≤600s eval

This submission is a **3-seed validated, eval-budget-compliant reproduction** of the neural-stack portion of [PR #1854](https://github.com/openai/parameter-golf/pull/1854) (ndokutovich), with `PHASED_TTT_PREFIX_DOCS` reduced from 2000 → 1500 to fit the 600s evaluation budget cleanly. **Beats merged-leaderboard SOTA PR #1493 (bigbag, 1.0810) by 0.01323 BPB** at ~13σ statistical significance.

The reported `val_bpb` is the **standard token-level NLL → byte conversion** (`val_loss / log(2) · tokens/bytes`) for the post-quantization, post-Phased-TTT model. **No byte-level mixture is claimed** — see "Note on byte-PPM mixture" below.

## Result (3 seeds, 8×H100 80GB SXM)

| Seed | val_bpb | Total bytes | Eval time |
|------|------:|------:|------:|
| 42   | 1.06686 | 15,952,086 | 374.6s |
| 1337 | 1.06893 | 15,949,941 | 371.0s |
| 314  | 1.06752 | 15,951,195 | 327.7s |
| **Mean** | **1.06777** | **15,951,074** | **357.8s** |
| **Std**  | **0.00106** | | |

- **vs merged leaderboard PR #1493 @bigbag (1.0810)**: **−0.01323 BPB** (12.5× std clearance over the 0.005-nat threshold; p ≪ 0.0001)
- All 3 artifacts under 16,000,000 bytes (max 15,952,086, margin 47,914)
- All 3 eval times under 600s wallclock (max 374.6s, margin 225.4s)

## What's new vs PR #1854

PR #1854's reported eval wallclock is **~700s** (per its own log breakdown: ttt_phased 516s + ppm_mix 116s + diagnostics 67s), which is over the 600s evaluation budget. This submission demonstrates that the same neural stack achieves **identical post-TTT val_bpb (~1.067)** while fitting cleanly under the 600s eval budget by reducing `PHASED_TTT_PREFIX_DOCS` from 2000 to 1500. Statistical evidence: 3-seed std of 0.00106 confirms the val_bpb is stable at this trim.

This decoupling matters because the 600s eval budget is an explicit contest constraint (closed PRs in Issue #677 cite eval over-budget as grounds for rejection — see PR #503 closure). A budget-compliant 1.067 record is a more defensible record candidate than a slightly lower over-budget one.

## Note on byte-PPM mixture (not claimed)

`train_gpt.py` includes our exploratory multibin-λ refinement of the byte-level PPM-D mixer (a 4-tier graduated gate: `[(0.95, 0.02), (0.85, 0.10), (0.75, 0.40), (0.0, 1.0)]`). When run with `PPM_ENABLED=1`, it produces `mix_bpb ≈ 0.861` on the val byte stream — a 0.04 BPB delta below the `quantized_ttt_phased val_bpb`.

We **do not claim `mix_bpb`** in this submission. The byte-PPM mixture relies on a per-byte spreading approximation `per_byte_logp = token_logp / n_bytes` whose normalization properties over the 256-byte alphabet are an open community question. Specifically: under the assumption that the NN's byte-level distribution is the geometric mean of the token's per-byte spread, the convex combination with PPM-D's normalized byte distribution is not unambiguously a Kraft-compliant prefix code. Until that interpretation is settled by the maintainers, we report only the standard `val_bpb` derived from token-level NLL, which has no such ambiguity.

The multibin mixer code is left in `train_gpt.py` so this submission is a single self-contained reproducible artifact. Setting `PPM_ENABLED=0` in the reproduction command produces only the standard `val_bpb` line and skips the mixer entirely.

## Issue #1017 four-condition compliance (for the standard val_bpb path)

| Condition | How this submission satisfies it |
|---|---|
| **C1 Causality** | Standard sliding-window eval; each token scored from prefix only. Phased TTT is score-first per PR #1413's protocol. |
| **C2 Normalized** | The model's softmax over the SP8192 token vocabulary is a proper normalized distribution. The reported `val_bpb` is `(total token NLL) / log(2) / total bytes`, the standard token-level codelength normalized by byte count. |
| **C3 Score-before-update** | Phased TTT scores each chunk under `torch.no_grad()` before any optimizer step. |
| **C4 Single pass** | One left-to-right pass; no rescoring or oracle selection. |

Additional compliance:
- **No SLOT, no token-level n-gram cache, no logit bias.** Inherited from PR #1854's stack.
- **No pre-quant TTT on val data.** Score-first phased TTT only (PR #1413 lineage).
- **No external network access** at eval time. Tokenizer unchanged from PR #1854's CaseOps SP8192.

## Normalization proof (Eppie / mhuen / abaybektursun litmus)

Issue #677 surfaced a class of invalid submissions where (a) the predictive distribution did not sum to 1 over the full token vocabulary Σ, or (b) the byte denominator was inflated by tokenizer encoding artifacts. The headline `val_bpb` reported here is clean on both axes. The explicit checks:

**(1) Full-vocab normalization over Σ.** The headline number is produced by the standard token-level path in `eval_val()` and the score-first phased TTT path. Per token position `t`:

```python
per_token_loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none")
val_loss_sum  += per_token_loss.to(torch.float64).sum()
```

`F.cross_entropy` is `−log softmax(logits)[y]` over the full V=8192 vocabulary. The softmax distribution sums to 1 by construction. There is no "score the correct token only" shortcut, no hash-bucket-only normalization, no eval-built n-gram cache contributing to the headline metric, and no `np.where(match, blend, p_model)` of the kind Eppie flagged in #913 / #933 / #944. The `_ppm_mixture_bpb` function is in the file but its output (`mix_bpb`) is **not** the reported number — see "Note on byte-PPM mixture" above.

**(2) Byte denominator equals original UTF-8 bytes.** The CaseOps tokenizer transforms text by inserting private-use sentinels `..` (each 3 UTF-8 bytes) that mark capitalization. Naïve `len(piece.encode("utf-8"))` would charge those sentinel bytes and inflate the denominator, lowering `val_bpb` artificially — the same class of bug that closed PR #1184 (Scylla, "byte accounting bug in candidate.meta.npz", per author's own closing comment).

This submission **bypasses** the inflated path. With `CASEOPS_ENABLED=1`, `eval_val` reads per-token original-byte counts from a precomputed sidecar (`fineweb_val_bytes_*.bin`, identical shard layout to `fineweb_val_*.bin`):

```python
# train_gpt.py:2618-2626
if val_data.caseops_enabled and val_data.val_bytes is not None:
    sidecar_slice = val_data.val_bytes[raw_start + 1 : raw_end].to(...)
    val_byte_count += sidecar_slice.to(torch.float64).sum()
```

The sidecar is built in `prepare_caseops_data.py:_byte_counts`. That function:

1. iterates the transformed string char by char,
2. **skips** the four case-op sentinels,
3. accumulates UTF-8 byte length only of non-sentinel characters into a prefix sum `prefix[]`,
4. attributes `prefix[end] − prefix[start]` original bytes to each token piece's surface span.

By telescoping, `Σ_i sidecar[i] = prefix[len(transformed)] = Σ_(non-sentinel char) utf8_bytes(char) = len(original_text.encode("utf-8"))`. BOS tokens are assigned 0 bytes (correct: they correspond to no original characters). The reported `val_bpb` is therefore `total_NLL_nats / (Σ_i sidecar[i] · ln 2)`, where the denominator is the actual UTF-8 byte length of the original validation text — not the inflated transformed-text byte length.

**(3) Eppie's reconstruction litmus.** The 16 MB artifact (compressed model + `train_gpt.py` + `lossless_caps.py` + `prepare_caseops_data.py` + tokenizer model) plus the published validation token stream can reconstruct the original validation text byte-for-byte: `decode_lossless_caps_v2(sp.decode(val_tokens))` is the exact inverse of the `prepare_caseops_data.py` pipeline. The reported BPB therefore corresponds to a real arithmetic-coding rate over the original UTF-8 stream, not over a transformed/inflated stream and not over an under-normalized auxiliary distribution.

**(4) One-line reviewer reproduction.** To verify the byte denominator independently, after running data prep:

```python
import glob, numpy as np
def _shard(f):
    n = int(np.fromfile(f, dtype=np.int32, count=256)[2])
    return np.fromfile(f, dtype=np.uint16, count=n, offset=256*4)
total_sidecar = sum(int(_shard(f).sum()) for f in sorted(glob.glob("data/.../fineweb_val_bytes_*.bin")))
# total_sidecar should equal len(original_validation_text.encode("utf-8"))
```

## Compute & artifact compliance

| Item | Value | Limit | Margin |
|---|--:|--:|--:|
| Training wallclock | 600s (cap-bound, all 3 seeds) | 600s | 0 (by cap) |
| Evaluation wallclock (max seed) | 374.6s | 600s | 225.4s |
| Artifact total bytes (max seed) | 15,952,086 | 16,000,000 | 47,914 |
| Code (uncompressed) | 161,565 | — | — |
| Code (pyminify + lzma) | 33,305 | — | — |
| Quantized model (int6 + brotli) | 15,918,781 | — | — |

## Lineage and credits

- **PR #549** (@abaybektursun) — score-first TTT framework
- **PR #1394** (@clarkkev) — SP8192 + multi-phase score-first TTT + GPTQ embeddings + SDClip
- **PR #1413** (@dexhunter) — legal score-first TTT with QK-Gain
- **PR #1493** (@bigbag) — 3-layer recurrence + parallel residuals + QK-Gain 5.25 (current merged leaderboard)
- **PR #1729** (@romeerp) — CaseOps bijective case transform
- **PR #1787** (@nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE
- **PR #1797** (@dexhunter) — Smear gate + LQER asymmetric (this submission's neural base)
- **PR #1854** (@ndokutovich) — PR #1797 base + PR #1835 PPM-D port (this submission's direct predecessor's neural stack)
- **PR #1835** (@anmarhindi) — original byte-level PPM-D mixture (we include a multibin-gate refinement in code, but do not claim its score)

This submission's contribution is twofold:
1. **Eval-budget-compliant 3-seed reproduction** of PR #1854's neural stack (val_bpb 1.06777 mean, std 0.00106) with `PHASED_TTT_PREFIX_DOCS=1500`, fitting cleanly under the 600s eval cap.
2. **Multibin-λ refinement** of the PR #1835 PPM-D mixer (included in code, runs at `PPM_ENABLED=1`). We document its measured `mix_bpb` of ~0.861 on the val byte stream but do not claim it as the headline `val_bpb` due to the byte-spread normalization question.

## Reproduction

### Data prep (run once, ~30 min on CPU pod)

```bash
python3 -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download(repo_id='willdepueoai/parameter-golf', \
    filename='datasets/docs_selected.jsonl', \
    repo_type='dataset', local_dir='./hf_cache')"

python3 prepare_caseops_data.py \
    --docs ./hf_cache/datasets/docs_selected.jsonl \
    --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    --workers 16 --max-docs 5000000
```

### 3-seed training + eval (~$54 on RunPod 8×H100 SXM)

```bash
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

for SEED in 42 1337 314; do
  CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=1500 PHASED_TTT_NUM_PHASES=3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 MIN_LR=0.1 \
  FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  TTT_WARM_START_A=1 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  PPM_ENABLED=1 PPM_ORDER=5 PPM_SUBSET_TOKENS=4000000 \
  DATA_PATH="$DATA_PATH" TOKENIZER_PATH="$TOKENIZER_PATH" \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > train_seed${SEED}.log 2>&1
done
```

The headline `val_bpb` for each run is logged as the `quantized_ttt_phased val_bpb:` field. The same logs also include the exploratory `mix_bpb` from the multibin mixer; that is not claimed.

To reproduce **only** the headline `val_bpb` (skipping the mixer entirely), set `PPM_ENABLED=0` in the env block above.

## Files

- `README.md` — this file
- `submission.json` — metadata
- `train_gpt.py` — PR #1854's neural stack with the multibin mixer addition (claimed `val_bpb` is the standard, mixer-independent path)
- `lossless_caps.py` — verbatim from PR #1854
- `prepare_caseops_data.py` — verbatim from PR #1854
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — verbatim
- `train_seed{42,1337,314}.log` — per-seed train+eval logs
- `final_model.int6.ptz` — quantized model artifact (best seed)
