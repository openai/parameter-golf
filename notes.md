# Parameter Golf — Rules and Key Facts

Source: `projects/parameter-golf/parameter-golf-upstream/README.md` + `data/README.md` + tokenizer specs. Distilled 2026-04-15.

## The game in one paragraph

Train the language model with the **lowest bits-per-byte on FineWeb validation**, subject to a **16 MB total artifact size** (code + compressed weights) and a **10-minute training budget on 8×H100 SXM**. This is explicitly the **L(N)** corner of scaling laws: minimize loss given a fixed parameter budget, unconstrained by data, steps, or architecture. Inspired by (and built on code from) `modded-nanogpt`'s speedrun. Challenge window **March 18 → April 30, 2026**.

## Hard rules (the constraints)

| Constraint | Value | Notes |
|---|---|---|
| Artifact size | **≤ 16,000,000 bytes (decimal)** | Not 16 MiB. Counted as `code bytes + compressed model bytes`. All code must live in `train_gpt.py`. |
| Training time | **≤ 10 minutes on 8×H100 SXM** | Specific hardware — SXM variant, not PCIe. For *record* submissions only. |
| Eval time | **≤ 10 minutes on 8×H100** | **Additional to the 10-min training budget** — eval doesn't have to fit inside training time. |
| Network / data access during eval | Disallowed | Artifact must be fully self-contained and reproducible. |
| Access to validation data during training | Disallowed | Classical "don't train on the test set." |
| Test-time training | **Allowed, but restricted** | You may TTT only on validation tokens you've **already evaluated on**. Those tokens have already been graded, so using them for state updates afterward is fair. Using unseen val tokens is cheating. |
| Python imports | Free | Any library (FlashAttention, etc.) is fine; import cost isn't counted in the 16 MB. Caveat: can't smuggle compute or capability via a custom library. |
| Evaluation sequence length | Free | Can eval at any seq length, independent of training. |
| Training sequence length | Free | Same. |
| External offline hyperparameter tuning | Judgment-based | Normal Adam tuning is fine. Brute-forcing seeds or clearly sneaking in extra compute can be disqualified. |

## Metric: `val_bpb` on FineWeb val

- **Dataset**: FineWeb (a filtered Common Crawl subset, published by HuggingFace).
- **Validation split**: fixed "first 50,000 documents" of `fineweb_val_*`. Always downloaded in full.
- **Metric**: **bits per byte** (bpb) on that fixed validation. Tokenizer-agnostic — a model with more tokens/byte can't cheat via a tinier vocab, because bpb re-normalizes.
- Rough math: `bpb = (loss_nats_per_token × tokens_per_byte) / ln(2)`.
- **Baseline target**: ~1.2 `val_bpb` after the 10-min reference run on the default sp1024 tokenizer.

## Dataset

- FineWeb cached via `python3 data/cached_challenge_fineweb.py --variant sp1024`
- Defaults: full val split + 80 train shards (~100M tokens each = ~8B train tokens).
- Tokenizer variants: the repo ships only `sp_bpe_1024` (1024-vocab SentencePiece BPE). Top leaderboard entries train their own **sp4096** and **sp8192** tokenizers from the published docs cache via `data/download_hf_docs_and_tokenize.py`. The docs set is frozen (sidecar `docs_sha256` for verification), so trained tokenizers are reproducible.
- Default published repo: `willdepueoai/parameter-golf`.

## Submission process (record track)

A PR that **only adds a new folder** to `/records/track_10min_16mb/` containing:
1. `README.md` explaining the submission.
2. `submission.json` with name, GitHub ID, `val_bpb`, metadata.
3. `train_gpt.py` + any dependencies (must run successfully from inside the records folder).
4. **Training logs demonstrating statistical significance.** Typically ≥3 runs averaged.

**SOTA threshold**:
1. Must beat prior SOTA by **≥ 0.005 nats**.
2. Must show at **p < 0.01** that this improvement is real (hence ≥3 seeds).
3. Must reproducibly run in ≤ 10 min on 8×H100 SXM.
4. Systems-only optimizations (same ML, faster) don't need the 0.005-nat improvement — speed wins are accepted without it.

**Special rule on tokenizer changes**: "Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score." So any tokenizer change needs airtight val_bpb computation.

## Non-record tracks (where creative / in-progress work goes)

- `/records/track_non_record_16mb/` — for ideas that beat baseline (or even just run cleanly within 16 MB) but aren't SOTA. **Strongly encouraged.** This is our target given "learn deeply, submit if ready."
- **Unlimited-compute track** — for runs that blow past the 10-min cap (e.g., the 4-hour baseline, the 1-bit 2-hour quantization demo). Valuable to see creative ideas taken further.

## Baseline config (`train_gpt.py` default)

```
9 layers, 512 embed dim, 1024 vocab (SP1024), tied embeddings, 4 KV heads
→ val_bpb ≈ 1.2244 after 10 min on 1×H100 or 8×H100
```

This is the reference point. Beating 1.2244 puts you in the running. Current SOTA (2026-04-09) is **1.0810**, a ~0.14 nat gap — roughly 14% relative improvement via stacked innovations.

## What's winning at the top (leaderboard 2026-04-09)

Observable themes in the top ~10 record entries, ordered by frequency:

| Technique | Appears in | Brief |
|---|---|---|
| **SP8192 tokenizer** | Top 6 entries | 8× larger vocab than baseline. Fewer tokens/byte → lower bpb for same per-token loss. |
| **Parallel residuals** | Top 5 | `x = x + attn(LN(x)) + mlp(LN(x))` — attention and MLP branches run simultaneously on same LN(x). Faster on GPU. |
| **Depth recurrence / layer tying** | Top 5 | Specifically looping layers 4-5 multiple times. Re-uses weights → more compute-per-param. |
| **Muon / MuonEq-R optimizer** | Top 5 | Non-AdamW optimizer, popular in modded-nanogpt. Faster convergence in fixed wallclock. |
| **Legal score-first TTT** | Top 4 | Test-time training on already-evaluated val tokens. "Legal" = compliant with the "only post-grading" rule. |
| **QK-Gain 5.0 (or 5.25)** | Top 3 | Attention scaling tweak — multiply QK product by a larger learned gain to sharpen attention patterns. |
| **GPTQ embeddings / INT6 mixed precision** | Top 3 | Quantize embeddings (GPTQ) and/or all weights to INT6. Major byte savings. |
| **Hessian-aware SDClip** | Top 1 | Gradient clipping variant using Hessian information. |

General recipe of a competitive entry = **custom tokenizer (SP8192) + parallel-residual blocks + depth recurrence + aggressive quantization (GPTQ/INT6) + non-standard optimizer (Muon variant) + optionally TTT**.

## Ideas OpenAI explicitly wants to see

From the "Requests for PRs" section (already checked off = someone has done it):

- [x] 1-bit quantization (Ciprian-Florin Ifrim, 106M params → 1.1239 on unlimited-compute track)
- [x] Ternary quantization (Ciprian-Florin Ifrim, 73.7M → 1.1570 in record track)
- [ ] JEPA
- [ ] Text diffusion
- [ ] H-net tokenization
- [ ] Universal Transformer (4-hour variant specifically — Redundant with depth-recurrence entries but wanted at longer training)
- [ ] Megakernels
- [ ] State-space models / E2E TTT / super-long context
- [ ] Learning adapters on random linear maps

**Implication for us**: unchecked items are likely high-signal for the non-record track. A clean JEPA-for-LM or text-diffusion attempt that just runs cleanly would be a valuable submission even at sub-SOTA bpb.

## Our constraints

- **Compute**: no local GPU (CPU-only VM). Confirmed `nvidia-smi` absent + not Apple Silicon, so local MLX path is also out. **All training must be cloud.**
- **Budget estimate**: 1×H100 on Runpod ~$2–3/hr. A 10-min iteration costs ~$0.50. Budget 20 iterations → ~$10. Final 8×H100 eval is ~$20/hr = $3.33 per 10-min pass. Rough ballpark: **$20–$50 of credits** would cover a meaningful experiment cycle.
- **$1M in OpenAI compute grants is available** via the form linked in the README (`openai/index/parameter-golf/#credit-form`). Email tied to OpenAI/ChatGPT account. **Should we apply?** — yes, there's no downside.

## Questions to answer today / tomorrow

1. Does the user want to **apply for the OpenAI compute grant**? (Very low effort, potentially covers everything.)
2. What's the **fallback if no grant**? Runpod spot with small personal credit — OK for iteration. Final eval on 8×H100 can wait.
3. **Which target?** (a) Beat baseline on the record track (val_bpb < 1.2244 with full 16 MB + 10-min constraints) — clean but won't move the leaderboard, or (b) pick a non-record-track direction (e.g. one of the unchecked "requests for PRs" — JEPA, text-diffusion, state-space) and aim for an interesting submission regardless of bpb. Option (b) is higher learning signal and matches "learn deeply."

## Misc

- Private challenge support channels: OpenAI Discord → #parameter-golf-discussions, #parameter-golf-announcements.
- Challenge Participant Form (optional, tied to hiring): `jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf`. **Should fill this out if we're serious** — it's how OpenAI connects your submission to you for recruiting.
- Hiring context: June 2026 cohort, targeting current undergrads / recent grads / Olympiad medalists. Phrasing is clearly a hiring funnel.
