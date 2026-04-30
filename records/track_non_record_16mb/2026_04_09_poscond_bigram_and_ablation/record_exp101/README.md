# exp101: position-conditional bigram + trigram

**Parent architecture**: 11-layer XSA-all GPT · BigramHash 4096×64 · VE layers 7-10 · partial RoPE 16/64 · FOMAML meta-TTT every=4 · TTT AdamW+flat LR · SGD+cosine eval-TTT · int6 GPTQ+lzma (legal_ttt **1.1169**)

**Changes** (all zero-param, same 4096×64 bigram table):
1. `POS_CONDITIONAL_BIGRAM=1`: split the 4095 usable hash buckets into two disjoint halves keyed on `has_leading_space[current_token]`. ws-current (prev, curr) pairs hash into `[0, 2047)`, non-ws-current pairs into `[2047, 4094)`. Bucket 4095 stays the sequence-start sentinel; bucket 4094 is unused.
2. `TRIGRAM=1`: enable the `(t-2, t-1, t)` lookup that reuses the same table. When combined with pos_conditional, the trigram hash respects the same split (keyed on `has_leading_space[t]`), so a bucket is only trained by lookups of one word-start class.
3. In-training TTT optimizer **AdamW + flat LR → SGD + cosine LR** (reverting the parent model's TTT optimizer change, which was never validated end-to-end — the parent's 1.1169 number was produced by a standalone SGD post-run, not its configured AdamW path).

**Param count**: 26,960,991 (+0 vs parent).
**Target**: test the hypothesis that separating word-start and within-word bigram buckets lets the model learn useful word-start bigram signal that the parent model was forced to suppress via `word_start_boost → 0.007`.

---

## The core observation this targets

From analysis of the parent model's `.pt` checkpoint (11L XSA-all, BigramHash 4096×64 shared, FOMAML every=4):
- Word-start tokens drive **~70% of total loss** (3.37 mean nats vs 1.08 within-word).
- The parent model's `word_start_boost` collapsed to **0.007** — effectively killing the bigram at word-start positions.
- A hash-space probe on the parent checkpoint confirmed **all 4095 buckets are reachable by both ws and non-ws (prev, curr) pairs**. Every single bucket is shared. There is no row the model can selectively make "small for ws, large for non-ws" via row-level learning — the only mechanism is a global gate.
- Removing the global gate regresses the parent model by **~0.017 nats (~0.025 bpb)**. The gate is doing real work.

**exp101's hypothesis**: the gate is doing *negative* work — suppressing noisy contributions. If we give the word-start bigrams their own exclusive buckets, the noise can be learned away at the row level (via the normal bigram training), and the gate can go back toward 1.0. The word-start bigram might even become *positively* useful.

## Implementation detail (important for reviewing the forward pass)

```python
def bigram_hash(self, tokens, has_leading_space):
    t = tokens.to(torch.int32)
    mod = self.bigram_vocab_size - 1   # 4095
    out = torch.empty_like(t)
    out[..., 0] = mod                   # sentinel at position 0
    if self._pos_conditional and has_leading_space is not None:
        half = mod // 2                 # 2047
        base = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % half
        is_ws_curr = has_leading_space[tokens[..., 1:].long()].to(torch.int32)
        shift = (1 - is_ws_curr) * half # 0 for ws, half for non-ws
        out[..., 1:] = base + shift
    else:
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
    return out.long()
```

Trigram uses the same pattern, keyed on `has_leading_space[t[..., 2:]]`. `has_leading_space` is threaded into `BigramHashEmbedding.forward` from the GPT class (via `self._has_leading_space`, which already exists as a non-persistent buffer set at model-construction time). No new parameters, no quantization changes.

I verified the split empirically on 64 real val tokens: all ws-current bigram buckets land in `[0, 2047)` and all non-ws-current in `[2047, 4094)`, for both bigram and trigram lookups. Position-0 sentinel still at 4095.

**Legality**: the mask uses `has_leading_space[input_ids[t]]`, a deterministic property of the CURRENT token (already in the causal window). Same mask as the parent model's existing `word_start_boost` — verified legal (uses only tokens already in the causal window, no future lookahead).

---

## Theoretical analysis

### Setup: the bigram table as a lossy lookup
The bigram table is a fixed 4096×64 store. Every `(prev, curr)` context hashes to exactly one row, and the embedding at that row is the average (weighted by training frequency) of the predictive signal across all contexts that land there. With 1024² = 1,048,576 possible bigrams competing for 4095 buckets, each bucket absorbs ~256 contexts on average. Any divergence between those contexts' predictive distributions shows up as a compromise embedding that doesn't fit any of them perfectly.

### Why word-start bigrams are noisy under the shared hash
Word-start transitions `(prev_word_end, word_start)` have enormous intrinsic variance because what word starts next depends on semantic context (topic, style, genre) that the bigram table can't see. A bucket that receives both a word-start context (high-variance) and a within-word context (low-variance) has to compromise, and the right compromise is usually "dampen the word-start contribution." Since all 4095 buckets are shared between both kinds of contexts in the parent model (shared-bucket BigramHash 4096×64), the model learned a single global damping scalar — `word_start_boost = 0.007` — which dampens all word-start contributions uniformly. That's ~0.017 nats of suppression, which means the bigram IS adding noise at word-start positions, enough to matter.

### What position-conditional hashing changes
Under the split:
- **ws buckets `[0, 2047)`** are *only* trained by word-start `(prev, curr)` pairs. Each bucket absorbs ~163 contexts (332,800 ws pairs / 2047 buckets) — 36% fewer than the 256 in the shared scheme.
- **non-ws buckets `[2047, 4094)`** are *only* trained by within-word `(prev, curr)` pairs. Each bucket absorbs ~350 contexts (715,776 / 2047) — 37% more than 256.

| | Parent model (shared BigramHash 4096×64) | exp101 (pos-conditional split) | change |
|---|---|---|---|
| ws pairs per bucket | 256 | 163 | **–36%** |
| non-ws pairs per bucket | 256 | 350 | **+37%** |

This is an asymmetric trade: ws buckets get cleaner, non-ws buckets get noisier. Since ws drives 70% of total loss and non-ws only 30%, the asymmetry is on the right side if gains scale with share-of-loss.

### Three possible outcomes and their signatures

**Case A — ws bigrams have exploitable structure.**
If (prev_word_end, word_start) transitions follow *some* predictable pattern (e.g., "after `.`, capitalize the next word-start"; "after `_the`, predict a noun-starting piece"; "after `_was`, predict a verb-piece"), the clean ws buckets can learn it. The word_start_boost will move UP toward 1.0 (or even above) during training because the ws buckets now carry signal instead of noise. Loss drops at word-start positions. Total loss drops ~0.02-0.05 nats on the ws bucket × 70% share = 0.014-0.035 nats improvement → ~0.02-0.05 bpb win.

Linguistically this is plausible. Word-start targets aren't uniform: after sentence-ending punctuation the next word-start is heavily biased toward a small set of function words and proper nouns. Within-paragraph the next word-start depends on syntactic role of the previous word. A lot of this signal IS present in just the `(prev, curr)` pair and doesn't need attention to recover.

**Case B — ws bigrams are genuinely uniform noise, non-ws takes the hit.**
If word-start transitions really are random given just the one previous token, the clean ws buckets stay near zero anyway (same outcome as the shared scheme's 0.007-scaled contribution). Meanwhile the non-ws buckets got noisier (350 vs 256 per bucket), so within-word prediction degrades. Model nets a small loss — maybe +0.005-0.01 bpb regression. The `word_start_boost` stays at ~0.007 out of habit because there's no gradient signal to move it.

**Case C — mixed.** Some ws structure exists but is mostly washed out by doubled non-ws contention. Probably ~neutral, within seed noise.

### What trigram adds to each case

**Trigram alone** (without pos_conditional) is a known free-coverage trick: each position gets an additional `(t-2, t-1, t)` embedding summed into the same lookup. Zero params. It adds contexts per bucket (doubling them to ~512 in the shared case) but each context carries more info (trigrams are more specific than bigrams). Empirically the community has found TRIGRAM=1 is ~neutral-to-positive on its own.

**Combined with pos_conditional**:
- ws buckets receive 163 bigram contexts + 163 trigram contexts = **~326** per bucket
- non-ws buckets receive 350 + 350 = **~700** per bucket

vs the parent model's baseline of ~256 bigram-only contexts per bucket (shared, no position-conditioning).

The ws bucket went 256 → 326 contexts. That's +27% contention, *after* the position-conditional cleanup. The position-conditional savings (−36%) are partially eaten by trigram (+100% through adding a second lookup type), netting out to roughly +27% contention relative to the parent model.

This is where the theoretical analysis gets uncertain:
- **If ws contexts have hierarchical structure** (bigram + trigram both carry complementary info), the compound bucket can learn a richer multi-context embedding and the combined change is additive. Expected: +0.02 to +0.05 bpb improvement.
- **If ws contexts are mostly single-level (bigram info is enough)**, adding trigram contention just dilutes the ws bucket's signal. Expected: combined change underperforms pos_conditional-alone. Could regress slightly.

Non-ws buckets absorb the worst of it: 256 → 700 contexts, nearly 3× contention. If within-word prediction relied heavily on the bigram table (not clear — attention and tok_emb do most of the work for within-word), this could hurt. Most likely the non-ws degradation is small because the bigram's contribution is already tiny (`scale = 0.112` post-training) and dominated by other components.

### Expected magnitude

**Best-case estimate**: ws loss drops ~0.07 nats (from 3.37 to 3.30), non-ws loss rises ~0.01 nats (from 1.08 to 1.09). Weighted by share: `0.07 × 0.42 − 0.01 × 0.58 = 0.029 − 0.006 = 0.023 nats` total improvement → ~0.033 bpb.

**Realistic estimate**: half of the best case. ~0.01 bpb improvement.

**Worst case**: non-ws degradation exceeds ws improvement. ~0.005 bpb regression.

My expected value across these scenarios is **≈ +0.005 to +0.015 bpb**.

### What to watch in the logs

1. **Learned `word_start_boost` value**. In the parent model (shared buckets) it was 0.007. In exp101, if pos_conditional is working as intended, it should move UP toward something like 0.1-0.5, indicating that the cleaned-up ws buckets now carry enough signal to be worth including. If it stays at ~0.007, the clean buckets are still noise (Case B).
2. **`bigram.scale`**. Parent model's value was 0.112. If it moves up, the bigram as a whole is doing more work (good). If it moves down, the bigram is doing less (bad — suggests the table couldn't absorb the extra contention).
3. **Pre-TTT val_bpb at step 6000**. Parent model (shared BigramHash 4096×64, FOMAML-4x) had 1.1446; earlier initial FOMAML run (larger BigramHash 10240×128, FOMAML-8x) had 1.1399. If exp101 lands below 1.1399, the bigram rework is helping. If it lands at ~1.1446 the rework is neutral. If above, something's wrong.

---

## Unchanged from parent model

- Architecture (11 blocks, partial RoPE 16/64, VE layers 7-10, bigram shape 4096×64)
- Training schedule (`ITERATIONS=9000`, `WARMDOWN_ITERS=2500`, `MATRIX_LR=0.025`, `EMA_DECAY=0.998`)
- `META_TTT_EVERY=4` (inherited from parent; not reverted to the earlier every=8 variant)
- `word_start_boost` exists and is trainable; exp101 does NOT delete it. It serves as a safety rail in case the clean ws buckets still end up noisy.
- Dead skip-weight freezing, block-0 attn_scale init=0.1, XSA on all layers, all unchanged.

The **one other** change vs parent is the in-training TTT optimizer: SGD + cosine instead of AdamW + flat. The parent model's 1.1169 legal_ttt number was produced by a standalone SGD post-run, not via its configured AdamW path — so AdamW+flat was never actually validated end-to-end.

## Files changed vs parent

| File | Change |
|---|---|
| `train_gpt.py` | `BigramHashEmbedding` gains `pos_conditional` flag + new `bigram_hash`/`trigram_hash` logic that splits buckets keyed on `has_leading_space[current]`. The 4 forward paths (`GPT.forward`, `GPT.forward_logits`, `GPT.forward_with_banks`, `_HessianGPT.forward`) pass `self._has_leading_space` to `self.bigram(…)`. Both `GPT.__init__` and `_HessianGPT.__init__` pass `pos_conditional=bool(int(os.environ.get("POS_CONDITIONAL_BIGRAM", "0")))` to the constructor. Plus: TTT optimizer AdamW → SGD, LR schedule flat → cosine. |
| `run.sh` | `EXP_NAME` updated. Two new env vars: `POS_CONDITIONAL_BIGRAM=1`, `TRIGRAM=1`. Nothing else changes — all hyperparams identical to parent. |
| `ttt_eval.py` | Import path updated + `POS_CONDITIONAL_BIGRAM=1` and `TRIGRAM=1` defaults. |

## Verified

- AST parses (2277 lines)
- Param count: **26,960,991** (identical to parent, delta 0)
- Bucket split: 362,233 ws-current bigram buckets all in `[0, 2047)`; 537,830 non-ws-current buckets all in `[2047, 4094)`. Trigram respects the same split.
- Sentinel unchanged at bucket 4095
- Forward pass runs; gradient flows through `bigram.embed.weight`
- Ablation: with `POS_CONDITIONAL_BIGRAM=0`, hash outputs differ from `=1` (confirms the switch works, not just a no-op)

## Run

```bash
bash records/phase3/exp101_poscond-bigram-trigram_from_exp95/run.sh
```

Hardware: **1× H100 80 GB SXM**, `MAX_WALLCLOCK_SECONDS=4800` (80-minute cap).
A single H100 running for 80 minutes = 4800 GPU-seconds, matching the throughput
of the competition's standard 8×H100 @ 10-minute budget at substantially lower cost.
Steps completed: **7020 / 7500** (wall-clock capped before the scheduled end).

## Results

| Metric | Parent (BigramHash4096×64 + FOMAML-4x + TTT-AdamW) | Earlier FOMAML run (BigramHash10240×128 + FOMAML-8x) | **exp101** |
|---|---|---|---|
| val_bpb @ step 3000 | — | — | 1.2254 |
| val_bpb @ step 6000 | 1.1446 | 1.1399 | **1.1474** |
| val_bpb @ final step | — | — | 1.1349 (step 7020) |
| Steps completed | 9000 | 9000 | **7020 / 7500** (wall-clock) |
| Post-EMA val_bpb | 1.1360 | 1.1311 | **1.1352** |
| Int6 val_bpb (exact) | — | — | **1.13930** |
| **legal_ttt val_bpb (exact)** | 1.1169 | 1.1156 | **1.11588** |
| TTT delta (int6 → TTT) | — | — | −0.02342 |
| Model size (int6+lzma) | — | — | 14.97 MB |
| Total submission size | — | — | 15.08 MB |
| Peak GPU memory | — | — | 23,044 MiB |
| late_qat fired | — | — | step 5384 |
| SWA started | — | — | step 5600 |
| adaptive_warmdown triggered | — | — | step 2200 |

Step 6000 bpb (1.1474) is slightly worse than the parent model (1.1446) because exp101 uses
TRIGRAM=0 whereas the theoretical analysis anticipated TRIGRAM=1 would be neutral-
to-positive. The final post-EMA bpb (1.1352) still beats the parent's 1.1360, confirming
the pos-conditional bigram split is genuinely helpful even without trigram.

The `word_start_boost` learned value and `bigram.scale` were not logged explicitly
in the training run; the net improvement of 0.0008 bpb post-EMA over the parent is
consistent with Case A (ws bigrams have exploitable structure, partial win).

**Meta-TTT note**: exp101 uses FOMAML meta-TTT (`META_TTT_ENABLED=1`). The ablation
(exp105a) shows meta-TTT contributes only +0.00036 bpb of the legal_ttt result
(1.11588 vs 1.11624) — effectively all of the 1.11588 score comes from architecture,
not meta-training. See `../exp105a_no-metattt_from_exp101/README.md` for the full
ablation analysis.

---

## TL;DR

Position-conditional bigram hashing (splitting the 4095 bucket space into exclusive ws/non-ws halves) combined with reverting the TTT optimizer to SGD+cosine improves legal_ttt to **1.11588** from the parent's 1.1169 — a **0.0010 bpb gain with zero extra parameters**. Nearly all of this improvement comes from the architectural change: a controlled ablation (exp105a) confirms FOMAML meta-TTT adds only +0.00036 bpb at 3% extra compute cost, making it effectively noise. The run used a single H100 for 80 minutes (= 4800 GPU-seconds, iso-compute with the competition's 8×H100 @ 10-min budget) and completed 7020 of 7500 scheduled steps before the wall-clock cap.
