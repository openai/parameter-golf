# Parameter Golf — Writeup Outline

Substack blog post. Tone: formal for Parts 0–1, semi-formal + funny for Parts 2–3.
Audience: half-technical, half non-technical.

---

## Part 0 — What Is This Competition?

- **The premise:** 16 MB artifact, 10 minutes training on 8×H100, scored by
  byte-level BPB on the FineWeb validation set (tokenizer-agnostic). OpenAI
  sponsoring $1M in compute grants.
- **The time budget:** 10 minutes training + 10 minutes evaluation — TTT comes
  out of the eval budget, not extra. If your TTT takes 8 minutes, you have 2
  minutes left for the actual scoring pass.
- **How scoring works:** the model outputs a probability distribution over its
  vocabulary at each token position → take log probability of the correct token
  → sum across all tokens → divide by total bytes in the document → that's
  bits-per-byte (BPB). (Internally the competition uses nats first, then
  converts: 1 nat = 1/ln(2) ≈ 1.4427 bits.)
- **The framing:** L(N) optimization — fixed parameters, unconstrained
  compute/data/architecture. Sister challenges: NanoGPT Speedrun (L(T)),
  NanoGPT Slowrun (L(D)).
- **The baseline:** 9-layer 512-dim 1024-vocab tied-embedding transformer →
  **1.2244 bpb**
- **The end:** 11-layer 512-dim SP8192+CaseOps transformer with depth
  recurrence, XSA, parallel residuals, SmearGate, LQER, phased TTT, and
  per-group lrzip compression → **1.0565 bpb** (clean SOTA, PR #2135)
- **Scoring:** what "byte-level BPB" means and why it matters — the model
  thinks in tokens, but the judge counts bytes, making the tokenizer choice a
  first-class compression lever.
- **The C1–C4 rules** (emerged mid-competition from Issue #1017, not present
  on day one):
  - C1: causal eval only — no peeking at future tokens. *Example violation:
    scoring position i using attention that reaches position i+1.*
  - C2: full normalized distribution — the scoring must be a real probability
    distribution over bytes, summing to exactly 1. *Example violation: the
    PPM-D uniform-spread construction, where p(byte) sums to >1.*
  - C3: score-before-update — in TTT, a chunk must be fully scored *before*
    any gradient step touches it. *Example violation: running one AdamW step
    on a chunk, then scoring it — the model has already adapted to those
    tokens.*
  - C4: single pass — each validation token is scored exactly once, no
    multi-pass re-scoring. *Example violation: scoring a token, updating on
    it, then re-scoring to get a lower loss.*
- **Statistical significance bar:** new records require a 3-seed mean beating
  prior SOTA with a p-value threshold; borderline submissions land as
  "non-record" entries. there are also just genuinly interesting non-record submmission that we will not cover). This writeup covers the record track only — there
  were many fascinating non-record submissions (text diffusion, Mamba hybrids,
  JEPA, 1-bit quantization) that deserve their own treatment.
---

## Part 1 — The Key Techniques

Four pillars. Each gets: the core insight, who pushed it, representative BPB
numbers, and what it contributed to the final model.

The submission pipeline every serious entrant ran: pre-compile (Triton
autotune) → training (10 min wallclock) → GPTQ quantization → artifact
compression → TTT at eval time. Each pillar maps onto a stage of this pipeline.
### 1. Tokenizer / Vocabulary

- SP1024 (baseline) → SP4096 → SP8192 — counterintuitive: a bigger vocabulary
  *helps* in a tight parameter budget because better tokenization is free
  compression before the model runs.
- The failed attempt first: **lossy casefold** — lowercasing all text before
  tokenizing gives a small vocabulary a big efficiency boost, but destroys
  case information. Ruled illegal (Issue #1604) because the scorer charges
  bytes on the original text and you can't recover them.
- **CaseOps** (romeerp, PR #1729): the lossless answer. Worth explaining the
  mechanics: standard SP8192 sees "Hello" and "hello" as two different tokens,
  wasting vocabulary slots on capitalization variants. CaseOps pre-processes
  the text before tokenizing — strip all case to lowercase, tokenize the
  compact lowercase stream, then store the original capitalization pattern in a
  tiny byte sidecar (a bitmask: 1 bit per alphabetic character saying
  "was this uppercase?"). The model trains and scores on the lowercase tokens;
  the sidecar bytes are scored by a simple fixed distribution (capitalization
  is highly predictable: sentence starts, proper nouns — you can mostly get it
  right with a few rules). Net effect: the model's vocabulary is far more
  efficient, each token encodes more semantic content per byte, and the sidecar
  cost is small because capitalization is easy. Drops the frontier from ~1.07
  to ~1.065 in one shot.
- Why byte-level BPB makes tokenizer choice load-bearing: the scorer charges
  per byte regardless of how the model tokenizes, so every token that encodes
  more bytes is a free win.
- Brief note on the token-vs-byte scoring controversy: some participants
  questioned whether scoring should be done at the byte level directly rather
  than converting from token-level log probs. For the official scorer they're
  equivalent — it's just accounting — but this became relevant when byte-level
  sidecars (like CaseOps) entered the picture.

### 2. Training Architecture

Start with what the baseline model already had — it was not naive. Worth
briefly explaining a few: **U-Net skip connections** (encoder layers feed
residuals directly into symmetric decoder layers, giving the model a natural
coarse-to-fine processing structure), **GQA** (grouped-query attention, fewer
KV heads than Q heads → big parameter saving), **BigramHash embeddings**
(learned embeddings for token bigrams, stored in a hash table → better local
context at low param cost), **Muon optimizer** (second-order-ish momentum on
matrix parameters). The competition started from a genuinely strong base.


Key architectural innovations in roughly the order they appeared:

- **XSA (Exclusive Self Attention)** — subtracts the "self-copy" component
  from each attention output. Attention tends to just copy a token's own value
  back to itself; XSA removes that self-bias, forcing the model to actually
  look at other positions. Applied to the deepest 3 layers first (#198), then
  eventually all 11. Biggest single jump of the middle phase (~0.01 bpb).
  Based on arXiv:2603.09078; good YouTube explainer exists.
- **Depth recurrence (Loop45, later Loop3-5)** — run the same layers multiple
  times per forward pass; free test-time compute within the 16 MB budget.
  Probably the single largest architectural change in the competition.
- **Variable-length (VarLen) attention** — pack documents of different lengths
  into one sequence without padding; enables per-document TTT boundaries.
- **Parallel residuals** — two-lane attention+MLP routing from layer 8+; stacks
  cleanly with recurrence.
- **SmearGate + BOS fix** — learned gate blending each token's hidden state
  with the previous token's, adding a lightweight bigram-level context signal.
  The BOS fix masks this gate at document-start positions to prevent
  cross-document leakage; the leak went unnoticed for weeks and cost ~0.002 bpb
  when finally fixed.
- **MuonEq-R / Polar-Express Newton-Schulz** — row-normalizes gradient matrices
  before Newton-Schulz orthogonalization; zero parameter cost, contributed
  ~0.001 bpb improvement.
- **Sparse attention gate** — narrow head-output gate (gate_window=12); late
  addition from PR #1787.
- **QK-Gain** — per-head learned scaling of query/key dot products, init 5.0+;
  in the final model and contributed meaningfully throughout the April phase.

Two more training dynamics worth mentioning:
- **Loop activation schedule** — depth recurrence doesn't run from step 1;
  it kicks in at `frac=0.35` (35% through the wallclock budget). Before that
  the model trains as a standard transformer, giving the weights time to settle
  before the recurrent computation graph is introduced.
- **Progressive context growth** — sequence length grows deliberately during
  training, from short sequences early on up to 3k tokens by the end (#2014).
  Lets the model learn local patterns first before being asked to handle long
  range. Combined with VarLen packing (no padding), this is a meaningful
  training-dynamics lever.

*End of section: lay out the full final model architecture from the #2135
README as a concrete snapshot of what all this stacking produced.*

### 3. Quantization
Comment: AR self-generated calibration data: not sure how much that matter.
The steady thread from day 1 to the end; every record touches quantization.

- int6 → all-int6 → full Hessian GPTQ (Cholesky error compensation, 64-batch
  calibration).
- **AR self-generated calibration data** — use the model's own outputs as GPTQ
  calibration data. Present in multiple records (#1060, #1204) but its isolated
  contribution is unclear; listed as a component but not ablated cleanly.
- **GPTQ embeddings** — quantize the embedding table too (int7); previously
  left in fp16.
- **AWQ-lite** — activation-aware weight quantization; identifies salient weight
  groups by activation magnitude and promotes them to int8 instead of int6.
  Stacks on top of GPTQ, appeared in #1908/#1945 and propagates into the final
  model.
- **Calib32** — increase GPTQ calibration batches from default to 32; cheap
  tuning that measurably improves quantization quality (#2135 is literally
  named after this).
- **LQER (Low-rank Quantization Error Reduction)** — after GPTQ, train a
  rank-4 low-rank residual to correct the rounding error on the top-3 tensors.
  A post-hoc quant repair, not a training-time technique. (~0.01 bpb gain)
- **Artifact compression** — lrzip zpaq + L1 similarity-sort row reordering +
  brotli; the final per-group pipeline in #1855 saved ~280 KB over plain
  brotli. The last few KB matter when you're bumping against 16 MB.
- QAT (quantization-aware training) as an alternative path explored early on.

### 4. TTT (Test-Time Training)

The most contested technique in the competition, and the one with the most
iterative rule-clarification history.

- **LoRA TTT (early, March 19)** — first appearance: fine-tune LoRA adapters
  on validation data during eval. LoRA TTT *is* legal — the adapter weights
  don't count against the 16 MB artifact limit because they're discarded after
  eval. The problem was not LoRA itself but *when* the gradient step happened
  relative to scoring (C3 violation in early implementations).
- **The legality fight** — C3 and C4 emerged directly from Issue #1017 after
  multiple submissions trained on val chunks before scoring them. The key
  distinction: you're allowed to *learn* from val tokens you've already been
  graded on, not from ones you're about to be graded on.
- **Score-first TTT** — the clean formulation: score each validation chunk
  completely, *then* run a gradient step. Settled by ~April 9, enabled a burst
  of records (PR #1514, #1529, #1530).
- **Phased TTT** — break eval into multiple phases; each phase scores a chunk
  then updates; multi-phase global SGD + per-doc LoRA reset. The mature form
  that ended up in the final model.
- **Warm-start-A TTT** — LoRA initialized from training rather than random;
  smaller effective LR needed.
- **Entropy-adaptive epochs** — vary TTT epochs by estimated document
  difficulty; more adaptation on harder chunks.
- **No-Q/V TTT mask** — during TTT adaptation, freeze Q and V weights; only
  adapt K and other params. Improves stability and reduces overfitting.
- **Short-doc TTT** — preferentially apply TTT to shorter documents, which
  have higher per-token uncertainty and respond better to adaptation (#2014).
- **Token-only n-gram tilt** — the legal form of n-gram that eventually worked.
  The original kernel (PR #1420) had *three* experts: a token expert (counts
  n-gram prefix matches in the already-scored window), a within-word expert
  (gates on whether the current position is mid-word), and a word-start expert
  (gates on whether it starts a word). The problem: the within-word and
  word-start experts read `boundary_lut[tokens[i]]` — the TARGET token's
  boundary type — which is non-causal (you can't know if position i is
  mid-word until you see what token lands at i). Two of three experts violate
  C1 by construction. The token expert is causal by construction — it only
  reads the prefix hash table. Disable the two broken ones, keep the one that
  works: closed-form `p'(a) = exp(β·1[a=h])·p(a)/Z`, prefix-only. The
  surprise ending to the n-gram saga.
- TTT contribution to the final model: pre-quant 1.064 → post-TTT 1.060,
  roughly ~0.004 bpb.

*Timeline of legality rulings would be a good visual here.*
---

## Part 2 — The Disqualification Zoo

Tone: semi-formal, wry.

### PPM-D: "Free Bits From Arithmetic That Doesn't Sum to 1"
- Background: PPM (Prediction by Partial Matching) is a classical byte-level
  compressor. Mix it with a neural model in probability space and you get a
  byte-level hybrid scorer.
- The cluster: ~8 PRs (including our own #1885) claiming 0.90–1.014 bpb via a
  NN + PPM-D mixture. Flagged in Issue #1872.
- The C2 violation: the uniform-spread construction (`p(t)^{1/n}` per byte)
  does not normalize. A toy example: two 2-byte tokens each with p=0.25 gives
  `p(first_byte) = 0.5 + 0.5 = 1.0`, not sharing space with any other byte.
- The punchline: under the correct conditional byte distribution, PPM is *not*
  better than the baseline — it's worse by ~0.038 bpb. The apparent 0.051 bpb
  gain is entirely from the broken scoring rule, not from PPM.

### Scylla: footnote
- PR #1184 appeared at 0.9485 bpb for three days, removed as "invalid record"
  — likely a GitHub merge accident rather than deliberate cheating. Not worth
  dwelling on.

### N-gram: Hard to Do Right
- Multiple attempts: n-gram eval cache, n-gram TTT, n-gram tilt.
- The original kernel (PR #1420) had three experts: token, within-word,
  word-start. The within-word and word-start experts read
  `boundary_lut[tokens[i]]` — the TARGET token's boundary type. You can't know
  if position i is mid-word until you see what token lands there, so two of
  three experts are non-causal by design. The C1 violation rate was ~95% of
  gated mass, because word/boundary gates fired on almost every token.
- The token expert is causal: it only queries the prefix hash table (tokens the
  model has already been graded on). Disabling within/word and keeping only the
  token expert gives the legal form.
- The surprise ending: once cleaned up, the token-only tilt *did* eventually
  work, and appears in the final SOTA record (#2135).
- Closing heuristic: the failure mode for classical n-gram/byte methods is
  almost always the same — the classical side either peeks forward (C1) or
  doesn't produce a normalized distribution (C2).



- General argument for why supplementing NNs with classical methods is so hard:
  a neural model already captures most of what an n-gram or PPM would tell you.
  The NN is well-calibrated on its own uncertainty — it knows when it doesn't
  know. For a classical model to help, it needs to provide *orthogonal*
  information: something the NN genuinely can't see. But both n-gram and PPM
  are trying to predict the same thing from the same context, so their signals
  are highly correlated with the NN's output. You're fighting for a thin slice
  of orthogonal signal. The one case where token-only n-gram tilt *does* eke
  out a gain is exact prefix repetition: a very strong prior that "this phrase
  appeared verbatim 3 tokens ago" that the softmax tends to smooth over. That's
  a real, narrow orthogonal lever — but most of the classical-method graveyard
  here is just correlated noise with a legality trap attached.
---

## Part 3 — Last Day Chaos: The CaseOps Val-Set Leak

Tone: funny, forensic.

- **CaseOps** arrives April 18–19 and is genuinely good (+0.003 bpb). Everyone
  adopts it immediately.
- **The bug:** `prepare_caseops_data.py` has a `--val-docs=10000` default.
  Nobody overrides it. All 34 CaseOps-lineage PRs. Zero overrides.
- **What leaked:** training starts at canonical-stream document 10,000, but the
  validation set is documents 0–49,999. Documents 10,000–49,999 (80% of the
  val set) are in both train and val.
- **The timeline:**
  - Introduced: PR #1736 (dexhunter, April 19) — first to use
    `prepare_caseops_data.py` with the default, while reporting a separately-
    regenerated 50k-doc val set
  - Fixed: PR #1851 (aquariouseworkman, April 27) — switched to the official
    HF dataset (`romeerp/parameter-golf-caseops-v1`), disjoint by construction
  - Re-introduced same day: PR #1855 (codemath3000, April 27) — rebuilt
    locally with the default, propagating the leak forward
- **The numbers:** claimed frontier #2118 at **1.04350** vs clean frontier
  #1851/#1855 at **~1.061**. A 0.016 bpb gap from memorizing 80% of your exam.
- **Our own position:** our research baseline (#1736) was leaky. All internal
  spec measurements (specs 008–302) are internally consistent but live in the
  leaked world. Absolute numbers not comparable to the clean leaderboard.

---

## Part 4 — Final Leaderboard and Some Thoughts

Two leaderboard updates cap the competition:
- **PR #1902 (cocohearts, April 29):** official retroactive update — applies
  the BOS fix, clarifies which CaseOps records are clean, establishes the
  accepted sequence through 1.0611.
- **May 2 update (not yet merged to main):** audits the late-April / May 1
  submissions and adds four more records.

The final accepted record sequence, bottom to top:

| BPB    | PR     | Author            | Key techniques |
|--------|--------|-------------------|----------------|
| 1.0565 | #2135  | codemath3000      | Calib32 + token-only n-gram tilt + AsymLogit ← **final SOTA** |
| 1.0567 | #2130  | TanishGudise      | Token-only n-gram tilt + AsymLogit + one-phase TTT |
| 1.0576 | #2014  | simonbissonnette  | Progressive context growth to 3k + short-doc score-first TTT |
| 1.0586 | #1953  | andrewbaggio1     | Long-context no-Q/V TTT + QK-Gain 5.25 |
| 1.0594 | #1945  | alertcat          | AWQ-lite GPTQ + AsymLogit on #1855 stack |
| 1.0611 | #1855  | codemath3000      | BOS-fixed SmearGate + LQER + SparseAttnGate + 9-hparam stack |
| 1.0614 | #1851/68 | aquariouseworkman | BOS-fixed SmearGate + LQER Asymmetric + Phased TTT |
| 1.0634 | #1787  | nprime06          | PolarNS + MIN_LR + SparseAttnGate + Warm-A TTT |
| 1.0645 | #1769  | dexhunter         | CaseOps + MLPClip12 + SmearGate/LoRA-TTT |
| 1.0655 | #1736  | dexhunter         | SP8192 + CaseOps + GatedAttn + Loop45 + Phased TTT |
| 1.0678 | #1729  | romeerp           | CaseOps tokenizer + tapered WD |
| 1.0714 | #1667  | MarioPaerle       | SmearGate + AttnOutGate + Legal TTT |
| 1.0719 | #1626  | dexhunter         | VarLen + fused MLP + multi-phase global SGD TTT |
| 1.0728 | #1610  | romeerp           | VarLenAttn + phasing TTT |
| 1.0734 | #1530  | samacqua          | VarLen FA3 + fused Triton MLP + doc-independent LoRA TTT |
| 1.0758 | #1529  | msisovic          | Parallel residuals + CUTLASS EVT + legal TTT |
| 1.0798 | #1514  | dexhunter         | SP8192 + Muon 0.97 + legal score-first TTT |
| 1.0810 | #1493  | bigbag            | SP8192 + 3-layer recurrence + parallel residuals + TTT |
| 1.0822 | #1477  | aryanbhosale      | SP8192 + parallel residuals + score-first TTT |
| 1.0828 | #1413  | dexhunter         | SP8192 + QK-Gain 5 + legal TTT |
| 1.0856 | #1394  | Kevin Clark       | SP8192 + GPTQ embeddings + Loop45×2 + SDClip |
| 1.0897 | #1334  | aryanbhosale      | SP4096 + depth recurrence + parallel residuals + MuonEq-R |
| 1.0912 | #1285  | dexhunter         | MuonEq-R + depth recurrence + WD=0.09 + all-int6 |
| 1.0979 | #1218  | Kevin Clark       | SP4096 + 4× MLP + high WD |
| 1.1063 | #1204  | msisovic          | Parallel residuals + mini depth recurrence |
| 1.1122 | #1060  | dexhunter         | Coprime loader + full Hessian GPTQ + XSA-all |
| 1.1147 | #1019  | abaybektursun     | AR self-gen GPTQ + XSA-all |
| 1.1194 | #549   | abaybektursun     | LeakyReLU² + legal TTT + parallel Muon |
| 1.1228 | #374   | signalrush        | EMA + GPTQ-lite + warmdown3500 |
| 1.1248 | #287   | jfprincz          | Partial RoPE + LN scale + EMA + XSA4 |
| 1.1271 | #198   | jfprincz          | XSA4 + EMA + int6 MLP3× |
| 1.1307 | #198   | unnir             | Efficient partial XSA (deepest 3 layers) |
| 1.1428–1.1556 | various | various      | BigramHash, SmearGate, int6 QAT, SWA, OrthoInit |
| 1.1630–1.1925 | various | various      | Sliding window eval, mixed quant, Muon WD, 10L |
| 1.2244 | baseline | OpenAI         | 9L 512d 1024-vocab |

- 1.2244 → 1.0565: a drop of **0.168 bpb** over 6 weeks of community effort.
- Reflection: what worked (stacking, tokenizer bets, VarLen+TTT, progressive
  context), what surprised (larger vocab helping in a constrained budget), what
  didn't (PPM, lossy casefold, anything requiring cheating to look good).
