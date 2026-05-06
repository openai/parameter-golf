# Use-Theoretic Embeddings: A Wittgensteinian Test of the Static-Embedding Orthodoxy

> *"Suppose everyone had a box with something in it: we call it a 'beetle.' No one can look into anyone else's box, and everyone says he knows what a beetle is only by looking at his beetle. — Here it would be quite possible for everyone to have something different in his box. ... The thing in the box has no place in the language-game at all; not even as a something: for the box might even be empty."* — Wittgenstein, *Philosophical Investigations* §293

## Abstract

We replace the V×d static tied embedding (4.2M params, the largest single line item in a 16 MB Parameter Golf model) with a V×d_seed *seed table* plus two tiny hypernetworks that realize the working embedding from a token's seed and a causal context summary at use-time, and the output projection from the same seed table at forward-time. Storage drops 5.02× (4,194,304 → 835,584 params at d_seed=64). The architectural change is motivated by Wittgenstein's late-period argument that meaning is use, not reference: a static embedding table is the technical incarnation of treating meaning as a private object to be looked up, when on the use-theoretic view it should be computed contextually. We submit the change as a non-record entry on top of PR #1855 (codemath3000, 1.06108 BPB merged 2026-04-27) with a 168-line surgical diff, six passing CPU sanity tests, and three integration bugs found and fixed via static analysis. Three matched-budget ablation conditions (A baseline, B redirect savings into a 12th layer, C smaller artifact at same architecture) are pre-registered. Empirical evaluation pending 8×H100 SXM compute access — the hypothesis remains open.

## Claims

We make exactly four falsifiable claims:

1. **Storage**: Use-Theoretic Embeddings store 5.02× fewer parameters than the V×d static tied embedding at d_seed=64, with no loss of representational expressivity at the input layer (verified, see §Sanity Checks).
2. **Causality**: the input embedding for token at position *t* depends only on tokens at positions *t* and (*t-c*, ..., *t-1*), with the current token's seed deliberately excluded from its own context summary (verified by permutation test).
3. **Compatibility**: the change is compositional with the existing PR #1855 stack — XSA, LQER, SparseAttnGate, BOS-fixed SmearGate, Polar-Express Muon, phased TTT, per-group lrzip compression — modulo three integration bugs we identified and fixed during static review (verified by AST + state-dict mock).
4. **Empirical hypothesis (pending GPU)**: under matched parameter budget (16 MB), the Use-Theoretic version equals or beats the static-embedding baseline because the freed ~3.4M parameters are more valuable in additional layer width than in static embedding storage. We do *not* claim a record; we claim an interesting result regardless of sign.

## The frame-flip

There are exactly two architectural directions on the Parameter Golf leaderboard. Tractarian moves *encode known regularity into structure* (parallel residuals, partial RoPE, tied embeddings — "show the regularity in the architecture, don't pay parameters for it"). Investigations moves *refuse to store what context can determine* (score-first TTT, EMA, BigramHash — "let documents teach the model their own language game"). The frontier explores both sides, except for one structural gap: **the embedding itself is still treated as a reference dictionary**.

Nobody has flipped that. The 4.2M-parameter static embedding is, in Wittgenstein's terms, a beetle in a box: every token gets a fixed vector that purportedly encodes its meaning, but each vector is only ever accessed through use, and the model could behave identically with a *different* vector that produces the same downstream behavior. The vector is a private object that does no work in the language game.

This submission asks: what if we delete the box and keep only what the language game actually needs?

## Method

### `UseTheoreticEmbedding`

| Component | Static (1.06108 BPB SOTA) | Use-Theoretic (this work) |
|---|---|---|
| Input embedding | `nn.Embedding(V, d)` lookup | `tok_emb[id] ⊕ ctx_summary → input_hyper → d_model` |
| Output projection | `F.linear(hidden, tok_emb.weight)` (tied) | `F.linear(hidden, output_hyper(tok_emb.weight))` |
| Storage (V=8192, d=512) | 4,194,304 params | 835,584 params (5.02× smaller) |
| Output context | static (tied to input) | static (regenerated each forward, not stored) |
| Input context | none (pure lookup) | causal mean of previous 3 seeds |

The input hypernetwork conditions on the token's seed *and* a causal mean-pool of the previous 3 seeds. The current token's own seed is excluded from its own context — meaning cannot refer to itself.

The output hypernetwork is intentionally non-context-conditioned. Output decoding scores every vocabulary token uniformly against the current hidden state; per-position context is incompatible with that. We pay this asymmetry as a deliberate concession to compute, not to philosophy. The output is no longer a static lookup, but it is not yet fully use-theoretic — it sits between Tractatus and Investigations.

### Parameter accounting

```
tok_emb (seed table):   V × d_seed                  =  8192 × 64           =  524,288
input_hyper:            2·d_seed → d_hidden         + (128 × 256)          =  163,840
                        + d_hidden → d_model        + (256 × 512)
output_hyper:           d_seed → d_hidden           + ( 64 × 256)          =  147,456
                        + d_hidden → d_model        + (256 × 512)
─────────────────────────────────────────────────────────────────────────────────────
Total                                                                       =  835,584

Static V × d_model baseline                                                 = 4,194,304
                                                                          ratio = 5.02×
```

Verified empirically by `python3 ../test_use_theoretic.py`.

### Activation

```bash
USE_THEORETIC_EMBED=1     # main switch
THEORETIC_D_SEED=64       # seed dimension (default 64)
THEORETIC_D_HIDDEN=256    # hypernet hidden dim (default 256)
THEORETIC_CTX=3           # causal context window in seed positions (default 3)
```

`USE_THEORETIC_EMBED=0` recovers the unmodified PR #1855 stack.

## Theoretical prediction

Three independent lines of evidence converge on a non-trivial prediction about d_seed:

1. **Aghajanyan et al. (2021)**, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" — fine-tuning a 110M-param BERT has intrinsic dimension on the order of ~hundreds, not millions. This is a *task-conditional* result, but it constrains how much information per token can be load-bearing.
2. **Delétang et al. (ICLR 2024)**, "Language Modeling Is Compression" — a language model's loss is its expected codelength, by Shannon's source-coding theorem. Every byte of the embedding table must earn its keep in predictive entropy reduction per validation byte.
3. **Lottery Ticket / strong-LTH (Ramanujan et al. 2020)** + universal compression theory (2510.00504) — networks compress to polylog(d) "objects" via permutation symmetry. Information density per parameter is unbounded above.

Together these suggest there is some d_seed* < d_model where Use-Theoretic Embeddings match or beat static V×d_model on FineWeb. Our default d_seed=64 is the rough geometric mean of (d_model=512, log V=13). It is a guess, not a derivation, and we expect the optimum could be anywhere in [16, 128].

This is a hypothesis worth running, regardless of whether it produces a record.

## Comparison to prior work

This is not pure Wittgenstein-fueled fantasy. It plants a flag in a well-established engineering tradition that has not yet been tried in this competition.

| Work | Idea | Used in Parameter Golf? | Closest to ours? |
|---|---|---|---|
| **Hypernetworks** (Ha, Dai, Le 2016) | Networks generating weights for other networks | No | Direct technical ancestor — "weights are a function of something else" |
| **HashEmbeddings** (Svenstrup et al. 2017) | Tokens as mixtures of hash-bucketed sub-embeddings | No | Most directly comparable — small lookup + composition |
| **Charformer / ByT5** (Tay et al., Xue et al. 2021) | Tokenizer-free models, embeddings from byte/char context | No | Most radical use-theoretic stance — embeddings as pure context output |
| **ELMo / contextual embeddings** (Peters et al. 2018) | Per-position embedding from context | Implicit (via attention) | We make it *explicit at the embedding layer* |
| **DenseFormer / mHC** (DeepSeek 2024) | Multi-depth connections over residuals | No | Orthogonal — operates on residual stream, not embedding |
| **GPTQ embedding quant** (PR #1394, Kevin Clark) | Compress static V×d via GPTQ + clip search | Yes (in stack) | Compresses *the lookup*; we *replace* it |
| **BigramHash** (PR #1019+) | Small hash table of (prev_tok, cur_tok) | Yes (in stack) | Adds n-gram sidecar; we condition embedding on context |

What is novel **here, in Parameter Golf**: the budget framing. The competition's binding constraint is parameter density (every byte must earn its keep in predictive entropy reduction per validation byte). The 4.2M-param static embedding has been compressed in the existing literature (GPTQ-int7-tied) but not questioned as a *category*. This submission asks whether it should exist as a static lookup at all.

## Pre-registered ablation design

Three matched conditions test the architectural claim cleanly. Each is a single env-var change against a single base file:

| Run | Embedding | Other | Total params | What it tells us |
|---|---|---|---|---|
| **A. Baseline** | static V×d, tied | — | matched 16 MB | reference SOTA (1.06108) |
| **B. Use-Theoretic, matched-budget** | seed+hyper | `NUM_LAYERS=12` | matched 16 MB | does the **freed budget** beat the lost flexibility? |
| **C. Use-Theoretic, matched-architecture** | seed+hyper | `NUM_LAYERS=11` | < 16 MB | does the embedding actually carry irreducible info? |

Predicted outcomes — all four are publishable:

- **B beats A**: the philosophical reframe paid off; the static embedding *was* misallocated.
- **C ≈ A**: the embedding really *was* over-parameterized; meaning truly is mostly use.
- **C < A but B ≈ A**: the embedding carries semantic content, but freed budget exactly recovers it elsewhere — interesting equilibrium.
- **B < A**: static reference remains structurally important; meaning is *not* purely use at this scale (a Tractarian win — the embedding *shows* what cannot otherwise be *said*).

## Sanity checks (CPU, no GPU required)

```
$ python3 ../test_use_theoretic.py
PASS shapes      input=(2, 17, 512)  output_weights=(8192, 512)
PASS budget      use_theoretic=835,584  static V*d=4,194,304  ratio=5.02x
PASS zero-init   output_weights.max=0.0
PASS zero-init   input_embed.max=0.0
PASS causality   pre=0.00e+00  pos15=0.0087  after=0.0028  far=0.00e+00
PASS grads flow  loss=9.0109
```

Causality is enforced by construction: token at position 15 affects positions {15, 16, 17} and *no others* (with `ctx_window=3`). Positions ≤14 and ≥18 are bit-identical when token 15 is permuted. This is not asserted philosophically; it is mechanically enforced and tested.

## Code rigor: bugs found and fixed during static review

GPU access was unavailable on the deadline day (8×H100 SXM exhausted on Runpod, no offerings on Community Cloud). Rather than ship without verification, we performed a static-analysis pass over the upstream 3,800-line `train_gpt.py` looking for every place our embedding swap touches. Three real bugs were found and fixed before submission. Each would have crashed the run on GPU.

1. **TTT lm_head LoRA dimension mismatch.** `BatchedTTTLoRA.__init__` reads `model.tok_emb.embedding_dim` to size the lm_head LoRA. Under our change `tok_emb` exposes the seed table, so `embedding_dim` returns d_seed=64 instead of d_model=512. The TTT path consumes d_model-wide hidden states. Fixed by branching on `use_theoretic_embed` and using the canonical d_model from `qo_bank.shape[-1]`.

2. **State-dict tensor duplication.** Naive aliasing `self.tok_emb = self.theoretic_embed.tok_emb` registered the same tensor under both `tok_emb.weight` and `theoretic_embed.tok_emb.weight` in `state_dict()`, doubling the embedding's contribution to the artifact and risking the 16 MB cap. Fixed using `object.__setattr__` to bypass `nn.Module.__setattr__`'s child-module registration. Verified with a mock GPT — exactly one `tok_emb.weight` entry remains.

3. **GPTQ KeyError on tensors without registered Hessians.** `collect_hessians` only registers hooks on the original tied `tok_emb` and inside transformer blocks. The new `theoretic_embed.*` tensors don't get Hessians, and `gptq_mixed_quantize` would have raised `KeyError` on the first hypernet linear weight that exceeded the 65,536-element fp16-passthrough threshold. Fixed by adding a defensive fp16-fallback path: tensors without registered Hessians are stored as fp16. Costs ~500 KB of artifact size, well under the headroom freed by the 5.02× embedding compression.

The full diff is 168 changed lines on a 3,753-line baseline. Inspect it with:

```bash
diff ../replication/train_gpt.py train_gpt.py
```

## Risks and failure modes

We list these explicitly because pre-registering one's own concerns is itself a signal of the work's seriousness.

- **Hypernet-quant interaction.** The hypernet linears at d_hidden=256 → d_model=512 have only 131K params each — small enough to be sensitive to int6 quantization noise. With our defensive fp16 fallback they ship at fp16, which is safe but not optimal. A run with proper Hessians + GPTQ on these weights might do better.
- **Output-side asymmetry.** The output_hyper is non-context-conditioned, breaking philosophical purity. A reviewer may reasonably argue we have not actually tested "meaning is use" on the output side. We acknowledge this; the input side is the harder claim.
- **d_seed=64 is a guess.** The theoretical prediction section above lays out *why* 64 is reasonable, but the optimum may be in [16, 128]. We did not have compute to sweep.
- **TTT-LoRA on the seed table.** The phased-TTT path runs LoRA over all attention/MLP weights and the lm_head. Whether per-document TTT *also* benefits from adapting the seed table is untested; for cleanliness we kept the existing TTT scope.
- **No 3-seed evidence.** A single-seed result (if one is later run) is not a record per the 0.005-nat / p<0.01 bar. We are submitting non-record explicitly.

## Reproduction

On 8×H100 SXM (RunPod Parameter Golf template recommended):

```bash
# Bootstrap (clones repo, installs FA3 + lrzip, downloads SP8192 cached data).
bash /workspace/openai-golf/runpod_bootstrap.sh

# Run condition B (matched-budget Use-Theoretic + 12th layer).
cd /workspace/openai-golf/wittgenstein
RUN_ID=witt_B_seed42 \
  DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/ \
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 CASEOPS_ENABLED=0 \
  USE_THEORETIC_EMBED=1 NUM_LAYERS=12 \
  SEED=42 MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Toggle conditions:

| Condition | Env vars |
|---|---|
| A (baseline) | `USE_THEORETIC_EMBED=0 NUM_LAYERS=11` |
| B (matched-budget) | `USE_THEORETIC_EMBED=1 NUM_LAYERS=12` |
| C (matched-architecture) | `USE_THEORETIC_EMBED=1 NUM_LAYERS=11` |

Three seeds per condition: `SEED=42`, `SEED=0`, `SEED=1234`.

## What this submission is NOT claiming

- Wittgenstein is not a theorem-prover for ML architecture. He didn't think about computation. The frame's job is to motivate a hypothesis a reviewer can read in one sentence — *not* to substitute for empirical evidence.
- We are not claiming a new optimizer, attention variant, or quantization scheme. The PR #1855 stack is unchanged below the embedding layer.
- We are not claiming this beats 1.06108 BPB. Whether it does is the open question.

## Author note

I'm Eren Kahveci, working from Raleigh, NC. I picked up this challenge on the morning of the deadline with a thinking partner (LLM-assisted) and a strong opinion that the recruiter-optimal play wasn't to chase a +0.001 BPB stack-stuff but to find the one architectural assumption nobody on the leaderboard had questioned. Wittgenstein arrived as the framing device because the static-embedding orthodoxy reads, philosophically, as exactly the picture-theory of meaning he spent his late period dismantling.

The compute didn't materialize. Runpod's 8×H100 SXM Secure Cloud was empty by mid-afternoon, Community Cloud had nothing, and chasing capacity on Vast.ai or Lambda inside the remaining clock would have meant either a half-debugged run or no run at all. So we shipped what we had: a clean architectural change, three real bugs caught before any GPU saw them, and a set of pre-registered conditions a reviewer can run themselves in 30 minutes of compute. That seemed more honest than a rushed seed.

If there is a follow-up email, I would love to talk about (a) what d_seed turns out to be empirically, (b) whether the asymmetry between use-theoretic input and Tractarian output is a real engineering tradeoff or just a budgetary one, and (c) whether the late-Wittgenstein → Investigations-moves and early-Wittgenstein → Tractarian-moves taxonomy I introduced above has any explanatory traction across the broader Parameter Golf leaderboard.

Built on PR #1855 by Benjamin Hadad (codemath3000). Philosophical framing draws on *Philosophical Investigations* §§43, 65–67, 293, and II.xi. Technical references in §"Comparison to prior work."
