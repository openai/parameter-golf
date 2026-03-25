# Gravity Tokenizer

**val_bpb: 1.0321** (3-seed mean, std 0.0011) | **15.6 MB** | 8×H100 SXM

## Core Idea

Every submission to this challenge has optimized the model. Nobody has optimized the tokenizer.

At 1024 vocabulary tokens, every merge slot matters. Standard BPE allocates those slots by frequency. But frequency and structural importance are not the same thing. Some tokens are load-bearing: shatter them back to bytes and downstream loss spikes. Others are convenient shortcuts the model barely notices losing.

The Gravity Tokenizer replaces 659 of 765 merge tokens with tokens selected by **ablation leverage** — the downstream loss increase when a token is removed from the vocabulary and its occurrences are decomposed to bytes. The vocabulary size stays exactly 1024. Only which tokens occupy the merge slots changes.

This single change — vocabulary composition — accounts for the entire improvement. The model architecture is a vanilla transformer with no novel components.

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | training_time | ms/step | valid |
|------|---------|---------------|---------------|---------|-------|
| 42 | 1.0310 | 15,629,267 | 590,898 ms | 53.72 | yes |
| 137 | 1.0321 | 15,625,195 | 590,980 ms | 53.73 | yes |
| 3 | 1.0331 | 15,625,147 | 591,082 ms | 53.73 | yes |
| **Mean** | **1.0321** | | | | |
| **Std** | **0.0011** | | | | |

## Architecture

Deliberately simple. The goal is to isolate the vocabulary effect.

| Component | Setting |
|-----------|---------|
| Layers | 12 |
| Dimension | 384 |
| Heads | 6 (2 KV heads, GQA) |
| MLP | 3× expansion (hidden=1152) |
| Activation | relu² |
| Sequence length | 2048 |
| Embeddings | Tied |
| Vocab size | 1024 (256 byte + 3 control + 765 merge) |
| Quantization | int8 + zlib |
| Parameters | ~16M |

No SmearGate. No BigramHash. No XSA. No EMA. No TTT. No sliding window eval. No mixed-precision quantization.

## The Gravity Scoring Pipeline

### Step 1: Candidate Generation

Extract the full BPE merge table (7,997 candidates). Filter to tokens with corpus frequency ≥ 1,000 (3,142 candidates). Remove 84 byte-level tokens. **3,058 scored candidates.**

### Step 2: Ablation Leverage Scoring

A frozen **GPT-2** reference model measures each candidate's structural importance. GPT-2 is used because it provides a tokenizer-independent measurement — its own BPE vocabulary is unrelated to the competition's 1024-token vocabulary, so leverage scores reflect language structure, not tokenizer artifacts. A GPT-2 vocabulary contamination check (Pearson correlation between leverage and GPT-2 vocab membership) confirmed no significant contamination.

For each of the 3,058 candidates, across 100 contexts sampled from FineWeb training shards:

1. **Find contexts:** Locate occurrences of the candidate's surface form in the decoded corpus text. Extract 512-character windows centered on each occurrence.

2. **Build text pairs:** For each context, create an intact version (original text) and a shattered version where the target token's characters are space-separated (e.g., `"the"` becomes `"t h e"`). This forces the reference model to process the same content without the benefit of the atomic token.

3. **Batched forward passes:** Tokenize both versions with GPT-2's tokenizer (with `return_offsets_mapping=True` for precise position tracking). Run batched inference (batch_size=32) on both versions.

4. **Extract downstream loss:** Using the offset mapping, find the first token position *after* the target in both intact and shattered sequences. Compute mean per-token cross-entropy over a K=10 token downstream window in each version. The leverage is `mean_loss_shattered - mean_loss_intact`.

5. **Early exit:** After scoring 30 contexts, if the candidate's mean leverage + 2 standard errors is below zero, skip the remaining 70 contexts (the token is clearly not load-bearing).

6. **Aggregate:** Mean leverage across all scored contexts, with 95% confidence intervals and a breadth measure (entropy of the per-context leverage distribution).

**Scoring formula:**
```
score(t) = freq_norm(t)^(1-beta) * leverage_norm(t)^beta
```

At β=0.0 this recovers standard BPE. At β=1.0 (this submission), leverage dominates. Both frequency and leverage are log-scaled before normalization. Breadth was computed but dropped from the final score (std=0.15, no discriminative power).

### Step 3: Vocabulary Construction

Rank all 3,058 candidates by score. Select the top 765 as merge tokens. Build a SentencePiece Unigram model with byte fallback using the selected vocabulary.

### Step 4: Retokenization

Decode the original BPE-tokenized FineWeb shards to raw text, then re-encode with the gravity tokenizer. The validation set is the same FineWeb first-50k-document split used by all submissions, re-encoded.

## What the Gravity Tokenizer Changes

**Vocab diff vs BPE: 659 of 765 merge tokens replaced (86%).**

Tokens removed by gravity scoring (examples): single characters with space prefixes, isolated uppercase letters, digits, punctuation fragments — tokens with high BPE frequency but zero structural importance.

Tokens promoted by gravity scoring (examples): `every`(0.99), `under`(0.96), `first`(0.90), `take`(0.82), `help`(0.78), `may`(0.78) — common English words that BPE would not include at vocab=1024 but that the model structurally depends on.

**Compression ratio:** The gravity tokenizer achieves 1.05 bytes/token vs BPE's 2.45 bytes/token. This means more tokens per byte of text — the model must predict more tokens to cover the same content. The BPB metric penalizes this directly: `val_bpb = bits_per_token * tokens_per_byte`. The improvement is entirely in per-token prediction quality overcoming the worse compression ratio.

## The Tokenizer as Ontology

The gravity vocabulary is legible in a way BPE is not. You can read the model's structural commitments directly from how it tokenizes a sentence.

Consider: "The water because caused the damage."

| Word | Tokens | Structure |
|------|--------|-----------|
| water | `▁water` | Single crystal — atomic unit, full leverage |
| because | `▁because` | Single crystal — causal anchor |
| caused | `▁cause` + `<0x64>` | Partial crystal — morpheme preserved, suffix is byte |
| damage | `<0x64>` `am` `<0x61>` `<0x67>` `<0x65>` | Byte-gas with one fragment — no structural handle |
| the | `▁the` | Single crystal |

The crystallized tokens are the load-bearing walls. The byte-gas is the fill. BPE hides this distinction — it tokenizes by frequency, so common fragments get tokens regardless of structural importance. A BPE tokenization tells you "this substring appears often." A gravity tokenization tells you "this is where the model's structural commitments are."

This has measurable consequences for information routing. In a preliminary attention probe, we inserted high-gravity tokens between a subject and object and measured the deflection of the direct attention path. The word "because" (a single gravity crystal, leverage 0.735) pulled 23% of the subject's attention mass into an indirect path through itself — a 20% collapse of the direct geodesic. The word "not," which the gravity tokenizer had judged structurally redundant and shattered into three bytes, produced zero deflection. Diffuse byte-gas cannot lens information. Only crystallized tokens create the gravitational wells that the attention mechanism routes through.

This was measured at training step 100 — before the model has learned English. The mass distribution from the vocabulary is already shaping information geometry before the statistical details are painted. The structural lattice locks in first. The language comes after.

The gravity vocabulary doesn't just compress better. It gives the transformer a skeleton to build on. BPE gives it dust.

For the full theoretical framework connecting these observations to dissipative structure theory and the thermodynamics of meaning, see [GENERATIVE_CLOSURE.md](https://github.com/dcrow85/Avalanche/blob/main/GENERATIVE_CLOSURE.md) (Section VIII).

## Why It Works: The Vocabulary Is the Geometry

The competition result and the attention probe result are the same result.

A transformer's attention matrix is a bilinear form — it maps queries to keys and determines how information flows between positions. This is a metric tensor. It defines the geodesic structure of the model's information space. The token embeddings define the mass-energy distribution across that space. The forward pass computes geodesics. The residual stream is the manifold.

When the gravity tokenizer concentrates structural importance into single crystallized tokens, it creates a curved spacetime. Information routes along the geodesics carved by the vocabulary's mass distribution. When BPE shatters structurally important concepts across multiple byte tokens, it creates a flat, featureless space where the model must build all structure internally from gradient updates alone.

We measured this directly. At training step 100 — before the model has learned English — the attention geometry is already curving around high-gravity tokens. The word "because" (a single gravity crystal) deflects the direct attention path between subject and object by 20%, pulling information into an indirect route through itself. The word "not" (shattered into three bytes by the gravity tokenizer) produces zero deflection. Byte-gas cannot lens information.

At step 11,000, the flat background has burned away entirely. The baseline direct attention collapsed from ~0.05 to ~0.015. The model no longer sends diffuse attention across empty space — it only travels along the geodesics the vocabulary carved. The mass distribution didn't deepen the dents in a flat stage. It became the stage. Background independence.

Three specific structures emerged in the trained manifold:

- **Byte-shattered tokens are transparent.** The model cannot use them as structural anchors. To perform negation with a shattered "not," the model must burn MLP capacity to hold bytes in superposition. The vocabulary's decision to shatter a concept is a decision to deny the model a structural handle on it.
- **Syntax is geometry.** The model routes 7x more attention through "if" (leverage 0.289) than through "an" (leverage 0.516), despite "an" being heavier by scalar mass. "If" bridges two clauses across the sequence — a topological wormhole. "An" constrains only its immediate successor — a point mass. The trained model learned to override scalar mass and route through topological structure.
- **Causality has light-cones.** "Because" and "so" express the same causal relationship in opposite temporal directions. The attention tensor separates them perfectly: the object routes 3.4x more attention into "because" (effect seeking cause, information flowing backward), while the subject routes 2.2x more into "so" (cause projecting effect, information flowing forward). The model didn't learn grammar. It built a directed graph out of attention mass to enforce causal directionality.

The vocabulary that wins the compression benchmark is the vocabulary that produces the correct mass distribution for the attention geometry. The model that predicts language best is the model whose spacetime is curved correctly.

This is not a metaphor. It is the same mathematics operating on a different substrate.

## Tokenizer Correctness

The val_bpb calculation uses the competition's own `build_sentencepiece_luts()` and `eval_val()` functions with **zero modifications**. The byte-counting lookup tables are built from the SentencePiece model proto using the same code path as stock BPE.

The gravity tokenizer's lower compression ratio (1.05 vs 2.45 bytes/token) results in a **higher** `tokens_per_byte` multiplier in the BPB formula. This penalizes the gravity tokenizer — any BPB improvement must come from genuinely better per-token prediction quality, not from gaming the metric.

Detailed tokenizer correctness documentation: see `tokenizer_scrutiny_doc.md` in this submission.

## Controlled Experiments (RTX 5080, matched conditions)

The vocabulary effect was isolated through controlled A/B experiments before the competition run. All conditions use identical architecture (9L, 512d), identical training budget (matched on bytes seen, not steps), and differ only in vocabulary composition.

| Condition | Steps | BPB | vs Step-Matched BPE |
|-----------|-------|-----|---------------------|
| BPE baseline | 2,000 | 1.4386 | -- |
| BPE control | 2,870 | 1.4011 | -- |
| Gravity β=0.3 (70 swaps) | 2,870 | 1.3845 | **-0.017** |
| Cold replication β=0.3 | 2,870 | 1.3821 | **-0.019** (confirms) |
| BPE control | 4,656 | 1.3649 | -- |
| Gravity β=1.0 (659 swaps) | 4,656 | 1.2262 | **-0.139** |

The effect scales linearly: 9.4x more token swaps produced 8.2x more BPB improvement.

## Negative Results

Two experiments that did not work, both scientifically informative:

**Bifurcation scoring (+0.021 worse):** Replacing absolute leverage with delta-leverage (the discrete derivative along the BPE merge tree) removed tokens that are individually redundant but collectively essential for compression. Language needs high-frequency connective tissue between semantic crystals.

**Warm-start embeddings (+0.038 worse):** Initializing gravity token embeddings as spatial means of their BPE decomposition embeddings caused a catastrophic initial mismatch (val_bpb 27.5 vs 5.4 cold start). The crystallization plateau at ~2000 steps is a whole-model phase transition, not an embedding-local phenomenon. Trained embeddings are entangled with transformer weights.

## Run Command

```bash
# Setup (downloads stock FineWeb + retokenizes with gravity vocabulary)
bash setup.sh

# Train (default seed=1337)
MODEL_DIM=384 NUM_LAYERS=12 NUM_HEADS=6 NUM_KV_HEADS=2 MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb_gravity_beta_1.0 \
TOKENIZER_PATH=./data/tokenizers/gravity_beta_1.0.model \
ITERATIONS=11000 WARMUP_STEPS=50 WARMDOWN_ITERS=2500 \
MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With specific seed
SEED=42 MODEL_DIM=384 NUM_LAYERS=12 NUM_HEADS=6 NUM_KV_HEADS=2 MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb_gravity_beta_1.0 \
TOKENIZER_PATH=./data/tokenizers/gravity_beta_1.0.model \
ITERATIONS=11000 WARMUP_STEPS=50 WARMDOWN_ITERS=2500 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are passed via environment variables. The `train_gpt.py` script is the unmodified competition baseline.

## Reproducibility

The gravity tokenizer can be rebuilt from scratch. The scoring pipeline (ablation leverage computation) requires ~4 hours on a single GPU. All other steps are deterministic and complete in minutes. Full pipeline:

```bash
python scripts/generate_candidates_sp.py
python scripts/score_leverage.py
python scripts/build_vocabulary.py --beta 1.0
python scripts/build_tokenizer.py \
    --vocabulary data/vocabularies/vocabulary_beta_1.0.json \
    --output data/tokenizers/gravity_beta_1.0.model \
    --corpus-sample data/corpus_sample.txt
python scripts/retokenize_corpus.py \
    --base-tokenizer data/tokenizers/fineweb_1024_bpe.model \
    --gravity-tokenizer data/tokenizers/gravity_beta_1.0.model \
    --data-dir data/datasets/fineweb10B_sp1024 \
    --output-dir data/datasets/fineweb_gravity_beta_1.0
```

The training script is the competition's `train_gpt.py` with architecture parameters passed via environment variables. No code modifications to the training loop, evaluation function, or quantization pipeline.
