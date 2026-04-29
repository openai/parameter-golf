# PR #1797 + SmearGate BOS Fix + 9-Hparam + EMBED_CLIP Relax

> **Track:** `track_non_record_16mb`
> **Headline:** 3-seed mean **val_bpb = 1.06122** (std 0.00104), **all 3 artifacts under 16 MB cap**.
> Statistically equivalent to PR #1797 (1.06157 ± 0.00066, Welch t = -0.49, p ≈ 0.32). Submitted
> as non-record because the 0.00091-nat improvement falls well below the 0.005-nat record bar.

This submission has two purposes:

1. **A clean, reproducible 3-seed run that fits the 16 MB cap** at the PR #1797 BPB floor,
   using PR #1855's 9-hparam stack with one tweaked clip threshold.
2. **A documented ablation stack** of five orthogonal techniques that *did not* transfer to
   this 5M-parameter / 16 MB / 600s budget regime, with first-principles explanations of why.

## 1. The headline configuration

| Component | Source |
|-----------|--------|
| SP8192 + CaseOps tokenization | PR #1729 @romeerp |
| 11-layer / 512-dim transformer + MuonEq-R + Polar Express NS + Sparse Attn Gate + Fused CE | PR #1787 @nprime06 |
| Smear gate (1-token causal lookback on first 12 residual dims) | PR #1797 self |
| LQER asymmetric rank-4 GPTQ correction | PR #1797 self |
| **SmearGate BOS-fix (mask gate to 0 at BOS positions)** | PR #1851 @aquariouseworkman |
| **9-hparam greedy stack** (`MLP_CLIP_SIGMAS=11.5`, `WARMDOWN_FRAC=0.85`, `BETA2=0.99`, `TTT_BETA2=0.99`, `TTT_WEIGHT_DECAY=0.5`, `TTT_LORA_RANK=80`, `SPARSE_ATTN_GATE_SCALE=0.5`, `PHASED_TTT_PREFIX_DOCS=2500`) | PR #1855 @codemath3000 |
| **EMBED_CLIP_SIGMAS=20.0** (instead of 14) — saves ~280 KB compressed; **THIS submission's contribution** | this PR |

### Per-seed numbers (8×H100 SXM, brotli q=11, ≤ 600 s eval)

| Seed | Pre-quant BPB | Post-quant BPB | TTT phased BPB | Artifact bytes | Eval time |
|------|--------------:|---------------:|----------------:|---------------:|----------:|
| 42   | 1.0635 | 1.0736 | **1.06099969** | 15,862,425 | 521.5 s |
| 314  | 1.0629 | 1.0730 | **1.06032332** | 15,860,925 | 459.9 s |
| 1234 | 1.0648 | 1.0751 | **1.06235764** | 15,861,284 | 457.2 s |
| **mean** | **1.0637** | **1.0739** | **1.06122**   | **15,861,545** | **479.5 s** |
| **std**  |        |        | **0.00104**     | 803            |          |

Direct seed-pair comparison vs PR #1797's published seeds (matched 42/314/1234):

| Seed | PR #1797 | this PR | Δ (BPB) |
|------|---------:|--------:|--------:|
| 42   | 1.06181 | 1.06100 | **−0.00081** |
| 314  | 1.06103 | 1.06032 | **−0.00071** |
| 1234 | 1.06209 | 1.06236 | +0.00027 |
| **mean** | 1.06157 | **1.06122** | **−0.00035** |

Welch t-test on the two 3-seed samples: t = −0.49, df ≈ 4, one-sided p ≈ 0.32. **Not a record**
(0.005-nat bar = 0.00194 BPB; we cleared 0.00091 nats / 0.00035 BPB only). Counts as a clean,
reproducible non-record reproduction with a documented compressed-artifact saving.

## 2. The EMBED_CLIP relaxation finding

PR #1855's 9-hparam stack with brotli q=11 produced 16,140,160-byte artifacts in the original
PR (just under their lrzip-augmented compressor's output of 15,907,550). Reproducing the exact
same training config in our codebase (no lrzip) reliably overran the 16 MB cap by 144 KB
(Phase G in the ablation log). After exhausting the obvious knobs (compressor swap, bit-width,
clip-tightening, vocab change), we found a counterintuitive single-knob fix:

> **Relaxing EMBED_CLIP_SIGMAS from 14 → 20 saves ~280 KB compressed.**

Phase G (EMBED_CLIP=14, MLP_CLIP=11.5): 16,144,312 bytes (over by 144 KB), mean 1.05969 BPB
Phase Q (EMBED_CLIP=20, MLP_CLIP=11.5): **15,861,545 bytes (under by 138 KB)**, mean 1.06122 BPB

The artifact savings cost ~1.5 milli-BPB on the mean (likely from less aggressive embedding
quantization). The mechanism appears to be that a *looser* clip preserves more of the natural
Gaussian structure of the embeddings, which brotli compresses better than the more uniform
post-clip distribution that a tighter clip produces. (We verified this is empirically robust
across both MLP_CLIP=10 and MLP_CLIP=11.5 settings.) This may be useful to other submissions
that hit the cap with the 9-hparam stack.

## 3. Five negative-result ablations

All ablations live in this submission's `train_gpt.py` behind env-var flags. The legality of
each is proven by 23/23 unit tests in `test_ngram_legality.py` (CPU-only, ~3 s).

### V1 — `NGramMixer` (prefix-only causal bigram with Dirichlet smoothing)

  - **Hypothesis:** Mix the neural softmax with a non-parametric bigram tallied online from
    already-scored tokens; binary-λ gate based on prefix entropy.
  - **Result:** All 11 sweep configs hurt by **+0.058 to +1.27 BPB** vs baseline. None helped.
  - **Lesson (first-principles):** Cold-start q_bi is uniform/8192 ≈ 1.2e-4. At λ=0.88 the
    mixer injects ~0.13 nats/token of pure noise into the NN output. The fix would be a
    PPM-D-style escape mechanism (backoff to lower-order context); a fixed-order bigram with
    uniform Dirichlet prior **cannot** do this without cold-start data.

### V2 — `TempScaler` (per-doc running-entropy temperature scaling)

  - **Hypothesis:** Adjust softmax temperature based on prefix-only running NLL to recalibrate
    over-/under-confident regions.
  - **Result:** Best of 14 sweep configs gave **−0.00003 BPB** (well within run-to-run noise).
  - **Lesson:** The PR #1797 model was trained with cross-entropy loss, which directly fits
    softmax sharpness. There is no calibration error to fix when train and val distributions
    are matched (FineWeb val is an i.i.d. sample of FineWeb train). Temperature scaling is a
    fix for systematic train/eval shift; it has zero headroom here.

### V3 — `TokenPPMMixer` (PPM-D order-2 over the token alphabet, clean Condition 2)

  - **Hypothesis:** Adapt the byte-level PPM-D mixture from PR #1835/#1850/#1854/#1873 (the
    "PPM-D byte cluster" currently disputed in [Issue #1872](https://github.com/openai/parameter-golf/issues/1872)
    on Condition 2 grounds) to operate on the SP8192 token alphabet. Defining the mixture
    distribution on Σ_token instead of the byte alphabet sidesteps the C2 dispute cleanly.
  - **Result on synthetic data:** **−3.2 nats/token** improvement over a uniform baseline on
    a stream with embedded repetition motifs (CPU unit test passes; PPM-D mechanism works).
  - **Result on real FineWeb:** **+0.0155 BPB regression** in eval_val. Token-level PPM does
    not help.
  - **Lesson (first-principles):** Byte-level PPM-D works because the 256-symbol byte
    alphabet *saturates* at order 5 — most 5-byte contexts repeat hundreds or thousands of
    times in 35M tokens of val. Confidence (max_count / denom) frequently exceeds the 0.9
    gate threshold, so the binary λ gate fires often. **The token alphabet (V=8192) has
    sparse contexts even at order 2** (V² = 67M cells, only 35M observations). The PPM-D
    confidence gate almost never fires; the mixer effectively defaults to its high-λ branch
    (mostly NN, slightly perturbed by a noisy q_PPM). The key ratio is
    *symbols_in_alphabet / total_observations*: byte-level is dense (256 / 140M ≈ 1.8e-6),
    token-level is sparse (8192 / 35M ≈ 2.3e-4) — two orders of magnitude denser at byte
    level. This is structural, not tunable.

### V4 — SP10240 + CaseOps (vocab progression hypothesis)

  - **Hypothesis:** The leaderboard's vocab progression 1024 → 2048 → 4096 → 8192 has been
    monotonically beneficial. Try SP10240 (and a custom CaseOps SP10240 tokenizer trained on
    the FineWeb-10B docs corpus, uploaded to `hf://FijaEE/parameter-golf-sp10240-caseops`).
  - **Result on a single seed:** Pre-quant val_bpb 1.06497 (−0.0015 vs SP8192 V2 baseline
    1.06321), but post-TTT val_bpb 1.06167 (+0.0017 vs SP8192 V2 baseline 1.05998), and
    artifact 16.49 MB (over cap by 494 KB).
  - **Lesson:** A 5M-parameter model cannot absorb the long-tail vocabulary required for a
    bigger SP. The embedding table grows ~25% (10240/8192), eating most of the artifact
    budget; the model gets noisier per-token softmax distributions. Net effect on FineWeb
    BPB: very slightly *worse* under TTT, plus invalidating the 16 MB cap. The vocab
    progression hypothesis transfers up to a parameter-budget-dependent ceiling.

### V5 — EMBED_BITS = 6 (artifact reduction probe)

  - **Hypothesis:** Drop the embedding from INT7 to INT6 to recover ~500 KB.
  - **Result:** Saves 530 KB on artifact, but adds **+3.9 milli-BPB** to TTT BPB.
  - **Lesson:** 6-bit embeddings damage the per-token softmax probabilities far more than
    6-bit attention/MLP weights do. The eval-time BPB is computed token-by-token; embedding
    quantization noise propagates directly into the scored output distribution. The
    cost/benefit ratio for embedding precision reduction is much worse than for matrix
    precision reduction.

## 4. Compressor empirical findings

Same exact training config (PR #1797 V2 + 9-hparam, MLP_CLIP=11.5), three compressors:

| Compressor | Artifact bytes | Δ vs brotli q=11 |
|------------|---------------:|-----------------:|
| brotli q=11 (default) | 16,144,312 | baseline |
| lzma preset=6         | 17,250,171 | **+1.1 MB** |
| lzma preset=9 EXTREME | 17,856,165 | **+1.7 MB** |

For PR #1797-style quantized weight blobs (post-byte-shuffle, mostly INT6 + per-row scales),
**brotli q=11 is the best off-the-shelf compressor**. lzma at any preset is worse, contrary to
what one might expect on text-like data. The leaderboard's most aggressive submissions
(PR #1855 etc.) chain brotli with custom preprocessing (per-group lrzip) to extract the
extra ~200 KB; in our submission we got that gain through the EMBED_CLIP relaxation instead.

## 5. Reproduction

```bash
# Inside an 8xH100 80GB SXM pod with HF_TOKEN exported:
git clone --branch submission/pr1797-ngram-mix https://github.com/Fija/parameter-golf.git
cd parameter-golf/records/track_non_record_16mb/2026-04-28_PR1797_EmbedClipRelax_AblationStack/

# Pull pre-tokenized SP8192 CaseOps shards (29 GB):
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='FijaEE/parameter-golf-sp8192-caseops',
                  repo_type='dataset', token='$HF_TOKEN',
                  local_dir='/workspace/data/datasets/fineweb10B_sp8192_caseops',
                  max_workers=16)
"

# Verify legality (CPU, 23/23 must pass):
python3 test_ngram_legality.py

# Run a seed (env vars below are the headline config):
DATA_PATH=/workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/ \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
VOCAB_SIZE=8192 CASEOPS_ENABLED=1 \
SEED=42 RUN_ID=phq_s42 QUANTIZED_MODEL_PATH=/workspace/runs/phq_s42/model.bin \
MAX_WALLCLOCK_SECONDS=620 \
FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
TTT_WARM_START_A=1 EMBED_BITS=7 MIN_LR=0.1 MATRIX_LR=0.026 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=20.0 \
WARMDOWN_FRAC=0.85 BETA2=0.99 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 SPARSE_ATTN_GATE_SCALE=0.5 \
COMPRESSOR=brotli \
NGRAM_MIX_ENABLED=0 TEMP_SCALE_ENABLED=0 PPM_MIX_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A full 3-seed driver with abort guards and pod auto-stop lives at
`runpod/phase_q_embedclip20.sh`.

## 6. Lineage

- PR #549, #1019, #1394 — merged baselines
- PR #1493 (@bigbag, 2026-04-09) — current merged SOTA at 1.0810
- PR #1729 (@romeerp) — CaseOps tokenizer
- PR #1736, #1787 (@nprime06) — Sparse Attn Gate, Polar Express NS, MIN_LR, Fused CE
- PR #1797 (@dexhunter) — SmearGate, LQER asymmetric, base for this submission
- PR #1851 (@aquariouseworkman) — SmearGate BOS-leak fix (inlined)
- PR #1855 (@codemath3000) — 9-hparam greedy stack (8 of 9 inlined; EMBED_CLIP swapped)
- PR #1812 (@EthanNing) — observed but not adopted (PR #1493's TTT path, not PR #1797's)
- Issue #1017 (@NoesisGenesis) — four conditions used in the legality unit tests
- Issue #1872 (@andrewbaggio1, @sharpobject) — C2 dispute that motivated the token-level PPM-D
  ablation in V3

## 7. Honest summary

We did not produce a record. We produced one valid 3-seed reproduction at the PR #1797 BPB
floor, plus five rigorously-tested orthogonal ablations whose negative results give a
first-principles explanation of why the obvious knobs (vocab size, embedding precision,
non-parametric mixers at the token alphabet, temperature scaling, lzma compression) do not
transfer to this regime. The EMBED_CLIP relaxation may be a small but useful tool for other
participants whose 9-hparam-style submissions are running just over the 16 MB cap.
