# Non-record: Confidence-Adaptive N-gram Boost on PR #2018 stack

**val_bpb = 1.05874** (seed 42, single-seed, ADAPTIVE_BOOST_GAMMA=1) | artifact 15,990,227 bytes | 8xH100 SXM | strict 600s train + eval

This is a non-record submission. It does not clear the 0.005-nat record threshold versus PR #1855 SOTA 1.06108 (margin is about 0.00162 nats/byte, below the 0.005 floor). It documents a clean novel addition to the strict-token-only n-gram tilt path from the PR #2018 stack and reports a small but consistent positive direction.

## What is novel

**Confidence-Adaptive N-gram Boost.** The token-only n-gram tilt in PR #2018 applies a fixed boost beta whenever the prefix-derived hint counter exceeds threshold. This submission scales beta per-position by the NN's own predictive confidence at the scored position:

```
beta_t = TOKEN_BOOST * (1 - q_hint_t)^gamma
```

where `q_hint_t = p(h_t | x_{<t})` is the NN distribution at position t evaluated on the hinted token, and `gamma` is a tunable exponent (env var `ADAPTIVE_BOOST_GAMMA`, default 0 = original behavior).

When the NN already places high probability on the hinted token (q_hint_t -> 1), beta -> 0 and we tilt almost not at all (NN already agrees). When the NN disagrees with the hint (q_hint_t -> 0), beta -> TOKEN_BOOST and we apply the full tilt. This matches the intuition that the n-gram expert is most useful as a corrective signal precisely when the NN is uncertain or wrong.

## Compliance

The closed-form tilt in `online_ngram_tilt.apply_tilt_to_ptl_torch_fast` is:

```
p'(a) = p(a) * exp(beta * 1[a == h]) / Z
Z     = 1 + q_hint * (exp(beta) - 1)
```

This is normalized over the vocab axis for any beta_t >= 0 that depends only on prefix-derived state. The adaptive scaling preserves all four legal-TTT conditions:

- **C1 causal:** q_hint_t is computed from the same prefix-only NN distribution that produces p_t. h_t comes from the prefix-only n-gram counter. No information from token t or later is used.
- **C2 normalized:** Z_t closed-form is the analytic normalizer of the per-position tilt for the per-position beta_t. Since beta_t is fixed for a given prefix, the resulting p'(.) sums to 1 over the SP8192 alphabet.
- **C3 score-before-update:** The tilt is applied to the per-token NLL at scoring time only. No parameter updates are driven by val tokens before they are scored.
- **C4 single pass:** Each val token is scored exactly once via the existing single-pass sliding eval inherited from the PR #2018 stack.

The base stack inherits PR #2018's strict token-only configuration (`WITHIN_BOOST=0 WORD_BOOST=0 AGREE_ADD_BOOST=0`), addressing the C1 concerns raised on PR #2118 and resolved on PR #2018 / PR #1514. Run logs confirm `within_gate=0 word_gate=0 agree2plus=0`.

## Result

| variant | seed | quantized_ttt_phased val_bpb | eval_time | artifact_bytes |
|---------|-----:|-----------------------------:|----------:|---------------:|
| baseline (gamma=0, our reproduction of PR #2018) | 42 | 1.05900 | 479.5s | 15,991,083 |
| **adaptive gamma=1 (this submission)** | 42 | **1.05874** | 449.9s | 15,990,227 |
| adaptive gamma=2 | 42 | 1.05878 | 450.4s | 15,990,227 |

The adaptive boost is monotonically positive across `gamma in {1, 2}` versus the constant-boost baseline on this stack. `gamma=1` is the best so far.

The reproduction of PR #2018's full pipeline on our pod lands at val_bpb 1.05900 vs the PR #2018 README's reported 1.04617 for seed 42, a +0.013 BPB gap before the adaptive change. The gap is consistent across pre-quant (1.06351 vs 1.04931), quantized (1.07123 vs 1.05773), and post-TTT (1.05900 vs 1.04617), so it's a base-model issue, not a tilt issue. We have not yet identified the source. The adaptive-boost layer is reported on top of our reproduction baseline.

## Compliance summary

- Train: 596,039 ms < 600,000 ms cap.
- Eval: 449,899 ms < 600,000 ms cap.
- Artifact: 15,990,227 bytes < 16,000,000 byte cap.
- Tokenizer: SP8192 lossless caps caseops v1 reserved (md5 b73929616bf6303b953396b767a29b99).
- 8xH100 80GB SXM.
- C1 / C2 / C3 / C4 all satisfied (see above).

## Files

- `README.md`
- `train_gpt.py` (PR #2018 train script + 2-line adaptive-boost wiring)
- `online_ngram_tilt.py` (PR #2018 tilt module + adaptive-gamma multiplication)
- `online_ngram_state.c` (unchanged from PR #2018)
- `prepare_caseops_data.py` (unchanged from PR #2018)
- `lossless_caps.py` (unchanged from PR #2018)
- `run.sh` (unchanged from PR #2018; set `ADAPTIVE_BOOST_GAMMA=1` to enable)
- `train_seed42.log` (single-seed run log)

## Why a non-record

To clear the 0.005-nat record threshold versus the merged PR #1855 SOTA (1.06108) we would need the 3-seed mean to be at most about 1.05387. Our single-seed adaptive-boost result on this PR #2018 reproduction is 1.05874. To get to record territory the reproduction gap (+0.013 BPB versus PR #2018's reported numbers) needs to be closed first. The adaptive boost on top of a properly reproduced PR #2018 baseline could plausibly land below the threshold versus PR #1855, but we are not confirming that here.

This submission is offered for the novel technique only.

## Reproduction

```bash
pip install brotli sentencepiece huggingface_hub numpy
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu129_torch291/
apt-get install -y lrzip

python3 -c "from huggingface_hub import snapshot_download;import os;snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', repo_type='dataset', local_dir='./data', max_workers=16)"

DATA_PATH=./data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
SEED=42 NPROC_PER_NODE=8 ADAPTIVE_BOOST_GAMMA=1 \
bash run.sh
```

## Credits

- PR #2018 by @simon-marcus: full base stack (strict token-only n-gram tilt + Gated XSA + LQER top-1 + AWQ-lite + AsymLogit + phased TTT). This submission is two small edits on top.
- PR #1145 / @AnirudhRahul (n-gram tilt origin), PR #1514 token-only legality precedent.
- PR #1855 / @codemath3000: prior architecture base.
- LQER (Yao et al. 2024), GPTQ (Frantar et al. 2023).

cc @cocohearts @valerio-oai @simon-marcus for visibility (non-record review).
