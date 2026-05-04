# Record: PR #1735 + CaseOps Tokenizer (V15) - val_bpb 1.0354

## Summary

- **val_bpb = 1.03540487** (3-seed mean, std 0.00056684) | **~16.0 MB** | 8xH100 SXM
- **Immediate stack:** PR #1735 parallel pre-quant AdamW TTT plus PR #1729 CaseOps tokenizer/byte-sidecar data path, integrated in PR #1738
- **Improvement:** -0.00750 BPB vs PR #1735 (1.04290), narrowly clearing the record threshold of -0.005 nats / -0.00721 BPB
- **Independent reproduction:** seed 1337 reproduced from this folder on 2026-04-28/29 at **1.03459029 BPB** with a **15,996,563 byte** artifact

This is an integration record, not a claim that every idea here originated in one PR. The important thing is the combination: the strongest available pre-quant TTT stack and a lossless tokenizer/data transform that makes the same model budget do less redundant work on casing.

## Why This Combination

The recent frontier PRs made the search space pretty clear. Small architecture knobs still matter, but the large steps came from two mostly orthogonal directions:

1. **Pre-quant TTT from PR #1735 / PR #1364** adapts the full-precision EMA model before GPTQ. It turns otherwise-unused evaluation budget into a better fixed artifact, then exports a quantized predictor.
2. **CaseOps from PR #1729** reduces case fragmentation in the token stream by representing casing as reversible operators over a lower-case lexical stream. It is still charged against the original UTF-8 bytes through byte sidecars.

Those two should compose: CaseOps makes the language modeling target cleaner, while pre-quant TTT spends the available time adapting the weights to that target before quantization. The one piece that had to be added for this specific record was byte-sidecar support inside the PR #1735 eval functions, because the transformed token stream cannot be evaluated with naive token-to-byte accounting.

## Results

| Seed | Sliding val_bpb | Artifact bytes |
|------|----------------:|---------------:|
| 1337 | 1.03484145 | 15,996,061 |
| 42   | 1.03618043 | 15,996,195 |
| 999  | 1.03519273 | 15,994,993 |
| **Mean** | **1.03540487** | **15,995,750** |
| Std | 0.00056684 | |

Current SOTA at the time of the record lineage was PR #1735 at 1.04290 BPB. This record improves that by 0.00750 BPB. The required threshold for a new record is 0.005 nats, about 0.00721 BPB, so the margin is small but positive.

## Independent Reproduction

| Date | Seed | Sliding val_bpb | Artifact bytes | Notes |
|------|-----:|----------------:|---------------:|-------|
| 2026-04-28/29 | 1337 | **1.03459029** | **15,996,563** | 8xH100 reproduction of this record folder |

Key reproduction checkpoints:

- Training stopped at the wallclock cap: `588132ms`, step `4568/20000`
- Pre-quantization post-EMA: `val_bpb=1.08389912`
- Pre-quant TTT epoch 21: `val_bpb=1.028560`
- After 21 pre-quant TTT epochs: `post-prequant-ttt val_bpb=1.02819756`
- Serialized full-precision model: `135,431,033` bytes
- Code size: `24,732` bytes
- GPTQ collected `67` Hessians in about `50s`
- Quantized model plus Brotli: `15,971,831` bytes
- Total submission size: `15,996,563` bytes
- Quantized non-sliding eval: `val_bpb=1.04801825`
- Quantized sliding-window eval: `val_bpb=1.03459029`, `eval_time=134105ms`

## What Changed In This Record

### CaseOps support inside the PR #1735 stack

PR #1735 did not know about CaseOps byte sidecars. CaseOps inserts private-use capitalization operators into the token stream, so counting bytes by decoding transformed tokens would charge the wrong denominator. This record adds `load_validation_token_bytes()` and threads the byte sidecar through:

- `eval_val()`
- `eval_val_sliding()`
- `eval_val_ttt()`

The eval path uses `fineweb_val_bytes_*.bin` when present and falls back to LUT-based byte counting for normal SP8192 data. `load_validation_tokens()` also excludes `_bytes_` files so validation token shards are not accidentally double-counted.

### CaseOps tokenizer/data path

CaseOps factorizes text into a lower-case lexical stream plus reversible case operators such as title-case, all-caps, cap-next, and escape. The model sees fewer redundant capitalization variants, but the original text remains exactly recoverable. Validation BPB is computed against original raw UTF-8 byte counts via sidecar files.

### Parallel pre-quant AdamW TTT

The pre-quant TTT path follows PR #1735: each of 8 ranks works on an interleaved subset of validation chunks, trainable weights are averaged across ranks after each epoch, and the LR decays across epochs rather than restarting every chunk. That makes 21 AdamW epochs feasible inside the time budget before GPTQ export.

## Technique Inventory

This specific record folder uses the following stack:

- SP8192 CaseOps tokenizer with reversible case-control operators
- Per-token original-byte sidecars for BPB accounting on transformed validation tokens
- 11-layer, 512d, 8-head / 4-KV-head transformer
- XSA on all layers
- 3-layer depth recurrence over layers 3-5, giving 17 virtual layers from 11 physical layers
- Parallel residual decoder path starting at layer 7
- QK-Gain initialized to 5.25
- LeakyReLU(0.5)^2 MLP with `mlp_mult=4.0`
- Skip gates, layer scaling, EMA, SWA, Muon-family optimization, high-WD compression pressure, and warmdown scheduling inherited through the record stack
- 8-GPU parallel pre-quant AdamW TTT for 21 epochs
- Full-Hessian GPTQ with SDClip-style row clipping for int6 model matrices
- Int8 embedding quantization
- Brotli-compressed artifact, with the code LZMA-wrapped, under the 16,000,000 byte limit
- Sliding-window evaluation with stride 64

## Lineage And Credits

I am not trying to compress the credits into a tiny shortlist. This record is a community stack, and the PRs below are the lineage I traced for the techniques that are actually used or directly led to the used integration.

| PR | Contributor | Role in this record lineage |
|----|-------------|-----------------------------|
| #1738 | @alertcat | Exact CaseOps V15 integration record: PR #1735 plus CaseOps byte-sidecar support. This folder is based on that record. |
| #1735 | @AjAnubolu | 8-GPU parallel pre-quant AdamW TTT, 21 epochs, federated averaging, epoch-level cosine LR, torch.compile acceleration. |
| #1729 | @romeerp | CaseOps lossless capitalization tokenizer/data export and validation byte-sidecar accounting. |
| #1626 | @dexhunter | Multi-phase global SGD TTT lineage used by the CaseOps PR; helped establish the score-first phased adaptation framing. |
| #1530 | @samacqua | VarLen attention, fused MLP, and doc-TTT base referenced by PR #1626. |
| #1610 | @romeerp | Phased TTT concept referenced by PR #1626. |
| #1493 | @bigbag | QK-Gain 5.25 and consolidation of the SP8192 + recurrence + residual + legal TTT frontier stack. |
| #1445 | @X-Abhishek-X | Tuned WD / matrix LR / EMA / warmdown settings cited by PR #1493. |
| #1412 | @Robby955 | Parallel residuals from layer 7 onward, plus Hessian-aware SDClip analysis that informed later quantization thinking. |
| #1331 | @dexhunter | 3-layer depth recurrence over layers 3-5 and the WD/LR compression tradeoff. |
| #1285 | @dexhunter | Earlier recurrence / WD-quantization synergy that #1331 extends. |
| #1394 | @clarkkev | SP8192 tokenizer stack, GPTQ embedding quantization, SDClip row-std clipping, Brotli packaging, simplified recurrence path. |
| #1218 | @clarkkev | 4096-vocab larger-model stack, higher WD compression logic, GPTQ Hessian-aware quantization path, sigmoid skip connections, QK-gain adoption. |
| #1217 | @bigbag | MuonEq-R row-normalized optimizer idea and QK-gain sweep context. |
| #1204 | @msisovic | Mini depth recurrence and parallel residual formulation used upstream. |
| #1179 | @dexhunter | Base stack used by #1204 and #1217. |
| #1125 | @jainpranjal97 | XSA-all and QK-Gain 4.0 hyperparameter findings that pushed attention gain upward. |
| #1105 | @abaybektursun | Mixed-quantization / autoregressive GPTQ path referenced by #1204. |
| #1089 | @clarkkev | Byte-shuffle/Brotli compression improvements and sigmoid-gated skip connections referenced by #1218. |
| #1060 | @clarkkev | GPTQ Hessian-aware quantization implementation referenced by #1218. |
| #1019 | @abaybektursun | AR self-generated GPTQ calibration, XSA-all, record architecture documentation, and the prior merged SOTA baseline for several later PRs. |
| #756 | @abaybektursun | Negative TTT / quantization experiments that helped motivate pre-quant rather than post-quant TTT. |
| #726 | @clarkkev | Coprime-stride loader lineage referenced before the simplified loader in #1394. |
| #609 | @saml212 | BigramHash and selective-pruning / GPTQ calibration lineage referenced by #1019. |
| #593 | multiple contributors | GPTQ calibration legality context referenced by #1019. |
| #569 | multiple contributors | GPTQ calibration legality context referenced by #1019. |
| #549 | @abaybektursun | LeakyReLU^2 plus legal score-first TTT and Parallel Muon record line. |
| #535 | @raahilshah | Full-Hessian GPTQ and QAT/export alignment lineage. |
| #518 | @sofiabod | LeakyReLU^2 follow-up credit in the #549 lineage. |
| #493 | @parinzee | 11-layer model, XSA, LeakyReLU(0.5)^2 MLP, EMA, int6 quantization, partial RoPE. |
| #478 | @gowtham0992 | XSA on all 11 layers and GPTQ-lite / EMA / late-QAT record line. |
| #461 | @Christopher-Lee-McClendon | Score-first TTT framework used by earlier legal TTT records. |
| #414 | @signalrush | Base model lineage for the #549 record stack. |
| #401 | @newjordan | EMA/SWA weight averaging lineage. |
| #399 | @abaybektursun | Parallel Muon optimizer lineage. |
| #364 | @shikhar1729 | Warmdown schedule lineage. |
| #315 | @jfprincz | Partial RoPE and layer-scale lineage. |
| #289 | contributor in PR #1019 lineage | U-Net skip connection lineage documented by #1019. |
| #286 | @chris-buckley | Late QAT / STE lineage documented by #1019. |
| #180 | @thwu1 | Early SOTA baseline credited by #493. |
| #162 | @raahilshah | BigramHash concept lineage documented by #1019. |
| #160 | @ChaseWNorton | Compression lineage documented by #1019. |
| #122 | @mtybadger | Flash Attention 3 / Hopper kernel dependency lineage documented by #1019. |
| #65 | @aquariouseworkman | SmearGate lineage documented by #1019, though later SP8192 stacks simplified parts of that path away. |

Some of the older entries above are not individually visible as isolated code blocks in this final compressed script because later record PRs folded, simplified, or removed pieces. I am listing them because the later PRs explicitly trace their ancestry through them, and I do not want the final record writeup to erase that chain.

## Compliance Analysis

This submission follows the same Track A framing as PR #1735 and PR #1738:

- The evaluated artifact is fixed after export: full-precision EMA model -> pre-quant TTT -> GPTQ -> compressed artifact.
- The final sliding-window evaluation uses the fixed quantized model.
- There is no eval-time cache, SLOT, RLS, ETLB, n-gram cache, or two-pass rescoring.
- The softmax is normalized and the attention path remains causal.
- CaseOps is a reversible preprocessing transform, and BPB is charged against original UTF-8 bytes through byte sidecars rather than transformed-token byte counts.
- All listed artifacts are under 16,000,000 bytes.
- Training is under 600 seconds. Eval is under 600 seconds.

The rule-sensitive part is pre-quant TTT itself. I am presenting this under the same interpretation as PR #1735 / PR #1738: adaptation is part of artifact generation, and the submitted predictor is fixed at scoring time. If maintainers decide that pre-quant TTT on validation chunks is outside Track A, this line should be judged consistently with those PRs.

## Dependencies And External Data

The rule text allows imports as long as they do not violate evaluation, compute, training-time, code-size, or other restrictions, and asks record folders to include dependency/setup notes. The repository README also says the official RunPod environment has the normal packages pre-installed and that `requirements.txt` is a reference for manual setup.

For this record:

- The **submitted artifact** is still self-contained: counted code bytes plus compressed model bytes. It does not download anything during final eval.
- The **final eval path** uses local validation shards and the fixed quantized artifact. There are no network calls, external services, or hidden files during scoring.
- The **training setup** needs the CaseOps tokenizer/data files. I used the public `romeerp/parameter-golf-caseops-v1` Hugging Face dataset export from PR #1729, downloaded before running `train_gpt.py`.
- The `train_gpt.py` runtime imports `torch`, `numpy`, `sentencepiece`, and `brotli`. It tries FlashAttention 3 if the official image has it, then falls back to the available PyTorch attention path.
- `huggingface-hub` and `hf_transfer` are listed for the dataset download step only; they are not part of the final artifact/eval dependency.

So the dependency story is: external packages and the public CaseOps data export are setup/training inputs, explicitly documented here; the actual scored artifact remains under 16,000,000 bytes and does not rely on network access during evaluation.

## Reproduction

```bash
# Install deps
pip install -r requirements.txt
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download CaseOps dataset
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='/workspace/caseops_data',
)
"

# Symlink to expected paths
cd /workspace/caseops_data/datasets/datasets/
ln -sf fineweb10B_sp8192_lossless_caps_caseops_v1_reserved fineweb10B_sp8192
cd /workspace/caseops_data/datasets/tokenizers/
ln -sf fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model fineweb_8192_bpe.model

# Run training
SEED=1337 \
  DATA_DIR=/workspace/caseops_data/datasets/ \
  TTT_EMA_ENABLED=0 \
  PREQUANT_TTT_ENABLED=1 \
  PREQUANT_TTT_EPOCHS=21 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Test Plan

- [x] 3-seed validation (1337, 42, 999)
- [x] Independent seed 1337 reproduction on 2026-04-28/29
- [x] All artifacts under 16,000,000 bytes
- [x] Training under 600s
- [x] Eval under 600s
- [x] Fixed predictor for final scoring
- [x] Full-Hessian GPTQ int6 + Brotli
- [x] CaseOps lossless reversibility via the public dataset/tokenizer export
- [x] Byte-sidecar BPB computation against original UTF-8 bytes
