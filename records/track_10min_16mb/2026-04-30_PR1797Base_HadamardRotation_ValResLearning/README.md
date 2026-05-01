# Record: PR1797Base + HadamardRotation + ValueResidualLearning

**val_bpb = 1.06172** (3-seed mean, std 0.0007) | **~15.97 MB** | 8xH100 SXM


## 3-Seed Results

| Seed | Steps | Pre-Quant BPB | Quantized BPB | **Post-TTT BPB** | Artifact |
|------|-------|---------------|---------------|------------------|----------|
| 314 | 4,879 | 1.06501204 | 1.07378518 | **1.06108303** | 15,968,887 |
| 2025 | 4,867 | 1.06534629 | 1.07427160 | **1.06159181** | 15,969,700 |
| 1 | 4,866 | 1.06633798 | 1.07522521 | **1.06249196** | 15,967,906 |
| **Mean** | | **1.06556543** | **1.07442733** | **1.06172226** | **15,968,831** |


## Key innovation — PR #1797 base + Hadamard Rotation + Value Residual Learning

1. **Native PR #1797 base stack** (PR #1787 base + Smear Gate + LQER Asym)

2. **Hadamard Rotation** : Hadamard rotation is applied as a post-training quantization pre-processing step to improve int6 GPTQ weight compression. Immediately before quantization, each weight matrix W is pre-rotated via W' = W @ R^T where R = diag(s) @ H_n / √n is an orthogonal matrix built from a Sylvester-Hadamard transform scaled by a deterministic ±1 sign vector. This rotation spreads weight magnitude uniformly across columns, eliminating large-magnitude outlier columns that would otherwise cause disproportionate GPTQ rounding error. After quantization, the rotation is exactly reversed during dequantization (Wq = W'q @ R),leaving model behavior unchanged. Each weight matrix gets its own unique rotation derived deterministically from a Blake2b hash of the weight name XOR'd with the base seed 0xc0ffee (12648430), ensuring reproducibility across serialize/deserialize cycles

3. **Value Residual Learning** : Value Residual Learning (ResFormer, arXiv:2410.17897) introduces a learned residual connection that blends the value tensor from layer 0 into every subsequent attention layer during training. At each layer l > 0, the value vector is mixed as v_l = σ(λ) · v_l + (1 - σ(λ)) · v_0 where λ is a per-layer learned scalar parameter (vres_lambda) initialised to 3.0 (giving σ(3.0) ≈ 0.95, a mild 5% blend at init). This provides each layer with a stable low-level value signal from the earliest representation, improving gradient flow and representation diversity across layers


## Architecture (inherits PR #1797 shape)

| Item | Value |
|------|------:|
| num_layers | 11 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 |
| mlp_mult | 4.0 |
| rope_base / rope_dims | 10000 / 16 |
| logit_softcap | 30.0 |
| loop_start / loop_end | 3 / 5 (NUM_LOOPS=2) |
| parallel_start_layer | 8 |
| eval_seq_len / eval_stride | 2048 / 64 |
| matrix_bits / embed_bits | 6 / 7 |
| LQER rank / top-K / A-bits / B-bits / asym group | 4 / 3 / 2 / 4 / 64 |
| smear gate window | 12 |
| compressor | brotli |


## Compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL**: For each one of the 3 seeds, the size is in the range 15,968,887–15,967,906 bytes.
- **train_time ≤ 600s**: all 3 seeds 599.555–599.616s (`stopping_early: wallclock_cap`).
- **total_eval_time ≤ 600s**: all 3 seeds 425855 415.184s–455.514s.
- **Reproducibility**: `train_gpt.py` is a single self-contained file; all mechanism flags are set via the Run Command environment.

- This PR inherits all other compliance requirements from  PR#1797


## Reproduction

1.Install the requirements 
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece triton numpy
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2110/
pip install python-minifier brotli
```

2.Data preparation

2.1. Run this command to download 80 shards of train data ,validation data and tokenizer model
```bash
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHMATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 cached_challenge_fineweb.py \
  --variant sp8192_lossless_caps_caseops_v1_reserved \
  --train-shards 80
```
2.2 Move the train data (80 .bin files) , validation data(2 .bin files) to the path 
/workspace/parameter-golf/records/track_10min_16mb/2026-04-30_PR1797Base_HadamardR
otation_ValResLearning/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/

2.3 Move the tokenizer (fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model file) to the path  
/workspace/parameter-golf/records/track_10min_16mb/2026-04-30_PR1797Base_HadamardR
otation_ValResLearning/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/


3. Training and validation

Run the command  

```bash 
for SEED in 314 2025 1 ; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
  MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 \
  MIN_LR=0.1 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  GATED_ATTN_ENABLED=0 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
  SMEAR_GATE_ENABLED=1 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  VAL_LOSS_EVERY=0 \
  HADAMARD_ROTATION_ENABLED=1 \
  VALUE_RESID_ENABLED=1 \
  VALUE_RESID_LAMBDA_INIT=3.0 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Credits

- **@dexhunter** — Base code (PR#1797)
- **@romeerp** — Caseops Data (PR#1530)


## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `cached_challenge_fineweb.py` (downloaded from PR#1530)
- `train_seed314.log`
- `train_seed2025.log`
- `train_seed1.log`