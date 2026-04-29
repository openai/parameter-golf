# Record: PR #1886 base + per-block MLP output gate (Linear, weight-learnable) — val_bpb 1.06872 (3-seed mean)

**val_bpb: 1.06872454** (3-seed mean, std ~0.00070), -0.01228 from SOTA

## Results (8×H100 80GB SXM, full pipeline with phased TTT, 10-min train / 10-min eval)

| Seed     |    Steps | Pre-EMA last val | Post-EMA pre-quant | Quantized (no TTT) |   **Post-TTT** |      Train |       Eval |  Artifact |
| -------- | -------: | ---------------: | -----------------: | -----------------: | -------------: | ---------: | ---------: | --------: |
| 42       |     4825 |           1.0795 |            1.06916 |            1.07938 | **1.06794764** |     599.4s |     398.4s |  ~15.9 MB |
| 1337     |     4827 |           1.0806 |            1.07070 |            1.08066 | **1.06931760** |     599.8s |     510.0s |  ~15.9 MB |
| 314      |     4825 |           1.0803 |            1.07014 |            1.08014 | **1.06890838** |     599.6s |     446.7s |  ~15.9 MB |
| **Mean** | **4826** |       **1.0801** |        **1.07000** |        **1.08006** | **1.06872454** | **599.6s** | **451.7s** |           |


All 3 seeds clear the 600s train, 600s eval, and 16 MB decimal artifact budgets.

### Head-to-head vs PR #1886 (matched seeds)

| Seed     |     This PR |    PR #1886 |  Δ (mBPB) |
| -------- | ----------: | ----------: | --------: |
| 42       |     1.06795 |     1.06920 | **−1.25** |
| 1337     |     1.06932 |     1.07010 | **−0.78** |
| 314      |     1.06891 |     1.06942 | **−0.51** |
| **Mean** | **1.06872** | **1.06957** | **−0.85** |

Every individual seed beats its matched PR #1886 counterpart.

## Novel contribution: per-block MLP output gate, input-dependent, weight-learnable

This idea came directly from the Attention Gate previously added by our team in PR #1667, and even if this adds minimal gain, it closes the Gate Research Story on Parameter Golf. 
Attention gate has much more effect, but the MLP_gate is adding only 143 additional params and gating token wise not headwise, so even if its effect is small, it's telling us a lot about what we could do without the specific constraints of this competition.

Additionally we're presenting this PR as a sort of update to the previous #1667 that was the first to introduce the Attention Gate (which was largely adopted by other PRs), but had a problem with the Smear Gate emerging from the discussion between @cocohearts, @msisovic, (cross document leakage) in PR #1797


```python
# In Block.__init__:
self.mlp_gate_out = CastedLinear(12, 1, bias=True)        # 13 params per block
self.mlp_gate_out._pos_bias_init = 5.0                     # init: w=0, b=+5

# In Block.forward, after self.mlp(...):
mlp_out = self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
gate = torch.sigmoid(self.mlp_gate_out(x_out[..., :12].contiguous()))  # (B, T, 1)
mlp_out = mlp_out * gate
x_out = x_out + self.mlp_scale[None, None, :] * mlp_out
```

**Initialization**: weight=0, bias=+5 → `sigmoid(+5) ≈ 0.993`, ≈ identity at start (do-no-harm bias init).

**Total new parameters**: 11 layers × (12 + 1) = **143 parameters** (negligible vs 35.99M model parameters).

## Rule compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL** (README FAQ + Issue #1017 §II.1): ✅ all seeds ≤ 16 MB.
- **train_time ≤ 600s**: ✅ all seeds 599.4–599.8s (`stopping_early: wallclock_cap`).
- **total_eval_time ≤ 600s**: ✅ all seeds 398.4–510.0s.
- **Issue #1017 Condition 3 (score-before-update)**: phased TTT unchanged — every chunk is scored under `inference_mode()` before any LoRA update.
- **No val data during training**: training uses only `fineweb_train_*.bin` shards.
- **No external network during eval**: self-contained `train_gpt.py` + tokenizer.
- **Reproducibility**: all hyperparameters set via env vars in the run command below.

## Reproduction

The MLP_GATE_OUT addition is hard-coded in `train_gpt.py` (no env-var flag — the gate is always present, since this is the record's defining change). All other env vars match the PR #1886 stack defaults; the explicit list below is conservative.

### Environment setup

```bash
pip install brotli sentencepiece python-minifier

# FlashAttention-3 
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

### Dataset

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192
```

### Training (3 seeds)

```bash
export DATA_DIR=/path/to/parameter-golf/data

for SEED in 42 1337 314; do
  NCCL_NET=Socket \
  GATED_ATTN_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 MIN_LR=0.10 \
  FUSED_CE_ENABLED=1 \
  TTT_WARM_START_A=1 TTT_WEIGHT_DECAY=2.0 \
  TTT_LORA_ALPHA=144 TTT_LORA_RANK=128 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Hardware

Trained on **RunPod 8×H100 80GB SXM**. PyTorch 2.9.1+cu128, FA3, Triton 3.5.1. Identical SP8192 SentencePiece tokenizer and FineWeb document selection as upstream `kevclark/parameter-golf` (the canonical PG `parameter-golf` validation split). No tokenizer mods.

## Lineage

- @nprime06 — PR #1787 (FusedCE / PolarNS / MIN_LR / SparseAttnGate base)
- @renqianluo — PR #1767 (warm-start LoRA), PR #1768 (GatedAttn), PR #1886 (WD=2.0 stability)
- @dexhunter — PR #1626 / PR #1736 (Multi-phase SGD, GPTQ trim, GatedAttn baseline)
- @samacqua — PR #1530 (VarLen + Fused MLP + doc-independent TTT)
- @bigbag — PR #1493 (3-layer recurrence + parallel residuals base)
- @MarioPaerle — PR #1667 (per-head attention output gate pattern, prior art for the "narrow gate Linear(12→1) + bias=+5" idiom used here on the MLP output)
- This submission — adds the per-block MLP output gate to the modern stack with the bug fix that makes the gate weight-learnable.

## Credits
This work was also possible thanks to the support provided by [Paradigma](https://paradigma.inc/) and the use of [Flywheel](https://flywheel.paradigma.inc/): their infrastructure for research.

- @MarioPaerle, @GabrieleCirillo, @CerovazS
- @renqianluo — PR #1886 base stack
- All upstream contributors as listed in lineage