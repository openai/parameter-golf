# Non-record submission: 11L mixed int5/int6 + working QAT + TTT + 8 additions

**Historical run:** `1.1466 val_bpb` (sliding window, stride=32, original post-TTT flow) | **14.7 MB** artifact | 8xH100 SXM, 605s train + 340s eval

Built on PR #315 (1.1248). Ran with PyTorch SDPA instead of FA3, so throughput was 110ms/step instead of 85ms. Got 5,129 steps instead of ~7,000. Score should drop with FA3.

Note: the historical `1.1466` number above came from the original pre-eval TTT flow in this run. The current script has been updated to report plain no-TTT metrics and causal TTT metrics separately so future runs do not adapt on unseen eval tokens before scoring them. That means the checked-in script should be rerun before using it for a fresh official score claim.

## What we added to PR #315

**1. Working QAT.** PR #315's late QAT is dead code because `torch.compile` constant-folds `CastedLinear._qat_enabled` at first trace. We swap the `forward` method to `forward_qat` per instance and recompile. QAT noise matches the export scheme: int5 STE for MLP, int6 STE for attention. The current script also exposes `QAT_ENABLED`, `QAT_START_STEP`, and `QAT_START_FRAC` so we can turn QAT on earlier instead of hoping it only catches the last few steps.

**2. Mixed int5/int6 quantization + magnitude pruning.** MLP weights get int5 ([-16, 15]), attention gets int6 ([-32, 31]), embeddings stay int8. 3% magnitude pruning before quantization. Result: 14.7MB with 1.3MB headroom.

**3. Test-time training.** This run originally used post-quantization SGD on validation tokens before final scoring. The script now also includes a causal TTT path that scores each eval chunk first and only then adapts on that chunk, which is the safer version for future experiments.

**4. BigramHash 10240.** Up from 2048 in PR #315.

**5. Memory tokens.** 64 learnable embeddings as global context. Overwritten during training (targets masked), prepended during eval (stripped after layers). 32K params.

**6. Backout connection.** Learned scalar (init=0.2) subtracts encoder/decoder boundary state from final output. One parameter.

**7. Per-head temperature.** Learned temperature per attention head. 88 params total.

**8. Eval stride 32.** Down from 64. Made no difference here (s32 and s64 both gave 1.1466).

## What we kept from PR #315

11 layers, U-Net skips, XSA on last 4, EMA (0.997), partial RoPE (16/64 dims), LN scale, 3x MLP relu-squared, SmearGate, ortho+muP init, Muon (0.025, 0.99, WD=0.04), NTK RoPE, seq 2048, softcap 30.

## Results

| Metric | Value |
|--------|-------|
| Steps | 5,129 (110ms/step, SDPA) |
| Pre-quant val_bpb | 1.1597 |
| Post-quant val_bpb | 1.1697 |
| Quant gap | +0.0100 |
| Historical post-TTT sliding s32 | **1.1466** |
| Historical no-TTT roundtrip | 1.1697 |
| Artifact | 14,706,424 bytes |
| TTT time | 83s |
| Peak memory | 25,777 MiB/GPU |

## What would help

- FA3 (30% more training steps)
- 12th layer with the 1.3MB budget headroom
- Earlier QAT so it gets hundreds to thousands of steps instead of 1

## Papers behind these ideas

- Low-bit quantization direction: [QQQ: Quality Quattuor-Bit Quantization for Large Language Models](https://arxiv.org/abs/2406.09904)
- Very low-bit training motivation: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- Test-time training: [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
- Faster Hopper attention kernels: [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/blog/2024/flash3/)

## How to run

```bash
cd /workspace
git clone -b submission/sota-attempt <your-fork-url> parameter-golf
cd parameter-golf
pip install huggingface-hub datasets sentencepiece tqdm zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024
ATTN_BACKEND=auto QAT_START_FRAC=0.8 \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-22_11L_MixedInt56_QAT_TTT_1.1466/train_gpt.py
```

If FA3 is installed, set `ATTN_BACKEND=fa3` to fail fast when the kernel is missing instead of silently falling back.

Single seed (1337), torch 2.4.1+cu124, 8xH100 SXM on RunPod.
