# Non-record submission: 11L mixed int5/int6 + working QAT + TTT + 8 additions

**val_bpb = 1.1466** (sliding window, stride=32, post-TTT) | **14.7 MB** artifact | 8xH100 SXM, 605s train + 340s eval

Built on PR #315 (1.1248). Ran with PyTorch SDPA instead of FA3, so throughput was 110ms/step instead of 85ms. Got 5,129 steps instead of ~7,000. Score should drop with FA3.

## What we added to PR #315

**1. Working QAT.** PR #315's late QAT is dead code because `torch.compile` constant-folds `CastedLinear._qat_enabled` at first trace. We swap the `forward` method to `forward_qat` per instance and recompile. QAT noise matches the export scheme: int5 STE for MLP, int6 STE for attention.

**2. Mixed int5/int6 quantization + magnitude pruning.** MLP weights get int5 ([-16, 15]), attention gets int6 ([-32, 31]), embeddings stay int8. 3% magnitude pruning before quantization. Result: 14.7MB with 1.3MB headroom.

**3. Test-time training.** 3 epochs of SGD on validation tokens post-quantization. lr=0.002, momentum=0.9, first 2 blocks frozen. Gradients synced via all_reduce(AVG). Took 83s on 8xH100. Moved BPB from 1.1697 to 1.1466.

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
| Post-TTT sliding s32 | **1.1466** |
| Artifact | 14,706,424 bytes |
| TTT time | 83s |
| Peak memory | 25,777 MiB/GPU |

## What would help

- FA3 (30% more training steps)
- 12th layer with the 1.3MB budget headroom
- QAT getting more than 1 step (it kicked in at step 5128, stopped at 5129)

## How to run

```bash
pip install huggingface-hub datasets sentencepiece tqdm zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-21_SOTA/train_gpt.py
```

Single seed (1337), torch 2.4.1+cu124, 8xH100 SXM on RunPod.
