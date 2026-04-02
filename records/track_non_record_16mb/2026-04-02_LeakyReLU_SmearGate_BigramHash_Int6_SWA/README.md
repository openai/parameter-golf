# LeakyReLU(0.5)^2 on the SmearGate + BigramHash + Int6 stack

Author: Ivo Brett (@oidebrett)

val_bpb: **1.1444** | Artifact: ~16.0 MB (int6+zstd-22) | 8xH100 SXM, 600s

## Background

I'm fairly new to ML training and this competition has been a massive learning experience. I started on a free Kaggle T4 just trying to get the baseline running (and fighting NaN losses for a full day), worked my way up to a 1xH100, and finally got time on 8xH100s.

This submission takes the excellent SmearGate + BigramHash + Int6 QAT + SWA stack from PR #162 (@raahilshah) and adds one small change: swapping `relu^2` for `leaky_relu(0.5)^2` in the MLP activation.

## The change

```python
# before (relu^2)
x = torch.relu(self.fc(x))

# after (leaky relu^2)
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
```

That's it. One line. The idea (from @abaybektursun's PR #549) is that regular ReLU kills all negative activations, so half the MLP capacity is wasted on dead neurons. LeakyReLU with slope 0.5 lets negative values contribute while the squaring still keeps outputs non-negative.

## Results

| Metric | Value |
|--------|-------|
| val_bpb (post-quant, sliding window) | **1.1444** |
| Pre-quant val_bpb | 1.1604 |
| Artifact size | 16,018,202 bytes |
| Steps | 7,276 |
| Step avg | 82.5 ms |
| SWA checkpoints averaged | 30 |
| Seed | 1337 |

For comparison, the same stack without LeakyReLU gives 1.1459 (I ran both back-to-back on the same pod to confirm).

## Architecture & hyperparams

- 9 layers, 512 dim, 8 heads (4 KV, GQA), MLP 3x
- Seq length 2048, batch 786k tokens
- SmearGate + BigramHash(4096, dim=128)
- Int6 QAT with per-row symmetric quantization
- SWA every 50 steps over last 50% of training
- Muon optimizer, WD=0.04, momentum warmup 0.92->0.99
- Orthogonal init, zstd-22 compression
- Sliding window eval stride=64

## Why non-record

Doesn't come close to beating the current SOTA of 1.1147. I ran out of RunPod credits before I could try more combinations (11 layers, XSA, better quantization, etc). Submitting anyway because I learned a ton and wanted to document what I found.

## Credit where it's due

Almost everything here is other people's work that I built on top of:

- SmearGate, BigramHash, MLP 3x, SWA, OrthoInit: @raahilshah (PR #162)
- Muon WD tuning: @thwu1 (PR #180)
- Int6 QAT: @signalrush (PR #414)
- LeakyReLU(0.5)^2 idea: @abaybektursun (PR #549)

Thanks to the whole Parameter Golf community for being so open with techniques. Coming from outside ML, being able to read and learn from everyone's submissions made this approachable.

## Compute grant

I've applied for the OpenAI Development Compute Grant. With more compute I'd like to keep experimenting - there are a lot of directions on the leaderboard I haven't had budget to try yet (more layers, better quantization schemes, architectural changes). This has been a great way to learn about training and I'd love to push further.

## Files

- `README.md` - this file
- `submission.json` - metadata
- `train_gpt.py` - training script (modified from PR #162)
- `train_seed1337.log` - full training log
