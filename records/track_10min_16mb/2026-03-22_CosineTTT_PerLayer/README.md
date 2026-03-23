# Cosine TTT with Per-Layer Learning Rates

**val_bpb = [TBD]** (3-seed mean) | 8×H100 SXM | 600s train + [TBD]s causal TTT eval

## Results

| Seed | Steps | Pre-TTT | Post-TTT | Artifact |
|------|-------|---------|----------|----------|
| 1337 | [TBD] | [TBD] | [TBD] | ~15.5 MB |
| 42 | [TBD] | [TBD] | [TBD] | ~15.5 MB |
| 7 | [TBD] | [TBD] | [TBD] | ~15.5 MB |

## Key finding

TTT learning rates should be proportional to per-weight quantization damage.

Analyzing a trained checkpoint, we measured that MLP output projections have 3.4× higher relative quantization error than input projections. Setting TTT lr to 3× for output projections and 0.5× for input projections, combined with cosine decay across epochs, improved TTT effectiveness by 23.5% over flat-lr AdamW in a 34-configuration ablation (4 rounds, covering optimizers, schedules, lr, epochs, freeze strategies, and loss functions). Details in FINDINGS.md §21b.

## Causal TTT (score-first, competition-legal)

Each validation chunk is scored under `torch.inference_mode()` before any weight update. Compliant with issue #402: tokens are never scored by a model that has trained on them.

```
for each 32K chunk:
    1. model.eval() + inference_mode → score chunk (loss recorded)
    2. model.train() → AdamW, 3 epochs, cosine lr reset per chunk
```

Per-chunk cosine decay resets lr to full at each new chunk and decays to zero across the 3 training epochs. This avoids both lr starvation on later chunks (which global cosine causes) and overshooting on the final epoch (which flat lr causes).

## TTT config

```
TTT_CAUSAL=1  TTT_OPTIMIZER=adamw  TTT_LR=0.0005  TTT_CHUNK_EPOCHS=3
TTT_COSINE=1  TTT_PERLAYER=1  TTT_FREEZE_BLOCKS=0  TTT_CHUNK_TOKENS=32768
TTT_BATCH_SEQS=64 (per GPU, 512 total with DDP gradient sharding)
```

## Training

11L, 512d, 8H/4KV (GQA), 3× MLP, U-Net skip connections, SmearGate, BigramHash(2048), OrthoInit, Partial RoPE (16/64), LN Scale, XSA on last 4 layers, Value Embedding (128d, layers 9-10), EMA(0.997), late QAT, warmdown 3500 steps. Int6 per-row + zstd-22.

## Reproduction

```bash
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout causal-ttt
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install zstandard sentencepiece huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024
bash run_causal_ttt.sh 1337
```

Hardware: 8×H100 SXM (RunPod), PyTorch 2.9.1+cu128, Flash Attention 3.

## Experimental history

See [FINDINGS.md](../FINDINGS.md) for 22 documented experiments including negative results on codebook quantization, depth recurrence, multi-token prediction, magnitude pruning, and the full 34-config TTT ablation.

## Acknowledgments

Training architecture: PRs #162, #180, #315. TTT: PRs #77, #398, #442, #461. DDP sharding: PRs #398, #417. Legal TTT protocol: PR #461. Muon and modded-nanogpt: foundation codebase.
