# Int6 QAT + 11L 512d + Sliding Window — val_bpb=1.2089

**Author:** darroney (GitHub: dibdabo)  
**Track:** Non-record (`track_non_record_16mb`)  
**val_bpb:** 1.2089 (post-roundtrip sliding window)  
**Artifact size:** 15,190,812 bytes  
**Hardware:** 8×H100, 10 minutes  
**Date:** 20 March 2026

---

## What I did

Started from the baseline and worked through the standard stack — int6 quantisation, STE QAT, sliding window eval, seq_len=4096, tuned Muon. Spent most of the day debugging rather than innovating, which taught me a lot about what actually matters here.

The architecture ended up at 11 layers × 512 dim × mlp_hidden=1024. I originally wanted mlp_hidden=1536 (3× expansion like PR #64) but it kept failing the budget. Turned out the issue is that Muon-trained weights are near-orthogonal and compress worse than you'd expect — zlib was only getting ~0.91× on my weights rather than the ~0.70× I was hoping for. Dropping to 11 layers brought the compressed size to 15.19 MB with room to spare.

---

## Techniques

**Int6 quantisation with flat tensor storage**
Large matrices (numel > 65,536) quantised to int6 per-row, 4 values packed into 3 bytes. The key thing I found: you have to concatenate all the packed int6 bytes into a single flat uint8 tensor before torch.save, otherwise pickle interleaves metadata between 73 separate tensors and zlib barely compresses anything. Storing them separately gave 0.90× compression and a 19.7 MB file. The flat format (int6_mixed_per_row_v2) fixed that.

**STE fake-int6 QAT**
Activates at step 200. Fake-quantises weights to the int6 grid before each forward pass, restores fp32 weights after backward. The weight restore is not optional — removing it breaks Muon completely (loss stuck at ~2.7 from step 400 onwards). Learned that the hard way.

**Sliding window evaluation**
ctx=4096, chunk=512, stride=64. Critical thing: the context window must match train_seq_len. Had it set to 1024 at one point with a model trained on seq_len=4096 and got 2.16 bpb — worse than flat eval. Fixed to ctx=4096 and the sliding window gave the expected improvement.

**Tuned Muon**
momentum=0.99, matrix_lr=0.02, scalar_lr=0.025, warmdown=3000, muon_momentum_warmup_start=0.92, muon_momentum_warmup_steps=1500. Warmdown fires correctly on 8×H100. On a single H100 with 600s budget it never fires — that cost me some early runs.

---

## What went wrong along the way

Quite a lot, honestly. Main bugs hit:

- Variable name collision — `scale` was used for both the LR multiplier and the QAT per-row quantisation scale. Caused a crash at step 200 on the first proper run.
- `load_quant_artifact` only handled format string `int6_mixed_per_row_v1`. When I introduced v2, the roundtrip crashed with `KeyError: 'quantized'` because it fell through to the int8 loader.
- Tried removing the QAT weight restore to see if it helped compression. It doesn't — it just breaks training entirely.
- Spent two runs trying to get mlp_hidden=1536 to fit by reducing MAX_WALLCLOCK_SECONDS. The compressed size barely changed because the compression ratio is determined by the weight entropy, not the step count.

---

## Results

| Metric | Value |
|--------|-------|
| Steps | 6,619 |
| Flat val_bpb (pre-quant) | 1.1856 |
| Sliding window val_bpb (pre-quant) | 1.1735 |
| Post-roundtrip sliding window val_bpb | **1.2089** |
| Artifact size | 15,190,812 bytes |
| Budget | Within |

---

## Reproduction

```bash
RUN_ID=v8_8xh100_submission_v4 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
NUM_LAYERS=11 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## What's next

Main thing I want to try is getting mlp_hidden=1536 to fit. Either via a compressor that handles high-entropy weights better (zstd looked promising in PR #107), or NorMuon which might reduce the orthogonalisation effect and allow the larger MLP to squeeze under budget.
