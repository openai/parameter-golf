This record brings the ideas from my last work (#87), which was high vocab size, NorMuon and mixed int6/int8 quantization, up to the frontier by copying a bunch of other people! Specifically, I take the STE and SWA ideas from @vmfunc (#89), the sliding window eval with seqlen=1024 and stride=64 from #50 @mattqlf (first), and #65 @aquariouseworkman, and the Momentum/LR tuning from #52 @spokane-way, #61 @saml212. I also use FA3, which decreases step time by about 10ms - a total free lunch!

The tradeoffs are getting tough. I'm sticking to my guns in losing a layer for higher vocab size, and I think everyone else is right that keeping embeddings in fp16 reduces the quant gap, which means I had to take my vocab size down to compensate. It's really a question about whether we want more diversity in vocab or more resolution in representation, and I think there's a better optimum in between yet to find.


Changes in this model from baseline:
- Vocab size 1024 -> 2048
- New "sp2048" tokenizer trained using `./data/download_hf_docs_and_tokenize.py   --output-root ./data   --tokenizer-config ./data/tokenizer_specs.json --max-train-tokens 8000000000 --tokenizer-train-docs 100000`, for a 50/50 val/train split. Tokenizers for sp1024, 2048, 4096 and 8192 with data available on [my huggingface](https://huggingface.co/sproos/parameter-golf-tokenizers/tree/main)
- NorMuon implementation from [the original paper](https://github.com/zichongli5/NorMuon), popularized by `modded-nanogpt`, replacing Muon
- The weights are quantized row-wise to int6, while the embeddings are kept in fp16. Quantization-aware trained; i.e. the weights are fake-quantized during training to make them more amenable later, and straight-through estimated on bkwd. Many people have done this - #42 @chonchiog (first), #66 @arjun-krishna1
- FlashAttention 3, thank u tri dao
- Sliding-window eval! It's a bit of a wheeze but everyone's doing it, and I wanted to see how it played with the rest of my stack. Stride=64, seqlen=1024. ty @mattqlf
- 3X MLP: hidden dim 1536 (up from 1024). ty @jfprincz
- SWA (stochastic weight averaging) - collects model checkpoints every 200 steps during the final warmdown phase and averages them. the averaged weights generalize slightly better than the final point estimate. 7 checkpoints averaged in the submitted run. ty @vmfunc!

Configuration:
- All hyperparams as in default NaiveBaseline except VOCAB_SIZE, TRAIN_SEQ_LEN, WARMDOWN_ITERS and NUM_LAYERS; unfortunately to get the increased vocab size we have to sacrifice a layer.
- Tested on Hyperbolic 8xH100 setup with SXM5; reproduced baseline with `step_avg:43.67ms` and `final_int8_zlib_roundtrip_exact val_bpb:1.22731147` immediately before.

Command:
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=verify_sp2048_w6e16_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp2048 \
TOKENIZER_PATH=./data/tokenizers/fineweb_2048_bpe.model \
VOCAB_SIZE=2048 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=4096 \
NUM_LAYERS=8 \
MLP_MULT=3 \
torchrun --standalone --nproc_per_node=8 ./records/track_10min_16mb/2026-03-19_SWE_VocabSize_NorMuon_Int6/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `11132/20000` steps due to the wallclock cap, which is further than before!.
- Pre-quant eval at stop: `val_loss:2.3953 val_bpb:1.1670 `
- Post-quant roundtrip eval: `val_loss:2.3982 val_bpb:1.1684 eval_time:1324ms`
- Post-quant rountrip sliding window eval: `val_loss:2.3780 val_bpb:1.1585 eval_time:205575ms`
- Train time: `train_time:600081ms step_avg:53.91ms`
- Serialized model w6e16+zstd22: `15289740 bytes`
- Code size: `63530 bytes`
- Total submission size w6e8+zlib: `15353270 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7224688640`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
