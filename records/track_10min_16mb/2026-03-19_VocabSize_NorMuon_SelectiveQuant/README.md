# 2023-03-19: Vocab Size, NorMuon, Selective Quantization

Day 1! This record contains three main new ideas, as well as some tweaks to the baseline, particularly vocab size. I had several ideas I wanted to try today, and these are the ones that worked - I want to chase further on quantization in the coming days.

Changes in this model:
- Vocab size 1024 -> 8192
- New "sp8192" tokenizer trained using 
```bash
./data/download_hf_docs_and_tokenize.py   --output-root ./data   --tokenizer-config ./data/tokenizer_specs.json --max-train-tokens 8000000000 --tokenizer-train-docs 100000
```
with this tokenizer_spec:
```json
{
  "tokenizers": [
    {
      "name": "sp_bpe_1024",
      "dataset_suffix": "sp1024",
      "vocab_size": 1024
    },
    {
      "name": "sp_bpe_8192",
      "dataset_suffix": "sp8192",
      "vocab_size": 8192
    }
  ]
}

```
with a 50/50 val/train split as a result. Tokenizers for sp1024, 2048, 4096 and 8192 with data are publicly available on [my huggingface](https://huggingface.co/sproos/parameter-golf-tokenizers/tree/main).
- NorMuon implementation from [the original paper](https://github.com/zichongli5/NorMuon), popularized by `modded-nanogpt`, replacing Muon
- Selective Quantization: the weights are quantized to int6, while the embeddings are kept at int8. Not sure if this is optimal and have seen plenty of weird behaviour from this, but I think it's in the right direction; I think being precise about precision will be really key to this challenge and I want to dig into it more. From now on there will be a lot of trading off precision between areas of the model!

Configuration:
- All hyperparams as in default NaiveBaseline except VOCAB_SIZE, TRAIN_SEQ_LEN, WARMDOWN_ITERS and NUM_LAYERS; unfortunately to get the increased vocab size we have to sacrifice a layer. I'm sure there's a better architectural setup here, but I don't know if it's recurrence.
- Tested on Hyperbolic Labs 8xH100 node with SXM5; reproduced baseline with `step_avg:43.67ms` and `final_int8_zlib_roundtrip_exact val_bpb:1.22731147` immediately before.

Command:
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=verify_sp8192_w6e8_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=600 \
WEIGHT_QUANTIZATION_BITS=6 \
EMBED_QUANTIZATION_BITS=8 \
WARMDOWN_ITERS=3000 \
TRAIN_SEQ_LEN=4096 \
NUM_LAYERS=8 \
torchrun --standalone --nproc_per_node=8 ./records/track_10min_16mb/2026-03-19_VocabSize_NorMuon_SelectiveQuant/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `9359/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:3.0261`, `val_bpb:1.1717`
- Post-quant roundtrip eval: `val_loss:3.06233041`, `val_bpb:1.18576208
- Train time: `600075ms` (`step_avg:64.12ms`)
- Serialized model w6e8+zlib: `14743224 bytes`
- Code size: `53612 bytes`
- Total submission size w6e8+zlib: `14796836 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `7224688640`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
