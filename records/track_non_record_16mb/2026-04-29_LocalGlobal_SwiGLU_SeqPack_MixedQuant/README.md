This record captures the `LocalGlobal_SwiGLU_SeqPack_MixedQuant` submission.

Trainer changes in this snapshot:

* repository `train_gpt.py` snapshot copied into the record folder
* depth-scheduled local/global attention (`LOCAL_ATTN_PATTERN="40,80,full"`)
* SwiGLU feedforward blocks replacing standard MLP
* randomized sequence packing with synchronized per-step offsets across DDP ranks
* selective mixed-bit quantization (`int6` on `attn.proj.weight`, `int8` elsewhere)
* SP-1536 tokenizer and dataset variant (`fineweb10B_sp1536`)
* 10-minute wallclock cap on `8xH100`
* periodic validation every `1000` steps on the full `fineweb_val_*` split

Configuration:

* Layout: `VOCAB_SIZE=1536 NUM_LAYERS=9 MODEL_DIM=480 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=1.625`
* Attention schedule: `LOCAL_ATTN_PATTERN="40,80,full"`
* Tied output/input embeddings: `TIE_EMBEDDINGS=1`
* Tied embedding LR: `TIED_EMBED_LR=0.040`
* Optimizer split: `MATRIX_LR=0.04 SCALAR_LR=0.035`
* Quantization: `USE_INT6=1 INT6_CLIP_PERCENTILE=99.998`
* Batching: `TRAIN_BATCH_TOKENS=327680 TRAIN_SEQ_LEN=2048`
* Warmdown schedule: `WARMDOWN_FRAC=0.75`

Command (track-relevant params):

```bash
NCCL_IB_DISABLE=1 \
EXPORT_ONLY=0 \
DATA_PATH="./data/datasets/fineweb10B_sp1536" \
TOKENIZER_PATH="./data/tokenizers/fineweb_1536_bpe.model" \
VOCAB_SIZE=1536 \
MODEL_DIM=480 \
NUM_LAYERS=9 \
MLP_MULT=1.625 \
LOCAL_ATTN_PATTERN="40,80,full" \
QK_GAIN_INIT=1.5 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=327680 \
WARMDOWN_FRAC=0.75 \
TIED_EMBED_LR=0.040 \
MATRIX_LR=0.04 \
SCALAR_LR=0.035 \
USE_INT6=1 \
INT6_CLIP_PERCENTILE=99.998 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log
```

Key metrics (from `train.log`):

* Timed training stopped at `15662/20000` steps due to the wallclock cap.
* Pre-quant eval at stop: `val_loss:2.2509`, `val_bpb:1.2005`
* Post-quant roundtrip eval: `val_loss:2.2609`, `val_bpb:1.2058`
* Exact printed metric: `final_mixed_quant_zstd_roundtrip_exact val_bpb:1.20584802`
* Train time: `600030ms` (`step_avg:38.31ms`)
* Peak memory: `6632 MiB allocated`, `7946 MiB reserved`
* Serialized model mixed-quant+zstd: `15481921 bytes`
* Code size: `56301 bytes`
* Total submission size mixed-quant+zstd: `15538222 bytes`

Training volume:

* Global batch: `327680` tokens/step
* Total train tokens seen: `5132124160`

Included files:

* `train_gpt.py` (code snapshot used for the run)
* `train.log` (exact remote training log)
* `submission.json` (submission metadata)
