This submission packages a near-frontier 10-minute run as a `track_non_record_16mb` entry.

It is intentionally submitted as a non-record result under the current rules. The run is under the decimal `16,000,000`-byte artifact cap, trains on `8xH100 SXM`, and evaluates cleanly, but it does not claim a new public SOTA because the live open-PR frontier is already slightly lower than this result and this package only includes one full leaderboard-grade seed. Under the current README rules, record submissions should beat the current SOTA PR by at least `0.005` nats with sufficient significance.

What this run is:
- A faithful reproduction of the public PR315-style 11-layer transformer line on RunPod `8xH100 SXM`, with native Hopper FlashAttention and `torch.compile`
- One cheap orthogonal addition: a learned Backout residual subtraction from the mid-network hidden state
- Seed `2025`, full `80`-shard SP-1024 training set, `600s` training cap, stride-64 sliding-window evaluation

Exact result:
- `final_int6_sliding_window_exact val_loss: 1.89896029`
- `final_int6_sliding_window_exact val_bpb: 1.12467423`
- `step_stop: 7048`
- `train_time: 600037ms`

Artifact accounting:
- Compressed model (`int6+zstd`): `15,472,918` bytes
- Submitted `train_gpt.py`: `72,744` bytes
- Total packaged artifact size: `15,545,662` bytes

Important note on code bytes:
- The original experiment log reports `Code size: 69975 bytes`, because the experiment version imported a sibling `flash_attn_interface.py`.
- For this submission folder, that helper has been inlined into `train_gpt.py` so the record is self-contained and more closely follows the repo guidance that counted code should live in `train_gpt.py`.
- The underlying model artifact is unchanged; only the packaged code bytes increase slightly.

Run details from `train.log`:
- Backend proof: `flash_attn_backend:native`
- Compile proof: `torch_compile:True`
- Stable throughput: about `85.14ms/step`
- Peak memory: `20693 MiB allocated`, `20748 MiB reserved`
- Post-quant roundtrip exact metric: `val_bpb: 1.14823337`
- Sliding-window exact metric: `val_bpb: 1.12467423`

Track-relevant command:
```bash
OMP_NUM_THREADS=1 \
RUN_ID=runpod-pr315-backout-seed2025-20260322 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
REQUIRE_NATIVE_FLASH_ATTN=1 \
ENABLE_TORCH_COMPILE=1 \
ITERATIONS=9000 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=0 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT=1 \
BACKOUT_ENABLED=1 \
BACKOUT_LAMBDA_INIT=0.2 \
BACKOUT_LAYER=-1 \
QAT_THRESHOLD=0.1 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Setup notes:
- This run was produced on the official RunPod Parameter Golf image with a native Hopper FlashAttention install available on the machine.
- The submitted script will run with fallback SDPA for local smoke tests if `REQUIRE_NATIVE_FLASH_ATTN=0`, but a faithful reproduction of this score expects native FA3 on `8xH100 SXM`.
- If you are self-provisioning instead of using the official template, install the Python packages in `requirements.txt` and make sure native Hopper FlashAttention is available to Python.

Included files:
- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- `requirements.txt`
