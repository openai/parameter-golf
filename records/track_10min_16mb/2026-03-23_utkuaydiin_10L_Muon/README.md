# 10L Muon + Int5/Int6 PTQ + BigramHash + SmearGate (15.46 MB)

**Submission details**
- Architecture: 10 layers, 448 dim, BigramHash(8192) + SmearGate, mixed Int5/Int6 post-training quant
- Optimizer: Muon + AdamW hybrid, WD=0.04, SWA starting ~step 3000
- Training: 10 min on 8×H100, global 5.5% prune, real FineWeb data
- Final size: 15.46 MB after zstd-22

**Reproducibility**
- Run command: `NCCL_IB_DISABLE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py`
- Data: FineWeb sp1024 (downloaded via cached_challenge_fineweb.py)
- Evaluation: Default sliding-window eval from the repo

Included files: submission.zst, train_gpt.py, train.log (this run), README.md, submission.json
