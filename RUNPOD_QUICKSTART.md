# RunPod Quick-Start — TTT Calibration Sweep

Assumes 8xH100 pod with PyTorch 2.9.x / CUDA 12.8.

## 1. Pod setup (~8 min)

```bash
cd /workspace
git clone https://github.com/newjordan/parameter-golf.git
cd parameter-golf
git checkout experiments/pr374-edge

pip install sentencepiece numpy zstandard

# FA3 selective build (~5 min)
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
mkdir -p flash_attn_3
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
python3 -m pip install --no-build-isolation -e .
cd /workspace/parameter-golf
```

## 2. Preflight (~30 sec)

```bash
cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"
python3 -c "
import torch; assert torch.cuda.device_count() == 8
from flash_attn_interface import flash_attn_func
import sentencepiece, zstandard
print(f'{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)} — OK')
"
ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l  # expect 80
ls data/tokenizers/fineweb_1024_bpe.model                        # must exist
```

## 3. Get GS checkpoint (if not already present)

If `final_model.int6.ptz` doesn't exist, generate it:
```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 GS/GS_train_gpt_v7_1.1206.py
# Takes ~15 min (10 min train + 5 min eval). Produces final_model.int6.ptz
cp final_model.int6.ptz GS_final_model.int6.ptz  # safety copy
```

If it already exists from a prior run, verify:
```bash
ls -lh final_model.int6.ptz  # expect ~15.5 MB
```

## 4. Run TTT sweep (~45 min)

```bash
nohup bash sweep_ttt_calibration.sh > ttt_sweep.log 2>&1 &
tail -f ttt_sweep.log
```

Monitor progress — each config prints `>>> TAG: val_bpb=X.XXXX` when done.

## 5. Pull results

Results are in `logs/ttt_sweep_*/results.csv`. To view sorted:
```bash
sort -t',' -k9 -n logs/ttt_sweep_*/results.csv | column -t -s','
```

To pull logs back to local machine (from local terminal):
```bash
# Option A: scp if available
scp <pod-ssh>:/workspace/parameter-golf/logs/ttt_sweep_*/results.csv .

# Option B: base64 over SSH (RunPod PTY workaround)
ssh <pod-ssh> "cat /workspace/parameter-golf/logs/ttt_sweep_*/results.csv | base64" | base64 -d > results.csv
```

## If something goes wrong

| Issue | Fix |
|-------|-----|
| FA3 import fails | `export PYTHONPATH=/workspace/parameter-golf/flash-attention/hopper:$PYTHONPATH` |
| `final_model.int6.ptz` not found | Run GS training first (step 3) or copy from prior pod |
| OOM during TTT eval | Reduce `TTT_BATCH_SEQS` (default 32, try 16) |
| torchrun hides error | Debug: `EVAL_ONLY=1 python3 ttt_eval_runner.py 2>&1 | head -50` |
| Data shards missing | `python3 data/cached_challenge_fineweb.py --variant sp1024` |
| Sweep dies mid-run | Results.csv has partial data. Re-run script — it overwrites, so note completed configs from the log first |

## Files needed on pod

These must be in `/workspace/parameter-golf/`:
- `ttt_eval_runner.py` — GS script with EVAL_ONLY mode
- `sweep_ttt_calibration.sh` — the 11-config sweep
- `final_model.int6.ptz` — GS checkpoint (generated or copied)

## Timeline

| Phase | Time |
|-------|------|
| Pod setup + FA3 build | ~8 min |
| Preflight | ~30 sec |
| GS training (if needed) | ~15 min |
| TTT sweep (11 configs) | ~45 min |
| **Total (cold start)** | **~70 min** |
| **Total (checkpoint exists)** | **~55 min** |
