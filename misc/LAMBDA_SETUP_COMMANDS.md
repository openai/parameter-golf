# Lambda 1xH100 Quick Runbook

```bash
export LAMBDA_HOST=ubuntu@192.222.52.79
export LAMBDA_KEY=/Users/alexzhao/.ssh/voltage-park-test
export REMOTE_DIR=/home/ubuntu/N-challenge
export LOCAL_DIR=/Users/alexzhao/code/N-challenge
```

## Setup (fresh instance)

```bash
ssh -i "$LAMBDA_KEY" -o StrictHostKeyChecking=accept-new "$LAMBDA_HOST" \
  'hostname; whoami; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'

rsync -az --delete -e "ssh -i $LAMBDA_KEY" --exclude '.git' --exclude '__pycache__' --exclude 'data/fineweb10B/' --exclude 'logs/' \
  "$LOCAL_DIR/" "$LAMBDA_HOST:$REMOTE_DIR/"

ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST" \
  'set -euxo pipefail; sudo apt-get update; sudo apt-get install -y python3-pip python3-venv git; \
   cd /home/ubuntu/N-challenge; python3 -m pip install --upgrade pip filelock; \
   pip3 install -r requirements.txt; pip3 install -r data/requirements.txt; \
   pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade; \
   python3 data/cached_fineweb10B.py 9'
```

## Sync (after local edits)

```bash
rsync -az --delete -e "ssh -i $LAMBDA_KEY" --exclude '.git' --exclude '__pycache__' --exclude 'data/fineweb10B/' --exclude 'logs/' \
  "$LOCAL_DIR/" "$LAMBDA_HOST:$REMOTE_DIR/"
```

## Launch (1x H100)

```bash
ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST" \
  'cd /home/ubuntu/N-challenge; timeout 4200 ./run_1xh100.sh |& tee run_1xh100.log'
```

## Quick checks

```bash
ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST" 'cd /home/ubuntu/N-challenge; grep -n "val_loss:" run_1xh100.log'
ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST" 'cd /home/ubuntu/N-challenge; grep -nE "OutOfMemoryError|CUDA out of memory|Traceback|FAILED|ChildFailedError|error:" run_1xh100.log | tail -n 50'
ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST" 'pgrep -af "[t]orchrun|[t]rain_gpt.py" || true'
```

## Quick fixes

- Missing shards after `rsync --delete`:
```bash
ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST" 'cd /home/ubuntu/N-challenge; python3 data/cached_fineweb10B.py 9'
```
- `huggingface_hub` lock mismatch:
```bash
ssh -i "$LAMBDA_KEY" "$LAMBDA_HOST" 'python3 -m pip install --upgrade filelock'
```
- `chz` import error on Python 3.10: re-sync latest code (`train_gpt_constants.py` has fallback).
