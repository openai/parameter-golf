#!/usr/bin/env bash
set -uo pipefail

DATA_DIR=/home/frosty40/parameter-golf-lab/data
TOK_MODEL="$DATA_DIR/tokenizers/fineweb_10240_bpe.model"
TOK_VOCAB="$DATA_DIR/tokenizers/fineweb_10240_bpe.vocab"
DATASET_DIR="$DATA_DIR/datasets/fineweb10B_sp10240"
LOG=/home/frosty40/parameter-golf-lab/sp10240_upload.log
HF=/home/frosty40/miniconda3/bin/hf
REPO=Frosty40/10k_golfer
BUILD_PID=2018905

echo "[upload] watcher started $(date -Iseconds)" | tee -a "$LOG"
echo "[upload] waiting for build PID $BUILD_PID to exit, then for outputs to materialize" | tee -a "$LOG"

# Wait for the build process to exit
while kill -0 "$BUILD_PID" 2>/dev/null; do
    sleep 30
done
echo "[upload] build PID $BUILD_PID exited at $(date -Iseconds)" | tee -a "$LOG"

# Wait for outputs to be visible (script may flush after exit)
for i in $(seq 1 20); do
    if [[ -s "$TOK_MODEL" && -s "$TOK_VOCAB" && -d "$DATASET_DIR" ]]; then
        shard_count=$(ls "$DATASET_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
        val_count=$(ls "$DATASET_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
        if [[ "$shard_count" -gt 0 && "$val_count" -gt 0 ]]; then
            echo "[upload] outputs ready: tokenizer + $shard_count train shards + $val_count val shards" | tee -a "$LOG"
            break
        fi
    fi
    echo "[upload] outputs not yet visible (try $i/20), sleeping 15s" | tee -a "$LOG"
    sleep 15
done

if [[ ! -s "$TOK_MODEL" || ! -d "$DATASET_DIR" ]]; then
    echo "[upload] FATAL: outputs not present after build exit. Check sp10240_build.log." | tee -a "$LOG"
    exit 1
fi

echo "[upload] creating repo $REPO (public, dataset)" | tee -a "$LOG"
"$HF" repo create "$REPO" --repo-type dataset 2>&1 | tee -a "$LOG" || \
    echo "[upload] repo create returned nonzero (likely already exists), continuing" | tee -a "$LOG"

echo "[upload] uploading tokenizer files" | tee -a "$LOG"
"$HF" upload "$REPO" "$TOK_MODEL" "fineweb_10240_bpe.model" --repo-type dataset 2>&1 | tee -a "$LOG"
"$HF" upload "$REPO" "$TOK_VOCAB" "fineweb_10240_bpe.vocab" --repo-type dataset 2>&1 | tee -a "$LOG"

echo "[upload] uploading dataset shards from $DATASET_DIR (large folder)" | tee -a "$LOG"
"$HF" upload-large-folder "$REPO" "$DATASET_DIR" --repo-type dataset 2>&1 | tee -a "$LOG"

echo "[upload] DONE at $(date -Iseconds). Repo: https://huggingface.co/datasets/$REPO" | tee -a "$LOG"
