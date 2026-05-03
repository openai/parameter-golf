#!/usr/bin/env bash
set -uo pipefail

DATA_DIR=/home/frosty40/parameter-golf-lab/data
SPEC=/home/frosty40/parameter-golf-lab/tokenizer_specs_sp10240.json
LOG=/home/frosty40/parameter-golf-lab/sp10240_build.log
PY=/home/frosty40/miniconda3/bin/python3
TS=$(date +%Y%m%d_%H%M%S)

MAN_BAK="$DATA_DIR/manifest.json.bak.before_sp10240_$TS"
TCE_BAK="$DATA_DIR/tokenizer_config.export.json.bak.before_sp10240_$TS"

cp "$DATA_DIR/manifest.json" "$MAN_BAK"
cp "$DATA_DIR/tokenizer_config.export.json" "$TCE_BAK"
echo "[wrapper] backed up index files to $MAN_BAK / $TCE_BAK" | tee -a "$LOG"

cleanup() {
    rc=$?
    echo "[wrapper] build exited rc=$rc; restoring index files from backup" | tee -a "$LOG"
    cp "$MAN_BAK" "$DATA_DIR/manifest.json"
    cp "$TCE_BAK" "$DATA_DIR/tokenizer_config.export.json"
    echo "[wrapper] index files restored. New tokenizer/dataset (if built) remain in place." | tee -a "$LOG"
}
trap cleanup EXIT

echo "[wrapper] starting sp10240 build at $TS" | tee -a "$LOG"
"$PY" "$DATA_DIR/download_hf_docs_and_tokenize.py" \
    --output-root "$DATA_DIR" \
    --tokenizer-config "$SPEC" \
    --skip-byte 2>&1 | tee -a "$LOG"
