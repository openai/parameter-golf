#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

ROOT="/workspace/final_pr1991_v6"
REPO="$ROOT/parameter-golf"
ART="$ROOT/artifacts"

REC_SRC="records/track_10min_16mb/2026-04-30_SP8192_PPMMixer_O5_TunedGate"
REC_FINAL="records/track_10min_16mb/2026-04-30_SP8192_PPMMixer_O5_V6Micro_Final3Seed"

mkdir -p "$ROOT" "$ART"

echo "=== FINAL 8xH100 PR1991 + V6 MICRO ==="
date
nvidia-smi || true

# Kill accidental old runs
pkill -TERM -f "torchrun|train_gpt" 2>/dev/null || true
sleep 3
pkill -KILL -f "torchrun|train_gpt" 2>/dev/null || true

# Clone PR1991
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/openai/parameter-golf.git "$REPO"
fi

cd "$REPO"
git fetch origin main
git reset --hard origin/main
git fetch origin pull/1991/head:pr1991_ppm_o5
git checkout pr1991_ppm_o5

# Dependencies
python3 -m pip install -U pip
python3 -m pip install -U brotli sentencepiece datasets huggingface_hub hf_transfer tqdm numpy

python3 -m pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ || true

test -f "$REC_SRC/train_gpt.py" || {
  echo "ERROR: missing PR1991 train_gpt.py: $REC_SRC/train_gpt.py"
  find records/track_10min_16mb -name train_gpt.py | tail -50
  exit 1
}

# Official FineWeb SP8192
if [ ! -f data/tokenizers/fineweb_8192_bpe.model ] || ! ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin >/dev/null 2>&1; then
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
fi

test -f data/tokenizers/fineweb_8192_bpe.model
ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin >/dev/null
ls data/datasets/fineweb10B_sp8192/fineweb_val_*.bin >/dev/null

# Build V6-only micro mix and FORCE hardcoded FineWeb path to point to V6-micro dataset.
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
import json, re, hashlib, os, shutil
import numpy as np
import sentencepiece as spm

repo = Path("/workspace/final_pr1991_v6/parameter-golf")
data_root = repo / "data/datasets"
base = data_root / "fineweb10B_sp8192"
orig = data_root / "fineweb10B_sp8192_ORIG_OFFICIAL"
dst = data_root / "fineweb10B_sp8192_V6MICRO"
tok = repo / "data/tokenizers/fineweb_8192_bpe.model"

assert base.exists(), base
assert tok.exists(), tok

# Preserve original FineWeb once.
if base.is_symlink():
    base.unlink()
if not orig.exists():
    base.rename(orig)

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

train_files = sorted(orig.glob("fineweb_train_*.bin"))
val_files = sorted(orig.glob("fineweb_val_*.bin"))
assert train_files and val_files, "missing FineWeb train/val files"

first = train_files[0]

# Copy first train shard for tiny sparse injection. Symlink rest and ALL official val files.
for p in train_files + val_files:
    target = dst / p.name
    if p == first:
        shutil.copy2(p, target)
    else:
        os.symlink(p.resolve(), target)

# Download V6 only.
v6src = Path("/workspace/final_pr1991_v6/v6_source")
snapshot_download(
    repo_id="8Planetterraforming/Parameter-Golf-V6-Privacy-Web-Filtering",
    repo_type="dataset",
    local_dir=str(v6src),
)

def clean(x):
    x = "" if x is None else str(x)
    return re.sub(r"\s+", " ", x).strip()

texts, seen = [], set()

for fp in v6src.rglob("*"):
    if not fp.is_file() or fp.suffix.lower() not in [".jsonl", ".json", ".txt", ".md"]:
        continue

    raw = fp.read_text("utf-8", errors="replace")
    rows = []

    if fp.suffix.lower() == ".jsonl":
        for line in raw.splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    pass
    elif fp.suffix.lower() == ".json":
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                rows += [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            pass
    else:
        rows.append({"text": raw})

    for obj in rows:
        candidates = []
        for k in [
            "plain_text_micro_mix_only", "training_text", "clean_payload",
            "payload", "text", "content", "target", "summary"
        ]:
            v = obj.get(k)
            if isinstance(v, str):
                candidates.append(v)

        if not candidates:
            for v in obj.values():
                if isinstance(v, str) and 60 <= len(v) <= 1200:
                    candidates.append(v)

        for t in candidates:
            t = clean(t)
            if len(t) < 50:
                continue
            if len(t) > 700:
                t = t[:700]

            # Avoid chat/instruction style contamination.
            bad = ["User:", "Assistant:", "Question:", "Answer:", "Do not reveal", "Audit this"]
            if any(b in t for b in bad):
                continue

            h = hashlib.sha256(t.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                texts.append(t)

# Very small: we do not want to wreck PR1991. This is one-card final risk.
texts = sorted(texts, key=lambda x: (len(x), x))[:384]
plain = "\n".join(texts) + "\n"

sp = spm.SentencePieceProcessor()
sp.Load(str(tok))
ids = [int(x) for x in sp.EncodeAsIds(plain) if 0 <= int(x) < 8192]
assert ids, "V6 tokenization produced zero ids"

# Tiny sparse train-only injection. No validation touched.
inject = int(os.environ.get("V6_INJECT_TOKENS", "8192"))
inject = max(2048, min(inject, 16384))
rep = np.resize(np.array(ids, dtype=np.uint16), inject)

mm = np.memmap(dst / first.name, dtype=np.uint16, mode="r+")

# Sparse blocks after warmup. Not prefix injection.
start = 3_000_000
block = 512
gap = 500_000
pos = start
written = 0

while written < inject and pos + block < len(mm):
    n = min(block, inject - written)
    mm[pos:pos+n] = rep[written:written+n]
    written += n
    pos += gap

mm.flush()

manifest = {
    "dataset": "V6 Privacy-Web-Filtering only",
    "source": "8Planetterraforming/Parameter-Golf-V6-Privacy-Web-Filtering",
    "method": "tiny sparse train-only micro injection",
    "base_fineweb": str(orig),
    "output_dataset": str(dst),
    "fineweb_validation": "untouched official fineweb_val symlinks",
    "texts": len(texts),
    "inject_tokens_requested": inject,
    "inject_tokens_written": written,
    "sha256_plain": hashlib.sha256(plain.encode()).hexdigest(),
    "legal_boundary": "V6 is train-only; not used for validation; not hidden eval data"
}

(dst / "v6_micro_manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")

# Force hardcoded path.
if base.exists() or base.is_symlink():
    if base.is_symlink():
        base.unlink()
    else:
        shutil.rmtree(base)
base.symlink_to(dst.resolve())

print(json.dumps(manifest, indent=2))
print("[OK] fineweb10B_sp8192 now points to:", base.resolve())
PY

mkdir -p "$REC_FINAL"
cp "$REC_SRC/train_gpt.py" "$REC_FINAL/train_gpt.py"
cp data/datasets/fineweb10B_sp8192/v6_micro_manifest.json "$REC_FINAL/v6_micro_manifest.json"

cat > "$REC_FINAL/README.md" <<'MD'
# SP8192 Byte-PPM O=5 + V6 Privacy-Web-Filtering Micro Final

Base:
- PR1991 SP8192 Byte-PPM Mixer O=5.

Modification:
- A tiny V6 train-only sparse micro-injection is applied to the first FineWeb train shard.
- FineWeb validation files are untouched official symlinks.
- V6 is not used as validation or evaluation data.
- No validation leakage is intended.

Risk:
- This is a final-time ablation. If it underperforms pure PR1991, pure PR1991 remains the safer baseline.
MD

cat > "$REC_FINAL/submission.json" <<'JSON'
{
  "name": "SP8192 Byte-PPM O=5 + V6 Privacy-Web-Filtering Micro",
  "author": "Sebastian Laskowski",
  "github_id": "Terraforming-Planet",
  "claiming_record_candidate": true,
  "val_bpb": null,
  "notes": "Final 8xH100 3-seed run. V6 is train-only sparse micro-injection; FineWeb validation untouched."
}
JSON

# Preflight legal boundary.
echo "=== PREFLIGHT DATASET PATHS ==="
readlink -f data/datasets/fineweb10B_sp8192
ls -lh data/datasets/fineweb10B_sp8192 | head
cat "$REC_FINAL/v6_micro_manifest.json"

# Run 3 seeds on 8xH100.
for S in 42 314 999; do
  LOG="$REC_FINAL/train_seed${S}.log"
  CONSOLE="$ROOT/final_v6_seed${S}.console.log"

  echo
  echo "=== FINAL SEED $S / 8xH100 ==="
  date
  nvidia-smi || true

  DATA_DIR="./data" \
  DATA_PATH="./data/datasets/fineweb10B_sp8192" \
  TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model" \
  VOCAB_SIZE=8192 \
  TRAIN_SHARDS_OVERRIDE=80 \
  TRAIN_BATCH_TOKENS=786432 \
  MAX_WALLCLOCK_SECONDS=600 \
  VAL_LOSS_EVERY=0 \
  TRAIN_LOG_EVERY=500 \
  MLP_MULT=3.96875 \
  V6_INJECT_TOKENS=8192 \
  RUN_ID="final_pr1991_v6_8xh100_seed${S}" \
  SEED="$S" \
  torchrun --standalone --nproc_per_node=8 "$REC_FINAL/train_gpt.py" \
    2>&1 | tee "$LOG" | tee "$CONSOLE"

  echo "=== SEED $S SUMMARY ==="
  grep -E "train_files|val_files|ppm_mixer val_bpb|Total submission size|quantized val_bpb|FAILED|ERROR|Root Cause" "$LOG" | tail -120 || true

  tar -czf "$ART/final_pr1991_v6_seed${S}_$(date -u +%Y%m%d_%H%M%S).tar.gz" \
    "$REC_FINAL" "$CONSOLE" 2>/dev/null || true

  ls -lh "$ART" | tail -10
done

echo
echo "=== FINAL 3-SEED SUMMARY ==="
grep -R -E "ppm_mixer val_bpb|Total submission size|quantized val_bpb|FAILED|ERROR|Root Cause" "$REC_FINAL" | tee "$ROOT/final_summary.txt" || true

tar -czf "$ART/final_pr1991_v6_ALL_$(date -u +%Y%m%d_%H%M%S).tar.gz" \
  "$REC_FINAL" "$ROOT/final_summary.txt" 2>/dev/null || true

echo
echo "=== DONE. DO NOT DELETE POD BEFORE COPYING ARTIFACTS ==="
ls -lh "$ART" | tail -20
