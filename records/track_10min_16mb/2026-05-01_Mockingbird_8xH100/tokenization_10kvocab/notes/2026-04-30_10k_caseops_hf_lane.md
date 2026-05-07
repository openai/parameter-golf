# SP10240 CaseOps HF Lane

Status timestamp: 2026-04-29T23:59:53-05:00 local host time.

Scope: local dataset/tokenizer build and Hugging Face upload prep only. No
training was launched, no remote `/workspace` files were touched, and active
pod processes were left alone.

## Tokenizer Finding

No true SP10240 CaseOps tokenizer was found locally.

Verified standard SP10240 tokenizer copies:

- `/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_10240_bpe.model`
- `/home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_10240_bpe.vocab`
- `/home/frosty40/sota_rascal/pod_pulls/8x_35002131_20260429_sp10240_mlp375_promising_20260429_202105/fineweb_10240_bpe.model`
- `/home/frosty40/sota_rascal/pod_pulls/8x_35002131_20260429_sp10240_mlp375_promising_20260429_202105/fineweb_10240_bpe.vocab`

Those standard SP10240 models map CaseOps operator codepoints U+E001..U+E004
to `<unk>` id 3, so they are not CaseOps tokenizers.

Verified PR1855/PR1797 CaseOps tokenizer:

- `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- vocab size: 8192
- reserved CaseOps operator ids: U+E001=4, U+E002=5, U+E003=6, U+E004=7
- embedded SentencePiece trainer settings: BPE, byte fallback enabled,
  split digits enabled, `nmt_nfkc`, no dummy prefix, pad/bos/eos/unk ids
  0/1/2/3, hard vocab limit disabled.

Missing artifact: I did not find the original explicit command/log that trained
the 8192 CaseOps tokenizer. The SP10240 CaseOps tokenizer is therefore a new
lane derived from the PR1855 model's embedded trainer spec plus the existing
standard SP10240 condition note `tokenizer_skip_docs=50000`.

## Added Local Scripts

- `scripts/prepare_sp10240_caseops_data.py`
  - trains `fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model` when
    `--train-tokenizer` is set
  - reserves U+E001..U+E004 and validates ids 4..7
  - applies PR1855 `lossless_caps_caseops_v1`
  - writes uint16 `fineweb_train_*.bin` and `fineweb_val_*.bin`
  - writes validation byte sidecars `fineweb_val_bytes_*.bin`
  - supports `--max-train-shards`
  - fails closed if target shard files already exist

- `scripts/build_sp10240_caseops_local.sh`
  - default docs: `/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl`
  - default output root:
    `/home/frosty40/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets`
  - default `MAX_TRAIN_SHARDS=80`, `VAL_DOCS=50000`,
    `SHARD_TOKENS=10000000`, `TOKENIZER_SKIP_DOCS=50000`

- `scripts/upload_sp10240_caseops_to_hf.sh`
  - default repo: `Frosty40/10k_caseops_golfer`
  - `check` mode validates local outputs and HF auth
  - `upload` mode creates the dataset repo if needed and runs
    `hf upload-large-folder`

## Smoke Test

Command:

```bash
python3 scripts/prepare_sp10240_caseops_data.py \
  --docs /home/frosty40/parameter-golf-lab/data/docs_selected.jsonl \
  --out /home/frosty40/sota_rascal/data/smoke_sp10240_caseops_20260429_235907 \
  --train-tokenizer \
  --tokenizer-skip-docs 50 \
  --tokenizer-train-docs 2000 \
  --val-docs 20 \
  --max-train-shards 1 \
  --shard-tokens 2000
```

Smoke result:

- tokenizer vocab size: 10240
- reserved CaseOps ids: `[4, 5, 6, 7]`
- train shards: 1
- val shards: 12
- val byte sidecars: 12
- manifest:
  `/home/frosty40/sota_rascal/data/smoke_sp10240_caseops_20260429_235907/caseops_manifest.json`

## Real Build

Started:

```bash
tmux new-session -d -s sp10240_caseops_build_20260429_235938 \
  'cd /home/frosty40/sota_rascal && PYTHONUNBUFFERED=1 scripts/build_sp10240_caseops_local.sh > notes/runtime_logs/sp10240_caseops_build_20260429_235938.log 2>&1'
```

Live status at start verification:

- tmux session: `sp10240_caseops_build_20260429_235938`
- bash PID: `1942873`
- python PID: `1942875`
- log: `notes/runtime_logs/sp10240_caseops_build_20260429_235938.log`
- output root:
  `/home/frosty40/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets`
- current stage: SentencePiece tokenizer training
- command in python process:
  `python3 scripts/prepare_sp10240_caseops_data.py --docs /home/frosty40/parameter-golf-lab/data/docs_selected.jsonl --out /home/frosty40/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets --train-tokenizer --val-docs 50000 --max-train-shards 80 --shard-tokens 10000000 --tokenizer-skip-docs 50000`

Expected completion artifact:

- `/home/frosty40/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/caseops_manifest.json`

## Hugging Face Upload

HF CLI:

- binary: `/home/frosty40/.local/bin/hf`
- version: 1.6.0
- auth check: logged in as `Frosty40`; token value was not printed.

Current upload state: not ready until the real build writes tokenizer, shards,
and `caseops_manifest.json`.

Check command:

```bash
scripts/upload_sp10240_caseops_to_hf.sh check
```

Upload command after check passes:

```bash
REPO=Frosty40/10k_caseops_golfer scripts/upload_sp10240_caseops_to_hf.sh upload
```

Alternate repo name if preferred:

```bash
REPO=Frosty40/10k_golfer_caseops scripts/upload_sp10240_caseops_to_hf.sh upload
```

## Blockers / Caveats

- The exact original 8192 CaseOps tokenizer training command/log was not found.
  The new 10k tokenizer spec is explicit and reproducible, but it is a new
  SP10240 CaseOps lane, not an exact reproduction of the missing 8192 training
  command.
- Upload should not start until `caseops_manifest.json` exists and
  `scripts/upload_sp10240_caseops_to_hf.sh check` exits 0.
