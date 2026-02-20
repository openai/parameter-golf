# N-challenge

Minimal reproducible commands for setup and launch.

## Fresh setup (repo already present)

```bash
cd ~/N-challenge
python3 -m pip install --upgrade pip filelock
pip3 install -r requirements.txt
python3 data/cached_fineweb10B.py 9
```

## Launch

8x H100:
```bash
cd ~/N-challenge
TORCHINDUCTOR_CUDAGRAPHS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py # internal
```

1x H100:
```bash
cd ~/N-challenge
USE_FLASH_ATTN=0 python3 -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py --config configs/train_gpt_1xh100.py
```

## Quick checks

```bash
cd ~/N-challenge
ls data/fineweb10B_sp4k/fineweb_val_000000.bin
ls data/fineweb10B_sp4k/fineweb_train_000009.bin
```

## Pure byte tokenizer (260 vocab)

Create tokenizer artifact (fixed mapping, no corpus training needed):

```bash
python3 data/create_pure_byte_tokenizer.py \
  --output_json data/tokenizers/fineweb_pure_byte_260.json
```

Build FineWeb shards with pure byte tokenization:

```bash
python3 data/fineweb_pure_byte.py \
  --version 10B \
  --tokenizer_json data/tokenizers/fineweb_pure_byte_260.json \
  --output_dir fineweb10B_byte260
```

Train with this dataset/tokenizer:

```bash
TRAIN_FILES=data/fineweb10B_byte260/fineweb_train_*.bin \
VAL_FILES=data/fineweb10B_byte260/fineweb_val_*.bin \
VOCAB_SIZE=260 \
BOS_ID=1 \
BIGRAM_VOCAB_SIZE=1920 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Matched multi-tokenizer export (same docs across all dumps)

Exports five datasets from one shared doc cache:
- pure byte (260)
- SentencePiece BPE 512
- SentencePiece BPE 1024
- SentencePiece BPE 2048
- SentencePiece BPE 4096

All five use the exact same raw doc sequence and the same doc-level val/train split.

```bash
python3 data/export_matched_fineweb_tokenizer_datasets.py \
  --version 10B \
  --num_docs 2000000 \
  --num_val_docs 50000 \
  --shuffle_seed 1337 \
  --sp_vocab_sizes 512,1024,2048,4096 \
  --output_root data/matched_10B_docs2m_seed1337
```

Output layout:
- `data/matched_10B_docs2m_seed1337/manifest.json`
- `data/matched_10B_docs2m_seed1337/tokenizers/*`
- `data/matched_10B_docs2m_seed1337/datasets/fineweb10B_byte260/`
- `data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp512/`
- `data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/`
- `data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp2048/`
- `data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp4096/`

## Notes

- Use this README as the source of truth for setup/repro commands.
- Use one dependency install path: `pip install -r requirements.txt`.
- Default training data is local shards at `data/fineweb10B_sp4k/`; trainer fails fast if missing.
- 1x launch should set `USE_FLASH_ATTN=0` on current runtime.
- For data rebuild/upload: `python3 data/build_upload_4096_bpe.py --repo_id cocohearts/4096-bpe --version 10B`.

# Internal

Run bbb cptree az://oaidatasets2/speedrunkits/fineweb10B/ data/fineweb10B/ instead of dataset loading.

bbb cptree az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337 data/matched_10B
