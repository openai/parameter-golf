# N-challenge

Minimal reproducible commands for setup and launch.

## Fresh setup (repo already present)

```bash
cd ~/N-challenge
python3 -m pip install --upgrade pip filelock
pip3 install -r requirements.txt
python3 data/cached_fineweb10B_sp4k.py 9
```

## Launch

8x H100:
```bash
cd ~/N-challenge
python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py --config configs/train_gpt_8xh100.py
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

## Notes

- Use this README as the source of truth for setup/repro commands.
- Use one dependency install path: `pip install -r requirements.txt`.
- Default training data is local shards at `data/fineweb10B_sp4k/`; trainer fails fast if missing.
- 1x launch should set `USE_FLASH_ATTN=0` on current runtime.
- For data rebuild/upload: `python3 data/build_upload_4096_bpe.py --repo_id cocohearts/4096-bpe --version 10B`.
