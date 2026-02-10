# OpenAI N-challenge

Train the best model you can with 50M parameters or less.

## Run

```bash
git clone <repo-url> N-challenge && cd N-challenge
pip install -r requirements.txt
python3 data/cached_fineweb10B.py 9
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
torchrun --standalone --nproc_per_node=8 train_gpt.py --config configs/train_gpt_8xh100.py
```

Alternatively, to run on 1x H100 use the follwoing training command instead:
```
USE_FLASH_ATTN=0 torchrun --standalone --nproc_per_node=1 train_gpt.py --config configs/train_gpt_1xh100.py
```

## Notes

- 1x profile uses fixed stage batch `3072`, `compile_model=False`, `empty_cache_every_steps=1`, and launches with `USE_FLASH_ATTN=0` in `run_1xh100.sh`.
- 1x model keeps depth:width closer to 8x (`8x: 11/768`, `1x: 6/512`).
- On this runtime, `USE_FLASH_ATTN=1` fails with `non-finite mean train loss` (latest probe at step 174). With `USE_FLASH_ATTN=0`, train and val stay finite through step 500 in current checks (`val_loss=9.6280` at step 250).
- Detailed architecture notes are in `AGENTS.md`.
- Lambda runbook is in `LAMBDA_SETUP_COMMANDS.md`.

# Internal

Run bbb cptree az://oaidatasets2/speedrunkits/fineweb10B/ data/fineweb10B/ instead of dataset loading.