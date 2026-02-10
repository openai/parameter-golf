# N-challenge

H100-focused GPT training repo.

Default launchers:
- `./run.sh` (8x H100)
- `./run_1xh100.sh` (1x H100)

## Running the current setup

### Internal debug
On 8xH100 Voltage Park, run
```bash
git clone <repo-url> N-challenge && cd N-challenge
pip install -r requirements.txt
pip install -r data/requirements.txt
python3 data/cached_fineweb10B.py 9
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
./run.sh
```
### End internal debug

For 1x H100:
```bash
pip install -r requirements.txt
pip install -r data/requirements.txt
python3 data/cached_fineweb10B.py 9
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
./run_1xh100.sh
```
`run_1xh100.sh` launches with `USE_FLASH_ATTN=0`.

## Configs

`train_gpt.py` reads `--config`:
```bash
torchrun --standalone --nproc_per_node=1 train_gpt.py --config configs/train_gpt_1xh100.py
```

## Notes

- 1x profile uses fixed stage batch `3072`, `compile_model=False`, `empty_cache_every_steps=1`, and launches with `USE_FLASH_ATTN=0` in `run_1xh100.sh`.
- 1x model keeps depth:width closer to 8x (`8x: 11/768`, `1x: 6/512`).
- On this runtime, `USE_FLASH_ATTN=1` fails with `non-finite mean train loss` (latest probe at step 174). With `USE_FLASH_ATTN=0`, train and val stay finite through step 500 in current checks (`val_loss=9.6280` at step 250).
- Detailed architecture notes are in `AGENTS.md`.
- Lambda runbook is in `LAMBDA_SETUP_COMMANDS.md`.
