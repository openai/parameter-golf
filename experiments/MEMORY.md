# Server Run Memory

## SSH
```
ssh -p 7122 rganapa@animal.netravi.net
# password: Summer02!
```

## SCP
```
scp -P 7122 <local_file> rganapa@animal.netravi.net:/data/backups/rganapa/parameter-golf/
```

## Base env vars
```
export PATH=/data/backups/rganapa/pylibs/bin:$PATH \
TMPDIR=/data/backups/rganapa/tmp \
PYTHONPATH=/data/backups/rganapa/pylibs \
TRITON_CACHE_DIR=/data/backups/rganapa/triton_cache \
TORCH_HOME=/data/backups/rganapa/torch_home \
WANDB_DIR=/data/backups/rganapa \
WANDB_API_KEY=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 \
WANDB_PROJECT=parameter-golf \
DATA_PATH=/data/backups/rganapa/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/backups/rganapa/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

## Model artifact location
```
/data/backups/rganapa/parameter-golf/final_model.int6.ptz
```

## Logs location
```
/data/backups/rganapa/parameter-golf/logs/
/data/backups/rganapa/*.log
```
