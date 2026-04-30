# Record: K_KVShare_Wider full-recipe FLA

**val_bpb: 1.0409** (3-seed mean, std 0.0011) | **3.1648 nats** | **8xH100 SXM, 600s** | **No TTT**

FLA / GatedDeltaNet candidate using `K_KVShare_Wider` on a fuller
upstream-style recipe. Nearest prior family reference: PR `#1370`.
The packaged script avoids runtime dependency downloads from `train_gpt.py`.

## Results

| Seed | Steps | Post-EMA BPB | **Quantized BPB** | val_loss (nats) | Artifact |
|------|------:|-------------:|------------------:|----------------:|---------:|
| 1337 | 1652 | 1.020660 | **1.03967403** | 3.16104735 | 15,762,406 |
| 42 | 1652 | 1.022042 | **1.04153708** | 3.16671180 | 15,870,797 |
| 2025 | 1583 | 1.023994 | **1.04148177** | 3.16654364 | 15,648,800 |
| **Mean** | **1629** | **1.022232** | **1.04089763** | **3.16476760** | **15,760,668** |

## Technique

- FLA / GatedDeltaNet family (`K_KVShare_Wider`)
- KV sharing is used to buy width rather than depth
- fuller upstream-style recipe
- EMA + SWA + late QAT + int6 artifact path
- final scored line in all logs is `final_int6_roundtrip_exact`

Not used:
- no TTT
- no SLOT
- no n-gram overlay
- no SWA/XSA final scoring path (`K_KVShare_Wider` has `num_swa_layers = 0`)

## Compliance Notes

- train uses `train_files`; scoring uses `val_files`
- no eval-time adaptation
- `train_gpt.py` does not download dependencies during evaluation
- dependencies are installed beforehand via `requirements.txt`
- max artifact bytes across reported seeds: `15,870,797`
- full packaged-folder audit remains under `16,000,000` bytes

## Reproducibility

Install dependencies before evaluation:

```bash
pip install -r requirements.txt
```

Prepare the SP8192 cached dataset/tokenizer as usual, then run one seed with:

```bash
SEED=$SEED ARCH_MODE=K MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 EVAL_COMPILE_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`EVAL_COMPILE_ENABLED=0` is an operational stability setting for final-eval
robustness; it does not change the model family or scored path.
