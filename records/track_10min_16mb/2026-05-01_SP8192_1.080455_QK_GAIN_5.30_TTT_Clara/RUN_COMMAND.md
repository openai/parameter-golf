# Run Command

The full shell launch line was not present verbatim in `train.log`; this command is reconstructed from the run log and environment.

The run log confirms:

```text
seed: 43
qk_gain_init: 5.3
ttt_enabled: True
ttt_lr: 0.005
ttt_epochs: 3
world_size: 8
```

Reconstructed launch command:

```bash
export SEED=43
export QK_GAIN_INIT=5.30
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export PYTHONUNBUFFERED=1
torchrun --standalone --nproc_per_node=8 train_gpt_py311_wrapper.py
```

Final result line from `train.log`:

```text
quantized_ttt val_loss:2.79019180 val_bpb:1.08017020 eval_time:277646ms
```

Final size line from `train.log`:

```text
Total submission size quantized+brotli: 15990341 bytes
```
