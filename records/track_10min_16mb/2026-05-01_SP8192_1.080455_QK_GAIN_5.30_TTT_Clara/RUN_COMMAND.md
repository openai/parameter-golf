# Run Command

The full shell launch line was not present verbatim in `train.log`; this command is reconstructed from the run log and environment.

The run log confirms:

```text
seed: 42
qk_gain_init: 5.3
ttt_enabled: True
ttt_lr: 0.005
ttt_epochs: 3
world_size: 8
```

Reconstructed launch command:

```bash
SEED=42 QK_GAIN_INIT=5.30 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Final result line from `train.log`:

```text
quantized_ttt val_loss:2.79092773 val_bpb:1.08045510 eval_time:925151ms
```

Final size line from `train.log`:

```text
Total submission size quantized+brotli: 15994385 bytes
```
