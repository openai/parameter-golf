# Spec 006 — checkpoints pointer file

All checkpoints live on AP-JP-1 volume `jlxvxeiol4` at `/workspace/runs/006-dense-ckpts/checkpoints/`. Each file is ~313 MB. Total: **15 GB across 49 files**.

Volume is persistent — stop/start the pod; files remain.

## Explicit milestone checkpoints (45 × `ckpt_event_stepNNNN.pt`)

Every 100 steps from 100 to 4500, inclusive:

```
ckpt_event_step100.pt    step1900.pt    step3700.pt
ckpt_event_step200.pt    step2000.pt    step3800.pt
ckpt_event_step300.pt    step2100.pt    step3900.pt
ckpt_event_step400.pt    step2200.pt    step4000.pt
ckpt_event_step500.pt    step2300.pt    step4100.pt
ckpt_event_step600.pt    step2400.pt    step4200.pt
ckpt_event_step700.pt    step2500.pt    step4300.pt
ckpt_event_step800.pt    step2600.pt    step4400.pt
ckpt_event_step900.pt    step2700.pt    step4500.pt
ckpt_event_step1000.pt   step2800.pt
... (all every-100 through 4500)
```

## Schedule-event checkpoints (2)

- `ckpt_warmdown_start_step1275.pt` — warmdown begins (predicted step ~1274 ✓)
- `ckpt_pre_recurrence_step1593.pt` — recurrence activates (predicted step ~1593 ✓)

## Final checkpoints (2)

- `ckpt_final_pre_ema_step4550.pt` — final weights, pre-EMA
- `ckpt_final_post_ema_step4550.pt` — final weights, post-EMA applied

## Access

Any future pod in AP-JP-1 attached to volume `jlxvxeiol4`:

```bash
ls /workspace/runs/006-dense-ckpts/checkpoints/
```

Load in Python:
```python
ckpt = torch.load("/workspace/runs/006-dense-ckpts/checkpoints/ckpt_event_step2000.pt",
                  map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
step = ckpt["step"]   # == 2000 for this example
ema = ckpt["ema_state"]   # dict of float32 EMA params on CPU
# optimizer_states also present
```
