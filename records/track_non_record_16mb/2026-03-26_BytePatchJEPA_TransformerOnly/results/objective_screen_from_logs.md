# Objective Screen Recovered From Logs

The final `results/objective_screen/summary.json` never synced back from RunPod, so the objective-screen headline values below were recovered from the copied-back final strong-probe logs before cleanup.

These are the numbers used in the README and PR body.

| Objective | Final strong-probe mode | Recovered `bpb` | Source log basename |
|------|------|------:|------|
| `slot_ema_teacher` | full | `2.3839` | `objective_transformer_rope_gqa_localglobal_slot_ema_teacher__final__strong.txt` |
| `slot_cosine` | full | `2.3885` | `objective_transformer_rope_gqa_localglobal_slot_cosine__final__strong.txt` |
| `slot_l2` | full | `2.3888` | `objective_transformer_rope_gqa_localglobal_slot_l2__final__strong.txt` |
| `slot_vicreg` | full | `2.3918` | `objective_transformer_rope_gqa_localglobal_slot_vicreg__final__strong.txt` |
| `masked_slot_jepa` | full | `2.5098` | `objective_transformer_rope_gqa_localglobal_masked_slot_jepa__final__strong.txt` |

Interpretation:

- `slot_ema_teacher` was the best objective in the Transformer-only family.
- `slot_cosine`, `slot_l2`, and `slot_vicreg` were tightly clustered.
- `masked_slot_jepa` was clearly worse.
