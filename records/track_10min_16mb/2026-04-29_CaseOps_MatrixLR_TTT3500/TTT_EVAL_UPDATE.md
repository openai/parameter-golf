# TTT Eval Update

Uses the original PR #1925 quantized artifacts and updates only score-first TTT eval:

```text
PHASED_TTT_PREFIX_DOCS=3500
PHASED_TTT_NUM_PHASES=1
TTT_LORA_LR=0.00008
```

The composite logs below contain the original training and quantization section, followed by an explicit eval-only continuation marker. No retraining or re-quantization occurs in the update section.

| Seed | Updated log | Old TTT BPB | Updated TTT BPB | Eval time | Delta |
|---:|---|---:|---:|---:|---:|
| 0 | `train_seed0_ttt_n1_lora8e5.log` | 1.06077210 | 1.06059202 | 370.1s | -0.00018008 |
| 42 | `train_seed42_ttt_n1_lora8e5.log` | 1.05925746 | 1.05906444 | 367.2s | -0.00019302 |
| 1234 | `train_seed1234_ttt_n1_lora8e5.log` | 1.06144340 | 1.06129561 | 367.8s | -0.00014779 |
| **Mean** | | **1.06049099** | **1.06031736** | | **-0.00017363** |

Compared with the seed-matched #1855 mean of 1.06107587, the updated composite result improves by 0.00075852 BPB.
