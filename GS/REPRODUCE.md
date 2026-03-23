# GOLD STANDARD — v7 GPTQ 1.1206 BPB (PR #508)

Best legal score: 1.1206 BPB (seed 1337), 15.56MB artifact.
3-seed mean: 1.1215 BPB.

## Reproduce and save checkpoint:

```bash
cd /workspace/parameter-golf
SEED=1337 torchrun --standalone --nproc_per_node=8 GS/GS_train_gpt_v7_1.1206.py
cp final_model.pt final_model_GS_v7_s1337.pt
```

## NEVER delete or overwrite these files.
