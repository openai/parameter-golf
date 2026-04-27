### Approach

This submission focuses on **parameter allocation**, not just export format. The current public frontier is clustered around a uniform `9 x 512` backbone with stronger quantization and evaluation tricks. I kept the proven surrounding recipe, but replaced the uniform `3x` FFN with **OpenELM-style layer-wise scaling** so later blocks get larger FFNs and earlier blocks get smaller ones at the same overall budget.

The exact FFN schedule is:

`768, 960, 1152, 1344, 1536, 1728, 1920, 2112, 2304`

The motivation is that, under a strict parameter budget, later layers are closer to the next-token loss and should benefit more from extra capacity than early layers. This is especially attractive in the Parameter Golf setting because the challenge is strongly **parameter-limited** relative to the 10-minute 8xH100 compute budget. Chinchilla-style reasoning says the available training compute could support a much larger model than the 16 MB artifact allows, so the right move is to spend the byte budget as efficiently as possible inside the model rather than leaving it uniformly distributed.

To support that architecture cleanly, I also implemented an **exact packed-int6 export path** with per-row fp16 scales. This keeps the submission self-contained and avoids relying on an external `zstd` dependency at evaluation time. I keep the tied embedding in fp16 because that tensor is especially quantization-sensitive and serves as both the input embedding and output projection.

Finally, the trainer includes **sliding-window evaluation** and a **CPU `DRY_RUN=1` mode** so the full train/export/eval pipeline can be verified locally without CUDA. Because I do not currently have GPU credits, I am not claiming a score yet and am leaving final eval metrics pending.

### Key Design Decisions

- Use **top-heavy FFN allocation** instead of a uniform `3x` FFN so later layers get more of the parameter budget.
- Keep the proven **9-layer, 512-dim, tied-embedding, GQA** backbone for a controlled comparison against public baselines.
- Export large matrices as **packed int6 + per-row fp16 scale + zlib** to make the artifact fully self-contained.
- Keep `tok_emb.weight` in **fp16** because it is the most quantization-sensitive tensor in public submissions.
- Retain **sliding-window eval** because it is allowed by the rules and is consistently helpful in public runs.
- Use **seed 42** by default and add a **10-step CPU dry-run** path for reproducibility without GPUs.

### What I Expect to Underperform / Risks

- The main risk is that layer-wise FFN scaling is theoretically motivated but not yet validated on this exact challenge objective.
- Packed int6 + zlib is self-contained, but the trained checkpoint still needs to prove it compresses comfortably under the 16 MB cap.
- If this challenge is dominated more by export/eval tricks than by parameter placement, the gain over a strong uniform-FFN baseline may be modest.

### Results

| Metric | Value |
|---|---|
| Baseline eval loss | `1.22436570` val_bpb (repo baseline) |
| Final eval loss | Pending — training to be run once compute credits are granted |
| Parameter count | `21,778,504` |
| Artifact size | Measured init-export: `4,273,390` bytes total; dense-random stress probe: `16,549,133` bytes total |
| Training time (estimated on 8×H100) | Expected to fit the 600s budget; public 21.8M-parameter 9x512 runs report roughly `45–57 ms/step`, implying about `10k–13k` steps in 10 minutes |

### Reproducibility

Seed: `42`.

Dry-run verified locally with:

`DRY_RUN=1 RUN_ID=topheavy_dryrun python train_gpt.py`

Full 8xH100 train/eval pending compute access.

### References

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022
- Mehta et al., *OpenELM*, 2024
