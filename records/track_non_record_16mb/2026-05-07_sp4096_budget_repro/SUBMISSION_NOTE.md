# Draft Non-Record Submission Note

This submission is a budget reproduction and engineering note for the SP4096 line of Parameter Golf submissions.

It starts from Kevin Clark's accepted `4096-Vocab + Larger Model + High WD + Simplifications` record and tests a constrained 1xH100 setup:

- 1x H100 SXM instead of 8xH100
- 86 SP4096 train shards due pod storage limits
- 3600 second train cap
- Full validation and sliding-window eval
- Reproducible artifact fitting under the decimal 16MB cap

The current result is valid but not SOTA:

```text
valid_sliding_bpb: 1.10743376
total_artifact_bytes: 15,987,195
```

The main implementation addition is an artifact-fit guard. If the compressed quantized artifact exceeds the byte cap, the script repeatedly coarsens selected low-impact quantized tensors and re-compresses. In the best RunPod result, repeated coarsening of `ve_shared.embed.weight` reduced the artifact enough to fit while changing sliding BPB by about `0.00012`.

The best valid local artifact is `runpod_results/final_model.qk45.valid.ptz`, produced from a `QK_GAIN_INIT=4.5` run. It improves the first 1xH100 budget baseline from `1.11043153` to `1.10743376` sliding BPB, but remains a non-record result relative to the public leaderboard.
