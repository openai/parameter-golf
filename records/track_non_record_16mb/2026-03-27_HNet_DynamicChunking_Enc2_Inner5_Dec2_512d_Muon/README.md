# H-Net with Dynamic Sequence Chunking

**Non-Record Submission (Research Contribution) | First H-Net Architecture in Parameter Golf**  
**Author:** Tim Shen ([@TimS-ml](https://github.com/TimS-ml))  

## Summary

First implementation of **H-Net (Hierarchical Network) with Dynamic Sequence Chunking** in Parameter Golf. H-Net learns content-dependent boundaries in the sequence, compresses the input into a shorter latent chunk sequence, runs most of its modeling capacity in that compressed space, and then upsamples back to full resolution for autoregressive prediction.

This implementation adapts the chunking mechanism from the [lucidrains reference](https://github.com/lucidrains/h-net-dynamic-chunking) and rewrites the transformer stack to match the competition baseline: **GQA + RoPE + ReLU-squared + CastedLinear + RMSNorm + Muon/Adam**.

## Architecture

H-Net splits the model into five parts:

**Encoder** (full resolution) -> **DynamicSequenceChunker** (learned downsampling) -> **Inner** (compressed sequence) -> **Upsample** -> **Decoder** (full resolution)

The chunker predicts boundaries using cosine distance in a projected QK space and downsamples a 512-token sequence into a much shorter latent sequence. In the strongest artifact-eligible setting here, the best results came from allocating more depth to the encoder/decoder interface than to a purely inner-heavy transformer. Empirically, **layout matters more than width**, and more aggressive compression continued to help when paired with the stronger layout.

## Results

### Simulated 8xH100 10-minute run
1xH100 80-minute run with x8 gradient_accumulation_steps, bpb: 1.4054; 1.4129 (int8+zlib). Artifact (int8+zlib): 11.9 MB.


### 1xH100 10-minute reference runs

These runs are **single-GPU ablations**, included to show the behavior of H-Net under the competition stack. They are useful for architecture analysis, but are **not directly comparable to official leaderboard submissions**, which are defined around 8xH100 / 10-minute training and a 16 MB artifact budget.

| Setting | model_dim | target_avg_len | layout | val_bpb | final_bpb | Note |
|---------|-----------|----------------|--------|---------|-----------|------|
| Best raw 1xH100 run | 640 | 10 | 2,5,2 | 1.4489 | 1.4552 | Best single-GPU result so far; exceeds current 16 MB record budget |
| Best budget-oriented 1xH100 run | 512 | 9 | 2,5,2 | 1.4601 | 1.4640 | Strongest current run in the smaller compressed-artifact setting |
| Layout comparison | 640 | 9 | 2,5,2 | 1.4514 | 1.4579 | Strong outer-interface layout |
| Layout comparison | 640 | 9 | 1,7,1 | 1.4887 | 1.4942 | Deeper inner trunk, worse result |

### Main observations from the 1xH100 runs

- **Layout mattered more than width.** Moving from `1,7,1` to `2,5,2` produced a much larger gain than moving from `d512` to `d640`.
- **More aggressive compression kept helping** on the stronger layout: for `2,5,2`, increasing `target_avg_len` from 7 -> 8 -> 9 -> 10 steadily improved the 1xH100 result.
- **The bottleneck appears to be the chunking / reconstruction interface, not just inner-model capacity.** Giving more depth to the encoder / decoder side was consistently better than concentrating depth in the inner transformer alone.
- Repeated runs showed the same qualitative pattern, suggesting that this is a real architectural effect rather than a one-off seed artifact.


## DDP and Multi-GPU Training

**Work in progress.**

H-Net's DynamicSequenceChunker produces variable-length inner sequences per sample, which complicates standard PyTorch DDP because batches across ranks no longer have uniform shapes. I am currently running longer experiments in this direction and will update this section once there is a stable multi-GPU training path.

Possible directions under investigation:

- **Padding-based batching**: pad inner sequences to the batch maximum
- **Bucket-based batching**: group samples with similar chunk counts
- **Ragged / nested tensor approaches**: promising long-term direction, but not yet mature enough for a clean competition implementation

## Next Steps

Near-term work is focused on three directions:

- validating the best compression settings under longer runs
- finishing a practical multi-GPU training path for 8xH100-compatible experiments
- combining H-Net with more aggressive quantization so wider models can fit within the record budget

Even in its current single-GPU form, this submission shows that **dynamic hierarchical sequence modeling is viable inside the Parameter Golf stack** and produces consistent, interpretable scaling trends.

## Credits

- Nawrot et al. (2024), *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*
- [lucidrains/h-net-dynamic-chunking](https://github.com/lucidrains/h-net-dynamic-chunking)
- modded-nanogpt / Parameter Golf community stack
- OpenAI for hosting the competition
