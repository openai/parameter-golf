# Depth-Recurrent Transformer

## Approach

Weight-shared depth recurrence: 5 unique transformer blocks looped 3x = 15 effective layers. Freed parameter budget reallocated to width (768 vs baseline 512).

- 5 physical blocks, 3 loops each = 15 effective depth
- dim=768, 12 heads, 6 KV heads (GQA), tied embeddings
- 21.4M params, ~13.9MB compressed
- U-Net skip connections across virtual layers
- Manual GQA KV-repeat for PyTorch 2.4 compat

## Results

Validated on 4xH100 SXM, 10min wallclock, 2651 steps:
- val_bpb: 1.2663
- val_loss: 2.1295

Model was still improving at cutoff (loss curve not plateaued). With 8xH100 and architecture sweep (narrower model = more steps), expect significant improvement.

## Key Insight

Depth recurrence lets you trade unique parameters for effective depth, freeing budget for width. The optimal width-depth-recurrence tradeoff within 16MB is the core research question. Current config may be too wide (too few steps); a narrower variant getting more training steps could beat baseline.

## Next Steps

1. Tokenizer optimization (sp4096/sp8192) for direct BPB reduction
2. Width/depth sweep to find Pareto-optimal config
3. Test-time training on validation sequence
4. QAT to close post-quantization gap
