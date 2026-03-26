## PR DRAFT — DO NOT SUBMIT YET (add image first)

### Title:
The Frugendorff Squared: Fractal Weight Sharing + MLP 4x (1.1478 BPB, 15.15MB)

### Body:

## Summary

Non-record submission exploring **fractal weight sharing** — a novel approach where 6 unique transformer blocks are looped 2× each, providing 12 effective layers of depth with only 6 blocks of stored parameters. The freed parameter budget enables **MLP 4x expansion**, which is the primary quality driver.

<!-- ADD YOUR IMAGE HERE -->

- **val_bpb: 1.1478** (sliding window stride=64) | **15.15 MB** | 8xH100 SXM, 600s
- 28.2M params, 4,390 steps at 136.7ms/step
- Full pipeline: Muon + SWA + Late QAT + Training Replay + Self-Distillation + EMA

## Key Insight

MLP 4x gives ~2% relative BPB improvement over MLP 3x, but doesn't fit in 16MB with 12 unique layers. Fractal weight sharing (6 unique × 2 loops) fits it in 15.15 MB. The weight sharing is the compression technique; the MLP 4x is the quality lever.

## Architecture

- 6 unique blocks × 2 loops = 12 effective depth
- dim=640, 10 heads, 5 KV (GQA), head_dim=64
- MLP 4x (hidden=2560), relu-squared
- Orthogonal loop positions, U-Net skips, SmearGate, BigramHash, VE128, XSA

## No TTT on validation data

All training uses training data only. Late replay buffers training batches. Self-distillation uses EMA teacher on training data. Fully compliant with issue #402.

## Test plan

- [x] 8xH100 SXM, 600s
- [x] Artifact under 16MB (15.15 MB)
- [x] No TTT on validation data (per issue #402)
- [x] Post-quant roundtrip verified
- [x] Sliding window eval (stride=64)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---
### Command to create PR (after adding image):
```
gh pr create --repo openai/parameter-golf \
  --title "The Frugendorff Squared: Fractal Weight Sharing + MLP 4x (1.1478 BPB, 15.15MB)" \
  --body "$(cat FRUGENDORFF_PR_BODY.md)"
```
