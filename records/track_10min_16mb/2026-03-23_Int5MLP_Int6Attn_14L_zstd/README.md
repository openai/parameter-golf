# Int5 MLP + Int6 Attention + zstd-22

val_bpb: 1.1996 (mean of 3 seeds)

## Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|----------------|-------|
| 42   | 1.1998  | 15,961,747     | yes   |
| 1337 | 1.1990  | 15,623,922     | yes   |
| 2024 | 1.1999  | 15,878,451     | yes   |
| Mean | 1.1996  |                |       |
| Std  | 0.0004  |                |       |

## What I changed

Started from the baseline and made three main changes:

**Quantization:** MLP weights use int5, attention weights use int6. 
Embeddings stay in fp16 since they're more sensitive to precision loss. 
The intuition is MLP weights compress more aggressively without hurting 
quality as much as attention does.

**Compression:** Swapped zlib for zstd at level 22. Straightforward change, 
saves about 1-2MB on the final artifact.

**Architecture:** Increased mlp_mult from 2 to 3 and num_layers from 9 to 11. 
The int5/int6 quantization freed up enough space to make this possible 
while staying under 16MB.

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads
- MLP 3x expansion
- Tied embeddings
- Trained on 8xH100 SXM, 10 min cap
