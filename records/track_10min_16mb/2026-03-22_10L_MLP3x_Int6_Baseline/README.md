# 10L MLP3x Int6 Baseline (non-record)

Non-record submission. Local MLX smoke test confirming pipeline works end-to-end.

## Config
- 10 layers, 512 dim, 8 heads, 4 KV heads
- MLP 3x expansion (hidden=1536), relu²
- int6 quantization, zlib-9 compression
- Trained on Apple Silicon (MLX), 200 iterations only

## Score
val_bpb: 2.3517 (200 iterations — not a competitive score)

## Planned improvements
- zstd-22 compression
- Sliding window eval (stride=64)
- Muon WD=0.04
- SmearGate + BigramHash
- SWA over last 40% of warmdown
- Full 10-min run on 8xH100
