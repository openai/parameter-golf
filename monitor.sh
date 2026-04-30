#!/usr/bin/env bash
REPO="/mnt/c/Users/wrc02/Desktop/Projects/NanoGPT-Challenge/repo"
cd "$REPO"

echo "=== Active python3 training processes ==="
ps aux | grep "python3 train_gpt" | grep -v grep || echo "(none)"

echo ""
echo "=== SOTA 10x512D MLP3 + bigram ==="
grep -E "^step:[0-9]|^qat_active|^swa:|^final_mixed_zstd_roundtrip_exact|^final_sliding_window_exact|^arch:" logs/sota_10x512_mlp3_bigram.txt 2>/dev/null | tail -10 || echo "(not started)"

echo ""
echo "=== LARGE 10x1024D MLP3 + bigram ==="
grep -E "^step:[0-9]|^qat_active|^swa:|^final_mixed_zstd_roundtrip_exact|^final_sliding_window_exact|^arch:" logs/large_10x1024_mlp3_bigram.txt 2>/dev/null | tail -10 || echo "(not started)"

echo ""
echo "=== RUN A (prev): 14x640 no MLP + bigram ==="
grep -E "^step:[0-9]|^qat_active|^swa:|^final_mixed_zstd_roundtrip_exact|^arch:" logs/attn_14x640_nomlp_bigram.txt 2>/dev/null | tail -5 || echo "(not started)"

echo ""
echo "=== RUN B (prev): 10x640 MLP expand=2 + bigram ==="
grep -E "^step:[0-9]|^qat_active|^swa:|^final_mixed_zstd_roundtrip_exact|^arch:" logs/attn_10x640_mlp2_bigram.txt 2>/dev/null | tail -5 || echo "(not started)"
