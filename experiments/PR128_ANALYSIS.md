# PR128 Analysis — What We Can Learn

## PR128 Result: 1.1594 sliding BPB, 15.16MB artifact, 10,535 steps @ 57ms

## Key Differences from Our Best (049: 1.1685)

### 1. Regular Muon, NOT NorMuon
PR128 uses **plain Muon** (same as baseline). NOT NorMuon like PR122.
This is significant — our 048 with NorMuon was WORSE than 038 with plain Muon (1.1990 vs 1.1935).
**NorMuon may be hurting us.** PR128 proves you can get 1.1594 without it.

### 2. relu² 3x MLP (h=1536), NOT SwiGLU
PR128 uses `relu(x).square()` with 3x expansion (hidden=1536).
Supports `MLP_HIDDEN` env var for fine-tuning the hidden dim.
2 matrices (fc, proj) instead of SwiGLU's 3 (up, gate, down).

### 3. Simpler int6 quantization — NO bit-packing
PR128 uses int6 but stores as **int8** (not bit-packed to 6 bits).
Still uses per-row quantization with range [-32, 31].
Embedding kept in fp16.
Compression: zstd-22 (if available) or zlib.

### 4. Same hyperparameters as our 049
- matrix_lr=0.02, scalar_lr=0.02, tied_embed_lr=0.03
- muon_momentum=0.99, warmdown_iters=3000
- train_seq_len=4096, train_batch_tokens=393216
- eval_stride=64
- vocab=1024, 9 layers, dim=512

### 5. Eval batch_seqs=16 (we use 32)
Smaller eval batch to fit in memory. May not affect BPB.

### 6. No SWA/LAWA
No weight averaging at all. Clean.

## What This Tells Us

### Action Items (ranked by expected impact):

1. **Switch back to plain Muon** — PR128 got 1.1594 with regular Muon. NorMuon appears to be hurting our SwiGLU config. This is the #1 thing to test.

2. **Test relu² 3x vs SwiGLU at equal params** — PR128 uses relu² h=1536. Our SwiGLU h=1024 has identical param count. Direct comparison: which activation is better with int6 QAT?

3. **Simplify quantization** — PR128 stores int6 as int8 (not bit-packed). Simpler code, may compress just as well with zstd. Our bit-packing adds complexity.

4. **Our 049 artifact is 290KB over** — need to trim. PR128 uses MLP_HIDDEN env var for fine-tuning. We could do SWIGLU_HIDDEN=960 or so.

## Proposed Experiments

### 052: Our best config but with plain Muon (not NorMuon)
- Same as 049 but swap NorMuon → Muon
- Hypothesis: plain Muon will improve BPB by ~0.005-0.01
- This is the cheapest test — just an env var or small code change

### 053: relu² 3x h=1536 (no SwiGLU) with plain Muon
- Directly replicate PR128's architecture with our script
- Compare against 052 to isolate SwiGLU vs relu² effect

### 054: SwiGLU h=960 with plain Muon (fits under 16MB)
- Same as 052 but trim hidden to fit artifact budget
- This would be our submission candidate

## Script Location
experiments/pr128_train_gpt.py (1354 lines, clean, no SWA/LAWA)
