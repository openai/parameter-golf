# BRLM SAGE Chunk4 D384

Chunk4 SAGE is the strongest observed trajectory, but the first `chunk4_1g_r1` log used the original non-causal within-chunk SAGE gate. This directory therefore promotes the architecture with the gate fixed before treating it as legal.

## Observed Reference

- Log: `logs/reference_leaky_gate_1g.txt`
- 1xH100 600s exact int4 BPB: `1.26460519`
- Submission: `13,181,861 B`
- Tokens seen: `702,578,688`
- Status: strong signal, not legal final evidence because the current-chunk `amax` gate used later token route signals.

## Legal Chunk4 Result

- Log: `logs/chunk4_causalgate_1g_r1.txt`
- 1xH100 600s exact int4 BPB: `1.36607248`
- Final step: `1007`
- Step avg: `596.38 ms`
- Tokens seen: `692,944,896`
- Submission: `13,191,221 B`
- Status: legal but neutral. It is slightly worse than the clean R fallback at `1.36443202`, so chunk4 SAGE does not rescue the architecture once the gate leak is removed.

## Legal Fix

The SAGE bus content was already shifted by one chunk and causal. The issue was only:

```python
chunk_gate = gate_chunks.amax(dim=2).repeat_interleave(chunk, dim=1)
```

That lets early positions in a chunk know whether later positions in the same chunk are high-entropy or structural anchors. The fixed root uses:

```python
chunk_gate = torch.cummax(gate_chunks, dim=2).values.reshape(bsz, seqlen)
```

Now position `i` can depend only on route signals at offsets `<= i`.

## Next Runs

Run one legal chunk8 check using the same file with `SAGE_CHUNK=8`. If chunk8 also ties or loses to clean R, demote SAGE and return to the localwide R/O chassis.

1. `SAGE_CHUNK=8 train_gpt.py`
2. `variants/train_gpt_chunk2.py`
3. `variants/train_gpt_chunk1.py`
4. `variants/train_gpt_chunk4_ffn3.py`
5. `variants/train_gpt_chunk4_decay098.py`
6. `variants/train_gpt_chunk4_block2.py`
7. `variants/train_gpt_chunk4_resid035.py`

Promotion bar: legal exact int4 BPB below `1.31691521` keeps the chunk-scale thesis alive; below `1.28` restores the target rescue range.
