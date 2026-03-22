# TTT Debug Status

## Confirmed
- TTT works on 1xH100, 200 steps, TORCH_COMPILE=0 (improved bpb by 0.105)
- TTT fails on 8xH100, full training, TORCH_COMPILE=1 (degrades bpb by ~0.09)
- SmearGate is NOT the cause (tested with minimal model, both with/without)
- Fresh model with correct dtypes (CastedLinear.float()) still fails
- Passing base_model directly also fails

## Hypothesis: torch.compile + BigramHash graph break
- BigramHash.bigram_hash() uses torch.bitwise_xor and .to(torch.int32)
- These are NOT compatible with torch.compile(fullgraph=True)
- May cause Dynamo to cache wrong graph or silently produce incorrect output
- Need to test: full 8xH100 run with TORCH_COMPILE=0

## Next test
Run on 8xH100 with TORCH_COMPILE=0 to confirm. Training will be slower
(~90ms/step vs 68ms, ~6700 steps vs 8700) but if TTT works, we confirm
the root cause and can then fix the compile interaction.
