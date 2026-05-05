# Experiment 082: 10L + clone fix — NO IMPROVEMENT

## Results
- FLAT+zstd: 16.89MB ❌ (identical to 080 without clone fix)
- Standard: 1.1626
- Sliding: *running*

## KEY FINDING
.clone().contiguous() made ZERO difference to artifact size.
The artifact size gap between platforms is from DIFFERENT WEIGHT VALUES
produced by different GPU floating point, not from tensor storage issues.
zstd is deterministic — same bytes = same output. Different bytes = different output.
Our weights simply have higher entropy than weights trained on Modal/Runpod hardware.

## CONCLUSION
Cannot fix artifact size gap with serialization tricks.
Must either:
1. Test on Runpod (actual submission platform)
2. Stay at 9 layers which fits on our hardware
3. Find ways to train weights with lower entropy (higher WD helps but not enough)
