# ABRAM_CHIP v2 — HECR Ultra Compact Language Model

val_bpb: ~0.50 (simulation dataset — FineWeb validation pending compute grant)

## Architecture

int16 only. No transformers. No gradients. No floats.

node_state = ((H*C + E*(100-R))[:, None] * emb) // 100

## Results

- bpb: ~0.50
- size: 34 KB
- space used: 0.21% of 16MB limit
- training time: ~36s CPU

Note: simulation dataset. FineWeb validation pending compute grant.

## Author

Abraham | H.A.S. Framework | Genoma Cognitivo
San Luis Potosí, México | March 22, 2026
github.com/abrahaw123-cell/abram_chip
