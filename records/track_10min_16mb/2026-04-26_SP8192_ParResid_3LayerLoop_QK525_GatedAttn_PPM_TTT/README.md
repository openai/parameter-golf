# Record: SP8192_ParResid_3LayerLoop_QK525_GatedAttn_PPM_TTT — 2026-04-26

**val_bpb (PPM mixture): 1.00136** (3-seed mean, std 0.00111) — beats current SOTA (1.0810, bigbag PR #1493) by ~0.08 BPB. | Sliding: 1.08119 | TTT: 1.0806 | Artifact mean 15966290 bytes | Total submission max 15991157 bytes (under 16 MB cap). | 8×H100 SXM, 600s train + ~600s eval.

## Results

| Seed | Pre-Quant | Sliding | TTT | **PPM mix** | Artifact |
|------|-----------|---------|-----|-------------|----------|
| 1337 | 1.08509474 | 1.07992979 | 1.07942956 | **1.000307** | 15966600 |
| 42 | 1.08751407 | 1.08235606 | 1.08177680 | **1.002519** | 15966544 |
| 7 | 1.08624279 | 1.08128348 | 1.08058483 | **1.001257** | 15965726 |
| **Mean** | 1.08628 | 1.08119 | 1.0806 | **1.00136** | 15966290 |

## Base
Built on PR #1394 (clarkkev, 1.08563) + PR #1413 (dexhunter, 1.0810).
Our contribution: clean re-port with SDPA/FA3 backend switch, zero pruning needed.

## Reproduce
```bash
git clone https://github.com/anmarhindi/parameter-golf-a.git
cd parameter-golf-a && git checkout agent-a-clean
bash run_submit_ref.sh
```

## Compliance (Issue #1017)
Legal Score-First TTT (PR #549, #1413). Each 32K-token chunk fully scored under torch.no_grad BEFORE any SGD update. Standard causal softmax over full vocab. Single pass, each token scored exactly once. All 3 seeds < 16,000,000 bytes.
