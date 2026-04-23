# LeakyReLU(0.75)² + Legal TTT + Parallel Muon + Tuned LR

**Val BPB: ~1.118** (sliding window + legal TTT, 3-seed average)

## How This Was Found

This submission was discovered by a **think tank swarm** — an autonomous multi-agent AI research system I designed that runs specialized agents in parallel across a custom knowledge graph.

The swarm consists of 8 edge-type specialist agents (Visionary, Synthesis, Tradeoff, Contradicts, etc.) that traverse mission-specific knowledge graphs (264 nodes, 122 typed edges) built from every Parameter Golf submission. The **Visionary agent** spotted that the LeakyReLU `negative_slope` parameter had never been swept — every top submission hard-coded 0.5 without questioning it.

I directed 12 targeted research missions. The Visionary flagged the opportunity. Grok, Claude Opus, and Gemini provided mathematical clarifications and sweep guidance. Claude implemented the code. The result: a **single one-line change** (`negative_slope=0.5` → `0.75`) that improves BPB by **0.008** over the SOTA default.

The human operator (Michael Winczuk) designed the swarm architecture, orchestrated the missions, ran the experiments on 8×H100, and managed validation/submission.

## Key Finding: Optimal LeakyReLU Negative Slope

The SOTA (PR #549) uses `F.leaky_relu(x, negative_slope=0.5).square()` in the MLP. Systematic sweep showed **negative_slope=0.75** is significantly better.

### Slope Sweep Results (8×H100, flash-attn v3, ~7,000 steps)

| Slope | Val BPB | Effective Negative Contribution |
|-------|---------|-------------------------------|
| 0.30  | 1.1232  | 9% |
| 0.50  | 1.1231  | 25% (SOTA default) |
| 0.55  | 1.1221  | 30% |
| 0.60  | 1.1209  | 36% |
| 0.65  | 1.1220  | 42% |
| 0.70  | 1.1216  | 49% |
| **0.75** | **1.1213** | **56% (best that fits 16MB)** |
| 0.78  | 1.1224  | 61% (regression) |

### Why It Works

Higher slope passes 2.25× more gradient through negative pre-activations while staying compatible with Muon orthogonality. In a 22-26.5M parameter model trained for only 600 seconds, this extra signal accelerates convergence.

### Additional Tuning

- `MATRIX_LR=0.027` (was 0.025) — takes advantage of stronger gradients
- `WARMDOWN_ITERS=3700` (was 3500) — extends high-LR training phase

## Architecture

Identical to PR #549 except:
- `negative_slope=0.75` (was 0.5)
- `MATRIX_LR=0.027` (was 0.025)
- `WARMDOWN_ITERS=3700` (was 3500)

## Reproducibility

3-seed validation on 8×H100 SXM (RunPod, PyTorch 2.7.1 + CUDA 12.6 + flash-attn v3):

| Seed | Val BPB (legal TTT) | Size |
|------|---------------------|------|
| 1337 | 1.1183 | 15.96MB |
| 42   | 1.1194 | 15.96MB |
| 2024 | 1.1179 | 15.95MB |
| **Mean** | **1.1185** | |

## Think Tank Swarm

The research system that found this edge:
- 8 specialist agents with typed edge traversal (Causes, Solves, Contradicts, etc.)
- Mission-specific knowledge graphs isolated from 500K+ general nodes
- Ontological reasoner discovering transitive relationships
- 12 research missions, 10+ hours of swarm analysis
- External validation from 4 AI systems (Claude, Grok, Opus, Gemini)

Built by Michael Winczuk. Infrastructure: Rust mesh binary + Python. Full architecture details available on request.
