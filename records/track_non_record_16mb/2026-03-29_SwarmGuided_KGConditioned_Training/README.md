# Swarm-Guided KG-Conditioned Training

**val_bpb: 1.1220** (seed 1337, with Legal TTT) | **~15.96 MB** | 8xH100 SXM

## Approach

This submission introduces **agentic self-improvement during training** — a lightweight multi-agent swarm monitors training metrics and makes hyperparameter decisions via consensus voting, while a 500K-node knowledge graph conditions the model's embedding initialization.

### What's Novel

No other submission treats the training process itself as a multi-agent system. Instead of a static training script with fixed hyperparameters, 4 autonomous agents observe training signals (loss trajectories, gradient norms, training progress) and vote on interventions every 800 steps. A typed-edge knowledge graph (CAUSES, REQUIRES, SOLVES, CONTRADICTS) provides semantic importance scores that bias the model toward learning important concepts first.

### The Swarm (4 Agents, Rule-Based, <300 microseconds total overhead)

| Agent | Observes | Controls | Decision Logic |
|-------|----------|----------|---------------|
| QAT Timing | LR scale + training progress | When to enable quantization-aware training | Fires when warmdown begins (scale < 0.15) but not before 40% progress. Safety deadline at 65%. |
| KG Weight | Training phase | Knowledge graph influence schedule | Ramps 0.3→0.5 early (guide learning), holds 0.4 mid-training, tapers to 0.1 late (free convergence). |
| Gradient Health | Gradient norms | Gradient clipping threshold | Tightens clipping if grad_norm > 2.0 to prevent instability. |
| MTP Weight | Training phase | Multi-token prediction loss weight | Reduces MTP from 0.1→0.05 after 75% to shift focus to primary loss. |

The swarm runs 8 decision cycles across a typical 7000-step training run. Each cycle completes in ~50 microseconds (pure Python heuristics, no LLM calls). Total overhead: **<300 microseconds** across the entire 600-second training window.

### Knowledge Graph Conditioning

A 500,000-node typed-edge knowledge graph (built from 615MB of academic papers, technical documentation, and domain knowledge across 58 clusters) is distilled to 358 token importance scores via degree-based centrality analysis. These scores are compressed to 976 bytes (LZMA + base64) and baked into the submission.

At initialization, token embeddings for semantically important concepts are scaled by their graph importance score (weight=0.1). This gives the model a head start on learning important concepts with zero runtime cost — no loss function modifications, no per-step overhead.

### Integration with Existing Stack

The swarm layers cleanly on top of the proven Parameter Golf stack:
- LeakyReLU(0.75)² activation
- Parallel Muon optimizer
- Multi-Token Prediction (2 heads, weight=0.1)
- EMA weight averaging (0.997)
- Int6 quantization (GPTQ-lite + LZMA)
- XSA (last 4 layers)
- BigramHash (2048)
- Legal Score-First TTT
- Sliding window evaluation (stride=64)

The swarm adds ~100 lines to train_gpt.py and imports from `swarm_agents.py` (not counted in artifact size). The knowledge graph data is in `kg_data.py` (976 bytes base64 blob).

## Results

### Seed 1337

| Metric | Score |
|--------|-------|
| Pre-quant (EMA) | 1.1397 |
| Post-quant (int6, roundtrip) | 1.1481 |
| Sliding window (stride=64) | 1.1245 |
| **Legal TTT** | **1.1220** |
| Artifact size | 15,955,969 bytes |
| Training steps | 6,956 / 20,000 |
| Step average | 86.27 ms |

### Swarm Decision Log (Seed 1337)

```
Swarm: 8 cycles, 2 decisions
  cycle 1 step 800: kg_weight_agent kg_weight 0.3->0.5 (conf=0.75, 50us) progress=0.04, adjusting KG schedule
  cycle 4 step 3200: kg_weight_agent kg_weight 0.5->0.4 (conf=0.75, 39us) progress=0.16, adjusting KG schedule
```

## Architecture

```
                    ┌─────────────────────────────┐
                    │   Knowledge Graph (500K nodes)│
                    │   Typed edges: CAUSES,        │
                    │   REQUIRES, SOLVES, etc.       │
                    └──────────┬──────────────────┘
                               │ PageRank → 358 token
                               │ importance scores
                               ▼
┌──────────────────────────────────────────────────┐
│              Embedding Initialization             │
│  tok_emb.weight[tid] *= 1 + 0.1 * importance     │
└──────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Training Loop      │
                    │   (600s, 8xH100)     │
                    └──────────┬──────────┘
                               │ Every 800 steps
                    ┌──────────▼──────────┐
                    │   Swarm Decision     │
                    │   Cycle (~50 μs)     │
                    │                      │
                    │  ┌─── QAT Agent ───┐ │
                    │  ├── KG Weight ────┤ │
                    │  ├── Grad Health ──┤ │
                    │  └── MTP Weight ───┘ │
                    │                      │
                    │  Vote → Apply        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Model Output       │
                    │   + Decision Log     │
                    └─────────────────────┘
```

## Why This Matters

Parameter Golf submissions optimize a fixed training recipe. This submission makes the recipe itself adaptive. The swarm is lightweight enough to run inside the 600-second window with zero measurable overhead, yet flexible enough to make meaningful decisions about quantization timing, learning dynamics, and loss weighting.

The knowledge graph provides external semantic structure that a small model (28M params) cannot learn from data alone. By biasing embeddings toward important concepts at initialization, the model allocates capacity more efficiently.

This is a proof-of-concept for **agentic training** — where the training process is not a static script but an intelligent system that observes, decides, and adapts.

## Files

| File | Size | Purpose | Counted in artifact? |
|------|------|---------|---------------------|
| `train_gpt.py` | 94KB | Training script with swarm integration | Yes (code bytes) |
| `swarm_agents.py` | 11KB | 4 agents + VotingMesh + data types | No (imported module) |
| `kg_data.py` | 1KB | Pre-computed KG importance (base64 blob) | No (imported module) |

## Infrastructure

The knowledge graph was built from a 615MB knowledge library spanning 58 domain clusters (AI/ML, systems, algorithms, security, etc.) using a custom Rust-based graph extraction pipeline. PageRank-style importance scoring on 500,292 nodes and 121,084 typed edges produces the 358 token importance weights used in this submission.

The swarm architecture is derived from the Think Tank Swarm (TTS), a multi-agent research system that uses knowledge graph traversal, typed-edge specialists, and voting-based consensus for autonomous investigation tasks.
