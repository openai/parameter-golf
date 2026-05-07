# Swarm-Guided Training: A Multi-Agent Think Tank Directs a Tiny Model Inside the 600s/16MB Limit

**Submission for OpenAI Parameter Golf (non-record / creative track)**

> A multi-agent Think Tank Swarm actively steers the training of a tiny GPT model in real time — dynamically adjusting hyperparameters and using a 500K-node knowledge graph to condition embedding initialization.

I designed and built this system from scratch — the swarm architecture, the knowledge graph structure, the agent roles, the decision logic, and the integration with the training loop are all my original work.

Instead of a static training script, a team of 4 specialist agents (QAT Timing, KG Weight, Gradient Health, MTP Weight) uses a typed 500K-node knowledge graph to make real-time decisions about quantization timing, knowledge graph influence scheduling, gradient clipping, and multi-token prediction weighting. The swarm makes 8 guided decision cycles during training, then outputs the final 16MB artifact with a full decision log.

This is the first submission (to my knowledge) where a multi-agent swarm acts as the live trainer and optimizer inside the strict Parameter Golf constraints.

### Core Idea
- The swarm is not just post-training analysis — it **steers the training loop itself**.
- Agents observe training metrics (loss trajectories, gradient norms, LR scale, training progress) and vote on changes.
- A 500K-node typed-edge knowledge graph (CAUSES, REQUIRES, SOLVES, CONTRADICTS) conditions the model's embedding initialization — giving semantically important tokens a head start.
- Everything is pure Python, rule-based heuristics — no LLM calls during training. Total swarm overhead: **<300 microseconds** across 600 seconds.
- The final model is a standard tiny GPT — the novelty is in **how** it was trained.

### Key Components I Designed
- **Think Tank Swarm** — multi-agent research system with typed-edge knowledge graph traversal, specialist agents, and voting-based consensus
- **Typed Knowledge Graph** — 500,292 nodes, 121,084 edges with 10 edge types (Solves, Causes, Requires, Enables, Contradicts, ConnectsTo, Improves, PartOf, TradeoffOf, AlternativeTo)
- **KG-Conditioned Embedding Init** — token embeddings scaled by PageRank importance from the knowledge graph at initialization (zero runtime cost)
- **VotingMesh** — consensus mechanism where agents propose changes with confidence scores, applied only above threshold
- **Swarm Decision Log** — full transparency into what the swarm decided, when, and why

The TTS research swarm (running on separate infrastructure with Phi-3 + Qwen2.5-7B local models) ran two investigation missions to design this approach before any training code was written. The knowledge graph was built from a 615MB library spanning 58 domain clusters.

All high-level architecture, agent roles, knowledge graph design, and system integration were mine. I used AI tools (Grok, Claude, Gemini) as implementation collaborators, but the vision and system design are original.

### The 4 Swarm Agents

| Agent | Observes | Controls | Decision Logic |
|-------|----------|----------|---------------|
| **QAT Timing** | LR scale + training progress | When to enable quantization-aware training | Fires when warmdown begins (scale < 0.15) but not before 40%. Safety deadline at 65%. |
| **KG Weight** | Training phase | Knowledge graph influence schedule | Ramps 0.3 → 0.5 early (guide learning), holds 0.4 mid-training, tapers to 0.1 late (free convergence). |
| **Gradient Health** | Gradient norms | Gradient clipping threshold | Tightens clipping from 0.3 to 0.15 if grad_norm > 2.0. |
| **MTP Weight** | Training phase | Multi-token prediction loss weight | Reduces MTP from 0.1 → 0.05 after 75% to shift focus to primary loss. |

### Results (Seed 1337)

| Metric | Score |
|--------|-------|
| Pre-quant (EMA) | 1.1397 |
| Post-quant (int6 roundtrip) | 1.1481 |
| Sliding window (stride=64) | 1.1245 |
| **Legal TTT** | **1.1220** |
| Artifact size | 15,955,969 bytes (under 16MB limit) |
| Training steps | 6,956 / 20,000 |
| Step average | 86.27 ms |
| Swarm cycles | 8 |
| Swarm decisions | 2 |
| Swarm overhead | <300 microseconds total |

### Swarm Decision Log

```
Swarm: 8 cycles, 2 decisions
  cycle 1 step 800:  kg_weight_agent  0.3 -> 0.5  (conf=0.75, 50us) progress=0.04, adjusting KG schedule
  cycle 4 step 3200: kg_weight_agent  0.5 -> 0.4  (conf=0.75, 39us) progress=0.16, adjusting KG schedule
```

### Architecture

```
                    ┌─────────────────────────────┐
                    │  Knowledge Graph (500K nodes) │
                    │  10 typed edge types           │
                    │  121K edges, 58 clusters        │
                    └──────────┬──────────────────┘
                               │ PageRank → 358 token
                               │ importance scores (976 bytes)
                               ▼
┌──────────────────────────────────────────────────┐
│              Embedding Initialization             │
│  tok_emb.weight[tid] *= 1 + 0.1 * importance     │
│  (one-time, zero runtime cost)                    │
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
                    │  ┌── QAT Timing ───┐ │
                    │  ├── KG Weight ────┤ │
                    │  ├── Grad Health ──┤ │
                    │  └── MTP Weight ───┘ │
                    │                      │
                    │  Observe → Vote →    │
                    │  Apply (if conf≥0.6) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Final Model +      │
                    │   Decision Log        │
                    └─────────────────────┘
```

### Base Stack

The swarm layers on top of the proven Parameter Golf recipe:
- 11 layers, 512d, 8 heads, 4 KV heads, 3x MLP
- LeakyReLU(0.75)² activation
- Parallel Muon optimizer
- Multi-Token Prediction (2 heads, weight=0.1)
- EMA weight averaging (0.997)
- BigramHash (2048) + SmearGate
- XSA (last 4 layers) + Partial RoPE + LN Scale
- Int6 quantization (GPTQ-lite + LZMA)
- Legal Score-First TTT
- Sliding window evaluation (stride=64)

### Why This Matters

Most submissions optimize a fixed training script. This submission shows a swarm can **dynamically steer** training decisions inside the tight constraints — opening the door to agentic self-improvement at the competition scale.

The knowledge graph provides external semantic structure that a 28M-parameter model cannot learn from data alone. By biasing embeddings toward important concepts, the model allocates capacity more efficiently from the start.

This is a proof-of-concept for **agentic training** — where the training process is not a static script but an intelligent system that observes, decides, and adapts.

### Files

| File | Size | Purpose | Counted in artifact? |
|------|------|---------|---------------------|
| `train_gpt.py` | 94KB | Training script + swarm integration | Yes |
| `swarm_agents.py` | 11KB | 4 agents + VotingMesh | No (imported) |
| `kg_data.py` | 1KB | Compressed KG importance data | No (imported) |
| `submission.json` | <1KB | Metadata | No |

### Reproducibility

```bash
LATE_QAT_THRESHOLD=0 TTT_ENABLED=1 KG_LOSS_WEIGHT=0.1 SEED=1337 \
  torchrun --nproc_per_node=8 train_gpt.py
```

Requires `swarm_agents.py` and `kg_data.py` in the same directory as `train_gpt.py`.

### Why This Matters to OpenAI

This submission demonstrates agentic self-improvement inside the strict 600s/16MB constraints — a step toward training systems that can reason about their own training process. It showcases a complete stack: multi-agent orchestration, typed knowledge graphs, rule-based consensus, and reliable execution under real production constraints.

**Built while competing in OpenAI Parameter Golf (March 2026).**
