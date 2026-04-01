# Parameter Golf Submission: GANGUS × NEXUS AI Orchestrator

## Approach: Multi-Model AI-Orchestrated Architecture Search

**Team:** Wojciech Kowalczyk ([@wojciechkowalczyk11to-tech](https://github.com/wojciechkowalczyk11to-tech))

### Core Idea

Instead of manually iterating on a single architecture, we use a **multi-tier AI orchestration system** to systematically explore, implement, evaluate, and refine model architectures — all within the 16MB/10min constraints.

The system treats Parameter Golf as a **search problem over architecture space**, where AI models at different cost/capability tiers handle different parts of the pipeline.

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  TIER 0: STRATEGIC BRAIN                                │
│  GPT-5.4 Codex (OpenAI flagship)                        │
│  → Plans experiments, analyzes results, decides next     │
│  → Delegates to specialized orchestrators below          │
└──────────────┬──────────────────────────────────────────┘
               │
    ┌──────────┼──────────────┬──────────────────┐
    ▼          ▼              ▼                  ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐
│GANGUS  │ │ N.O.C  │ │GigaGrok  │ │ NEXUS MCP    │
│AI      │ │Backend │ │Telegram  │ │ Server       │
│Orchest.│ │8 LLMs  │ │Grok 4.20 │ │ 54 tools     │
└───┬────┘ └───┬────┘ └────┬─────┘ └──────┬───────┘
    │          │           │               │
    ▼          ▼           ▼               ▼
 Multi-model  RAG +      Deep research   Infrastructure:
 routing +    pgvector   + X/Twitter     GCP, Cloudflare,
 cost optim.  semantic   search for      GitHub, RunPod,
 token guard  search     latest papers   Shell, Docker
```

### Orchestrators & Tools

| System | URL | Role in Competition |
|--------|-----|---------------------|
| **NEXUS MCP Server** | [mcp.nexus-oc.pl](https://mcp.nexus-oc.pl) | 54-tool control plane: RunPod pod management, GitHub ops, experiment logging, AI delegation to 8 providers |
| **Gangus AI** | [github.com/.../Gangus_Ai](https://github.com/wojciechkowalczyk11to-tech/Gangus_Ai) | Multi-model orchestrator with cost optimizer, evaluator layer, token guard. Routes tasks to cheapest capable model |
| **N.O.C (Nexus Omega Core)** | [Cloud Run (live)](https://nexus-backend-r56g4gr2da-uc.a.run.app) | Backend with 8 LLM providers, RAG pipeline with pgvector, multi-agent runtime |
| **GigaGrok Bot** | [github.com/.../gigagrok-bot](https://github.com/wojciechkowalczyk11to-tech/gigagrok-bot) | Grok 4.20 multimodal agent with xAI Collections (925-doc knowledge base), web+X search |
| **xAI Knowledge Base** | Vector store (925 tools indexed) | Semantic search over ML tools, architectures, training techniques |
| **Vertex AI Search** | Google Cloud | Semantic search across research documents and experiment logs |

### Competition Strategy

**Phase 1: Research (AI-driven literature review)**
- Grok 4.20 with web_search + x_search: scan latest efficient LM papers, X/Twitter threads from ML researchers
- xAI Collections: search 925-tool knowledge base for relevant architectures
- Vertex AI Search: cross-reference with indexed research corpus
- N.O.C RAG pipeline: semantic search over collected papers

**Phase 2: Architecture Search (multi-model code generation)**
- GPT-5.4 Codex: strategic planning, experiment design
- DeepSeek ($0.14/1M tokens): bulk code generation of architecture variants
- Grok 4.1-fast: analysis of results, suggest next experiments
- Mistral/Devstral-2: GitHub-native code review and optimization

**Phase 3: Training & Evaluation (automated pipeline)**
- RunPod pods: GPU training (1xH100 for iteration, 8xH100 for final)
- NEXUS MCP `shell_exec`: orchestrate training runs programmatically
- NEXUS MCP `runpod_deploy_from_template`: spawn/destroy pods on demand
- Automated eval pipeline: train → compress → measure bpb → log → iterate

**Phase 4: Compression & Submission**
- Train over-budget fp32 → quantize (int8/int4) → prune → fine-tune
- LZMA compression of final weights
- Automated artifact size verification (<16MB)
- PR submission via NEXUS MCP GitHub tools

### Technical Approaches Under Exploration

1. **Cross-layer weight sharing** — share weights every 2nd layer, ~40% param reduction
2. **Grouped Query Attention (GQA)** — fewer KV heads = fewer params in attention
3. **SwiGLU FFN** — better parameter efficiency per FLOP vs standard FFN
4. **RoPE** — zero learned positional parameters
5. **Hybrid transformer + linear attention** — RWKV-style layers for memory efficiency
6. **Post-training quantization** — train fp32 → int8 with calibration
7. **Magnitude pruning + recovery** — 10-20% sparsity with fine-tune pass
8. **Vocabulary optimization** — analyze if 1024 tokens is optimal or if sub-word strategies help

### Infrastructure

```
AI Models Available:
├── Claude Opus 4.6 (synthesis, strategy)
├── GPT-5.4 Codex (planning, delegation)
├── Grok 4.20 (deep research, multimodal)
├── Grok 4.1-fast (2M context, fast iteration)
├── DeepSeek-chat (cheap code generation, $0.14/1M)
├── Mistral/Devstral-2 (GitHub-native, free tier)
├── Gemini 2.5 Flash (backup)
└── OpenRouter (fallback routing)

Infrastructure:
├── NEXUS MCP Server: 54 tools, live 24/7
├── GCP VM: orchestration node
├── RunPod: GPU training (H100 pods)
├── Cloudflare: DNS, tunnels, CDN
├── GitHub Actions: CI/CD pipeline
└── xAI Vector Store: 925-doc knowledge base
```

### Why This Approach

Traditional ML competition workflow: human thinks → human codes → human trains → human analyzes → repeat.

Our workflow: **AI thinks → AI codes → AI trains → AI analyzes → AI iterates** — with human oversight at strategic checkpoints. This allows us to explore 10-100x more architecture variants in the same wall-clock time.

The 16MB constraint makes this especially interesting: the search space is small enough that systematic AI-driven exploration can cover it more thoroughly than manual iteration.

### Links

- NEXUS MCP Server (public): https://github.com/wojciechkowalczyk11to-tech/nexus-mcp-server-public
- Gangus AI: https://github.com/wojciechkowalczyk11to-tech/Gangus_Ai
- N.O.C: https://github.com/wojciechkowalczyk11to-tech/N.O.C
- GigaGrok Bot: https://github.com/wojciechkowalczyk11to-tech/gigagrok-bot
- Portfolio: https://nexus-oc.pl
