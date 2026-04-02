# Local LLM Setup

## Model: Qwen 3.5 4B (Abliterated/Heretic)

- **Model:** `huihui_ai/qwen3.5-abliterated:4B`
- **Base:** Qwen 3.5 (latest Qwen release)
- **Parameters:** 4B
- **Disk:** ~2-3 GB
- **VRAM:** ~3-4 GB (tested on RTX 3070 8GB)
- **Context:** 256K
- **Source:** [Ollama](https://ollama.com/huihui_ai/qwen3.5-abliterated:4B)

Abliteration performed with [Heretic](https://github.com/p-e-w/heretic) — removes safety filtering while preserving model intelligence via directional ablation + TPE optimization.

## Installation

```bash
# 1. Install Ollama (requires sudo)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the model
ollama pull huihui_ai/qwen3.5-abliterated:4B

# 3. Run
ollama run huihui_ai/qwen3.5-abliterated:4B
```
