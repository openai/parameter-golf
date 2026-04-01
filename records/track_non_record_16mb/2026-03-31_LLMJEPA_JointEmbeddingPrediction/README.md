# Non-record: LLM-JEPA — Joint Embedding Prediction for Language Modeling

**val_bpb: 2.2020** | 1×RTX 5090, 180s

Adds a JEPA prediction head alongside AR loss. Context embeddings (first 75%) predict target span embeddings (last 25%) via lightweight MLP. Stop-gradient on targets prevents collapse. JEPA predictor stripped from export (training-only).

Implements OpenAI's requested 'JEPA' direction.
