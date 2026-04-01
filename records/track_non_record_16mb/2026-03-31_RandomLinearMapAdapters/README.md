# Non-record: Learning Adapters on Random Linear Maps

**val_bpb: 2.2017** | 1×RTX 5090, 180s

All Q/K/V/proj/up/down weights are frozen random orthogonal matrices. Only diagonal scale+shift vectors are trainable (~0.5% of params). Tests whether random projections provide sufficient structure for language modeling from scratch.

Implements OpenAI's requested 'Learning adapters on random linear maps' direction.
