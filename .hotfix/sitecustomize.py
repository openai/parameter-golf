import os

try:
    import torch._dynamo as d
    # Keep compile enabled, but avoid known DDP graph optimizer crash path.
    d.config.optimize_ddp = False
    # If a graph still fails, fall back instead of killing the entire run.
    d.config.suppress_errors = True
except Exception:
    pass
