# This package-level re-export is only a convenience shim so callers can import
# `train_gpt_mlx_lib.main` without knowing that the actual implementation lives
# in `runner.py`. Keeping `__init__` minimal avoids hiding any real logic here.
from .runner import main

__all__ = ["main"]
