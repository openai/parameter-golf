"""Compatibility wrapper -- delegates to src/crucible/training/torch_backend.py

The training loop has been extracted into the crucible.training module.
This file remains for backward compatibility so that existing scripts,
fleet configs, and documentation that reference ``train_gpt.py`` keep working.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crucible.training.torch_backend import main

if __name__ == "__main__":
    main()
