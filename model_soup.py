"""
Model Soup — average weights from multiple independently trained models.
Based on Wortsman et al. (2022): "Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference cost."

Usage: python model_soup.py model1.pt model2.pt [model3.pt ...]
"""
import sys, os, torch, glob, math, numpy as np
from pathlib import Path
import torch.nn.functional as F
import sentencepiece as spm
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim, sl, vs = 512, 1024, 1024

model_paths = sys.argv[1:]
if not model_paths:
    # Auto-discover model files
    model_paths = sorted(glob.glob('best_model*.pt'))
    if not model_paths:
        print("Usage: python model_soup.py model1.pt model2.pt")
        sys.exit(1)

print(f'Model Soup: averaging {len(model_paths)} models', flush=True)
for p in model_paths:
    print(f'  {p}', flush=True)

# Load and average
avg_state = None
for i, path in enumerate(model_paths):
    sd = torch.load(path, map_location='cpu')
    if avg_state is None:
        avg_state = {k: v.float() for k, v in sd.items()}
    else:
        for k in avg_state:
            avg_state[k] += sd[k].float()
    print(f'  Loaded {path}', flush=True)

for k in avg_state:
    avg_state[k] /= len(model_paths)

# Save averaged model
torch.save(avg_state, 'model_soup.pt')
print(f'Saved averaged model to model_soup.pt', flush=True)
print(f'Run: python full_eval.py model_soup.pt', flush=True)
