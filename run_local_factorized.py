"""Wrapper to run train_gpt_factorized.py on a single GPU (RTX 5070 Ti workarounds)."""
import torch
torch.compile = lambda fn, *a, **kw: fn
exec(open("train_gpt_factorized.py", encoding="utf-8").read())
