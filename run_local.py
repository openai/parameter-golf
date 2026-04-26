"""Wrapper to run train_gpt.py on a single GPU (RTX 5070 Ti workarounds).
Disables torch.compile which has dynamo issues with GQA on Blackwell GPUs.
"""
import torch
torch.compile = lambda fn, *a, **kw: fn
exec(open("train_gpt.py", encoding="utf-8").read())
