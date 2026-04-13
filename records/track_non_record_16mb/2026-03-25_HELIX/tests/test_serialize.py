"""Test that _classify_param correctly classifies HELIX param names."""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault('flash_attn_interface', types.SimpleNamespace(flash_attn_func=None))
sys.modules.setdefault('sentencepiece', types.SimpleNamespace(SentencePieceProcessor=None))
sys.modules.setdefault('torch.distributed', types.SimpleNamespace())
sys.modules.setdefault('torch.nn.parallel', types.SimpleNamespace(DistributedDataParallel=None))
sys.modules.setdefault('zstandard', types.SimpleNamespace())
import train_gpt as tg

cases = [
    ("blocks.0.dtpa.W_O.weight",         "attn"),
    ("blocks.0.swiglu.gate.weight",       "mlp"),
    ("blocks.0.swiglu.fc.weight",         "mlp"),
    ("blocks.0.swiglu.proj.weight",       "mlp"),
    ("tok_emb.weight",                    "embed"),
    ("bigram.embed.weight",               "other"),
    ("blocks.0.pre_norm_attn.weight",     "other"),
]

for name, expected in cases:
    got = tg._classify_param(name)
    assert got == expected, f"_classify_param({name!r}) = {got!r}, expected {expected!r}"
print("PASS: _classify_param OK")
