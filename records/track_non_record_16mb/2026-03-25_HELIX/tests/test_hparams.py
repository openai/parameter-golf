"""Sanity-check that Hyperparameters loads without error and fields exist."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
# Avoid importing CUDA-dependent and missing modules by patching
import types, unittest.mock
sys.modules.setdefault('flash_attn_interface', types.SimpleNamespace(flash_attn_func=None))
sys.modules.setdefault('sentencepiece', types.SimpleNamespace(SentencePieceProcessor=None))
sys.modules.setdefault('torch.distributed', types.SimpleNamespace())
sys.modules.setdefault('torch.nn.parallel', types.SimpleNamespace(DistributedDataParallel=None))
sys.modules.setdefault('zstandard', types.SimpleNamespace())
import train_gpt as tg
args = tg.Hyperparameters()
assert hasattr(args, 'dtpa_rank'), "dtpa_rank missing"
assert hasattr(args, 'num_unique_blocks'), "num_unique_blocks missing"
assert hasattr(args, 'num_iterations'), "num_iterations missing"
assert hasattr(args, 'ffn_hidden'), "ffn_hidden missing"
assert hasattr(args, 'mor_lb_weight'), "mor_lb_weight missing"
assert hasattr(args, 'mor_lb_decay_steps'), "mor_lb_decay_steps missing"
assert args.dtpa_rank == 4
assert args.num_unique_blocks == 5
assert args.model_dim == 768
assert "mor_gate" in tg.CONTROL_TENSOR_NAME_PATTERNS
print("PASS: Hyperparameters OK")
