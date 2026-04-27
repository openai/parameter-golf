"""
Validate the _select_ttt_params filter logic against a mock model that has the
same parameter-name structure as the real GPT (from PR #1493 SOTA).

Run locally on CPU. Doesn't import torch.cuda or flash-attn.
"""
from collections import OrderedDict
import torch
import torch.nn as nn

CONTROL_TENSOR_NAME_PATTERNS = (
    'attn_scale', 'attn_scales', 'mlp_scale', 'mlp_scales',
    'resid_mix', 'resid_mixes', 'q_gain',
    'skip_weight', 'skip_weights', 'skip_gates',
)


def _select_ttt_params(model, filt):
    named = list(model.named_parameters())
    if filt == 'all':
        return [p for _, p in named]
    if filt == 'scales':
        return [p for n, p in named
                if any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if filt == 'scales+embed':
        return [p for n, p in named
                if 'tok_emb' in n
                or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if filt.startswith('last_n_layers:'):
        K = int(filt.split(':', 1)[1])
        L = len(model.blocks)
        keep = tuple(f'blocks.{i}.' for i in range(L - K, L))
        return [p for n, p in named if any(n.startswith(k) for k in keep)]
    if filt == 'attn_only':
        return [p for n, p in named if '.attn.' in n]
    if filt == 'mlp_only':
        return [p for n, p in named if '.mlp.' in n]
    raise ValueError(f"Unknown TTT_PARAM_FILTER: {filt}")


class MockAttn(nn.Module):
    def __init__(self, dim=64, num_heads=8, num_kv_heads=4):
        super().__init__()
        head_dim = dim // num_heads
        kv_dim = num_kv_heads * head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(num_heads))


class MockMLP(nn.Module):
    def __init__(self, dim=64, mlp_mult=4):
        super().__init__()
        h = int(dim * mlp_mult)
        self.fc = nn.Linear(dim, h, bias=False)
        self.proj = nn.Linear(h, dim, bias=False)


class MockBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = MockAttn(dim)
        self.mlp = MockMLP(dim)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]))


class MockGPT(nn.Module):
    def __init__(self, num_layers=11, dim=64, vocab=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([MockBlock(dim) for _ in range(num_layers)])
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2 + 1, dim))
        self.skip_gates = nn.Parameter(torch.zeros(num_layers // 2 + 1, dim))


def fmt(params, named_lookup):
    """Return sorted unique param names from list of param tensors."""
    by_id = {id(p): n for n, p in named_lookup}
    return sorted({by_id[id(p)] for p in params})


def main():
    m = MockGPT()
    named = list(m.named_parameters())
    total_floats = sum(p.numel() for p in m.parameters())
    print(f"Mock model: {len(named)} param tensors, {total_floats:,} floats")
    print()

    cases = [
        ('all',          None,       'should select every named parameter'),
        ('scales',       'control',  'should select only fp32 control tensors'),
        ('scales+embed', 'control+e','scales plus tok_emb'),
        ('last_n_layers:3', 'last3', 'only blocks 8,9,10'),
        ('attn_only',    'attn',     'only .attn.* params'),
        ('mlp_only',     'mlp',      'only .mlp.* params'),
    ]

    for filt, _, desc in cases:
        sel = _select_ttt_params(m, filt)
        sel_floats = sum(p.numel() for p in sel)
        sel_names = fmt(sel, named)
        print(f"{filt:24s} -> {len(sel):3d} tensors, {sel_floats:>9,} floats   [{desc}]")
        if filt == 'scales':
            # Expected: 11 attn_scale, 11 mlp_scale, 11 resid_mix, 11 q_gain,
            # plus skip_weights, skip_gates. No tok_emb, no .weight matrices.
            for n in sel_names:
                if any(bad in n for bad in ('c_q.weight', 'c_k.weight', 'c_v.weight',
                                            'attn.proj.weight', 'mlp.fc.weight',
                                            'mlp.proj.weight', 'tok_emb')):
                    print(f"  FAIL: {n} should not be in 'scales'")
            assert any('q_gain' in n for n in sel_names), "missing q_gain"
            assert any('attn_scale' in n for n in sel_names), "missing attn_scale"
            assert any('mlp_scale' in n for n in sel_names), "missing mlp_scale"
            assert any('resid_mix' in n for n in sel_names), "missing resid_mix"
            assert any('skip_weights' in n for n in sel_names), "missing skip_weights"
            assert any('skip_gates' in n for n in sel_names), "missing skip_gates"
        if filt == 'last_n_layers:3':
            for n in sel_names:
                # If it starts with blocks., must be 8/9/10
                if n.startswith('blocks.'):
                    idx = int(n.split('.')[1])
                    assert idx in (8, 9, 10), f"{n} outside last 3 blocks"
        if filt == 'attn_only':
            for n in sel_names:
                assert '.attn.' in n, f"{n} not in .attn.*"
        if filt == 'mlp_only':
            for n in sel_names:
                assert '.mlp.' in n, f"{n} not in .mlp.*"
        if filt == 'all':
            assert len(sel) == len(named), "all should match every named param"

    print()
    print("All filter assertions passed.")

    # Summary table — useful for understanding adapt budget
    print("\nAdapt-surface size:")
    print(f"  all              {sum(p.numel() for p in _select_ttt_params(m,'all')):>10,} floats")
    print(f"  scales           {sum(p.numel() for p in _select_ttt_params(m,'scales')):>10,} floats")
    print(f"  scales+embed     {sum(p.numel() for p in _select_ttt_params(m,'scales+embed')):>10,} floats")
    print(f"  last_n_layers:3  {sum(p.numel() for p in _select_ttt_params(m,'last_n_layers:3')):>10,} floats")
    print(f"  attn_only        {sum(p.numel() for p in _select_ttt_params(m,'attn_only')):>10,} floats")
    print(f"  mlp_only         {sum(p.numel() for p in _select_ttt_params(m,'mlp_only')):>10,} floats")


if __name__ == '__main__':
    main()
