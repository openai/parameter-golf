"""
Experiment 01: Weight Sharing / Depth Recurrence

HYPOTHESIS: Sharing transformer block weights across layers gives us more
effective depth for fewer parameters. The freed parameter budget can go
toward wider layers or more unique blocks.

KEY INSIGHT: Shared weights produce identical byte patterns in the state dict.
zlib compresses repeated patterns nearly to zero. So weight sharing gives us
BOTH parameter efficiency AND compression efficiency. Double win.

APPROACH:
1. Define N unique blocks (e.g., 3)
2. Repeat them K times (e.g., 3×3 = 9 effective layers)
3. Add lightweight per-layer adapters (scalar gains only, ~dim params each)
4. With 3 unique blocks instead of 9, we save ~6× the block parameters
5. Use that budget for wider model (768 or 1024 dim instead of 512)

MODIFICATIONS TO train_gpt.py:
- GPT.__init__: Create N unique blocks, reference them K times
- GPT.forward: Index into unique blocks with modular arithmetic
- Add per-layer scalar adapters (attn_scale, mlp_scale per repetition)
- Increase MODEL_DIM to use freed parameter budget

VARIANTS:
- 01a: 3 unique × 3 repeats, 512 dim (parameter savings → smaller artifact)
- 01b: 3 unique × 3 repeats, 768 dim (parameter savings → wider model)
- 01c: 1 unique × 9 repeats, 768 dim (maximum sharing)
- 01d: 3 unique × 4 repeats = 12 effective layers, 512 dim (more depth)

EXPECTED IMPACT: 0.01-0.03 BPB improvement
RISK: Shared weights may underperform unique weights. The adapter overhead
      may not be enough to differentiate layers.
"""

# Code changes to apply to train_gpt.py for variant 01b:
#
# In class Hyperparameters:
#   num_unique_blocks = 3
#   num_repeats = 3  # effective layers = 9
#   model_dim = 768  # wider with freed params
#
# In class GPT.__init__:
#   self.unique_blocks = nn.ModuleList([Block(...) for _ in range(num_unique_blocks)])
#   self.layer_adapters = nn.ParameterList([
#       nn.Parameter(torch.ones(2, model_dim)) for _ in range(num_unique_blocks * num_repeats)
#   ])
#
# In class GPT.forward:
#   for i in range(total_layers):
#       block = self.unique_blocks[i % num_unique_blocks]
#       adapter = self.layer_adapters[i]
#       x = block(x, x0)
#       x = x * adapter[0] + adapter[1]  # per-layer scale + shift
