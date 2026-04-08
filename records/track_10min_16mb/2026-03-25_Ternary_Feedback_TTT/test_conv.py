import torch
import torch.nn.functional as F

B, T, S = 2, 10, 3
W = 4

# Mock input
x_gated = torch.ones(B, T, S)
# S dims
D = torch.tensor([0.5, -0.5, 0.9])

# Method 1: Loops
x_padded_1 = F.pad(x_gated, (0, 0, W - 1, 0))
h1 = torch.zeros(B, T, S)
D_pow = torch.ones(S)
for w in range(W):
    shifted = x_padded_1[:, W - 1 - w:W - 1 - w + T, :]
    h1 = h1 + D_pow * shifted
    D_pow = D_pow * D

# Method 2: Conv1d
x_gated_t = x_gated.transpose(1, 2)
weight = torch.zeros((S, 1, W))
D_w = torch.ones(S)
for w in range(W):
    weight[:, 0, W - 1 - w] = D_w
    D_w = D_w * D

x_padded_2 = F.pad(x_gated_t, (W - 1, 0))
h2_t = F.conv1d(x_padded_2, weight, groups=S)
h2 = h2_t.transpose(1, 2).contiguous()

print("Max diff:", (h1 - h2).abs().max().item())
