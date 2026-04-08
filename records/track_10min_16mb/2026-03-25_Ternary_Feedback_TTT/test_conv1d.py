import torch
import torch.nn.functional as F

B, T, S = 2, 8, 3
K = 3
x = torch.rand(B, T, S)
mixer_conv = torch.rand(K, S)

# Original loop
x_padded_1 = F.pad(x, (0, 0, K - 1, 0))
h1 = torch.zeros_like(x)
for k in range(K):
    h1 = h1 + mixer_conv[k] * x_padded_1[:, K - 1 - k:K - 1 - k + T, :]

# Conv1d
weight = mixer_conv.flip([0]).T.unsqueeze(1)
x_padded_2_t = F.pad(x.transpose(1, 2), (K - 1, 0))
h2_t = F.conv1d(x_padded_2_t, weight, groups=S)
h2 = h2_t.transpose(1, 2).contiguous()

print("Conv1D Max Diff:", (h1 - h2).abs().max().item())
