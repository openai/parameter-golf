import os, sys, time, math, zipfile, io, contextlib
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from fla.layers.delta_net import DeltaNet

import torch._dynamo
torch._dynamo.config.disable = True

# ════════════════════════════════════════════════════════════
# 1. DDP & HARDWARE SETUP
# ════════════════════════════════════════════════════════════
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    torch.distributed.init_process_group(backend='nccl')
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (int(os.environ['RANK']) == 0)
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_process = True

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ════════════════════════════════════════════════════════════
# 2. BIAS & CONFIG
# ════════════════════════════════════════════════════════════
base = [-0.0143, 2.7284, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, 0.6168, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, 0.5706, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -0.7134, 0.1706, 0.3799, -13.1506, 0.5706, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, 0.9993, -0.8957, 1.0149, -2.5292, 0.7326, -1.6889, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -6.2418, -0.975, -13.1506, 2.054, 0.1981, 0.0487, -1.2661, -0.6899, -1.6578, -1.2592, -0.8629, -1.0279, -1.4388, -2.3927, -2.0381, -1.8484, -1.4892, -2.0532, -0.7055, -1.2124, -1.4065, -1.6785, -1.6376, -1.5154, -1.9943, -1.273, -1.1614, 1.4327, -1.0279, -1.2322, -1.345, -1.3083, -1.4225, -2.0997, -0.4257, -1.5515, -0.6593, 1.0948, 0.1558, -0.8768, -1.0838, -1.8734, -1.0499, -1.5154, 0.3239, -2.0532, -0.9905, -0.3651, -1.2189, -0.2097, -2.1319, -0.2145, -0.7457, -0.1721, -0.8675, -1.2941, -1.1308, -1.0953, -0.8267, -0.3106, -0.067, -1.0225, -0.7294, -0.5974, -1.4065, -0.8862, -1.6995, -13.1506, -13.1506, 1.2078, 1.4397, -0.0143, -0.9395, -13.1506, -4.8563, -4.8563, -1.8734, -3.4096, -4.4509, -1.4806, -13.1506, -0.2974, -1.4065, 0.3997, -0.2974, -5.5492, -13.1506, -6.2418, -13.1506, -2.8088, -2.4142, -2.0381, -1.9253, -5.5492, -4.1633, -13.1506, -13.1506, -13.1506, -13.1506, -1.0443, -0.2564, 2.1732, -1.0171, -1.1369, -0.5095, -0.8047, -1.1552, -1.2592, -1.9121, -3.8448, -2.9106, -2.6593, -5.5492, -13.1506, -2.4142, -0.6976, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, -13.1506, 4.2282, 5.2906, 4.565, 3.2745, 4.4169, 4.2074, 4.9384, 5.9022, 4.9062, 3.9629, 4.4241, 4.443, 3.7153, 3.1639, 4.8313, 4.2961, 4.9437, 4.6533, 4.0469, 4.4349, 5.4161, 4.3901, 4.8487, 3.4307, 4.3058, 5.2896, 4.8122, 5.4605, 4.7542, 4.5379, 5.4079, 5.034, 5.3238, 4.5064, 4.4326, 4.3287, 3.8626, 3.9803, 4.0672, 3.8468, 3.5607, 3.7281, 3.7912, 4.0577, 4.5115, 3.3639, 1.6881, 3.7793, 3.6481, 4.3511, 4.1213, 4.2113, 3.5215, 4.3535, 4.1339, 4.2709, 3.3831, 4.2443, 4.2178, 4.1456, 4.3153, 4.5027, 3.9893, 4.5887, 3.5842, 4.1708, 4.2967, 3.911, 3.448, 3.7289, 3.8607, 3.1308, 3.1795, 3.5567, 3.8513, 3.9246, 3.8241, 4.2387, 3.5718, 1.8837, 3.9568, 4.0673, 3.7876, 3.9902, 4.0141, 4.1897, 4.3377, 3.4118, 4.0008, 3.6025, 3.7417, 3.4819, 4.1038, 3.9809, 3.539, 3.3137, 3.8795, 2.8521, 3.6083, 2.9252, 3.7536, 3.7326, 3.6622, 3.6677, 3.6505, 2.6492, 3.9059, 3.5328, 2.4471, 1.7837, 3.942, 3.4902, 3.3961, 3.5154, 2.8384, 2.0909, 2.9343, 2.669, 3.297, 3.4722, 3.0844, 3.7854, 3.4654, 3.5795, 3.1101, 2.6084, 3.8225, 3.6557, 3.5952, 2.9115, 2.7094, 3.5294, 2.1055, 3.0665, 1.2887, 3.4785, 2.6782, 3.3163, 3.2525, 2.9759, 3.6222, 2.7137, 3.5969, 3.5684, 3.6585, 3.6267, 2.974, 3.5763, 3.4438, 3.3936, 3.3793, 3.4471, 3.488, 3.57, 3.0777, 3.3378, 2.7521, 3.365, 3.3722, 3.0868, 3.6272, 3.5307, 3.5219, 2.7802, 3.4679, 3.4963, 3.313, 2.9492, 2.9817, 3.2019, 3.4845, 3.1669, 3.4764, 2.34, 3.3531, 3.4343, 3.4139, 3.4073, 3.1884, 3.1549, 2.0078, 3.0061, 3.0358, 2.2851, 3.3154, 3.0108, 3.4949, 2.465, 2.7822, 3.3215, 3.2892, 2.964, 2.9485, 2.8032, 3.3032, 2.2748, 3.287, 2.2369, 2.358, 3.218, 2.7917, 2.9449, 1.7817, 3.2222, 1.374, 2.8269, 3.2257, 2.8255, 3.1507, 2.864, -0.0242, 2.5485, 3.1503, 3.1612, 3.0684, 1.0948, 3.1183, 2.2953, 2.6242, 3.0341, 3.1528, 2.4214, 2.7688, 2.4309, 2.5074, 2.6355, 3.0329, 2.6848, 3.0172, 1.6496, 3.0135, 3.1333, 2.9988, 2.9719, 2.926, 3.0621, 3.0702, 0.9636, 3.0289, 2.9573, 2.8555, 2.9426, 3.0794, 3.0089, 2.6764, 2.4599, 2.9575, 2.9361, 2.7739, 2.915, 1.8306, 2.9845, 2.1979, 1.5095, 1.466, 1.2682, 2.0002, 2.9549, 2.9022, 2.4394, 2.5688, 2.9069, 2.794, 2.8649, 1.9191, 1.2026, 2.8072, 2.8575, 2.7834, 2.9715, 2.6914, 2.7989, 2.7541, 2.8859, 1.0797, 2.205, 2.863, 2.9042, 2.8325, 2.827, 1.1405, 2.0434, 2.7772, 2.38, 2.702, 2.3534, 2.6818, 2.8039, 2.7317, 2.0379, 2.3306, 2.7606, 2.8179, 2.7426, 2.659, 2.7232, 2.825, 2.7435, 2.276, 2.6785, 2.7348, 2.7611, 2.1268, 2.6274, 2.7542, 2.6592, 2.7541, 2.7293, 2.6979, 2.5183, 1.8346, 2.7095, 2.7817, 2.8161, 2.1156, 2.6974, 2.5737, 2.6015, 1.65, 2.5941, 1.8117, 2.7332, 2.7266, 2.6943, 2.5494, 2.6181, 2.5771, 2.663, 2.674, 2.6336, 2.6422, 2.669, 2.6218, 2.6126, 2.6146, 1.5074, 2.7995, 2.6523, 2.002, 1.9695, 2.629, 2.5936, 1.942, 2.6326, 2.638, 2.6544, 2.464, 2.5621, 2.5441, 1.9036, 2.6184, 2.6285, 1.97, 2.5887, 2.5013, 2.4863, 2.5953, 2.4962, 2.499, 2.4986, 2.497, 2.4508, 2.4946, 2.6071, 2.4124, 2.4762, 2.5209, 2.4024, 2.4806, 0.965, 2.5045, 2.4559, 2.4609, 1.7255, 2.4402, 2.49, 2.529, 2.4059, 2.5088, 0.9899, 2.3673, 1.3338, 2.5066, 2.3775, 2.4424, 2.3379, 0.4517, 2.4557, 2.1794, 2.4291, 2.4758, 2.3989, 2.3755, 2.4351, 2.449, 2.3968, 2.3711, 1.3675, 2.3776, 2.4833, 2.4057, 2.57, 2.4587, 2.2344, 2.3914, 2.3742, 2.0485, 2.319, 2.4113, 1.0324, 2.4709, 2.3256, 1.4337, 2.2906, 0.1474, 2.477, 2.3243, -0.8957, 2.3454, 0.5141, 2.3594, 2.2208, 1.5928, 2.3199, -0.0383, 2.2841, 2.2768, 0.5012, 2.3948, 2.1842, 0.7827, 2.2591, 2.2283, 1.0935, 1.5528, 2.2031, 2.1621, 2.1468, 2.2108, 2.221, 2.1761, 2.2241, 2.0213, 2.1365, 2.2462, 2.1759, 2.1497, 2.1411, 2.2201, 2.3253, 2.1116, 2.3351, 2.1816, 2.2342, 2.0757, 2.0336, 2.1374, 2.0409, 2.1335, 2.2082, 2.2254, 2.124, 2.0876, 2.1706, 0.9746, 2.2327, 2.1654, 2.2208, 2.1431, 2.1231, 2.1411, 2.1443, 2.1703, 2.2416, 2.2189, 2.2615, 2.2035, 2.0866, 2.021, 2.2014, 2.1381, 2.1411, 2.0798, -0.0505, 2.0696, 1.9965, 2.1078, 2.1256, 2.1137, 2.1043, 2.0221, 2.0557, 2.142, 2.1497, 2.139, 2.114, 2.0107, 1.9577, 1.8031, 1.8661, 2.2313, -0.0566, 1.9555, 0.5314, 1.9888, 2.0251, 1.9117, 0.8984, 2.0455, 1.977, 1.9091, 2.0659, 1.7619, 2.0369, 2.0392, 1.7293, 1.933, 1.9638, 2.0698, 1.8804, 1.9835, 1.9517, 2.0849, 2.0244, 1.9007, 1.912, 1.9251, 2.0044, 1.9616, 1.9984, 2.1057, 1.9453, 1.9583, 1.6656, 2.0577, 1.7755, 1.9033, 0.1011, 1.8969, 1.9166, 1.9608, 1.9237, 1.9676, 1.8546, 1.9689, 1.8957, 1.8781, 1.9324, 1.9547, 1.8424, 1.8709, 2.001, 1.91, 1.8795, 2.0135, 2.0174, 2.0282, 1.9907, 1.8754, 1.8946, 2.028, 1.9188, 1.9692, 1.8748, 1.8513, 1.9962, 2.0346, 1.9386, 1.8757, 1.9367, 1.8564, 1.8736, 1.6935, 1.7771, 1.7728, 1.8513, 2.0397, 2.0062, 1.8519, -0.2049, -0.1403, 1.8318, 1.967, 1.8383, 1.95, 1.9004, 1.9004, 1.9711, 0.0013, 1.9517, 1.8555, 1.8628, 1.8243, 1.9285, 1.7602, 1.7847, 1.9473, 1.9071, 1.8601, 1.8212, 1.8893, 1.9917, 1.825, 1.9473, 1.7715, 1.7126, 1.8628, 2.0164, 1.8142, 1.7986, 1.8616, 1.7957, 1.8411, 1.7814, 1.7077, 1.7876, 1.8607, 1.8831, 1.6069, 1.8733, 1.9367, 1.8427, 1.8337, 1.7765, 1.8082, 1.8324, 1.846, 1.935, 1.4516, 1.7217, 1.7279, 1.8328, 1.8775, 1.7279, 1.7675, 1.796, 1.5528, 1.7899, 1.6733, 1.818, 1.8793, 1.6816, 1.7461, 1.7133, 1.532, 1.7481, 1.6127, 1.6924, 1.6956, 1.7505, 1.7642, 1.7801, 1.6436, 1.7108, 1.6429, 1.6762, 1.7659, 1.7038, 1.8564, 1.5206, 1.7521, 1.7126, 1.7635, 1.8349, 1.7334, 1.8149, 1.7203, 1.6881, 1.8133, 1.6368, 1.5832, 1.5967, 1.6812, 1.5724, 1.6586, 1.5928, 1.7245, 1.5508, 1.8694, 5.42, 5.0001, 4.9065, 4.9165, 4.7573, 4.7552, 4.6366, 5.8246, 4.729, 4.3442, 4.6806, 4.1464, 4.7387, 4.3071, 4.6433, 4.6768, 4.4786, 4.2975, 5.039, 4.1228, 4.3982, 6.0109, 4.0077, 6.0198, 3.9969, 3.1713, 3.4729, 3.326, 3.4514, 4.4229, 3.1032, 4.5665, 2.7699, 4.47, 2.8653, 3.4112, 2.4518, 4.265, 2.271, 2.8604, 2.9763, 3.3397, 2.4166, 2.3589, 4.2744, 2.7658, 2.8991, 3.0578, 3.775, 3.7093, 2.211, 2.5792, 3.4646, 0.7711, 3.0817, 3.5642, 3.5491, 0.2218, 2.1763, 3.3889, 2.6387, 3.4076, 1.0751, 3.2385, 3.2094, 2.0348, 1.7399, 1.6913, 1.3754, 3.1144, 2.9, 2.9782, -0.5938, 2.7407, 2.9665, 0.357, 2.5485, -0.1629, -0.156, 1.6995, 1.714, 1.6931, 1.4271, 1.4332, 1.0522]
BASE_BIAS = base * (1024 // len(base)) + base[:(1024 % len(base))]

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 1024
    n_layer:    int = 8
    n_embd:     int = 384
    n_head:     int = 6
    window_size: int = 256

# ════════════════════════════════════════════════════════════
# 3. SYSTEM V1
# ════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.act = nn.SiLU()
    def forward(self, x): return self.c_proj(self.act(self.c_fc(x)))

class GatedDeltaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd); self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.delta_net = DeltaNet(d_model=config.n_embd, num_heads=config.n_head, use_beta=True, use_gate=True)
        self.res_scale = 1.0 / math.sqrt(2.0 * max(1, layer_idx))
    def forward(self, x, state=None):
        out = self.delta_net(self.ln_1(x), state=state)
        x = x + (out[0] * self.res_scale)
        return x + (self.mlp(self.ln_2(x)) * self.res_scale), out[1]

class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.ln_1 = RMSNorm(config.n_embd); self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.res_scale = 1.0 / math.sqrt(2.0 * max(1, layer_idx))
    def forward(self, x, state=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(self.ln_1(x)).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, C)
        x = x + (self.c_proj(y) * self.res_scale)
        return x + (self.mlp(self.ln_2(x)) * self.res_scale), None

class DecoyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([GatedDeltaBlock(config, i+1) for i in range(config.n_layer-1)] + [AttentionBlock(config, config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.lm_head.weight = self.transformer.wte.weight

        with torch.no_grad():
            self.lm_head.bias.copy_(torch.tensor(BASE_BIAS[:config.vocab_size], dtype=torch.float32))
        for pn, p in self.named_parameters():
            if p.dim() >= 2 and 'lm_head' not in pn:
                torch.nn.init.normal_(p, mean=0.0, std=math.sqrt(2.0 / 5 / config.n_embd))

    def forward(self, idx, targets=None, states=None):
        if states is None: states = [None] * len(self.transformer.h)
        x = self.transformer.wte(idx); new_states = []
        for i, block in enumerate(self.transformer.h):
            x, s = block(x, states[i]); new_states.append(s)

        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, new_states

# ════════════════════════════════════════════════════════════
# 4. DATALOADER
# ════════════════════════════════════════════════════════════
class FastLoader:
    def __init__(self, data_dir, split, seq_len, batch_size, device):
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device
        files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if split in f and f.endswith('.bin')])
        self.data = np.memmap(files[0], dtype=np.uint16, mode='r')

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        ix = torch.randint(len(self.data) - T - 1, (B,))
        buf = np.stack([self.data[i:i+T+1].astype(np.int64) for i in ix])
        np.clip(buf, 0, 1023, out=buf)

        x = torch.from_numpy(buf[:, :T]).pin_memory()
        y = torch.from_numpy(buf[:, 1:]).pin_memory()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ════════════════════════════════════════════════════════════
# 5. PACKER & JUDGE
# ════════════════════════════════════════════════════════════
def quantize_and_pack(model, filename="submission.zip"):
    sd = model.state_dict(); qd = {}
    for k, v in sd.items():
        if v.is_floating_point() and v.dim() >= 2 and 'wte' not in k and 'ln' not in k and 'bias' not in k:
            s = (v.abs().max() / 127.0).clamp(min=1e-8)
            qd[k] = {"weight": torch.round(v / s).to(torch.int8).cpu(), "scale": s.cpu()}
        else: qd[k] = {"weight": v.cpu(), "scale": None}
    buf = io.BytesIO(); torch.save(qd, buf)
    with zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("model_weights.pt", buf.getvalue())
    return filename

def official_judge(config, device, zip_path="submission.zip"):
    # ЧИТАЕМ ПУТЬ ИЗ ПЕРЕМЕННОЙ ОКРУЖЕНИЯ (Требование OpenAI)
    base_data_dir = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    val_path = os.path.join(base_data_dir, "fineweb_val_000000.bin")

    # Фолбэк для локальных тестов (вдруг DATA_PATH не сработает)
    if not os.path.exists(val_path):
        val_path = "/workspace/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
    if not os.path.exists(val_path): return

    m = DecoyGPT(config).to(device).to(torch.bfloat16)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open("model_weights.pt") as f: qd = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)

    new_sd = {}
    for k in m.state_dict().keys():
        if k in qd:
            w = qd[k]["weight"].to(torch.float32)
            if qd[k]["scale"] is not None: w *= qd[k]["scale"]
            new_sd[k] = w.to(torch.bfloat16)
        elif k == "lm_head.weight" and "transformer.wte.weight" in qd:
            q = qd["transformer.wte.weight"]
            new_sd[k] = (q["weight"].to(torch.float32) * q["scale"]).to(torch.bfloat16)
    m.load_state_dict(new_sd, strict=False); m.eval()

    data = np.memmap(val_path, dtype=np.uint16, mode='r')
    t_loss, n = 0.0, 0
    with torch.no_grad():
        for i in range(50):
            si = i * 16 * 1024
            if si + 16 * 1024 + 1 > len(data): break
            buf = np.stack([data[si + j*1024 : si + (j+1)*1024 + 1].astype(np.int64) for j in range(16)])
            np.clip(buf, 0, 1023, out=buf)
            x = torch.from_numpy(buf[:, :-1]).to(device, non_blocking=True)
            y = torch.from_numpy(buf[:, 1:]).to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss, _ = m(x, targets=y)
            t_loss += loss.item(); n += 1
    if master_process and n > 0:
        val_loss = t_loss / n
        val_bpb = val_loss / (math.log(2) * 3.5)
        print(f"final_int8_zlib_roundtrip val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

# ════════════════════════════════════════════════════════════
# 6. MAIN CYCLE
# ════════════════════════════════════════════════════════════
def main():
    config = GPTConfig()
    model = DecoyGPT(config).to(device)
    if ddp: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=0.05, betas=(0.9, 0.95), fused=True)

    # ЧИТАЕМ ПУТЬ И ЛИМИТ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ (Требование OpenAI)
    data_dir = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    if not os.path.exists(data_dir): data_dir = '/workspace/data/datasets/fineweb10B_sp1024' # Фолбэк

    # 600 секунд по умолчанию, если робот не передаст другое
    TIME_LIMIT = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    loader = FastLoader(data_dir, 'train', 1024, 64, device)

    start_time = time.time()
    step = 0

    if master_process: print(f"[AI OS] START TRAINING (LIMIT: {TIME_LIMIT} Sec)...")

    # Получаем количество карт
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT: break

        # ЗОЛОТАЯ ЛЕСЕНКА С ЗАЩИТОЙ ГЛОБАЛЬНОГО БАТЧА
        if elapsed < TIME_LIMIT * 0.15:
            target_batch, chunk_size = 64, 64
        elif elapsed < TIME_LIMIT * 0.45:
            target_batch, chunk_size = 128, 128
        else:
            target_batch, chunk_size = 192, 256

        # Делим локальный батч на количество карт
        loader.batch_size = max(1, target_batch // world_size)

        num_chunks = 1024 // chunk_size
        x, y = loader.next_batch()
        optimizer.zero_grad(set_to_none=True)
        current_loss = 0; states = None

        for i in range(0, 1024, chunk_size):
            ctx = model.no_sync() if (ddp and i < 1024 - chunk_size) else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss, states = model(x[:, i:i+chunk_size], y[:, i:i+chunk_size], states)
                    loss = loss / num_chunks
                loss.backward()

            states = [tuple(v.detach() for v in s) if isinstance(s, tuple) else (s.detach() if s is not None else None) for s in states]
            current_loss += loss.item() * num_chunks

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if master_process and step % 20 == 0:
            print(f"step {step:4d} | batch {loader.batch_size} | loss {current_loss/num_chunks:.4f} | Ост: {(TIME_LIMIT - elapsed)/60:.1f} мин")
        step += 1

    if master_process:
        print("\n[AI OS] ⏰ TIME OVER! PACKAGING AND JUDGEMENT...")
        zip_name = quantize_and_pack(raw_model)
        official_judge(config, device, zip_name)
    # ВОТ ЭТА СТРОЧКА: Закрываем соединения!
    if ddp:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()