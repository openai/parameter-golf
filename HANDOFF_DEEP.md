# 深度交接文档：论文、代码仓、竞赛技术与动态步长

---

## 一、看过的论文

### H-net（核心论文）
- **标题**: "Hierarchical Autoregressive Language Models with Learnable Tokenization"（近似名称）
- **核心思想**: 动态分层 tokenization，byte → chunk → word，每层用 Mamba（SSM）处理
- **关键机制**:
  - **RoutingModule**: 计算相邻隐状态的余弦相似度，相似度低 → 边界（boundary）
  - **ChunkLayer**: 根据 boundary_mask 选出边界 token，打包成更短序列
  - **DeChunkLayer**: 用 EMA（指数移动平均）把词级输出扩散回字节级
  - **STE（Straight-Through Estimator）**: 让 hard argmax 的梯度能反向传播
  - **load_balancing_loss**: 惩罚边界分布不均，推动 boundary_mean → 1/N

### Mamba / Mamba2
- **Mamba**: Selective State Space Model，O(N) 复杂度，用输入决定 SSM 参数（B, C 依赖 x）
- **Mamba2**: 改进版，把 SSM 写成矩阵乘法形式（SSD），可用 Triton kernel 加速（`mamba_chunk_scan_combined`）
- **核心优势**: 相同参数量下长序列比 Attention 更高效，推理是 O(1)/step

### 其他参考
- **ALBERT**: 跨层权重共享思路（我们用于 encoder-decoder 共享）
- **ByT5**: Google 的 byte-level T5，证明字节级模型可以达到竞争力

---

## 二、看过的代码仓

### 1. 官方 H-net 代码仓
- **本地路径**: `/Users/maomao/Documents/parameter golf/notes/hnet/`
- **关键文件**:
  - `hnet/modules/dc.py` — RoutingModule, ChunkLayer, DeChunkLayer 实现
  - `hnet/models/hnet.py` — HNet 递归架构，encoder→routing→chunk→main→dechunk→residual→decoder
  - `hnet/modules/isotropic.py` — Isotropic 类，用 arch_layout 字符串（如 "m2T4"）构建序列模型
  - `hnet/utils/train.py` — load_balancing_loss 官方实现
  - `hnet/models/config_hnet.py` — HNetConfig, SSMConfig, AttnConfig
- **重要设计**:
  - `arch_layout` 例子: `["m2", ["T4"], "m2"]` → 2层Mamba编码器，4层Attention主网络，2层Mamba解码器
  - `pos_idx` 决定 Isotropic 读取 arch_layout 的哪个位置（0=encoder, 1=main, 2=decoder）
  - `stage_idx` 决定读取哪层的配置（0=最外层字节级，1=词级）

### 2. 竞赛参考仓（modded-nanogpt 及 PR）
- **位置**: 竞赛 GitHub，records 文件夹有各个参赛者的提交
- **最值得看的实现**:
  - **Depth Recurrence PR**: 循环若干层，11层物理→17层虚拟，不加参数加深度
  - **GPTQ + SDClip PR**: Hessian-aware 量化 + 统计裁剪离群值，当前 SOTA 量化方案
  - **3-Layer Recurrence (SOTA)**: SOTA 1.0810 bpb，SP8192 + 3层循环 + TTT
  - **MTP PR (#2046)**: 文档表明 MTP 是**失败实验**，600s 预算下 overhead > 收益

---

## 三、竞赛 Leaderboard 值得借鉴的技术

### 最重要（直接对我们有用）

| 技术 | 为什么有用 | 怎么加 |
|------|-----------|--------|
| **Depth Recurrence** | 不加参数就能加深，完美契合 16MB 限制 | 某几层 Mamba 循环 2-3 次 |
| **GPTQ + SDClip** | 我们现在是简单 int6，GPTQ 能减少量化误差，同样大小更好 | 需要校准数据集，训练后做 GPTQ |
| **byte-shuffle** | Brotli 前做字节重排，压缩率提升 5-10%，直接腾出更多空间 | 在序列化时加 `np.frombuffer(data).reshape(-1,4).T.flatten()` |
| **MuonEq-R** | 比 AdamW 对矩阵权重更高效，我们现在只用 AdamW | 替换优化器 |
| **Warmdown** | 训练末期 LR 降到 0，模型更稳定，bpb 小幅提升 | 已有 WARMDOWN_ITERS 参数，确认有效 |

### 中等优先级

| 技术 | 说明 |
|------|------|
| **QK-Gain** | chunk transformer 的 attention head 加可学习缩放，收益约 0.002 bpb |
| **更大 FFN (4x)** | chunk_ffn_dim 加到 4x chunk_model_dim，已经是这样了 |
| **EMA** | 训练末期对权重做指数移动平均，更稳定 |
| **Logit softcap** | tanh 软限幅 logits，代码里已有 |

### 对 byte 模型特别相关

| 技术 | 说明 |
|------|------|
| **更长序列** | byte 模型序列长，训练时用 2048 比 1024 上下文更好 |
| **SP8192 作对比** | SOTA 都用大 vocab BPE；byte 模型能到 1.97，仍有差距，但方向独特 |

---

## 四、动态步长怎么做（核心问题）

### 为什么固定步长能跑通但动态路由跑不好

**根本原因**：RoutingModule 用余弦相似度检测边界，需要相邻 token 的隐状态已经编码了"属于同一个词"的信息。
- 2 层 Mamba encoder 太浅，字节表示还没聚合成词级语义
- 所有字节对之间的余弦相似度都偏低，路由无法分辨真正的边界
- boundary_mean 停在 0.31，收敛不到目标 0.125

### 方向 1：更深的 Encoder（最直接）

```
arch: m4-T3-m2  (encoder 4层，chunk 3层，decoder 2层)
```

- 4 层 Mamba 后字节表示有更丰富的上下文，cosine sim 信号更清晰
- 总层数不变（9层），参数量类似
- 命令: `ENCODER_LAYERS=4 CHUNK_LAYERS=3 DECODER_LAYERS=2 NUM_LAYERS=9`

### 方向 2：可学习阈值（Soft Routing）

当前的 RoutingModule 直接 argmax boundary_prob，是 hard decision。可以改成：

```python
# 不用 argmax，用一个可学习的温度参数
temperature = nn.Parameter(torch.tensor(1.0))
boundary_prob = sigmoid(cos_sim_logit / temperature)
# 训练时用 soft mask（概率加权），推理时用 hard mask
```

好处：梯度更流畅，不需要 STE；温度参数能学到合适的阈值。

### 方向 3：预训练 Routing（两阶段训练）

1. **第一阶段**：先不用 routing，跑一个固定步长模型让 encoder 学好字节表示
2. **第二阶段**：加入动态 routing，此时 encoder 已经有能力提供有意义的边界信号

适合我们的设定：先用 FIXED_STRIDE=8 的 checkpoint，再 fine-tune 开启动态路由。

### 方向 4：改变 Routing 信号来源

余弦相似度不是唯一选择，可以用：
- **预测误差**：encoder 预测下一个字节的 loss 高的位置 → 边界（信息量跳变的地方）
- **熵**：某个位置的隐状态对应输出分布熵高 → 边界
- **直接学习分类器**：在 encoder 输出上加一个小的 binary classifier，有监督地用规则标注边界（如空格位置）

### 方向 5：分层 Soft Boundary（最接近论文原意）

论文的 DeChunkLayer 用 EMA 做 upsample，EMA 的衰减系数 p 来自 boundary_prob。核心公式：
```
dt = log(1 / (1 - p))
output = mamba_chunk_scan_combined(x / dt, dt, A=-1, B=p, C=1)
```

这个 EMA 可以 differentiable 地传播梯度。关键改进：
- 不做 hard argmax，直接用连续的 boundary_prob 做 soft chunk
- ChunkLayer 也改成 weighted pooling 而不是 binary selection
- 整个 pipeline 全可导，不需要 STE

这是最难实现的方向，但理论上最优雅。

---

## 五、当前代码状态总结

| 文件 | 说明 |
|------|------|
| `train_hnet_repo_pg.py` (本地) | Codex 原版 + int6/brotli（基线）|
| `train_hnet_repo_pg_experimental.py` (本地) | 加了 FixedStrideHNetLM + FIXED_STRIDE 参数 |
| `train_hnet_v2.py` (本地+服务器) | 加了 SHARE_ENC_DEC 权重共享 |
| 服务器 `train_hnet_repo_pg.py` | 实际是 experimental 版本（带 FixedStrideHNetLM）|

**当前最佳**: 固定步长 stride=8，val_bpb=1.9733，15.3MB，927步/600秒

**下一个实验**: SHARE_ENC_DEC=1 + MODEL_DIM=608，看 bpb 能否低于 1.97

**动态路由下一步建议**: 先试 ENCODER_LAYERS=4，如果 boundary_mean 能收敛到 0.15 以下就说明方向对
