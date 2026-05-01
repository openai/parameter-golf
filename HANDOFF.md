# Parameter Golf 交接文档

## 比赛背景
- **比赛**: OpenAI Parameter Golf — 16MB 模型文件限制，在 8×H100 上训练 600 秒，用 FineWeb 数据集，按 bpb（bits per byte）评分
- **评估**: bpb = cross_entropy_loss / log(2)，越低越好
- **tokenizer**: byte260，纯字节 tokenizer，vocab=260，每个 token = 1 个字节

## 当前最佳结果
| 版本 | val_bpb | 压缩后大小 | 备注 |
|------|---------|-----------|------|
| 固定步长 H-net (stride=8) | 1.9733 | 15.3MB | 旧基线 |
| 固定步长 H-net + Muon + compile (stride=8, m2-T4-m2) | 1.9370 | 9.07MB | |
| 固定步长 H-net + Muon + compile (stride=8, m2-T6-m2) | 1.9058 | 13.04MB | 10分钟单卡 |
| 固定步长 H-net + Muon + compile (stride=8, m2-T4-m2, 80min) | **1.5629** | **15.92MB** | **当前最好**；80min×1H100≈竞赛8H100×600s |
| 动态路由 H-net (routing_loss=1.0) | 3.485 | 13.3MB | routing 不收敛 |
| 动态路由 H-net (routing_loss=3.0) | 6.29 | 14.8MB | 调坏了 |

## 实验记录（train_hnet_v4.py，固定步长）

| 架构 | stride | val_bpb（量化后） | 压缩后大小 | 结论 |
|------|--------|-----------------|-----------|------|
| m2-T4-m2 | 8 | 1.9733 | 15.3MB | **基线最好** |
| m2T1-T3-m2 | 8 | 3.0113 | 14.2MB | encoder 加 T1 有害，bpb 反升 |
| m2-T6-m2 | 8 | 2.0213 | 20.7MB | 超出 16MB 限制，且 bpb 更差 |
| m2-T6-m2 | 4 | 2.0004 | 20.5MB | 超出 16MB 限制；stride=4 比 stride=8 略好但仍超限 |
| m2-T4-m2 | 6 | 2.0085 | 14.58MB | 在限制内，但 bpb 比 stride=8 差 |
| m2-T4-m2 + Muon + compile | 8 | **1.9370** | **9.07MB** | 当前最好；Muon 让权重更可压缩 |
### 关键规律
- 参数量控制：m2-T4-m2 约 24.5M 参数 → 15.3MB；m2-T6-m2 约 39.5M → 20.7MB，超限
- 要加深必须同时缩 MODEL_DIM 或 CHUNK_MODEL_DIM 保持在预算内
- encoder 加 byte 级 Attention（m2T1）没有帮助，600s 内反而变差
- **Muon + compile 有效**：同配置下 bpb 1.9733→1.9370，且压缩后大小从 15.3MB 降到 9.07MB
- **训练时间 vs 压缩率**：同一模型（2-6-2 + Muon）10分钟→13MB（3.0x），30分钟→18.7MB（2.1x）——训练越久权重越特化，brotli 压缩率下降，反而超出 16MB 限制
- **结论**：对 Muon 模型，600s（10分钟）是压缩预算的甜点，继续训练会超限
- **固定步长结构结论（2026-05-01）**：`stride=8, m2-T4-m2` 仍是当前结构基线；`stride=4` 最终约 `2.035`，`stride=6` 最终约 `2.000`，都不如 `stride=8` 的最佳结果
- **chunk 深度结论（2026-05-01）**：把 chunk core 从 `T4` 加深到 `T6` 没有带来收益，`m2-T6-m2` 在同预算下不如 `m2-T4-m2`
- **压缩结论（2026-05-01）**：当前主瓶颈已经从结构转到压缩；有些结构训练态更低，但 roundtrip 掉分明显。后续优先优化量化/压缩损伤，再考虑继续加深
- **Muon / tie 结论（2026-05-01）**：`USE_MUON=1` 需要只作用于严格 2D 且非空矩阵参数；`Muon + tie_embeddings` 在当前配置下出现过严重 roundtrip 退化，不宜直接作为主线
- 下一步方向：在 600s 内加大模型（预算还剩 3MB）、探索更好的压缩方案

## 实验记录（动态路由）

| 架构 | 脚本 | OUTER_LR_MULT | INNER_LR_MULT | val_bpb | 压缩后大小 | 结论 |
|------|------|--------------|--------------|---------|-----------|------|
| m4-T4-m2 | v6 | 3.0 | 1.7 | 3.3820 | 17.2MB | 超限；boundary_mean=0.17 未收敛（10min） |
| m4-T4-m2 + Muon | v5 | — | — | 2.8475 | 18.69MB | 超限；2小时 routing 收敛但 bpb 仍差 |
| 2stage: m1-[T1m1-T2-m1T1]-m1 (OUTER=2.0,MID=1.3,INNER=0.9) | v7 | 2.0 | 0.9 | 3.8230 | 4.68MB | boundary_mean=0.35 未收敛；模型很小（10min） |
| 2stage: m2-[T1m2-T4-m2T1]-m2 dim=512/512/640 (OUTER=2.0,MID=1.3,INNER=0.9) | v7 | 2.0 | 0.9 | 3.2606 | 26.57MB | 超限；2小时 bpb 仍差 |

## 模型架构：固定步长 H-net（最佳）

```
输入字节 → Embedding
→ Encoder: 2层 Mamba2（byte级别）
→ 残差分支保存
→ 固定步长选取：每8个字节取最后1个（causal）
→ Chunk Transformer: 4层 Attention（word级别，stride=8相当于128 tokens）
→ Causal upsample: 右移1位再 repeat_interleave，word k的输出给第k+1组的字节用
→ 合并残差
→ Decoder: 2层 Mamba2（byte级别）
→ LM head → loss
```

**为什么用固定步长而不是动态路由**：
- H-net 原论文的动态路由用余弦相似度检测边界
- byte 输入下，encoder 只有2层太浅，字节表示还没聚合成词级别语义，cosine sim 信号弱
- boundary_mean 一直停在 0.31，无法收敛到目标 0.125
- 固定步长完全绕开这个问题，训练稳定，bpb 大幅提升

## 服务器信息
- **旧 Pod**: `ssh -p 17937 -i ~/.ssh/id_ed25519 root@103.207.149.135`
- **新 Pod**: `ssh root@216.243.220.230 -p 10560 -i ~/.ssh/id_ed25519`
- **工作目录**: `/workspace/parameter-golf`
- **数据**: `/workspace/parameter-golf/data/datasets/fineweb_byte260/`（两个 pod 都有）
- **tokenizer**: `./data/tokenizers/fineweb_pure_byte_260.json`（文件不需要真实存在，代码只检查 .json 后缀）
- **byte260 下载源**: Hugging Face `LightSpeedUp/parameter-golf-data`，目录名是 `fineweb_byte260/`（不是 `fineweb10B_byte260`）
- **byte260 下载命令**: `hf download LightSpeedUp/parameter-golf-data --repo-type dataset --include "fineweb_byte260/*" --local-dir ./data`
- **只下前 10 个 train shard**: 优先用 `data/cached_challenge_fineweb.py --variant byte260 --train-shards 10`，但要先把脚本里的 `fineweb10B_byte260` 改成 `fineweb_byte260`；或者用 `hf download` 分别指定 `fineweb_val_*` 和前 10 个 `fineweb_train_*.bin`

## 新 Pod 环境配置

新 pod 的 PyTorch 是 `cu128`，系统 CUDA 12.8，用 `--no-build-isolation` 绕过版本检查：

```bash
# 需要 --no-build-isolation 的包（防止 pip 拉入 cu130 torch 导致 CUDA 版本冲突）
pip install mamba-ssm causal-conv1d --no-build-isolation --break-system-packages

# 普通包
pip install brotli sentencepiece optree numpy --break-system-packages

# 传 hnet 库和训练脚本（从本地执行）
scp -P <PORT> -i ~/.ssh/id_ed25519 -r ./hnet root@<IP>:/workspace/parameter-golf/hnet
scp -P <PORT> -i ~/.ssh/id_ed25519 train_hnet_v5.py root@<IP>:/workspace/parameter-golf/

# 验证
python -c "from mamba_ssm import Mamba2; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; import brotli, sentencepiece, optree; print('all ok')"
```

## 关键文件
| 文件 | 位置 | 说明 |
|------|------|------|
| `train_hnet_repo_pg.py` | 本地+服务器 | Codex 原版 + int6/brotli，动态路由，跑出 1.97 bpb 的那个 |
| `train_hnet_repo_pg_experimental.py` | 本地+服务器(覆盖为train_hnet_repo_pg.py) | 加了 FixedStrideHNetLM 的版本 |
| `train_hnet_v2.py` | 本地+服务器 | 加了 encoder-decoder 权重共享（SHARE_ENC_DEC=1）|

**注意**: 服务器上 `train_hnet_repo_pg.py` 实际是 experimental 版本（带 FixedStrideHNetLM）

## 压缩方案
- int6 量化（Mamba SSM 动态参数 A_log/dt_proj/conv1d/D 保持 int8）
- Brotli-11 压缩
- 典型压缩比：约 2x（相比 float16 原始大小）

关键代码在 `train_hnet_repo_pg.py`：
- `INT6_ENABLED`, `MAMBA_SSM_INT8_PATTERNS`
- `quantize_float_tensor()` 函数
- 文件末尾的 brotli 压缩和验证逻辑

## 当前正在跑的实验
**train_hnet_v2.py**，encoder-decoder 共享权重 + 加宽 model_dim：
```bash
FIXED_STRIDE=8 SHARE_ENC_DEC=1 MODEL_DIM=608 VOCAB_SIZE=260 \
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
DATA_PATH=./data/datasets/fineweb_byte260 \
MAMBA_STATE=32 CHUNK_MODEL_DIM=576 CHUNK_ROTARY_DIM=48 \
NUM_LAYERS=8 ENCODER_LAYERS=2 CHUNK_LAYERS=4 DECODER_LAYERS=2 \
VAL_LOSS_EVERY=200 python train_hnet_v2.py 2>&1 | tee /workspace/hnet_v2_shared.log
```

## 下一步方向（优先级排序）
1. **encoder-decoder 共享 + 加宽 model_dim**：省出 decoder 参数，加宽到 model_dim=608，看 bpb 能否提升
2. **更深编码器的动态路由**：`ENCODER_LAYERS=4 CHUNK_LAYERS=3 DECODER_LAYERS=2`，更深的编码器让 routing 有更好的信号
3. **byte-shuffle + Brotli**：字节重排后再压缩，提升压缩比，腾出空间放更大模型
4. **加深整体**：用共享省出的空间多加 chunk transformer 层

## 超参数说明
```
FIXED_STRIDE=8        固定步长（0=动态路由）
SHARE_ENC_DEC=1       encoder-decoder 权重共享
MODEL_DIM=512         字节级别 embedding 维度
CHUNK_MODEL_DIM=576   词级别 transformer 维度
CHUNK_FFN_DIM=2304    词级别 FFN 维度（=4×chunk_model_dim）
ENCODER_LAYERS=2      encoder Mamba 层数
CHUNK_LAYERS=4        chunk transformer 层数
DECODER_LAYERS=2      decoder Mamba 层数
MAMBA_STATE=32        Mamba2 state 维度
ROUTING_LOSS_WEIGHT   动态路由专用，固定步长不需要
VAL_LOSS_EVERY=200    每200步验证一次
```
