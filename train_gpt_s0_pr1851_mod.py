# =====================================================================
# 导入模块
# =====================================================================
# 标准库：base64编码、有序字典、深拷贝、文件锁、文件通配、内存IO、
# LZMA压缩、数学运算、操作系统接口等基础工具
import base64, collections, copy, fcntl, glob, io, lzma, math, os
from pathlib import Path
# 随机数、正则表达式、子进程、系统接口、计时器、UUID生成器、
# NumPy数值计算、SentencePiece分词器、PyTorch深度学习框架、
# 分布式训练接口 (dist)、函数式API (F) — 均为训练流程的核心依赖
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
# Flash Attention 3 接口 — 高效注意力计算的CUDA实现
# flash_attn_func: 定长序列的闪存注意力
# flash_attn_varlen_func: 变长序列的闪存注意力（用于打包批次，避免填充浪费）
from flash_attn_interface import (
    flash_attn_func as flash_attn_3_func,
    flash_attn_varlen_func,
)
# 线程池执行器 — 用于异步IO操作（如数据预取、日志写入）
from concurrent.futures import ThreadPoolExecutor
# Triton — OpenAI开发的GPU编程语言，用于编写自定义高性能内核
# triton.language (tl) 提供向量化操作原语（load/store/reduce等）
import triton
import triton.language as tl
# TensorDescriptor — Triton的张量描述符工具，用于高级内存访问模式
from triton.tools.tensor_descriptor import TensorDescriptor


# =====================================================================
# 融合软上限交叉熵损失（Triton实现）— 仅训练路径使用
# =====================================================================
# 背景：标准做法是先对logits施加softcap变换（防止logits过大导致数值不稳定），
# 再计算交叉熵损失。朴素实现需要两步：
#     logits_softcap = softcap * tanh(logits / softcap)   # 第一步：软上限
#     loss = F.cross_entropy(logits_softcap.float(), targets)  # 第二步：交叉熵
# 这种两步方式需要在GPU全局显存中存储完整的softcapped logits矩阵（形状为
# [batch_size, vocab_size]），带来大量显存开销和额外的全局存取延迟。
#
# 本实现将两步操作融合为单个Triton内核：logits只从显存读取一次，softcap变换
# 在GPU寄存器中就地完成，然后直接在同一pass中计算log-sum-exp(LSE)和loss。
# 反向传播内核也以同样方式融合，因此无需保存中间的softcapped logits张量。
#
# 数学等价性：采用 softcap * tanh(x/softcap) = 2C * sigmoid(2x/C) - C 的恒等变换，
# 其中 C = softcap。sigmoid形式更适合GPU计算（一次指数运算 vs tanh的两次）。
# 结果在fp32累加精度范围内与朴素实现完全一致。
#
# 性能收益：减少一次全局显存读写、降低显存峰值、改善内核融合效率。
# ===== Fused softcapped cross-entropy (Triton) — training-only path =====
# Replaces the eager
#     logits_softcap = softcap * tanh(logits / softcap)
#     F.cross_entropy(logits_softcap.float(), targets, reduction="mean")
# sequence with a single fused kernel that reads logits_proj once, applies
# softcap in-register, and computes (LSE, loss) in one streaming pass. The
# backward kernel mirrors the forward so there's no stored softcapped logits.
# Numerically identical to the eager path up to fp32 accumulation differences.
# 注册到torch.library的自定义算子库名称，用于前向和反向算子的命名空间
_FUSED_CE_LIBRARY = "pgsubmission1draft7fusedce"
# 每个线程块处理的词表维度块大小（1024个元素/块）
# 选择1024是在寄存器压力和并行度之间的平衡点
_FUSED_CE_BLOCK_SIZE = 1024
# 每个线程块使用的warp数量（4个warp = 128个线程）
# 4个warp足以隐藏内存延迟同时避免寄存器溢出
_FUSED_CE_NUM_WARPS = 4


# ---------------------------------------------------------------
# 前向内核：计算每行（每个token）的softcapped交叉熵损失
# ---------------------------------------------------------------
# GPU并行策略：
#   - 每个program（线程块）处理logits矩阵的一行（对应一个token的词表分布）
#   - grid大小 = n_rows（token数量），即每行一个线程块
#   - 每个线程块内部循环遍历词表维度，每次处理block_size个元素
#   - 使用在线算法（online algorithm）逐块累积max和sum_exp，
#     避免需要两次遍历词表（先求max再求softmax）
#
# 数学推导：
#   softcap * tanh(x / softcap)
#   = softcap * (2*sigmoid(2x/softcap) - 1)
#   = 2*softcap*sigmoid(2x/softcap) - softcap
#   令 A = 2*softcap, inv_C = 2/softcap
#   则 z = A * sigmoid(x * inv_C) 是偏移后的softcapped值
#   （常数偏移 -softcap 在LSE中抵消，不影响交叉熵）
#
# 输出：
#   losses[i] = LSE(z_i) - z_i[target_i]  即交叉熵损失
#   lse[i] = log(sum(exp(z_i)))  保存供反向传播使用
@triton.jit
def _softcapped_ce_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, softcap,
    block_size: tl.constexpr,
):
    # 获取当前线程块的行索引（每行对应一个token）
    row_idx = tl.program_id(0).to(tl.int64)
    # 计算当前行的起始指针
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    # 在线LSE算法的累积变量：跟踪当前最大值和指数和
    max_val = -float("inf")
    sum_exp = 0.0
    # 预计算常量：A = 2*softcap, inv_C = 2/softcap
    # 这将 tanh 形式转换为等价的 sigmoid 形式以提高计算效率
    A = 2.0 * softcap
    inv_C = 2.0 / softcap
    # 分块遍历词表维度，每次处理block_size个词表项
    for off in range(0, n_cols, block_size):
        cols = off + tl.arange(0, block_size)
        # 边界掩码：防止越界访问（最后一块可能不满）
        mask = cols < n_cols
        # 从全局显存加载一块logits到寄存器，越界位置填-inf
        val = tl.load(
            logits_row_ptr + cols * stride_logits_v,
            mask=mask, other=-float("inf"),
        ).to(tl.float32)
        # 在寄存器中就地计算softcap变换: z = 2C * sigmoid(2x/C)
        z = A * tl.sigmoid(val * inv_C)
        z = tl.where(mask, z, -float("inf"))
        # 在线最大值更新算法（数值稳定的log-sum-exp）
        # 每处理一个新块，都重新调整之前的累积指数和
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        # 关键步骤：用 exp(old_max - new_max) 缩放旧的sum_exp，然后加上新块的贡献
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max
    # 最终LSE = max + log(sum_exp)
    lse = max_val + tl.log(sum_exp)
    # 保存LSE供反向传播使用
    tl.store(lse_ptr + row_idx, lse)
    # 加载目标token的索引，计算目标token对应的softcapped logit
    target = tl.load(targets_ptr + row_idx).to(tl.int32)
    target_val = tl.load(logits_row_ptr + target * stride_logits_v).to(tl.float32)
    target_z = A * tl.sigmoid(target_val * inv_C)
    # 交叉熵损失 = LSE - 目标token的softcapped logit
    tl.store(losses_ptr + row_idx, lse - target_z)


# ---------------------------------------------------------------
# 反向内核：计算损失对原始logits的梯度
# ---------------------------------------------------------------
# GPU并行策略与前向相同：每个program处理一行。
#
# 梯度推导（链式法则）：
#   loss = LSE(z) - z[target]
#   其中 z_j = A * sigmoid(x_j * inv_C)
#
#   d(loss)/d(z_j) = softmax(z)_j - 1{j==target}  即 (probs_j - indicator_j)
#   d(z_j)/d(x_j) = A * inv_C * sigmoid(u) * (1 - sigmoid(u))
#                    其中 u = x_j * inv_C
#
#   最终: d(loss)/d(x_j) = grad_loss * (probs_j - indicator_j) * dz/dx_j
#
# 注意：此内核重新读取logits并重新计算sigmoid和softcapped值，
# 而不是从前向传播保存中间结果。这是经典的"重计算换显存"策略。
@triton.jit
def _softcapped_ce_bwd_kernel(
    grad_logits_ptr, grad_losses_ptr, lse_ptr, logits_ptr, targets_ptr,
    stride_logits_n, stride_logits_v,
    stride_grad_n, stride_grad_v,
    n_rows, n_cols, softcap,
    block_size: tl.constexpr,
):
    # 获取当前行索引
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_logits_ptr + row_idx * stride_grad_n
    # 加载前向传播保存的LSE值和上游梯度
    lse = tl.load(lse_ptr + row_idx)
    grad_loss = tl.load(grad_losses_ptr + row_idx).to(tl.float32)
    # 加载目标token索引
    target = tl.load(targets_ptr + row_idx).to(tl.int32)
    # 预计算常量
    A = 2.0 * softcap
    inv_C = 2.0 / softcap
    # dz/dx 的缩放因子 = A * inv_C = 4/1 = 4（当softcap=2时）
    dz_dx_scale = A * inv_C
    # 分块遍历词表维度计算梯度
    for off in range(0, n_cols, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n_cols
        # 重新加载原始logits（重计算策略，节省显存）
        val = tl.load(
            logits_row_ptr + cols * stride_logits_v,
            mask=mask, other=0.0,
        ).to(tl.float32)
        # 重新计算sigmoid和softcapped值
        sigmoid_u = tl.sigmoid(val * inv_C)
        z = A * sigmoid_u
        # 从LSE计算softmax概率: prob_j = exp(z_j - LSE)
        probs = tl.exp(z - lse)
        # d(loss)/d(z_j) = prob_j - 1{j==target}
        grad_z = grad_loss * (probs - tl.where(cols == target, 1.0, 0.0))
        # 应用链式法则: d(loss)/d(x_j) = d(loss)/d(z_j) * dz/dx
        # 其中 dz/dx = A * inv_C * sigmoid(u) * (1 - sigmoid(u))
        grad_x = grad_z * (dz_dx_scale * sigmoid_u * (1.0 - sigmoid_u))
        # 写入梯度到全局显存
        tl.store(grad_row_ptr + cols * stride_grad_v, grad_x, mask=mask)


# ---------------------------------------------------------------
# 输入验证函数：确保Triton内核的输入满足所有前置条件
# ---------------------------------------------------------------
# 检查内容包括：
#   1. logits必须是2D张量 (batch, vocab_size)
#   2. targets必须是1D张量 (batch,)
#   3. 行数必须匹配
#   4. 必须在CUDA设备上（Triton只支持GPU）
#   5. softcap必须为正数
#   6. logits的dtype必须是fp16/bf16/fp32之一
# 同时确保内存连续性（contiguous）以便Triton高效访问，
# 并将targets统一转为int64类型
def _validate_softcapped_ce_inputs(
    logits: Tensor, targets: Tensor, softcap: float,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits.ndim=2, got {logits.ndim}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets.ndim=1, got {targets.ndim}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Expected matching rows, got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    if not logits.is_cuda or not targets.is_cuda:
        raise ValueError("softcapped_cross_entropy requires CUDA tensors")
    if softcap <= 0.0:
        raise ValueError(f"softcap must be positive, got {softcap}")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    logits = logits.contiguous()
    targets = targets.contiguous()
    if targets.dtype != torch.int64:
        targets = targets.to(dtype=torch.int64)
    return logits, targets


# ---------------------------------------------------------------
# 前向自定义算子注册：将Triton内核包装为torch.library自定义算子
# ---------------------------------------------------------------
# 使用torch.library.custom_op注册使得该算子可被torch.compile追踪，
# 也可参与自动微分图（autograd graph）。mutates_args=()表示不原地修改任何输入。
# 返回值：(losses, lse) 其中losses是每个token的损失，lse供反向传播使用。
# grid=(n_rows,)表示启动n_rows个线程块，每个处理词表上的一行。
@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce", mutates_args=())
def softcapped_ce_op(logits: Tensor, targets: Tensor, softcap: float) -> tuple[Tensor, Tensor]:
    logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
    n_rows, n_cols = logits.shape
    # 预分配输出张量（fp32精度，确保损失累加的数值稳定性）
    losses = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    lse = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    # 启动Triton内核：grid=(n_rows,) 即每个token一个线程块
    _softcapped_ce_fwd_kernel[(n_rows,)](
        logits, losses, lse, targets,
        logits.stride(0), logits.stride(1),
        n_rows, n_cols, float(softcap),
        block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
    )
    return losses, lse


# ---------------------------------------------------------------
# Fake（抽象）实现：供torch.compile的形状推断和FakeTensor模式使用
# ---------------------------------------------------------------
# torch.compile在编译期需要知道算子的输出形状和dtype，
# 但不需要实际执行计算。register_fake提供这种"虚拟"实现，
# 只返回正确形状和dtype的空张量。
@softcapped_ce_op.register_fake
def _(logits: Tensor, targets: Tensor, softcap: float):
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("softcapped_ce fake impl expects 2D logits and 1D targets")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Expected matching rows, got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    n_rows = logits.shape[0]
    return (
        logits.new_empty((n_rows,), dtype=torch.float32),
        logits.new_empty((n_rows,), dtype=torch.float32),
    )


# ---------------------------------------------------------------
# 反向自定义算子注册：将反向Triton内核包装为torch.library自定义算子
# ---------------------------------------------------------------
# 接收前向保存的logits、targets、lse，以及上游传来的grad_losses，
# 返回logits的梯度。梯度形状与logits相同 (n_rows, n_cols)。
# 注意grad_logits使用与logits相同的dtype（可能是bf16/fp16），
# 因为反向传播不需要fp32精度的梯度存储。
@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce_backward", mutates_args=())
def softcapped_ce_backward_op(
    logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float,
) -> Tensor:
    logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
    # 确保lse和grad_losses内存连续且为fp32
    lse = lse.contiguous()
    grad_losses = grad_losses.contiguous().to(dtype=torch.float32)
    if lse.ndim != 1 or grad_losses.ndim != 1:
        raise ValueError("Expected 1D lse and grad_losses")
    if lse.shape[0] != logits.shape[0] or grad_losses.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Expected row-aligned lse/grad_losses, got logits={tuple(logits.shape)} "
            f"lse={tuple(lse.shape)} grad_losses={tuple(grad_losses.shape)}"
        )
    # 分配梯度输出张量（与logits同dtype同形状）
    grad_logits = torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    # 启动反向Triton内核
    _softcapped_ce_bwd_kernel[(n_rows,)](
        grad_logits, grad_losses, lse, logits, targets,
        logits.stride(0), logits.stride(1),
        grad_logits.stride(0), grad_logits.stride(1),
        n_rows, n_cols, float(softcap),
        block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
    )
    return grad_logits


# 反向算子的Fake实现：返回与logits同形状的空张量供形状推断
@softcapped_ce_backward_op.register_fake
def _(logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float):
    if logits.ndim != 2 or targets.ndim != 1 or lse.ndim != 1 or grad_losses.ndim != 1:
        raise ValueError("softcapped_ce_backward fake impl expects 2D logits and 1D row tensors")
    if (
        logits.shape[0] != targets.shape[0]
        or logits.shape[0] != lse.shape[0]
        or logits.shape[0] != grad_losses.shape[0]
    ):
        raise ValueError("softcapped_ce_backward fake impl expects row-aligned tensors")
    return logits.new_empty(logits.shape)


# ---------------------------------------------------------------
# PyTorch自动微分（autograd）集成
# ---------------------------------------------------------------
# setup_context: 在前向传播后保存反向传播所需的张量到ctx
# 保存的内容：原始logits、targets、以及前向计算的LSE
# 注意：losses不需要保存，因为反向传播不依赖losses值本身
def _softcapped_ce_setup_context(
    ctx: torch.autograd.function.FunctionCtx, inputs, output,
) -> None:
    logits, targets, softcap = inputs
    _losses, lse = output
    ctx.save_for_backward(logits, targets, lse)
    ctx.softcap = float(softcap)


# 反向传播函数：从ctx中取出保存的张量，调用反向Triton内核
# grad_lse被丢弃（del grad_lse），因为LSE仅是反向传播的辅助值，
# 不需要对其求梯度。返回元组中的两个None对应targets和softcap的梯度（不可微）。
def _softcapped_ce_backward(
    ctx: torch.autograd.function.FunctionCtx, grad_losses: Tensor, grad_lse: "Tensor | None",
):
    del grad_lse
    logits, targets, lse = ctx.saved_tensors
    grad_logits = torch.ops.pgsubmission1draft7fusedce.softcapped_ce_backward(
        logits, targets, lse, grad_losses, ctx.softcap
    )
    return grad_logits, None, None


# 将自动微分钩子注册到前向算子上，使PyTorch知道如何对该自定义算子反向传播
softcapped_ce_op.register_autograd(
    _softcapped_ce_backward, setup_context=_softcapped_ce_setup_context,
)


# ---------------------------------------------------------------
# 用户友好的包装函数：softcapped_cross_entropy
# ---------------------------------------------------------------
# 这是融合softcapped交叉熵的对外接口，API风格与F.cross_entropy一致。
# 支持三种归约方式：
#   "none" — 返回每个token的损失（不归约）
#   "sum"  — 返回所有token损失之和
#   "mean" — 返回所有token损失的平均值（默认）
# 内部通过torch.ops命名空间调用注册的自定义算子。
def softcapped_cross_entropy(
    logits: Tensor, targets: Tensor, softcap: float, reduction: str = "mean",
) -> Tensor:
    losses, _lse = torch.ops.pgsubmission1draft7fusedce.softcapped_ce(
        logits, targets, float(softcap)
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction={reduction!r}")


# =====================================================================
# 超参数配置类
# =====================================================================
# 所有超参数通过环境变量配置，支持灵活的实验管理。
# 设计理念：类属性在导入时即求值，所有参数都有合理的默认值，
# 可通过设置环境变量覆盖默认值而无需修改代码。
# 这种模式便于：(1) 命令行实验切换 (2) 分布式训练配置 (3) 超参搜索脚本
class Hyperparameters:
    # --- 基础训练配置 ---
    # 数据根目录：存放数据集和分词器的基础路径
    data_dir = os.environ.get("DATA_DIR", "./data/")
    # 随机种子：确保实验可复现
    seed = int(os.environ.get("SEED", 1337))
    # 运行ID：每次运行的唯一标识符，默认为UUID，用于日志和模型文件命名
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    # 总训练迭代次数
    iterations = int(os.environ.get("ITERATIONS", 20000))
    # warmdown阶段起始点（占总迭代的比例），即从75%处开始降低学习率
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.75))
    # 学习率预热步数：训练初期线性增加学习率，防止初始梯度过大
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    # 每个训练批次的总token数量（约786K tokens）
    # 实际batch_size = train_batch_tokens / train_seq_len
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    # 是否启用融合softcapped交叉熵（上面定义的Triton内核）
    # 仅用于训练路径；评估路径仍使用朴素的softcap+F.cross_entropy
    # 默认开启，经验证性能至少不劣于朴素实现
    # Fused softcapped CE (Triton). Training-only — forward_logits eval path still uses
    # eager softcap+F.cross_entropy. Default ON since validated as at-worst neutral.
    fused_ce_enabled = bool(int(os.environ.get("FUSED_CE_ENABLED", "1")))
    # 训练序列长度（token数）
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    # 每隔多少步打印一次训练日志
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    # 最大墙钟时间（秒），超时后停止训练。600秒=10分钟，
    # 这是parameter-golf竞赛的时间限制
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 6e2))
    # 验证集每批次的token数量（约524K tokens）
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    # 评估序列长度
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    # 每隔多少步执行一次验证集评估
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    # --- 模型架构参数 ---
    # 词表大小：SentencePiece BPE分词器的词表大小
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    # Transformer层数
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    # 交叉序列注意力(XSA)使用的最后N层数量
    # 当等于num_layers时表示所有层都使用XSA
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    # 模型隐藏维度（d_model）
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    # KV头数量（用于分组查询注意力GQA）
    # num_heads=8, num_kv_heads=4 意味着每2个查询头共享1个KV头
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    # 查询头数量
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    # MLP隐藏层维度相对于model_dim的倍数
    # MLP隐藏维度 = model_dim * mlp_mult = 512 * 4 = 2048
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    # 是否启用残差连接的跳跃门控（learnable skip gates）
    # 门控机制允许模型学习每层残差连接的权重
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    # 是否共享输入嵌入和输出投影的权重（weight tying）
    # 共享权重可显著减少参数量（vocab_size * model_dim个参数）
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    # 输出logits的软上限值：限制logits的最大幅度为±30
    # 防止logits过大导致softmax数值不稳定
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 3e1))
    # --- RoPE (旋转位置编码) 参数 ---
    # RoPE基频：控制不同维度的旋转频率分布
    rope_base = float(os.environ.get("ROPE_BASE", 1e4))
    # RoPE编码的维度数（每个头中用于位置编码的维度）
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    # RoPE训练时的参考序列长度，用于YaRN外推时的缩放计算
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 2048))
    # 是否启用YaRN（Yet another RoPE extensioN）位置编码外推
    rope_yarn = bool(int(os.environ.get("ROPE_YARN", "0")))
    # LayerNorm是否使用可学习的缩放参数（scale/gamma）
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    # QK增益的初始值：用于缩放Q和K的投影输出，
    # 较大的初始值使注意力分布在训练初期更尖锐
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))
    # --- 循环层（Looping Layers）参数 ---
    # 循环层机制：将中间的一组层重复执行多次，模拟更深的网络
    # 这相当于用相同权重构建"虚拟"深层，增加计算深度但不增加参数量
    # 循环次数：层[loop_start:loop_end]将被重复执行num_loops次
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    # 循环起始层索引（包含）
    loop_start = int(os.environ.get("LOOP_START", 3))
    # 循环结束层索引（不包含），即循环层3和层4
    loop_end = int(os.environ.get("LOOP_END", 5))
    # 在训练进度达到此比例后才启用循环层（渐进式启用）
    # 0.35表示训练进行到35%时开始循环，避免训练初期的不稳定
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    # --- 并行层（Parallel Layers）参数 ---
    # 从此层开始使用并行执行（多条"车道"同时处理）
    parallel_start_layer = int(os.environ.get("PARALLEL_START_LAYER", 8))
    # 并行车道的最终合并策略："mean"表示取各车道输出的平均值
    parallel_final_lane = os.environ.get("PARALLEL_FINAL_LANE", "mean")
    # --- 优化器与学习率参数 ---
    # 学习率调度的最小学习率（warmdown阶段的下限）
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    # 嵌入层学习率（独立嵌入时使用，较高以加速嵌入学习）
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    # 共享嵌入（tied embeddings）的学习率，显著低于独立嵌入
    # 因为共享权重同时影响输入和输出，需要更保守的更新
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    # 共享嵌入的初始化标准差（较小值确保初始输出不过大）
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    # 矩阵参数（线性层权重）使用Muon优化器的学习率
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.026))
    # 标量参数（如gate、scale等1D参数）使用Adam优化器的学习率
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    # --- Muon优化器参数 ---
    # Muon是一种基于矩阵正交化的优化器，专为矩阵参数设计
    # 动量系数：控制历史梯度的衰减速率
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.97))
    # Muon后端的Newton-Schulz正交化迭代次数
    # 更多步数使更新方向更接近正交，但增加计算开销
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    # 动量预热的起始值（从0.92线性增加到0.97）
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    # 动量预热步数：在此期间动量从warmup_start线性增加到muon_momentum
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    # 是否对Muon更新进行逐行归一化
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    # --- Adam优化器参数（用于标量参数和嵌入） ---
    # 一阶矩（均值）的指数衰减率
    beta1 = float(os.environ.get("BETA1", 0.9))
    # 二阶矩（方差）的指数衰减率
    beta2 = float(os.environ.get("BETA2", 0.95))
    # Adam的epsilon值，防止除零
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-08))
    # 梯度裁剪的最大范数，防止梯度爆炸
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    # 评估时的滑动窗口步长（token数），用于评估长序列时的采样间隔
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    # --- 权重衰减（Weight Decay）参数 ---
    # Adam优化器的权重衰减系数
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    # Muon优化器的权重衰减系数（矩阵参数使用更强的正则化）
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    # 嵌入层的权重衰减系数
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    # --- EMA (指数移动平均) ---
    # EMA衰减率：用于维护模型参数的指数移动平均副本
    # 评估和最终模型使用EMA参数，通常比训练参数更稳定
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    # --- TTT (Test-Time Training / 测试时训练) 参数 ---
    # TTT是一种推理时自适应技术：在评估时对模型的部分参数（通过LoRA）
    # 进行梯度更新，使模型能够适应当前输入的分布。
    # 这在分布偏移的场景下特别有效。
    # 是否启用TTT
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    # TTT使用的LoRA秩（低秩适配的维度），96是相对较高的秩
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 96))
    # TTT内部优化器的学习率
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.0001))
    # TTT处理的chunk大小（token数），将序列分成chunk逐段适配
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 48))
    # TTT评估时的序列长度
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 2048))
    # TTT的batch大小（每次适配的序列数）
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))
    # TTT每个chunk的梯度更新步数
    ttt_grad_steps = int(os.environ.get("TTT_GRAD_STEPS", 1))
    # TTT内部优化器的权重衰减
    ttt_weight_decay = float(os.environ.get("TTT_WEIGHT_DECAY", 1.0))
    # TTT内部Adam的beta1（设为0表示不使用一阶矩，类似SGD+二阶矩自适应）
    ttt_beta1 = float(os.environ.get("TTT_BETA1", 0))
    # TTT内部Adam的beta2
    ttt_beta2 = float(os.environ.get("TTT_BETA2", 0.999))
    # 是否对K(Key)投影使用LoRA适配
    ttt_k_lora = bool(int(os.environ.get("TTT_K_LORA", "1")))
    # 是否对MLP层使用LoRA适配
    ttt_mlp_lora = bool(int(os.environ.get("TTT_MLP_LORA", "1")))
    # 是否对O(Output)投影使用LoRA适配
    ttt_o_lora = bool(int(os.environ.get("TTT_O_LORA", "1")))
    # TTT使用的优化器类型（"adam"或其他）
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "adam")
    # TTT评估的批次列表（逗号分隔的批次索引），空字符串表示评估所有批次
    ttt_eval_batches = os.environ.get("TTT_EVAL_BATCHES", "")
    # --- 评估与压缩参数 ---
    # 验证集使用的文档比例（1.0=全部，<1.0用于快速调试）
    val_doc_fraction = float(os.environ.get("VAL_DOC_FRACTION", 1.0))
    # 模型权重/提交文件的压缩算法（brotli提供高压缩率）
    compressor = os.environ.get("COMPRESSOR", "brotli")
    # --- GPTQ (训练后量化) 参数 ---
    # GPTQ是一种逐层量化方法，需要校准数据来估计Hessian矩阵
    # 校准批次数：用于收集Hessian统计的数据批次数量
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 16))
    # 为GPTQ量化预留的墙钟时间（秒），确保在时间限制内完成量化
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 4.0))
    # --- 分阶段TTT参数 ---
    # 分阶段TTT的前缀文档数：每个阶段使用的文档数量
    phased_ttt_prefix_docs = int(os.environ.get("PHASED_TTT_PREFIX_DOCS", 2000))
    # 分阶段TTT的阶段数
    phased_ttt_num_phases = int(os.environ.get("PHASED_TTT_NUM_PHASES", 1))
    # --- 全局TTT参数 ---
    # 全局TTT与上面的per-sequence TTT不同：它在整个验证集上进行多遍适配，
    # 类似于在验证数据上做少量epoch的微调，但只更新LoRA参数
    # 全局TTT的学习率
    global_ttt_lr = float(os.environ.get("GLOBAL_TTT_LR", 0.001))
    # 全局TTT的SGD动量
    global_ttt_momentum = float(os.environ.get("GLOBAL_TTT_MOMENTUM", 0.9))
    # 全局TTT的epoch数（遍历验证集的次数）
    global_ttt_epochs = int(os.environ.get("GLOBAL_TTT_EPOCHS", 1))
    # 每个chunk处理的token数（32K tokens/chunk）
    global_ttt_chunk_tokens = int(os.environ.get("GLOBAL_TTT_CHUNK_TOKENS", 32768))
    # 每批处理的序列数
    global_ttt_batch_seqs = int(os.environ.get("GLOBAL_TTT_BATCH_SEQS", 32))
    # 全局TTT学习率预热的起始值
    global_ttt_warmup_start_lr = float(os.environ.get("GLOBAL_TTT_WARMUP_START_LR", 0.0))
    # 全局TTT预热的chunk数（0表示无预热）
    global_ttt_warmup_chunks = int(os.environ.get("GLOBAL_TTT_WARMUP_CHUNKS", 0))
    # 全局TTT的梯度裁剪阈值
    global_ttt_grad_clip = float(os.environ.get("GLOBAL_TTT_GRAD_CLIP", 1.0))
    # 是否在全局TTT中尊重文档边界（不跨文档构建序列）
    global_ttt_respect_doc_boundaries = bool(int(os.environ.get("GLOBAL_TTT_RESPECT_DOC_BOUNDARIES", "1")))
    # --- 量化位宽与裁剪参数 ---
    # 用于GPTQ训练后量化的目标位宽和异常值裁剪策略
    # 矩阵参数（线性层权重）的量化位宽：6位
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    # 嵌入层的量化位宽：8位（嵌入对量化更敏感，使用更高精度）
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    # 量化前裁剪的sigma数：超过此倍标准差的异常值会被裁剪
    # 不同类型的权重有不同的裁剪阈值，反映其值分布的差异
    # 矩阵权重的裁剪阈值（12.85σ，较宽松，保留大部分值域）
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    # 嵌入权重的裁剪阈值（20σ）
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 2e1))
    # MLP权重的裁剪阈值（10σ，较严格）
    mlp_clip_sigmas = float(os.environ.get("MLP_CLIP_SIGMAS", 10.0))
    # 注意力权重的裁剪阈值（13σ）
    attn_clip_sigmas = float(os.environ.get("ATTN_CLIP_SIGMAS", 13.0))
    # --- AttnOutGate (注意力输出门控) ---
    # 来源：PR #1667 (MarioPaerle)
    # 功能：在每个注意力头的输出上施加一个可学习的乘性门控
    # 初始化策略：权重初始化为0，使得 2*sigmoid(0)=1（初始时门完全打开，
    # 即"透明"状态），训练过程中逐渐学习哪些头的输出需要衰减
    # 门控输入源：'proj'使用块输入x（默认），'q'使用Q投影输出
    # AttnOutGate (per-head multiplicative output gate, PR #1667 MarioPaerle).
    # Zero-init weight: 2*sigmoid(0)=1 -> transparent at start. Source defaults to
    # block input x ('proj'); 'q' uses raw Q projection output.
    attn_out_gate_enabled = bool(int(os.environ.get("ATTN_OUT_GATE_ENABLED", "0")))
    attn_out_gate_src = os.environ.get("ATTN_OUT_GATE_SRC", "proj")
    # --- SmearGate (前向token平滑门控) ---
    # 来源：modded-nanogpt @classiclarryd (PR #1667)
    # 功能：输入依赖的token间信息传递，将前一个token的表示"涂抹"到当前token
    # 公式：x_t <- x_t + λ * sigmoid(W * x_t[:gate_window]) * x_{t-1}
    # 初始化：λ=0, W=0 → 初始时无涂抹效果（"透明"状态）
    # 这种机制允许模型学习一种轻量级的局部时序混合，
    # 补充注意力机制捕获不到的相邻token关系
    # SmearGate (input-dependent forward-1 token smear, modded-nanogpt @classiclarryd
    # via PR #1667). x_t <- x_t + lam * sigmoid(W*x_t[:gate_window]) * x_{t-1}.
    # lam=0 + W=0 -> transparent at init.
    smear_gate_enabled = bool(int(os.environ.get("SMEAR_GATE_ENABLED", "0")))
    # 门控窗口大小：使用输入向量的前GATE_WINDOW个维度来计算门控信号
    # 只使用部分维度而非全部，减少参数量（W的形状为 gate_window -> 1）
    # Window: first GATE_WINDOW dims of the source feed the gate projection.
    gate_window = int(os.environ.get("GATE_WINDOW", 12))
    # --- Gated Attention (门控注意力) ---
    # 来源：Qwen团队, NeurIPS 2025最佳论文 (arXiv:2505.06708)
    # 功能：在SDPA输出之后、output projection之前添加逐头的sigmoid门控
    # 门控输入：完整的块输入x（论文中的headwise G1变体）
    # W_g形状：(num_heads, dim)，使用普通sigmoid激活
    # 初始化：近零初始化使得 g ≈ sigmoid(0) = 0.5（初始时保留一半注意力输出）
    # 配合per-block的attn_scale（初始化为1.0）在训练中补偿
    # 命名中包含"attn_gate"以便参数路由系统将其分配给标量AdamW优化器
    # Gated Attention (Qwen, NeurIPS 2025 Best Paper, arXiv:2505.06708;
    # qiuzh20/gated_attention). Per-head sigmoid gate on SDPA output, BEFORE
    # out_proj. Gate input = full block input x (paper's headwise G1 variant
    # driven from hidden_states). W_g shape (num_heads, dim), plain sigmoid.
    # Near-zero init gives g~0.5 at step 0 (half attention output); per-block
    # attn_scale (init 1.0) compensates during training. Name contains
    # "attn_gate" so CONTROL_TENSOR_NAME_PATTERNS routes it to scalar AdamW.
    gated_attn_enabled = bool(int(os.environ.get("GATED_ATTN_ENABLED", "0")))
    # 门控权重的初始化标准差（0.01 → 近零初始化）
    gated_attn_init_std = float(os.environ.get("GATED_ATTN_INIT_STD", 0.01))
    # 是否对attn_gate_w张量使用专用的int8逐行量化
    # 这些张量很小 ((num_heads, dim) = (8, 512) = 4096个参数)，
    # 默认会绕过GPTQ（通过numel<=65536直通分支）以fp16存储（每层8KB）。
    # int8逐行量化可将原始张量减半，对BPB影响可忽略：
    # 每头8个缩放值，对称量化到[-127, 127]范围。
    # 不需要Hessian（门控权重不在collect_hessians()中）。
    # Dedicated int8-per-row quantization for `attn_gate_w` tensors. These are
    # small ((num_heads, dim) = (8, 512) = 4096 params) and bypass GPTQ via the
    # numel<=65536 passthrough branch -> stored as fp16 (8 KB/layer, ~65 KB total
    # compressed). int8-per-row cuts the raw tensor in half with negligible BPB
    # impact: scales per head (8 values), symmetric quant over [-127, 127].
    # No Hessian needed (gate weights not in collect_hessians()).
    gated_attn_quant_gate = bool(int(os.environ.get("GATED_ATTN_QUANT_GATE", "0")))
    # --- Sparse Attention Gate (稀疏注意力门控) ---
    # 来源：modded-nanogpt风格的变体
    # 与GatedAttn的区别：门控输入不使用完整的dim维度，
    # 而只使用残差流的前GATE_WINDOW个维度
    # W_g形状: (num_heads, gate_window) = (8, 12) = 96参数/层
    # 对比密集GatedAttn的 (8, 512) = 4K参数/层，参数量大幅减少
    # 共享"attn_gate_w"命名，因此量化路由和int8门控直通自动兼容
    # 与ATTN_OUT_GATE_ENABLED和GATED_ATTN_ENABLED互斥（只能启用一种门控）
    # Sparse Attention Gate (modded-nanogpt-style). Keeps dense SDPA and only
    # swaps the output-gate input to the first GATE_WINDOW residual dims.
    # W_g: (num_heads, gate_window) = (8, 12) = 96 params/layer (~44K total),
    # vs dense GatedAttn's (8, 512) = 4K/layer (~44K diff). Name "attn_gate_w"
    # is shared so quant routing and int8 gate passthrough Just Work. Gate
    # passthrough int8 still applies via GATED_ATTN_QUANT_GATE=1.
    # Mutually exclusive with ATTN_OUT_GATE_ENABLED and GATED_ATTN_ENABLED.
    sparse_attn_gate_enabled = bool(int(os.environ.get("SPARSE_ATTN_GATE_ENABLED", "0")))
    # 稀疏门控权重的初始化标准差（0.0 = 零初始化）
    sparse_attn_gate_init_std = float(os.environ.get("SPARSE_ATTN_GATE_INIT_STD", 0.0))
    # 稀疏门控的输出缩放因子
    sparse_attn_gate_scale = float(os.environ.get("SPARSE_ATTN_GATE_SCALE", 1.0))
    # --- LQER (低秩量化误差修正) ---
    # 来源：PR #1530 v2 port
    # 原理：对量化误差矩阵 E = W_fp - W_quant 进行SVD分解，
    # 取前r个奇异值对应的低秩近似 A*B^T 作为修正项。
    # 推理时：W_dequant = W_quant + A * B^T，用少量额外参数补偿量化损失。
    # 因子矩阵A和B本身也被量化为低位宽整数（INT2/INT4/INTk），
    # 进一步减少存储开销。
    # LQER asymmetric rank-k correction on top-K quant-error tensors (PR #1530 v2 port).
    # Computes SVD of E = W_fp - W_quant, packs top-r A,B as INT2/INT4 (asym) or INTk (sym).
    # 是否启用LQER误差修正
    lqer_enabled = bool(int(os.environ.get("LQER_ENABLED", "1")))
    # SVD分解保留的秩（奇异值数量），秩越高修正越精确但参数越多
    lqer_rank = int(os.environ.get("LQER_RANK", 4))
    # 对量化误差最大的前K个权重矩阵应用LQER修正
    # 只修正误差最大的层，在精度和存储之间取得平衡
    lqer_top_k = int(os.environ.get("LQER_TOP_K", 3))
    # LQER因子矩阵A和B的量化位宽
    lqer_factor_bits = int(os.environ.get("LQER_FACTOR_BITS", 4))
    # 是否使用非对称量化（有零点偏移），非对称量化比对称量化更精确
    lqer_asym_enabled = bool(int(os.environ.get("LQER_ASYM_ENABLED", "1")))
    # 非对称量化的分组大小：每64个元素共享一组量化参数（缩放因子+零点）
    # 更小的组提供更精细的量化但增加元数据开销
    lqer_asym_group = int(os.environ.get("LQER_ASYM_GROUP", "64"))
    # --- 分布式训练参数 ---
    # 通过检测环境变量RANK和WORLD_SIZE来判断是否为分布式训练模式
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    # 当前进程的全局排名（0-indexed）
    rank = int(os.environ.get("RANK", "0"))
    # 总进程数（GPU数量）
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # 当前进程在本地节点上的排名（用于多节点训练）
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # 是否为主进程（只有主进程负责日志记录和模型保存）
    is_main_process = rank == 0
    # 梯度累积步数：8个micro-batch除以GPU数量
    # 例如单卡时累积8步，双卡时各累积4步，总有效batch_size保持不变
    grad_accum_steps = 8 // world_size
    # --- CaseOps 集成 ---
    # CaseOps是一种大小写感知的分词优化方案。
    # 当启用时，使用专用的数据集路径和分词器（包含大小写保留的BPE词表）。
    # 核心区别：加载一个per-token的字节侧车文件(fineweb_val_bytes_*.bin)，
    # 该文件与val_*.bin有相同的分片布局，用作BPB(Bits Per Byte)评估的
    # 标准原始字节预算。这完全替代了build_sentencepiece_luts的字节计数路径。
    # CaseOps integration: optional override of dataset root + tokenizer path.
    # When CASEOPS_ENABLED=1, the wrapper loads a per-token byte sidecar
    # (fineweb_val_bytes_*.bin, identical shard layout to val_*.bin) and uses
    # it as the canonical raw-byte budget for BPB accounting. The sidecar
    # REPLACES the build_sentencepiece_luts byte-counting path entirely.
    caseops_enabled = bool(int(os.environ.get("CASEOPS_ENABLED", "0")))
    # CaseOps专用数据集的默认路径
    _default_caseops_data = os.path.join(
        data_dir,
        "datasets",
        "fineweb10B_sp8192_caseops",
        "datasets",
        "datasets",
        "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved",
    )
    # CaseOps专用分词器的默认路径
    _default_caseops_tok = os.path.join(
        data_dir,
        "datasets",
        "fineweb10B_sp8192_caseops",
        "datasets",
        "tokenizers",
        "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
    )
    # 根据是否启用CaseOps选择不同的数据集和分词器路径
    # 两种模式都支持通过环境变量DATA_PATH和TOKENIZER_PATH自定义覆盖
    if caseops_enabled:
        datasets_dir = os.environ.get("DATA_PATH", _default_caseops_data)
        tokenizer_path = os.environ.get("TOKENIZER_PATH", _default_caseops_tok)
    else:
        # 标准模式：使用与vocab_size匹配的数据集
        datasets_dir = os.environ.get(
            "DATA_PATH",
            os.path.join(data_dir, "datasets", f"fineweb10B_sp{vocab_size}"),
        )
        tokenizer_path = os.environ.get(
            "TOKENIZER_PATH",
            os.path.join(data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model"),
        )
    # --- 文件路径配置 ---
    # 训练数据文件的glob模式（分片存储，支持多worker并行加载）
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    # 验证数据文件的glob模式
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    # CaseOps的字节侧车文件（记录每个token对应的原始字节数，用于BPB计算）
    val_bytes_files = os.path.join(datasets_dir, "fineweb_val_bytes_*.bin")
    # 产出物（日志、模型）的存放目录，空字符串时使用默认位置
    artifact_dir = os.environ.get("ARTIFACT_DIR", "")
    # 训练日志文件路径
    logfile = (
        os.path.join(artifact_dir, f"{run_id}.txt")
        if artifact_dir
        else f"logs/{run_id}.txt"
    )
    # 训练完成后保存的全精度模型路径
    model_path = (
        os.path.join(artifact_dir, "final_model.pt")
        if artifact_dir
        else "final_model.pt"
    )
    # GPTQ量化后的模型路径（.ptz表示压缩后的量化模型）
    # int6表示6位量化，这是提交竞赛时的最终模型格式
    quantized_model_path = (
        os.path.join(artifact_dir, "final_model.int6.ptz")
        if artifact_dir
        else "final_model.int6.ptz"
    )


# ============================================================================
# 日志工具 (Logging Utilities)
# ============================================================================
# 全局变量，存储日志相关的超参数（如是否为主进程、日志文件路径等）。
# 初始为 None，表示尚未配置，此时 log() 直接 print 到控制台。
_logger_hparams = None


# 设置日志超参数：将包含 is_main_process、logfile 等属性的超参数对象保存到全局变量。
# 通常在训练初始化阶段调用一次。
def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


# 统一日志函数：根据配置决定输出到控制台和/或日志文件。
# - 如果尚未配置超参数（_logger_hparams is None），直接 print 并返回。
# - 否则仅在主进程（rank 0）上执行日志输出，避免多进程重复打印。
# - console 参数控制是否同时输出到标准输出；logfile 不为 None 时追加写入文件。
def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


# ============================================================================
# 验证数据集 (ValidationData)
# ============================================================================
# 封装验证集的 token 数据和用于计算 BPB（Bits Per Byte）指标的查找表。
# BPB 是语言模型的标准评估指标，衡量每个原始字节的信息量。
# 需要知道每个 token 对应多少个 UTF-8 字节，因此要从 SentencePiece 分词器
# 构建字节数查找表。
# 可选地支持 CaseOps（大小写操作）的字节 sidecar 数据。
class ValidationData:
    def __init__(self, h, device):
        # 加载 SentencePiece 分词器模型，用于构建 token → 字节数的映射关系
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        # 校验：确保超参数中的词表大小与分词器一致，否则 BPB 计算会出错
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        # 加载验证集 token 序列（从二进制 shard 文件读取，截断到 eval_seq_len 对齐长度）
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        # 构建三个查找表（LUT），放在 GPU 上以便快速索引：
        # - base_bytes_lut: 每个 token ID 对应的 UTF-8 字节数
        # - has_leading_space_lut: token 是否以 ▁（空格前缀）开头
        # - is_boundary_token_lut: 是否为边界 token（控制/未知/未使用 token）
        (
            self.base_bytes_lut,
            self.has_leading_space_lut,
            self.is_boundary_token_lut,
        ) = build_sentencepiece_luts(self.sp, h.vocab_size, device)
        # CaseOps: when enabled, load per-token byte sidecar and stash it as a
        # CPU tensor aligned 1:1 with self.val_tokens. eval_val/eval_val_ttt
        # branches use this as the canonical raw-byte budget per token.
        # CaseOps（大小写操作）功能：启用时，加载与 val_tokens 一一对应的字节 sidecar，
        # 提供每个 token 在原始文本中占用的真实字节数。用于更精确的 BPB 计算。
        self.caseops_enabled = bool(getattr(h, "caseops_enabled", False))
        self.val_bytes = None
        if self.caseops_enabled:
            self.val_bytes = load_validation_byte_sidecar(
                h.val_bytes_files, h.eval_seq_len, self.val_tokens.numel()
            )


# 构建 SentencePiece 查找表 (BPB 字节计数所需)
# 遍历分词器的所有 token，为每个 token 计算：
#   1. base_bytes: 该 token 对应的 UTF-8 字节数（不含 ▁ 前缀的空格字节）
#   2. has_leading_space: 该 token 是否以 ▁ 开头（▁ 代表一个空格字节，需单独计数）
#   3. is_boundary_token: 是否为控制/未知/未使用 token（这类 token 不计入字节总数）
# 返回三个 GPU 张量，形状为 (table_size,)，可直接用 token ID 索引。
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    # 确保分词器有独立的 ▁ token，否则 BPB 中的空格字节计数会出错
    assert (
        sp.piece_to_id("▁") != sp.unk_id()
    ), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    # table_size 取词表大小和分词器词表大小的最大值，以处理可能的填充 token
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    # 默认所有 token 都是边界 token，然后在下方循环中把有效 token 标记为 False
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        # 跳过控制、未知、未使用 token（它们保持 is_boundary=True，字节数=0）
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        # 单字节回退 token（byte fallback）：固定为 1 字节
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        # 检测并去除 ▁ 前缀，▁ 本身代表 1 个空格字节，单独用 has_leading_space 标记
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        # 去掉 ▁ 后的字符串编码为 UTF-8，其长度即为该 token 的基础字节数
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


# ============================================================================
# 数据加载函数 (Data Loading Functions)
# ============================================================================

# 加载验证集 token 序列。
# 从 glob 模式匹配的二进制 shard 文件中读取并拼接所有 token。
# 截断到 seq_len 的整数倍 +1（+1 是因为语言模型需要 input[:-1] 和 target[1:]）。
# 过滤掉 CaseOps 字节 sidecar 文件（文件名包含 "_bytes_"），它们共享相同的 glob 模式。
def load_validation_tokens(pattern, seq_len):
    # Filter out CaseOps byte sidecar shards which share the val_*.bin glob.
    files = [
        Path(p)
        for p in sorted(glob.glob(pattern))
        if "_bytes_" not in Path(p).name
    ]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # 拼接所有 shard 的 token 为一个连续张量
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    # 计算可用长度：(总长-1) 向下取 seq_len 的整数倍，保证能整除成完整序列
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    # 返回 usable+1 个 token（多出的 1 个用作最后一个序列的 target 末尾）
    return tokens[: usable + 1]


# 加载 CaseOps 字节 sidecar 文件。
# sidecar 文件与 token shard 共享相同的二进制格式（256 个 int32 头部 + uint16 数组），
# 但每个元素不是 token ID，而是该位置 token 在原始文本中占用的字节数。
# 用于 CaseOps 模式下的精确 BPB 计算。
# 返回的张量与 val_tokens 长度一致（截断到 expected_len）。
def load_validation_byte_sidecar(pattern, seq_len, expected_len):
    """Load CaseOps per-token byte sidecar(s). Same shard layout as token shards
    (256 int32 header + uint16 array). Each entry = canonical raw-text byte
    budget for that token in the corresponding val shard. Returns a CPU
    int16 tensor sliced to match expected_len (i.e. val_tokens length)."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No byte sidecar files for pattern: {pattern}")
    shards = [load_data_shard(file) for file in files]
    # load_data_shard 返回 uint16 张量——与 sidecar 的存储格式一致
    # load_data_shard returns uint16 — that's exactly what the sidecar stores.
    bytes_full = torch.cat(shards).contiguous()
    if bytes_full.numel() < expected_len:
        raise ValueError(
            f"Byte sidecar too short: {bytes_full.numel()} < val_tokens {expected_len}"
        )
    # 转为 int32 以避免后续计算中的溢出
    return bytes_full[:expected_len].to(torch.int32)


# 从二进制 shard 文件加载 token 数据。
# shard 文件格式：
#   - 头部：256 个 int32（小端序），共 1024 字节
#     - header[0] = 20240520  （魔数，用于验证文件格式）
#     - header[1] = 1          （版本号）
#     - header[2] = num_tokens （该 shard 包含的 token 数量）
#   - 数据：num_tokens 个 uint16（小端序），每个 token ID 占 2 字节
# 返回 uint16 类型的 PyTorch 张量。
def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize  # 头部固定 1024 字节
    token_bytes = np.dtype("<u2").itemsize  # 每个 token 占 2 字节
    header = np.fromfile(file, dtype="<i4", count=256)
    # 校验头部：魔数、版本号必须匹配
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    # 校验文件大小是否与头部声明的 token 数量一致
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    # 跳过头部，读取 token 数据
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


# ============================================================================
# Shard 缓存与内存映射 (Shard Caching & Memory Mapping)
# ============================================================================
# 以下缓存机制用于 ShuffledSequenceLoader：避免重复读取 shard 头部和重复创建 memmap。

# shard 头部固定大小（256 个 int32 = 1024 字节）
_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
# 缓存：文件路径 → token 数量，避免重复读取头部
_SHARD_NTOKENS_CACHE = {}
# 缓存：文件路径 → numpy memmap 对象，避免重复创建内存映射
_MMAP_CACHE = {}


# 读取 shard 文件头部以获取 token 数量，带缓存。
# 第一次读取时解析头部并缓存结果，后续调用直接返回缓存值。
def _read_num_tokens(file):
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


# 获取 shard 文件的内存映射（memmap），带缓存。
# 内存映射允许按需从磁盘读取数据，而非将整个文件加载到内存。
# 这对于 ShuffledSequenceLoader 的随机访问模式特别高效。
def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    # 以只读模式创建 memmap，跳过 1024 字节头部，直接映射 token 数据区
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


# BOS（Beginning Of Sequence）token 的 ID。
# 初始为 None，在 DocumentPackingLoader._init_shard() 中被设置为 1。
# 用于在 packed document 模式中定位文档边界。
BOS_ID = None


# 向上取整到 n 的倍数。例如 get_next_multiple_of_n(65, 64) = 128。
# 用于对 cu_seqlens 数组进行填充对齐。
def get_next_multiple_of_n(v, n):
    return ((v + n - 1) // n) * n


# 构建累积序列长度数组 (cu_seqlens)，用于 flash_attn_varlen_func。
# 在 document packing 模式下，一个 batch 内可能包含多个长度不等的文档。
# Flash Attention 的 varlen 变体需要 cu_seqlens 来指定每个文档的起止位置，
# 从而确保注意力不会跨越文档边界。
#
# 参数:
#   bos_pos: BOS token 在当前 batch 中的位置列表（即文档起始位置）
#   total_len: batch 的总 token 数
#   device: 目标设备
#   max_doc_len: 若 > 0，则将超长文档拆分为不超过此长度的段（限制注意力跨度）
#   bucket_size: cu_seqlens 数组的对齐粒度（填充到 bucket_size 的倍数以提高 GPU 效率）
#
# 返回:
#   cu: 填充后的累积序列长度张量，形状 (padded_len,)
#   max_seqlen: 所有段中最长段的长度（Flash Attention 需要此值）
def _build_cu_seqlens(bos_pos, total_len, device, max_doc_len=0, bucket_size=64):
    # 确保第一个文档从位置 0 开始
    if not bos_pos or bos_pos[0] != 0:
        bos_pos = [0] + bos_pos
    seg_starts = []
    starts_with_end = bos_pos + [total_len]
    for i in range(len(starts_with_end) - 1):
        start = starts_with_end[i]
        end = starts_with_end[i + 1]
        # 如果设置了 max_doc_len，将长文档拆分为多个不超过 max_doc_len 的段
        if max_doc_len > 0:
            pos = start
            while pos < end:
                seg_starts.append(pos)
                pos += max_doc_len
        else:
            seg_starts.append(start)
    # boundaries 是所有段的起始位置 + 末尾位置，即 cu_seqlens 的内容
    boundaries = seg_starts + [total_len]
    # 填充到 bucket_size 的倍数，多余位置填 total_len（不影响注意力计算）
    padded_len = get_next_multiple_of_n(len(boundaries), bucket_size)
    cu = torch.full((padded_len,), total_len, dtype=torch.int32, device=device)
    cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    # 计算所有段中最大的段长度
    seg_ends = seg_starts[1:] + [total_len]
    max_seqlen = max(end - start for start, end in zip(seg_starts, seg_ends))
    return cu, max_seqlen

# ============================================================================
# 文档打包数据加载器 (DocumentPackingLoader)
# ============================================================================
# 这是训练时使用的主要数据加载器。核心思想是"文档打包"（document packing）：
# 将多个文档紧密拼接在同一个序列中，通过 cu_seqlens 告诉 Flash Attention
# 各文档的边界，使注意力不会跨越文档边界。
#
# 优势：
#   - 相比固定长度序列（ShuffledSequenceLoader），没有 token 浪费（无需 padding）
#   - 每个 batch 的 token 利用率接近 100%
#
# 预取策略：
#   - 使用 ThreadPoolExecutor 异步预读下一个 shard（_shard_pool）
#   - 使用 ThreadPoolExecutor 异步准备下一个 batch（_batch_pool）
#   - 这样 CPU 上的数据准备与 GPU 上的计算可以重叠执行
class DocumentPackingLoader:
    # 单线程池，用于在后台异步加载下一个 shard 文件
    _shard_pool = ThreadPoolExecutor(1)

    def __init__(self, h, device, cu_bucket_size=64):
        self.rank = h.rank
        self.world_size = h.world_size
        self.device = device
        # cu_seqlens 数组的对齐粒度
        self.cu_bucket_size = cu_bucket_size
        self.max_seq_len = h.train_seq_len
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files
        self.file_iter = iter(self.files)
        # 加载第一个 shard 并初始化
        self._init_shard(load_data_shard(next(self.file_iter)))
        # 立即在后台开始加载下一个 shard
        self._next_shard = self._submit_next_shard()
        # 单线程池，用于异步准备下一个 batch
        self._batch_pool = ThreadPoolExecutor(1)
        self._next_batch = None

    # 初始化当前 shard：存储 token 张量，找到所有 BOS token 的位置（文档边界）
    def _init_shard(self, tokens):
        global BOS_ID
        self.tokens = tokens
        self.shard_size = tokens.numel()
        # BOS_ID 默认设为 1（SentencePiece 中通常 BOS=1）
        if BOS_ID is None:
            BOS_ID = 1
        # 找出所有 BOS token 的位置索引，用于后续定位文档边界
        self.bos_idx = (
            (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        )
        # 游标从第一个 BOS 位置开始（跳过 shard 开头可能的不完整文档片段）
        self.cursor = int(self.bos_idx[0])

    # 在后台线程中提交加载下一个 shard 的任务
    # 如果文件迭代器已耗尽，返回 None
    def _submit_next_shard(self):
        try:
            path = next(self.file_iter)
            return self._shard_pool.submit(load_data_shard, path)
        except StopIteration:
            return None

    # 切换到下一个 shard。如果预取的 shard 为 None（所有文件已遍历），
    # 则重新从头开始遍历文件列表（循环读取）。
    def _advance_shard(self):
        if self._next_shard is None:
            # 所有文件已遍历完，重新开始（epoch 循环）
            self.file_iter = iter(self.files)
            self._next_shard = self._shard_pool.submit(
                load_data_shard, next(self.file_iter)
            )
        # 等待预取的 shard 加载完成，然后初始化
        self._init_shard(self._next_shard.result())
        # 继续预取下一个 shard
        self._next_shard = self._submit_next_shard()

    # 在当前 shard 的 bos_idx 中，用二分查找找出 [local_start, local_start+total_len)
    # 范围内的所有 BOS 位置，并转换为相对于 local_start 的偏移量。
    # 这些偏移量将传给 _build_cu_seqlens 以构建注意力掩码边界。
    def _local_doc_starts(self, local_start, total_len):
        lo = np.searchsorted(self.bos_idx, local_start, side="left")
        hi = np.searchsorted(self.bos_idx, local_start + total_len, side="left")
        return (self.bos_idx[lo:hi] - local_start).tolist()

    # 准备一个 batch 的数据。
    # 在多 GPU（DDP）场景下，每个 rank 从 shard 中取不同的连续区间。
    # per_rank_span = num_tokens_local + 1 (+1 是因为 target 比 input 右移一位)
    # global_span = per_rank_span * world_size（所有 rank 总共消耗的 token 数）
    def _prepare_batch(self, num_tokens_local, max_seq_len):
        per_rank_span = num_tokens_local + 1
        global_span = per_rank_span * self.world_size
        # 如果当前 shard 剩余 token 不够一个 global batch，切换到下一个 shard
        while self.cursor + global_span > self.shard_size:
            self._advance_shard()
        # 每个 rank 取自己的区间：rank 0 取前 per_rank_span，rank 1 取接下来的，以此类推
        local_start = self.cursor + self.rank * per_rank_span
        buf = self.tokens[local_start : local_start + per_rank_span]
        # input = buf[:-1]，target = buf[1:]（标准语言模型的 next-token 预测格式）
        inputs = buf[:-1].to(dtype=torch.int64).pin_memory()
        targets = buf[1:].to(dtype=torch.int64).pin_memory()
        # 构建 cu_seqlens：找出本 rank 区间内的文档边界，生成累积序列长度
        starts = self._local_doc_starts(local_start, inputs.numel())
        cu_seqlens, max_seqlen = _build_cu_seqlens(
            starts, inputs.numel(), inputs.device, max_seq_len, self.cu_bucket_size
        )
        # pin_memory() 使后续 .to(device, non_blocking=True) 可以异步传输
        cu_seqlens = cu_seqlens.pin_memory()
        # 移动全局游标，下一次调用从新位置开始
        self.cursor += global_span
        return inputs, targets, cu_seqlens, max_seqlen

    # 获取下一个 batch，支持异步预取。
    # 第一次调用时同步准备 batch，之后每次返回上一步异步准备好的 batch，
    # 同时在后台开始准备下一个 batch，实现 CPU-GPU 流水线。
    def next_batch(self, global_tokens, grad_accum_steps):
        # 计算每个 rank 每次梯度累积步骤需要的 token 数
        num_tokens_local = global_tokens // (self.world_size * grad_accum_steps)
        if self._next_batch is not None:
            # 获取异步准备好的 batch 结果
            inputs, targets, cu_seqlens, max_seqlen = self._next_batch.result()
        else:
            # 第一次调用，同步准备
            inputs, targets, cu_seqlens, max_seqlen = self._prepare_batch(
                num_tokens_local, self.max_seq_len
            )
        # 在后台线程中开始准备下一个 batch（预取）
        self._next_batch = self._batch_pool.submit(
            self._prepare_batch, num_tokens_local, self.max_seq_len
        )
        # 添加 batch 维度 [None] 并异步传输到 GPU
        return (
            inputs[None].to(self.device, non_blocking=True),
            targets[None].to(self.device, non_blocking=True),
            cu_seqlens.to(self.device, non_blocking=True),
            max_seqlen,
        )


# ============================================================================
# 随机序列数据加载器 (ShuffledSequenceLoader)
# ============================================================================
# 与 DocumentPackingLoader 不同，此加载器将数据切分为固定长度（seq_len）的序列，
# 然后随机打乱顺序。不使用 document packing 和 cu_seqlens，因此注意力会跨越
# 文档边界（这是一种简化方案，适用于不需要严格文档边界隔离的场景）。
#
# 每个 rank 分到不同的 shard 文件子集（交错分配），实现数据并行。
# 使用 numpy memmap 进行随机访问，避免一次性将所有数据加载到内存。
class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        # 交错分配 shard 文件给各 rank：rank 0 取第 0, world_size, 2*world_size... 个文件
        self.files = all_files[h.rank :: h.world_size]
        # 每个 rank 使用自己的随机数生成器（以 rank 为种子），确保各 rank 数据不重复
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        # 预读每个 shard 的 token 数量（仅读头部，使用缓存）
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        # 为每个 shard 生成随机打乱的序列起始索引列表
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    # 重置某个 shard 的序列索引：
    # 1. 随机选一个 phase（起始偏移），使每次 epoch 的切分位置不同，增加数据多样性
    # 2. 计算该 shard 能切出多少个完整序列
    # 3. 生成随机排列的序列起始位置列表
    def _reset_shard(self, si):
        # phase 最大为 seq_len-1，确保不浪费超过一个序列长度的数据
        max_phase = min(
            self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1)
        )
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        # 减 1 是因为 target 需要多一个 token（右移一位）
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        # 生成所有序列的起始索引（已打乱顺序）
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    # 获取下一个 batch。
    # 从多个 shard 中按剩余序列数的概率采样序列，保证各 shard 被均匀消耗。
    def next_batch(self, global_tokens, grad_accum_steps):
        # 计算本设备需要的 token 数和 batch size
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        # remaining[si] = 第 si 个 shard 还剩多少个未使用的序列
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            # 如果所有 shard 的序列都用完了，重置所有 shard（开始新 epoch）
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array(
                    [len(s) for s in self.start_inds], dtype=np.float64
                )
                total = remaining.sum()
            # 按各 shard 剩余序列数的比例采样，确保均匀消耗
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            # 从选中的 shard 中弹出一个序列起始索引
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            # 通过 memmap 随机访问读取 seq_len+1 个 token
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(
                np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64)
            )
            # input/target 错位一个 token
            x[bi] = window[:-1]
            y[bi] = window[1:]
        # 异步传输到 GPU
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# ============================================================================
# 基础模块 (Basic Modules)
# ============================================================================

# RMSNorm（Root Mean Square Normalization）—— 一种比 LayerNorm 更轻量的归一化方式。
# 与 LayerNorm 不同，RMSNorm 不减去均值，只除以 RMS（均方根），也没有可学习的 bias。
# 这里直接调用 PyTorch 内置的 F.rms_norm，无可学习参数（无 gamma/beta）。
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


# CastedLinear —— 带自动类型转换的线性层。
# 在混合精度训练中，权重可能以 float32 存储（用于优化器更新的精度），
# 但前向传播时需要转换为输入的 dtype（通常是 bfloat16）以节省显存和计算量。
# 继承自 nn.Linear，仅重写 forward 方法添加类型转换逻辑。
class CastedLinear(nn.Linear):
    def forward(self, x):
        # 将权重和偏置转换为输入的 dtype（例如 bfloat16）
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


# ============================================================================
# Triton 融合矩阵乘 + Leaky ReLU² 内核
# (Fused Linear + Leaky ReLU Square Triton Kernel)
# ============================================================================
# 这是一个使用 Triton TMA（Tensor Memory Accelerator）描述符的融合内核，
# 将矩阵乘法与 Leaky ReLU Square 激活函数合并在一个 GPU kernel 中执行。
#
# Leaky ReLU Square 激活函数定义：
#   f(x) = x²          当 x > 0 时
#   f(x) = (0.5x)²     当 x <= 0 时  （即 leaky_relu 后平方）
#
# 融合的好处：避免中间结果写回全局显存，减少显存带宽消耗。
#
# 前向传播 (FORWARD=True)：
#   1. 计算矩阵乘 c = a @ b.T
#   2. 对 c 应用 leaky_relu_square 激活，结果存入 aux（用于反向传播）
#   3. c 存储未激活的原始矩阵乘结果（pre-activation）
#
# 反向传播 (FORWARD=False)：
#   1. 计算 grad_output @ weight 的矩阵乘
#   2. 乘以 leaky_relu_square 的导数（从 aux 中读取前向保存的 pre-activation 值）
#   导数：d/dx [leaky_relu(x)²] = 2*leaky_relu(x) * leaky_relu'(x)
#         = 2x (x>0) 或 0.5x (x<=0)（即 tl.where(pre>0, 2*pre, 0.5*pre)）
#
# 输出张量 c 的布局：采用 (M, 2, N//2) → permute → (M, N//2, 2) → split 的交错布局，
# 将 N 维度的奇偶列分开处理，可能是为了优化内存访问模式或配合后续计算。
@triton.jit
def linear_leaky_relu_square_kernel(
    # TMA 描述符：描述张量的形状和分块方式，Triton 自动处理 DMA 加载/存储
    a_desc,       # 输入矩阵 A 的 TMA 描述符，形状 (M, K)
    b_desc,       # 权重矩阵 B 的 TMA 描述符，形状 (N, K)
    c_desc,       # 输出矩阵 C 的 TMA 描述符，形状 (M, N//2) × 2 半块
    aux_desc,     # 辅助矩阵的 TMA 描述符（前向存激活值，反向存 pre-activation）
    M,            # A 的行数（batch × seq_len）
    N,            # B 的行数 / 输出列数（隐藏层维度）
    K,            # 内积维度（输入特征维度）
    BLOCK_SIZE_M: tl.constexpr,  # M 维度的分块大小
    BLOCK_SIZE_N: tl.constexpr,  # N 维度的分块大小
    BLOCK_SIZE_K: tl.constexpr,  # K 维度的分块大小（内积累积）
    NUM_SMS: tl.constexpr,       # GPU 上的 SM（流多处理器）数量
    FORWARD: tl.constexpr,       # 编译期常量：True=前向，False=反向
):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    # 计算 M 和 N 维度各需要多少个 tile
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    # 持续流式调度：每个 SM 依次处理多个 tile，步长为 NUM_SMS（SM 总数）
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # 将一维 tile_id 映射到 (pid_m, pid_n) 二维 tile 坐标
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        # 在 float32 精度下累积矩阵乘（避免 bfloat16 精度损失）
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # 沿 K 维度分块累积：a[M_block, K_block] @ b[N_block, K_block].T
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        offs_am_c = offs_am
        offs_bn_c = offs_bn
        # 将累积结果从 (M, N) 重排为交错布局 (M, N//2, 2)，然后拆分为两半
        # 这种布局优化了后续对 N 维度偶数/奇数列的独立处理
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        c1 = acc1.to(dtype)
        if not FORWARD:
            # 反向传播：从 aux 中加载前向保存的 pre-activation 值
            pre0 = aux_desc.load([offs_am_c, offs_bn_c])
            pre1 = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            # 乘以 leaky_relu_square 的导数：
            # d/dx [leaky_relu(x)]² = 2*x (x>0) 或 2*0.25*x = 0.5*x (x<=0)
            c0 = c0 * tl.where(pre0 > 0, 2.0 * pre0, 0.5 * pre0)
            c1 = c1 * tl.where(pre1 > 0, 2.0 * pre1, 0.5 * pre1)
        # 将结果的两半分别存储到输出矩阵的前半段和后半段
        c_desc.store([offs_am_c, offs_bn_c], c0)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        if FORWARD:
            # 前向传播：计算 leaky_relu_square 激活值并存入 aux
            # leaky_relu(x) = x (x>0) 或 0.5*x (x<=0)
            aux0 = tl.where(c0 > 0, c0, 0.5 * c0)
            aux1 = tl.where(c1 > 0, c1, 0.5 * c1)
            # 平方后存储：aux = leaky_relu(x)²，这是该层的最终激活输出
            aux_desc.store([offs_am_c, offs_bn_c], aux0 * aux0)
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], aux1 * aux1)


# Triton 内核的 Python 包装函数。
# 负责分配输出张量、创建 TMA 描述符、计算 grid 大小，然后启动 Triton 内核。
#
# 参数:
#   a: 输入矩阵 (M, K)
#   b: 权重矩阵 (N, K)，注意是 (N, K) 而非 (K, N)
#   aux: 可选。为 None 时表示前向传播（分配新的 aux 张量存储激活值）；
#        不为 None 时表示反向传播（aux 包含前向保存的 pre-activation 值）。
#
# 返回:
#   前向: (c, aux) — c 是 pre-activation，aux 是 leaky_relu_square 激活输出
#   反向: c — 已乘以激活导数的梯度
def linear_leaky_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    forward = aux is None
    if aux is None:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 获取当前 GPU 的 SM 数量，用于持续流式调度
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    # 分块大小：M=128, N=256, K=64（针对 bfloat16 矩阵乘的经验值）
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    # 前向使用 4 级流水线，反向 3 级（反向需要额外读取 aux，寄存器压力更大）
    num_stages = 4 if forward else 3
    # 创建 TMA 描述符：告诉硬件每个分块的形状，实现高效的 DMA 传输
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    # c 和 aux 以半宽块存储（N//2），配合内核中的交错布局
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    # grid 大小 = min(SM 数量, tile 总数)，实现持续流式调度
    grid = lambda _meta: (
        min(num_sms, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),
    )
    linear_leaky_relu_square_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        aux_desc,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=num_sms,
        FORWARD=forward,
        num_stages=num_stages,
        num_warps=8,  # 每个 block 使用 8 个 warp（256 个线程）
    )
    if forward:
        return c, aux
    return c


# ============================================================================
# 融合 MLP 的 Autograd 包装 (Fused MLP Autograd Function)
# ============================================================================
# 将完整的两层 MLP（线性→激活→线性）封装为 PyTorch 自定义 autograd Function，
# 以便利用上面的 Triton 融合内核。
#
# MLP 前向计算流程：
#   hidden = leaky_relu_square(x @ w1.T)   ← 由 Triton 内核融合完成
#   output = hidden @ w2.T                  ← 普通 F.linear
#
# 反向传播推导：
#   dw2 = grad_output.T @ hidden  （w2 的梯度）
#   d_hidden = grad_output @ w2   （hidden 的梯度）
#   dpre = d_hidden * leaky_relu_square'(pre)  ← Triton 内核融合完成
#   dw1 = dpre.T @ x              （w1 的梯度）
#   dx = dpre @ w1                 （输入 x 的梯度）
class FusedLinearLeakyReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        # 展平为 2D 矩阵：(batch*seq, dim)
        x_flat = x.reshape(-1, x.shape[-1])
        # pre = x @ w1.T（pre-activation），post = leaky_relu_square(pre)
        pre, post = linear_leaky_relu_square(x_flat, w1)
        # 第二层线性变换
        out = F.linear(post, w2)
        # 保存中间结果用于反向传播
        ctx.save_for_backward(x, w1, w2, pre, post)
        # 恢复原始形状（除最后一维外保持不变）
        return out.view(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, w2, pre, post = ctx.saved_tensors
        x_flat = x.reshape(-1, x.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        # 计算 w2 的梯度：dw2 = grad_output.T @ post
        dw2 = grad_output_flat.T @ post
        # 融合计算：(grad_output @ w2) * leaky_relu_square'(pre)
        # 传入 aux=pre 表示反向模式，内核会自动乘以激活函数的导数
        dpre = linear_leaky_relu_square(grad_output_flat, w2.T.contiguous(), aux=pre)
        # 计算 w1 的梯度和输入 x 的梯度
        dw1 = dpre.T @ x_flat
        dx = dpre @ w1
        return dx.view_as(x), dw1, dw2


# 便捷别名：直接调用 FusedLinearLeakyReLUSquareFunction.apply(x, w1, w2) 即可
# 用法：output = FusedLeakyReLUSquareMLP(x, mlp_w1, mlp_w2)
FusedLeakyReLUSquareMLP = FusedLinearLeakyReLUSquareFunction.apply


# ============================================================================
# 旋转位置编码 (Rotary Position Embedding, RoPE)
# ============================================================================
# RoPE 通过对 Q/K 向量施加旋转变换来注入位置信息，使注意力分数自然地包含
# 相对位置关系。此实现还支持 YaRN (Yet another RoPE extensioN) 缩放，
# 用于在推理时将序列长度外推到训练长度之外。
class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0, yarn=True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.yarn = yarn
        # rope_dims 允许只对 head_dim 的前 rope_dims 维度施加旋转编码，
        # 剩余维度不参与旋转（部分旋转 RoPE），当 rope_dims=0 时默认对全部维度旋转
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        # 计算逆频率向量：theta_i = 1 / (base^(2i/d))，这是 RoPE 的核心频率参数
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 缓存机制：避免重复计算 cos/sin 矩阵，仅在序列长度变化时更新
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        # 仅在首次调用、序列变长或设备变更时重新计算 cos/sin 缓存
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            # YaRN 外推：当推理序列长度超过训练长度时，动态调整 base 频率
            # 公式: new_base = base * (seq_len/train_seq_len)^(d/(d-2))
            # 这种缩放方式能保持高频分量不变、平滑拉伸低频分量，从而实现
            # 更好的长度外推效果（相比直接的 NTK-aware 缩放）
            if self.yarn and seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd
                )
            else:
                inv_freq = self.inv_freq.float().to(device)
            # 外积计算每个位置 t 与每个频率的乘积，得到旋转角度矩阵
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            # 缓存形状 [1, seq_len, 1, rope_dims//2]，用于广播到 [B, T, H, D//2]
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :seq_len].to(dtype=dtype), self._sin_cached[:, :seq_len].to(dtype=dtype)


# 将旋转位置编码应用到输入张量 x 上
# 旋转公式：将 x 分成前后两半 (x1, x2)，计算:
#   x_rot = [x1*cos + x2*sin, -x1*sin + x2*cos]
# 这等价于对每对维度施加一个 2D 旋转矩阵。
# 当 rope_dims < head_dim 时，只对前 rope_dims 个维度旋转，其余维度直接透传
# （部分旋转 RoPE，类似 GPT-NeoX 的做法）
def apply_rotary_emb(x, cos, sin, rope_dims=0):
    # 部分旋转：只对前 rope_dims 维度施加 RoPE
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    # 全维度旋转
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


# ============================================================================
# 因果自注意力 (Causal Self-Attention)
# ============================================================================
# 核心自注意力模块，包含以下特性:
# - 分组查询注意力 (GQA): 多个 Q 头共享一组 K/V 头，节省参数和内存
# - QK-Norm: 对 Q 和 K 分别做 RMSNorm，稳定注意力分数的数值范围
# - RoPE 旋转位置编码: 注入相对位置信息
# - Flash Attention: 使用高效 IO-aware 注意力算法
# - XSA (跨子空间注意力): 可选的注意力输出正交化，去除 V 方向的冗余分量
# - 三种互斥的门控机制 (每次只能启用一种):
#   1. AttnOutGate: 基于投影输出或输入的窗口门控
#   2. GatedAttn: 全维度 sigmoid 门控 (arXiv:2505.06708, Qwen 风格)
#   3. SparseAttnGate: 稀疏窗口门控 (modded-nanogpt 风格)
class CausalSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, yarn=True,
        attn_out_gate=False, attn_out_gate_src="proj", gate_window=12,
        gated_attn=False, gated_attn_init_std=0.01,
        sparse_attn_gate=False, sparse_attn_gate_init_std=0.0, sparse_attn_gate_scale=1.0,
    ):
        super().__init__()
        # 验证维度可整除关系
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        # 三种门控机制互斥，最多只能启用一种
        if int(attn_out_gate) + int(gated_attn) + int(sparse_attn_gate) > 1:
            raise ValueError(
                "attn_out_gate, gated_attn, and sparse_attn_gate are mutually exclusive"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        # q_gain: 每个头的可学习增益系数，缩放 Q 向量来控制注意力分数的温度
        # 初始值由 qk_gain_init 控制，用 fp32 确保精度
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len, yarn=yarn)
        # XSA (跨子空间注意力) 标志，默认关闭，由 GPT.__init__ 中根据 xsa_last_n 设置
        self.use_xsa = False
        # AttnOutGate (PR #1667 MarioPaerle): per-head multiplicative gate on attention
        # output. CastedLinear so restore_fp32_params casts back to fp32 for GPTQ.
        # _zero_init -> 2*sigmoid(0)=1 -> transparent at init.
        self.attn_out_gate = attn_out_gate
        self.attn_out_gate_src = attn_out_gate_src
        self.gate_window = gate_window
        if attn_out_gate:
            self.attn_gate_proj = CastedLinear(gate_window, num_heads, bias=False)
            self.attn_gate_proj._zero_init = True
        # Gated Attention (arXiv:2505.06708, Qwen, NeurIPS 2025). Per-head sigmoid
        # gate on SDPA output, BEFORE out_proj. Gate projection W_g: (num_heads, dim).
        # Name "attn_gate_w" contains "attn_gate" substring so it matches
        # CONTROL_TENSOR_NAME_PATTERNS and routes to the scalar AdamW group.
        # fp32 Parameter -> restore_fp32_params path covers it via the ndim<2 OR
        # name-pattern check (name matches "attn_gate"). Cast to x.dtype on use.
        # 门控注意力 (Gated Attention): 来源于 arXiv:2505.06708 和 Qwen 模型
        # 对每个头的 SDPA 输出乘以 sigmoid 门控值: g = sigmoid(x @ W_g^T)
        # W_g 形状 (num_heads, dim)，门控值 g 形状 [B,T,H]，广播到 [B,T,H,D]
        # 使用 fp32 参数，在计算时 cast 到输入 dtype
        self.gated_attn = gated_attn
        if gated_attn:
            W = torch.empty(num_heads, dim, dtype=torch.float32)
            nn.init.normal_(W, mean=0.0, std=gated_attn_init_std)
            self.attn_gate_w = nn.Parameter(W)
        # Sparse attention head-output gate (modded-nanogpt style). Keeps dense SDPA
        # and only narrows the gate input to the first gate_window residual dims.
        # W_g: (num_heads, gate_window). y_{t,h} <- sigmoid(scale * W_g_h @ x_t[:gate_window]) * y_{t,h}.
        # Shares attn_gate_w name with dense GatedAttn so the quant routing
        # (CONTROL_TENSOR_NAME_PATTERNS / attn_gate_w int8 passthrough) is unchanged.
        # 稀疏注意力门控 (Sparse Attention Gate, modded-nanogpt 风格):
        # 与 GatedAttn 类似但更轻量——门控输入只取残差流的前 gate_window 个维度
        # W_g 形状 (num_heads, gate_window)，远小于 GatedAttn 的 (num_heads, dim)
        # 公式: y_{t,h} <- sigmoid(scale * W_g_h @ x_t[:gate_window]) * y_{t,h}
        # 零初始化或小方差初始化确保训练初期门控近似透明 (sigmoid(0)=0.5)
        self.sparse_attn_gate = sparse_attn_gate
        self.sparse_attn_gate_scale = sparse_attn_gate_scale
        if sparse_attn_gate:
            W = torch.empty(num_heads, gate_window, dtype=torch.float32)
            if sparse_attn_gate_init_std > 0:
                nn.init.normal_(W, mean=0.0, std=sparse_attn_gate_init_std)
            else:
                nn.init.zeros_(W)
            self.attn_gate_w = nn.Parameter(W)

    # XSA (Cross-Subspace Attention, 跨子空间注意力):
    # 从注意力输出 y 中减去其在 V 方向上的投影分量。
    # 原理: 在 GQA 中，同一组内的多个 Q 头共享一组 K/V 头。注意力输出 y 可能
    # 在 V 方向上有较大的冗余分量。XSA 通过正交投影去除这个方向上的分量，
    # 迫使每个头学到与 V 空间正交的互补信息，从而提高头的多样性。
    # 数学: y_hat = y - (y · v_norm) * v_norm  (格拉姆-施密特正交化)
    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv  # 每组 KV 头对应的 Q 头数量
        y_g = y.reshape(B, T, Hkv, group, D)  # 按 KV 组重新分组
        vn = F.normalize(v, dim=-1).unsqueeze(-2)  # 归一化 V 向量作为投影方向
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn  # 计算 y 在 v 方向上的投影
        return (y_g - proj).reshape(B, T, H, D)  # 减去投影，保留正交分量

    # 前向传播: 完整的自注意力计算流程
    # 注意: Q/K/V 的权重矩阵从外部传入（来自 GPT 的权重银行），而非模块自身持有
    # 这种设计允许权重在循环层 (looping) 间复用
    def forward(self, x, q_w, k_w, v_w, out_w, cu_seqlens=None, max_seqlen=0):
        bsz, seqlen, dim = x.shape
        # 步骤1: QKV 线性投影
        # q_raw 保留未 reshape 的原始 Q 投影，用于 AttnOutGate 的 src='q' 模式
        q_raw = F.linear(x, q_w.to(x.dtype))
        q = q_raw.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        # 步骤2: QK-Norm (对 Q 和 K 做 RMSNorm)
        # 稳定注意力分数，防止 softmax 的输入值过大导致梯度消失
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # 步骤3: 应用 RoPE 旋转位置编码
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        # 步骤4: Q 增益缩放 (可学习的每头温度参数)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        # 步骤5: Flash Attention 计算
        # cu_seqlens 不为 None 时使用 varlen 版本，支持打包的变长序列（多文档训练）
        if cu_seqlens is not None:
            y = flash_attn_varlen_func(
                q[0],
                k[0],
                v[0],
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1),
            )[None]
        else:
            # 标准 Flash Attention 3 (IO-aware, causal mask)
            y = flash_attn_3_func(q, k, v, causal=True)
        # 步骤6: 可选的 XSA (跨子空间注意力正交化)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        # AttnOutGate inlined (PR #1667). Inline + .contiguous() barrier so torch.compile
        # fullgraph=True is happy (this avoids the @torch.compiler.disable trap that
        # crashed gates v3). Per-head gate on (B,T,H,D) tensor: g shape [B,T,H], broadcast
        # over D via [..., None]. zero-init weight -> 2*sigmoid(0)=1 -> transparent.
        if self.attn_out_gate:
            gate_src = q_raw if self.attn_out_gate_src == "q" else x
            gate_in = gate_src[..., : self.gate_window].contiguous()
            g = 2.0 * torch.sigmoid(self.attn_gate_proj(gate_in))
            y = y * g[..., None]
        # Gated Attention (arXiv:2505.06708 G1). Inline + .contiguous() barrier so
        # torch.compile fullgraph=True is happy. Per-head gate on (B,T,H,D): g shape
        # [B,T,H], broadcast over D via [..., None]. Paper: g = sigmoid(x @ W_g.T)
        # where W_g: (H, dim). .to(x.dtype) on fp32 param before broadcast with bf16.
        if self.gated_attn:
            x_c = x.contiguous()
            g = torch.sigmoid(F.linear(x_c, self.attn_gate_w.to(x.dtype)))
            y = y * g[..., None]
        # Sparse head-output gate: narrower (gate_window) input, same shape g as GatedAttn.
        if self.sparse_attn_gate:
            gate_in = x[..., : self.gate_window].contiguous()
            g = torch.sigmoid(
                self.sparse_attn_gate_scale
                * F.linear(gate_in, self.attn_gate_w.to(x.dtype))
            )
            y = y * g[..., None]
        # 步骤7: reshape 回 [B, T, dim] 并执行输出投影
        y = y.reshape(bsz, seqlen, dim)
        # _last_proj_input 用于量化校准 (GPTQ 等)，正常训练时为 None
        self._last_proj_input = y.detach() if getattr(self, "_calib", False) else None
        return F.linear(y, out_w.to(x.dtype))


# ============================================================================
# MLP (前馈网络)
# ============================================================================
# 使用 LeakyReLU-Square 激活函数的两层前馈网络: down(leaky_relu(up(x))^2)
# LeakyReLU-Square: 先做 LeakyReLU(slope=0.5) 再平方。平方操作使激活函数
# 变为多项式形式，增加了表达能力（类似 SwiGLU 的效果但更简单）。
# 训练时使用融合 kernel (FusedLeakyReLUSquareMLP) 以减少中间张量的内存占用
# 和 GPU 读写次数。推理时用非融合版本以便兼容量化校准。
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.use_fused = True  # 训练时使用自定义 autograd 融合算子

    def forward(self, x, up_w, down_w):
        # 训练路径: 融合 kernel 将 up_proj -> leaky_relu -> square -> down_proj
        # 合并为一个 autograd Function，避免保存巨大的中间激活
        if self.training and self.use_fused:
            return FusedLeakyReLUSquareMLP(x, up_w.to(x.dtype), down_w.to(x.dtype))
        # 推理/校准路径: 分步计算以便提取 down_proj 的输入用于量化校准
        hidden = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5).square()
        self._last_down_input = hidden.detach() if getattr(self, "_calib", False) else None
        return F.linear(hidden, down_w.to(x.dtype))


# ============================================================================
# Block (Transformer 块)
# ============================================================================
# 一个完整的 Transformer 层，包含:
# 1. 残差混合 (resid_mix): 将当前隐藏状态 x 与初始嵌入 x0 加权混合
#    mix[0]*x + mix[1]*x0，初始化为 [1,0] 即只用当前状态
# 2. 注意力子层: RMSNorm -> CausalSelfAttention -> 残差连接 (带 attn_scale 缩放)
# 3. MLP 子层: RMSNorm -> MLP -> 残差连接 (带 mlp_scale 缩放)
# 4. ln_scale_factor: 可选的层归一化缩放因子 1/sqrt(layer_idx+1)，
#    随层数递增而递减，帮助深层网络稳定训练 (类似 DeepNet 的 Post-Norm 缩放思想)
# 注意: Q/K/V/Out/Up/Down 权重从外部传入，Block 本身不持有这些大型权重矩阵
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        train_seq_len,
        layer_idx=0,
        ln_scale=False,
        yarn=True,
        attn_out_gate=False,
        attn_out_gate_src="proj",
        gate_window=12,
        gated_attn=False,
        gated_attn_init_std=0.01,
        sparse_attn_gate=False,
        sparse_attn_gate_init_std=0.0,
        sparse_attn_gate_scale=1.0,
    ):
        super().__init__()
        # 注意力子层和 MLP 子层各自的 Pre-Norm (RMSNorm)
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, yarn=yarn,
            attn_out_gate=attn_out_gate, attn_out_gate_src=attn_out_gate_src, gate_window=gate_window,
            gated_attn=gated_attn, gated_attn_init_std=gated_attn_init_std,
            sparse_attn_gate=sparse_attn_gate,
            sparse_attn_gate_init_std=sparse_attn_gate_init_std,
            sparse_attn_gate_scale=sparse_attn_gate_scale,
        )
        self.mlp = MLP(dim, mlp_mult)
        # attn_scale / mlp_scale: 可学习的逐维度残差缩放因子，初始化为 1
        # 允许模型自适应地调节每个维度上注意力/MLP 贡献的比例
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        # resid_mix: 残差混合系数 [2, dim]，控制输入到本层时 x 与 x0 的混合比例
        # mix[0] 是当前状态权重 (初始=1)，mix[1] 是初始嵌入权重 (初始=0)
        # 允许模型学习"回看"初始嵌入 x0，类似一种持久的跳跃连接
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        # ln_scale_factor: 层归一化输出的缩放因子
        # 当 ln_scale=True 时为 1/sqrt(layer_idx+1)，深层的缩放更小
        # 这帮助控制深层网络中信号幅度的增长
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    # Block 前向传播: x 是当前层输入, x0 是初始归一化嵌入 (用于 resid_mix)
    # 权重矩阵 q_w/k_w/v_w/out_w/up_w/down_w 从外部权重银行传入
    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=None, max_seqlen=0):
        # 将当前隐藏状态与初始嵌入按学习到的比例混合
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # 注意力子层: Pre-Norm -> Attention -> 缩放残差加
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w, k_w, v_w, out_w,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        # MLP 子层: Pre-Norm -> MLP -> 缩放残差加
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[
            None, None, :
        ] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out

# ============================================================================
# GPT 模型
# ============================================================================
# 核心语言模型架构，融合了多项高级技术:
#
# 【权重银行 (Weight Banks)】
#   所有层的 QKV/Out/MLP 权重不是放在各层 Block 中，而是集中存储在 4 个大的
#   参数张量中 (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank)。
#   qo_bank[i] = Q 权重, qo_bank[n+i] = O 权重 (output projection)
#   kv_bank[i] = K 权重, kv_bank[n+i] = V 权重
#   这种设计便于权重循环复用 (looping) 和统一的初始化策略。
#
# 【编码器/解码器分割 (Encoder-Decoder Split)】
#   模型的层被分为前半的"编码器"和后半的"解码器"。编码器层的输出
#   以 LIFO (后进先出) 的方式通过跳跃连接注入到解码器层中，
#   形成类似 U-Net 的对称结构。这种跳跃连接帮助梯度流动，
#   并让解码器可以直接访问不同深度的特征。
#
# 【层循环 (Looping)】
#   可以将部分层 [loop_start, loop_end] 重复执行 num_loops 次，
#   实现深度上的参数复用——同一组权重被多次使用，
#   增加了有效深度而不增加参数量。
#
# 【并行双通道 (Parallel Lanes)】
#   从 parallel_start_layer 开始，计算分裂为两条并行通道:
#   - lane0 (注意力通道): 接收注意力输入
#   - lane1 (MLP 通道): 接收 MLP 输入
#   两条通道通过可学习的 lambda 系数交叉混合:
#   lane0 = resid_attn * lane0 + post_attn[0] * attn_out + post_mlp[0] * mlp_out
#   lane1 = resid_mlp * lane1 + post_attn[1] * attn_out + post_mlp[1] * mlp_out
#   最终通过 parallel_final_lane 选择输出通道或取平均。
#
# 【SmearGate (嵌入层向前涂抹门控)】
#   对嵌入序列做一步前向涂抹: x_t <- x_t + lam * sigmoid(W * x_t[:w]) * x_{t-1}
#   让当前 token 的嵌入可以混入前一个 token 的信息，
#   提供一种轻量级的局部上下文预混合。BOS token 位置被 mask 掉以保持因果性。
class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        # logit_softcap: 对 logits 做 softcap 限幅 (tanh 缩放)，防止 logits 过大
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.fused_ce_enabled = bool(h.fused_ce_enabled)
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.num_layers = h.num_layers
        head_dim = h.model_dim // h.num_heads
        kv_dim = h.num_kv_heads * head_dim
        hidden_dim = int(h.mlp_mult * h.model_dim)
        # 权重银行 (Weight Banks):
        # qo_bank: 前 n 层存 Q 权重 [dim, dim]，后 n 层存 Output 投影权重 [dim, dim]
        self.qo_bank = nn.Parameter(torch.empty(2 * h.num_layers, h.model_dim, h.model_dim))
        # kv_bank: 前 n 层存 K 权重 [kv_dim, dim]，后 n 层存 V 权重 [kv_dim, dim]
        self.kv_bank = nn.Parameter(torch.empty(2 * h.num_layers, kv_dim, h.model_dim))
        # mlp_up_bank / mlp_down_bank: MLP 上投影和下投影权重
        self.mlp_up_bank = nn.Parameter(torch.empty(h.num_layers, hidden_dim, h.model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(h.num_layers, h.model_dim, hidden_dim))
        # 编码器/解码器分割: 前半层为编码器，后半层为解码器
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        # 实例化所有 Transformer 块 (层归一化、注意力、MLP 的非权重参数)
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                    yarn=h.rope_yarn,
                    attn_out_gate=h.attn_out_gate_enabled,
                    attn_out_gate_src=h.attn_out_gate_src,
                    gate_window=h.gate_window,
                    gated_attn=h.gated_attn_enabled,
                    gated_attn_init_std=h.gated_attn_init_std,
                    sparse_attn_gate=h.sparse_attn_gate_enabled,
                    sparse_attn_gate_init_std=h.sparse_attn_gate_init_std,
                    sparse_attn_gate_scale=h.sparse_attn_gate_scale,
                )
                for i in range(h.num_layers)
            ]
        )
        # 如果指定了 rope_dims > 0，则为每个块重新创建 Rotary 模块以支持部分旋转 RoPE
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim,
                    base=h.rope_base,
                    train_seq_len=h.train_seq_len,
                    rope_dims=h.rope_dims,
                    yarn=h.rope_yarn,
                )
        # 最终层归一化
        self.final_norm = RMSNorm()
        # 语言模型头: 如果 tie_embeddings 则复用 tok_emb.weight，否则单独的线性层
        # 零初始化确保训练初期输出分布均匀，减少初始损失的波动
        self.lm_head = (
            None
            if h.tie_embeddings
            else CastedLinear(h.model_dim, h.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        # XSA (跨子空间注意力): 对最后 xsa_last_n 层启用
        # 只在深层使用，因为浅层需要保留 V 方向的信息来建立基础表征
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        # 层循环 (Layer Looping): 通过重复执行部分层来增加有效深度
        # 例如: 8 层网络, loop_start=2, loop_end=5, num_loops=1
        # 原始索引: [0,1,2,3,4,5,6,7]
        # 循环后: [0,1, 2,3,4,5, 2,3,4,5, 6,7] (层 2-5 重复 2 次)
        # 然后按半分拆成编码器/解码器索引
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):  # +1 因为原始也算一次
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        # U-Net 风格跳跃连接 (Skip Connections):
        # 编码器层的输出按 LIFO 顺序与对应的解码器层输入融合
        # skip_weights: 逐维度缩放跳跃连接 (初始=1)
        # skip_gates: 可选的 sigmoid 门控，通过 lerp 控制跳跃连接的混合强度
        #   当 skip_gates 启用时: lane = lerp(w*skip, lane, sigmoid(gate))
        #   当 skip_gates 未启用时: lane = lane + w*skip
        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
            )
            if h.skip_gates_enabled
            else None
        )
        # 并行双通道 (Parallel Lanes) 配置:
        # parallel_start_layer: 从哪一层开始分裂为双通道
        # parallel_final_lane: 最终输出选择 "attn"/"mlp"/"avg"
        self.parallel_start_layer = h.parallel_start_layer
        self.parallel_final_lane = h.parallel_final_lane.lower()
        # parallel_post_lambdas [num_layers, 2, 2]: 控制 attn_out 和 mlp_out 对两条通道的贡献
        #   [layer, 0, :] = attn 通道的 [attn_post, mlp_post] 系数
        #   [layer, 1, :] = mlp 通道的 [attn_post, mlp_post] 系数
        self.parallel_post_lambdas = nn.Parameter(
            torch.ones(h.num_layers, 2, 2, dtype=torch.float32)
        )
        # parallel_resid_lambdas [num_layers, 2]: 两条通道各自的残差缩放系数
        # 初始值 1.1 (略大于 1) 以轻微放大残差路径，帮助早期训练稳定
        self.parallel_resid_lambdas = nn.Parameter(
            torch.full((h.num_layers, 2), 1.1, dtype=torch.float32)
        )
        # SmearGate (PR #1667 / modded-nanogpt @classiclarryd):
        #   x_t <- x_t + lam * sigmoid(W * x_t[:gate_window]) * x_{t-1}.
        # Per-token forward-1 smear of the embedding lane. W zero-init + lam=0 ->
        # transparent at init. Uses CastedLinear so restore_fp32_params handles dtype.
        self.smear_gate_enabled = h.smear_gate_enabled
        if self.smear_gate_enabled:
            self.smear_window = h.gate_window
            self.smear_gate = CastedLinear(self.smear_window, 1, bias=False)
            self.smear_gate._zero_init = True
            self.smear_lambda = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self._init_weights()

    # 权重初始化策略:
    # - Q/K/V 权重: 正交初始化 (gain=1.0)，保持信号的范数不变
    # - O/Down 权重: 零初始化后乘以 1/sqrt(2n) 的缩放因子
    #   零初始化 O 和 Down 让残差分支初期对主干贡献为零，
    #   1/sqrt(2n) 缩放确保随层数增加不会爆炸 (类似 fixup 初始化)
    # - Up 权重: 正交初始化
    # - 标记 _zero_init 的 Linear: 零初始化 (如 lm_head, SmearGate)
    # - 其他足够大的 Linear: 正交初始化
    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        # proj_scale: 残差输出投影 (O/Down) 的缩放因子，防止深层残差累加过大
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            # Q 权重: 正交初始化
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            # O 权重 (输出投影): 零初始化 + 缩放，训练初期注意力分支贡献为零
            nn.init.zeros_(self.qo_bank.data[n + i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            # K/V 权重: 正交初始化
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
        for i in range(n):
            # Up 权重: 正交初始化
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            # Down 权重: 零初始化 + 缩放，训练初期 MLP 分支贡献为零
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.mlp_down_bank.data[i].mul_(proj_scale)
        # 其他子模块的初始化
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    # 从权重银行中提取第 i 层所需的全部权重
    # 返回顺序: (Q权重, K权重, V权重, O权重, MLP上投影, MLP下投影)
    # 注意索引映射: Q=qo_bank[i], O=qo_bank[n+i], K=kv_bank[i], V=kv_bank[n+i]
    def _bank_weights(self, i):
        n = self.num_layers
        return (
            self.qo_bank[i],       # Q 权重
            self.kv_bank[i],       # K 权重
            self.kv_bank[n + i],   # V 权重
            self.qo_bank[n + i],   # Output 投影权重
            self.mlp_up_bank[i],   # MLP 上投影权重
            self.mlp_down_bank[i], # MLP 下投影权重
        )

    # 并行双通道块 (Parallel Block):
    # 与标准 Block 不同，这里注意力和 MLP 从不同通道读取输入并行计算，
    # 然后通过可学习的 lambda 系数将结果交叉混合回两条通道。
    # 这种设计让注意力和 MLP 在信息流上部分解耦，
    # 各自专注于不同类型的特征提取，同时通过交叉项保持信息交互。
    def _parallel_block(
        self, block_idx, lane0, lane1, x0,
        q_w, k_w, v_w, out_w, up_w, down_w,
        cu_seqlens=None, max_seqlen=0,
    ):
        block = self.blocks[block_idx]
        # 注意力从 lane0 (注意力通道) 读取，并与 x0 混合
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        attn_out = block.attn(
            block.attn_norm(attn_read) * block.ln_scale_factor,
            q_w, k_w, v_w, out_w,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        # MLP 从 lane1 (MLP 通道) 读取
        mlp_read = lane1
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * block.mlp(
            block.mlp_norm(mlp_read) * block.ln_scale_factor, up_w, down_w
        )
        # 获取此层的可学习混合系数
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        # 交叉混合: 每条通道 = 残差缩放*自身 + attn贡献 + mlp贡献
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1

    # 从两条并行通道中选择最终的隐藏状态
    # "mlp": 只用 MLP 通道; "attn": 只用注意力通道; 默认: 两者取平均
    def _final_parallel_hidden(self, lane0, lane1):
        if self.parallel_final_lane == "mlp":
            return lane1
        if self.parallel_final_lane == "attn":
            return lane0
        return 0.5 * (lane0 + lane1)

    # 核心隐藏状态计算: 从 token 嵌入经过编码器/解码器栈到最终 RMSNorm
    # 此方法被 forward_logits (推理) 和 forward (训练) 共用
    def _forward_hidden(self, input_ids, cu_seqlens=None, max_seqlen=0):
        """Run the encoder/decoder stack to the final RMSNorm; returns pre-projection hidden.
        Shared by eval (softcap+projection via forward_logits) and train (fused CE path)."""
        x = self.tok_emb(input_ids)
        # SmearGate (PR #1667). Inline gate compute with .contiguous() on the slice fed
        # to the projection so torch.compile fullgraph is happy. lam=0 + W=0 -> identity
        # at init. This block runs unconditionally on the smear path; the cat keeps
        # position 0 untouched so causality holds.
        if self.smear_gate_enabled:
            sl = self.smear_lambda.to(dtype=x.dtype)
            gate_in = x[:, 1:, : self.smear_window].contiguous()
            g = sl * torch.sigmoid(self.smear_gate(gate_in))
            bos_mask = (input_ids[:, 1:] == 1).unsqueeze(-1)
            g = g.masked_fill(bos_mask, 0.0)
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
        # 嵌入后 RMSNorm，并保存 x0 作为后续 resid_mix 的初始锚点
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x  # 初始归一化嵌入，贯穿所有层用于 resid_mix
        skips = []  # 编码器层输出的栈，用于 U-Net 跳跃连接
        # 根据是否启用 looping 来决定编码器/解码器的层索引迭代器
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else range(self.num_encoder_layers)
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else range(
                self.num_encoder_layers,
                self.num_encoder_layers + self.num_decoder_layers,
            )
        )
        # ---- 编码器阶段 ----
        # 每层输出压入 skips 栈，后续解码器以 LIFO 方式弹出使用
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            skips.append(x)
        # ---- 解码器阶段 ----
        # 解码器层依次处理，同时消费 skips 栈中的跳跃连接
        # 当层索引 >= parallel_start_layer 时切换到并行双通道模式
        psl = self.parallel_start_layer
        lane0 = None  # 注意力通道 (并行模式)
        lane1 = None  # MLP 通道 (并行模式)
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            # 并行双通道路径: 层索引 >= parallel_start_layer 时激活
            if i >= psl and psl > 0:
                # 首次进入并行模式时，用当前 x 初始化两条通道
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                # 跳跃连接注入到 lane0 (注意力通道)
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        # 门控跳跃连接: lerp(加权skip, lane0, gate)
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                # 并行块处理: 注意力和 MLP 分别从不同通道读取并交叉混合
                lane0, lane1 = self._parallel_block(
                    i, lane0, lane1, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                )
            else:
                # 串行路径 (parallel_start_layer 之前的解码器层)
                # 跳跃连接直接加到 x 上
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        # 如果使用了并行双通道，合并为最终隐藏状态
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        # 最终层归一化
        x = self.final_norm(x)
        return x

    # 将隐藏状态投影到词表维度
    def _project_logits(self, hidden):
        if self.tie_embeddings:
            return F.linear(hidden, self.tok_emb.weight)  # 与嵌入层共享权重
        return self.lm_head(hidden)

    # 推理路径: 返回 softcapped logits
    # softcap: logit_softcap * tanh(logits / logit_softcap)，将 logits 限制在 [-cap, cap] 范围内
    def forward_logits(self, input_ids, cu_seqlens=None, max_seqlen=0):
        hidden = self._forward_hidden(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        logits_proj = self._project_logits(hidden)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    # 训练路径: 计算交叉熵损失
    # 支持两种模式:
    # 1. 融合 CE: 使用自定义 Triton kernel 将 softcap + cross_entropy 融合为一步
    # 2. 非融合 CE: 先在 fp32 中计算 softcap，再调用标准 cross_entropy
    def forward(self, input_ids, target_ids, cu_seqlens=None, max_seqlen=0):
        hidden = self._forward_hidden(input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        logits_proj = self._project_logits(hidden)
        flat_targets = target_ids.reshape(-1)
        # Fused softcapped-CE kernel (training path only). Applies softcap inside the
        # Triton kernel; takes pre-softcap logits_proj. Non-fused path matches stock
        # PR-1736 numerics exactly (softcap in fp32, then F.cross_entropy on fp32).
        if self.fused_ce_enabled:
            return softcapped_cross_entropy(
                logits_proj.reshape(-1, logits_proj.size(-1)),
                flat_targets,
                self.logit_softcap,
                reduction="mean",
            )
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            flat_targets,
            reduction="mean",
        )

    # ============================================================================
    # TTT (Test-Time Training) 前向传播
    # ============================================================================
    # TTT 在推理时对模型进行在线微调，通过 LoRA 适配器在每个文档上做临时的
    # 参数更新。此方法是 TTT 阶段的前向传播，与标准 forward 的区别:
    # 1. 每层的 Q/V/O/MLP 投影叠加了 LoRA 残差: out = base_proj(x) + lora(x)
    # 2. K 投影可选地也叠加 LoRA
    # 3. 最终 logits 也叠加了 lm_head 的 LoRA 残差
    # 4. 返回逐 token 的损失 [B, T] (而非 mean)，供外层 TTT 优化器使用
    # 5. 不使用融合 CE kernel (TTT 需要逐 token 损失)
    # slot 变量跟踪当前的 LoRA 槽位索引 (在循环层中同一物理层可能有多个槽位)
    def forward_ttt(self, input_ids, target_ids, lora):
        x = self.tok_emb(input_ids)
        # SmearGate on the TTT path — same inline compute as forward_logits.
        if self.smear_gate_enabled:
            sl = self.smear_lambda.to(dtype=x.dtype)
            gate_in = x[:, 1:, : self.smear_window].contiguous()
            g = sl * torch.sigmoid(self.smear_gate(gate_in))
            bos_mask = (input_ids[:, 1:] == 1).unsqueeze(-1)
            g = g.masked_fill(bos_mask, 0.0)
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else list(range(self.num_encoder_layers))
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else list(
                range(
                    self.num_encoder_layers,
                    self.num_encoder_layers + self.num_decoder_layers,
                )
            )
        )
        # slot: LoRA 槽位计数器，每经过一个物理层递增
        # 在 looping 模式下，同一物理层被多次执行时使用不同的 slot
        slot = 0
        # ---- 编码器阶段 (TTT with LoRA) ----
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w)
            slot += 1
            skips.append(x)
        # ---- 解码器阶段 (TTT with LoRA) ----
        # 逻辑与 _forward_hidden 完全一致，区别在于使用 _block_with_lora 和
        # _parallel_block_with_lora 替代原生的 block.forward 和 _parallel_block
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                # 跳跃连接 (同 _forward_hidden)
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                # 并行块 + LoRA
                lane0, lane1 = self._parallel_block_with_lora(
                    i, lane0, lane1, x0, lora, slot,
                    q_w, k_w, v_w, out_w, up_w, down_w,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                # 串行块 + LoRA
                x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w)
            slot += 1
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        x = self.final_norm(x)
        # Logits 计算: 基础投影 + lm_head 的 LoRA 残差
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        # LoRA 对 lm_head 的增量调整
        logits = logits + lora.lm_head_lora(x)
        # Softcap 限幅
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        bsz, sl, V = logits.shape
        # 返回 [B, T] 形状的逐 token 损失 (reduction="none")
        # TTT 优化器需要逐 token 损失来计算梯度
        return F.cross_entropy(
            logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
        ).reshape(bsz, sl)

    # ============================================================================
    # _block_with_lora: 带 LoRA 适配器的串行 Transformer 块
    # ============================================================================
    # 与 Block.forward 的逻辑完全对应，但在以下投影上叠加了 LoRA 残差:
    # - Q 投影: q = base_q + lora.q_loras[slot](n)
    # - K 投影: k = base_k + lora.k_loras[slot](n)  (可选)
    # - V 投影: v = base_v + lora.v_loras[slot](n)
    # - O 投影: attn_out = base_o + lora.o_loras[slot](n)  (可选)
    # - MLP: mlp_out = base_mlp + lora.mlp_loras[slot](n)  (可选)
    # 所有门控机制 (AttnOutGate, GatedAttn, SparseAttnGate) 的行为
    # 与训练路径完全一致，确保 TTT 时的语义匹配
    def _block_with_lora(self, block, x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w):
        mix = block.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = block.attn_norm(x_in) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        # Q 投影 = 基础投影 + LoRA 残差 (Q 的 LoRA 始终启用)
        q_raw = F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)
        q = q_raw.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
        # K 投影: LoRA 可选 (某些配置可能不对 K 做 LoRA)
        k = F.linear(n, k_w.to(n.dtype))
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        # V 投影 = 基础投影 + LoRA 残差
        v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        # QK-Norm -> RoPE -> Q 增益 (与标准路径完全一致)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # AttnOutGate (TTT path) — inline + .contiguous() barrier, same as the eval path.
        if attn.attn_out_gate:
            gate_src = q_raw if attn.attn_out_gate_src == "q" else n
            gate_in = gate_src[..., : attn.gate_window].contiguous()
            g = 2.0 * torch.sigmoid(attn.attn_gate_proj(gate_in))
            y = y * g[..., None]
        # Gated Attention (TTT path). Gate input is n (post-norm block input), same
        # as eval path. .to(n.dtype) on fp32 param before bf16 broadcast.
        if attn.gated_attn:
            n_c = n.contiguous()
            g = torch.sigmoid(F.linear(n_c, attn.attn_gate_w.to(n.dtype)))
            y = y * g[..., None]
        # Sparse attention head-output gate (TTT path) — must match the eval path in
        # forward() exactly, else training (which applied the gate) and TTT eval (which
        # skipped it) produce mismatched representations and catastrophic BPB regression.
        if attn.sparse_attn_gate:
            gate_in = n[..., : attn.gate_window].contiguous()
            g = torch.sigmoid(
                attn.sparse_attn_gate_scale
                * F.linear(gate_in, attn.attn_gate_w.to(n.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        # O 投影 + 可选的 LoRA 残差
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        # 注意力残差连接 (带 attn_scale 缩放)
        x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        # MLP 子层 + 可选的 LoRA 残差
        mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        x_out = x_out + block.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        return x_out

    # ============================================================================
    # _parallel_block_with_lora: 带 LoRA 适配器的并行双通道块
    # ============================================================================
    # 与 _parallel_block 的逻辑完全对应，但在 Q/K/V/O/MLP 投影上叠加 LoRA 残差。
    # 注意力从 lane0 读取，MLP 从 lane1 读取，结果通过 lambda 系数交叉混合回两条通道。
    def _parallel_block_with_lora(
        self, block_idx, lane0, lane1, x0, lora, slot,
        q_w, k_w, v_w, out_w, up_w, down_w,
    ):
        block = self.blocks[block_idx]
        # 注意力从 lane0 读取 (与 _parallel_block 一致)
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        n = block.attn_norm(attn_read) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        # QKV 投影 + LoRA 残差 (与 _block_with_lora 相同的模式)
        q_raw = F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)
        q = q_raw.reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
        k = F.linear(n, k_w.to(n.dtype))
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        # QK-Norm -> RoPE -> Q 增益 -> Flash Attention (与标准路径一致)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        # 门控机制 (TTT 并行路径): 必须与训练路径完全匹配
        # AttnOutGate (TTT parallel path) — inline + .contiguous() barrier.
        if attn.attn_out_gate:
            gate_src = q_raw if attn.attn_out_gate_src == "q" else n
            gate_in = gate_src[..., : attn.gate_window].contiguous()
            g = 2.0 * torch.sigmoid(attn.attn_gate_proj(gate_in))
            y = y * g[..., None]
        # Gated Attention (TTT parallel path). Gate input is n (post-norm block input).
        if attn.gated_attn:
            n_c = n.contiguous()
            g = torch.sigmoid(F.linear(n_c, attn.attn_gate_w.to(n.dtype)))
            y = y * g[..., None]
        # Sparse attention head-output gate (TTT parallel path) — must match the
        # eval path in forward() to keep train/eval semantics in sync.
        if attn.sparse_attn_gate:
            gate_in = n[..., : attn.gate_window].contiguous()
            g = torch.sigmoid(
                attn.sparse_attn_gate_scale
                * F.linear(gate_in, attn.attn_gate_w.to(n.dtype))
            )
            y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        # O 投影 + 可选 LoRA 残差
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        # MLP 从 lane1 (MLP 通道) 读取 + 可选 LoRA 残差
        mlp_read = lane1
        mlp_n = block.mlp_norm(mlp_read) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
        # 可学习的 lambda 系数交叉混合 (与 _parallel_block 一致)
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1


# =============================================================================
# BatchedLinearLoRA — 批量低秩适配器（Batched LoRA）
# =============================================================================
# 这是一个支持"批次维度"的 LoRA 模块，用于 Test-Time Training (TTT) 场景。
# 每个批次样本拥有独立的 A/B 低秩矩阵，因此可以对不同文档做独立的 per-doc 适配。
#
# 核心设计：
#   - 参数 A 的形状为 (bsz, rank, in_features)，B 的形状为 (bsz, out_features, rank)
#   - 前向计算: y = (x @ A^T @ B^T) * (alpha / rank)
#   - alpha/rank 缩放使得有效输出幅度与 rank 解耦——调整 rank 不需要同步调整学习率
#   - _WARM_START_A：如果开启，在文档边界处只清零 B、保留 A，
#     让 A 在同一 TTT 阶段跨文档积累有用的特征方向
# =============================================================================
class BatchedLinearLoRA(nn.Module):
    # PR-1767: rank-scaled output (alpha/rank), like standard LoRA. Decouples
    # effective magnitude from rank so changing rank does not change LR scale.
    # alpha 超参数，控制 LoRA 输出的缩放因子（默认 144）
    _ALPHA = float(os.environ.get("TTT_LORA_ALPHA", "144"))
    # PR-1767: optionally keep A warm across per-doc resets (only B is zeroed).
    # Accumulates useful feature directions across documents within a TTT phase.
    # 热启动标志：为 True 时 reset() 只清零 B，保留 A 的学习到的特征投影
    _WARM_START_A = bool(int(os.environ.get("TTT_WARM_START_A", "1")))

    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        # Kaiming 均匀初始化的边界值 1/sqrt(in_features)
        self._bound = 1.0 / math.sqrt(in_features)
        # LoRA 标准缩放系数 alpha/rank，确保不同 rank 下输出幅度一致
        self._scale = self._ALPHA / rank
        # A 矩阵: (bsz, rank, in_features)，用均匀分布初始化
        self.A = nn.Parameter(
            torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound)
        )
        # B 矩阵: (bsz, out_features, rank)，零初始化
        # 这样初始状态下 LoRA 的增量输出为零，不影响原始模型行为
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    # 重置 LoRA 适配器，在文档边界处调用
    def reset(self):
        with torch.no_grad():
            # 如果未启用热启动，则重新随机初始化 A
            if not self._WARM_START_A:
                self.A.uniform_(-self._bound, self._bound)
            # B 始终清零——这保证重置后 LoRA 增量为零
            self.B.zero_()

    # 前向计算: x @ A^T -> (bsz, seq, rank), 再 @ B^T -> (bsz, seq, out_features)
    # 最后乘以 alpha/rank 缩放因子
    def forward(self, x):
        return ((x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)) * self._scale


# =============================================================================
# BatchedTTTLoRA — Test-Time Training 中每层的 LoRA 适配器集合
# =============================================================================
# 为 Transformer 的各子模块（Q/K/V/O 投影 + MLP + LM Head）各创建一组
# BatchedLinearLoRA 适配器。每层对应一个 "slot"（若开启 looping 则 slot 数 =
# encoder + decoder 层数，否则 = 总 block 数）。
#
# TTT (Test-Time Training) 在推理期间对这些 LoRA 参数进行在线梯度更新，
# 使模型能动态适应当前输入文档的分布。
# =============================================================================
class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz, model, rank, k_lora=True, mlp_lora=True, o_lora=True):
        super().__init__()
        self.bsz = bsz
        dim = model.qo_bank.shape[-1]           # 模型隐藏维度
        vocab = model.tok_emb.num_embeddings     # 词表大小
        # 确定 slot 数量（每个 slot 对应一组 Q/K/V/O/MLP 的 LoRA）
        if getattr(model, "looping_active", False):
            num_slots = len(model.encoder_indices) + len(model.decoder_indices)
        else:
            num_slots = len(model.blocks)
        # KV 投影的输出维度 = num_kv_heads * head_dim（可能 != dim，因 GQA 分组注意力）
        kv_dim = model.blocks[0].attn.num_kv_heads * (
            dim // model.blocks[0].attn.num_heads
        )
        embed_dim = model.tok_emb.embedding_dim
        # LM Head 的 LoRA：从 embed_dim 投影到 vocab，用于适配输出分布
        self.lm_head_lora = BatchedLinearLoRA(bsz, embed_dim, vocab, rank)
        # 每层的 Q 投影 LoRA：dim -> dim
        self.q_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
        )
        # 每层的 V 投影 LoRA：dim -> kv_dim
        self.v_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
        )
        # 每层的 K 投影 LoRA（可选）：dim -> kv_dim
        self.k_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
            )
            if k_lora
            else None
        )
        # 每层的 MLP LoRA（可选）：dim -> dim
        self.mlp_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if mlp_lora
            else None
        )
        # 每层的 O（输出投影）LoRA（可选）：dim -> dim
        self.o_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if o_lora
            else None
        )

    # 重置所有 LoRA 适配器，在文档边界处调用
    # 遍历 lm_head + 所有层的 Q/V/K/MLP/O LoRA 并逐一 reset
    def reset(self):
        with torch.no_grad():
            self.lm_head_lora.reset()
            for loras in [self.q_loras, self.v_loras, self.k_loras,
                          self.mlp_loras, self.o_loras]:
                if loras is not None:
                    for lora in loras:
                        lora.reset()


# =============================================================================
# Polar Express Newton-Schulz 迭代系数
# =============================================================================
# Newton-Schulz 迭代用于近似计算矩阵的"零幂"（即极分解 polar decomposition）：
#   给定矩阵 G，求 U = G (G^T G)^{-1/2}，即 G 的正交/酉部分。
#
# 传统 Muon 使用固定系数 (a=3.4445, b=-4.775, c=2.0315) 来执行以下迭代：
#   X_{k+1} = a * X_k + (b * X_k X_k^T + c * (X_k X_k^T)^2) X_k
# 这本质上是一个矩阵多项式 p(X X^T) X，其中 p(t) = a + b*t + c*t^2。
#
# "Polar Express" (PR #1344) 的改进是：对每次迭代使用不同的最优系数 (a_i, b_i, c_i)，
# 这些系数通过 minimax 优化得到，使得在给定迭代次数下逼近精度最大化。
# 第一步的系数偏大（快速粗略缩放），后续步骤逐渐精细化。
# 当 backend_steps > 5 时，超出部分使用最后一组系数（已收敛）。
# =============================================================================
# Polar Express per-iteration minimax Newton-Schulz coefficients (PR #1344).
# Replaces the fixed (3.4445, -4.775, 2.0315) coefficients of stock Muon.
# Applied at backend_steps=5 — taking more than 5 iterations from this list
# falls back to the final (converged) tuple via the slice guard below.
_PE_COEFFS = (
    # 第 1 步：大系数，快速将谱范数粗略归一化
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    # 第 2 步起逐步精细化，系数趋于收敛
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    # 第 5 步：最终收敛系数，超出 5 步时将重复使用此组
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


# =============================================================================
# zeropower_via_newtonschulz5 — 通过 Newton-Schulz 迭代计算矩阵极分解的正交因子
# =============================================================================
# 给定梯度矩阵 G，计算其"零幂" U = G (G^T G)^{-1/2}，
# 这等价于极分解 G = U S 中的正交矩阵 U。
#
# 直觉：将梯度矩阵的每个奇异值都映射到 1（保留方向、去除尺度），
# 这使得 Muon 优化器对所有方向施加相同幅度的更新，避免梯度尺度不均的问题。
#
# 算法步骤：
#   1. 先对 G 做 Frobenius 范数归一化，使谱范数 ≈ 1
#   2. 然后执行 steps 次 Newton-Schulz 迭代：
#      A = X @ X^T                     (Gram 矩阵)
#      B = b * A + c * A^2             (多项式近似 (XX^T)^{-1/2} 的修正项)
#      X = a * X + B @ X               (更新)
#   3. 迭代收敛后 X 的奇异值全部趋近 1
#
# 注意：使用 bfloat16 计算以提升速度，torch.compile 编译优化
# =============================================================================
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    # 支持 2D 输入（单个矩阵）和 3D 输入（批量矩阵）
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    # 如果行数 > 列数，转置后处理（确保 "宽" 矩阵，加速计算）
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    # Frobenius 范数归一化，将谱范数缩放到 ≈ 1，这是 Newton-Schulz 收敛的前提
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    # 选择每步不同的 Polar Express 系数（最多 5 步各不同，超出用最后一组）
    coeffs = _PE_COEFFS[:steps] if steps <= len(_PE_COEFFS) else _PE_COEFFS
    for a, b, c in coeffs:
        A = X @ X.mT                     # Gram 矩阵 X X^T
        B = b * A + c * (A @ A)          # 二阶多项式修正
        X = a * X + B @ X               # Newton-Schulz 更新步
    # 恢复原始形状
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


# =============================================================================
# Muon 优化器 — 动量 + Newton-Schulz 正交化的优化器
# =============================================================================
# Muon 的核心思想：
#   1. 对梯度施加动量（Nesterov 或经典动量）
#   2. 通过 Newton-Schulz 迭代对动量更新做极分解（polar decomposition），
#      将所有奇异值映射为 1，只保留梯度方向信息
#   3. 这相当于对参数矩阵做"谱归一化"的梯度更新，
#      所有方向的更新步长相同，避免大奇异值主导训练
#
# 分布式支持：
#   - 参数沿第 0 维（batch/output 维度）分片到各 worker
#   - 使用 reduce_scatter 聚合梯度分片
#   - 使用流水线式 all_gather 收集正交化后的更新
#   - reduce_scatter 和 all_gather 都是异步执行，与计算重叠
#
# 额外特性：
#   - row_normalize: 可选的行级范数归一化（在 Newton-Schulz 之前）
#   - weight_decay: 解耦的权重衰减（直接缩放参数）
#   - scale 因子: max(1, rows/cols)^0.5，补偿非方阵的行列比例
# =============================================================================
class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
        row_normalize=False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )
        self._built = False

    # _build: 延迟初始化，首次 step 或 launch_reduce_scatters 时调用
    # 为每个参数分配分布式通信缓冲区和动量缓冲区
    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                # 将第 0 维 pad 到 world_size 的整数倍，便于均匀分片
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    "p": p,                 # 原始参数引用
                    "B": B,                 # 原始第 0 维大小（未 pad）
                    # padded_grad: 存放 pad 后的梯度，供 reduce_scatter 使用
                    "padded_grad": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    # shard: 本 worker 负责的梯度分片（reduce_scatter 输出）
                    "shard": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    # shard_mom: 本 worker 分片上的动量缓冲区
                    "shard_mom": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    # full_update: all_gather 后的完整更新（pad 过的大小）
                    "full_update": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    # scale 因子: 对非方阵做行列比例补偿
                    # 当 rows > cols 时，正交化矩阵的 Frobenius 范数偏大，需缩放补偿
                    "scale": max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        # 按参数元素数量降序排列，使大参数的通信先启动（更好地重叠计算与通信）
        self._bank_meta.sort(key=lambda m: -m["p"].numel())
        self._built = True

    # launch_reduce_scatters: 异步启动所有参数的 reduce_scatter 操作
    # reduce_scatter 将所有 worker 的梯度取平均后，每个 worker 只接收自己负责的分片
    # 这比 all_reduce 更节省内存：每个 worker 只需存储 1/world_size 的梯度
    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m["padded_grad"]
            # 将实际梯度复制到 pad 缓冲区（转换为 bf16 节省带宽）
            pg[: m["B"]].copy_(p.grad.bfloat16())
            # pad 区域清零，避免影响平均值
            if pg.shape[0] > m["B"]:
                pg[m["B"] :].zero_()
            # 异步 reduce_scatter: 对所有 worker 的 pg 求平均，结果分片到 m["shard"]
            fut = dist.reduce_scatter_tensor(
                m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True
            )
            self._rs_futures.append(fut)

    # step: Muon 优化器的核心更新步骤
    # 流程：
    #   1. 等待上一个参数的 all_gather 完成，应用其更新（流水线）
    #   2. 等待当前参数的 reduce_scatter 完成（获取本分片的平均梯度）
    #   3. 施加动量（Nesterov 或经典）
    #   4. 可选行归一化
    #   5. Newton-Schulz 正交化（zeropower_via_newtonschulz5）
    #   6. 异步启动 all_gather 将正交化结果广播到所有 worker
    #   7. 处理最后一个参数的 all_gather（循环外收尾）
    #
    # 流水线设计：当前参数的 Newton-Schulz 计算与上一个参数的 all_gather 通信重叠
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            row_normalize = group.get("row_normalize", False)
            # 流水线状态：追踪上一个参数的 all_gather handle
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, "_rs_futures")
            for idx, m in enumerate(self._bank_meta):
                p = m["p"]
                if p.grad is None:
                    continue
                # 流水线：等待上一个参数的 all_gather 完成，然后应用其更新
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m["p"]
                    # 只取未 pad 的部分（前 B 行）
                    upd = prev_m["full_update"][: prev_m["B"]]
                    # 解耦权重衰减：直接乘以 (1 - lr * wd)
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    # 应用正交化更新，乘以 scale 补偿非方阵比例
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
                # 获取梯度：分布式模式下从 reduce_scatter 分片获取，否则直接用本地梯度
                if sharded and self._rs_futures[idx] is not None:
                    self._rs_futures[idx].wait()
                    g = m["shard"]         # 本 worker 的梯度分片
                    buf = m["shard_mom"]   # 本 worker 的动量缓冲区
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                # 经典动量更新: buf = momentum * buf + g
                buf.mul_(momentum).add_(g)
                # Nesterov 动量: update = g + momentum * buf（比经典动量多一步前瞻）
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                # 可选：行级范数归一化（在 Newton-Schulz 之前）
                if row_normalize:
                    rn = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                    update = update / rn.to(update.dtype)
                # 核心操作：Newton-Schulz 正交化，将更新矩阵的所有奇异值映射为 1
                # 这使得参数在所有方向上获得相同幅度的更新
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    # 异步 all_gather：将本 worker 的正交化结果收集到所有 worker
                    # 流水线：启动后立即继续处理下一个参数的 Newton-Schulz
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m["full_update"], update, async_op=True
                    )
                    prev_m = m
                else:
                    # 非分布式路径：直接应用更新
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
            # 流水线收尾：处理最后一个参数的 all_gather
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m["p"]
                upd = prev_m["full_update"][: prev_m["B"]]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
            # 清理 reduce_scatter futures，避免下次 step 误用旧 handle
            if hasattr(self, "_rs_futures"):
                del self._rs_futures
        return loss


# 控制张量的名称模式列表
# 这些张量是模型中的缩放因子、门控参数、残差混合系数等小型控制参数。
# 它们在优化时与矩阵权重区分对待：使用 AdamW 而非 Muon，
# 因为这些参数通常是 1D 标量/向量，不适合 Newton-Schulz 正交化。
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,parallel_post_lambdas,parallel_resid_lambdas,attn_gate_proj,attn_gate_w,smear_gate,smear_lambda",
    ).split(",")
    if pattern
)

# 小参数打包 all-reduce 的阈值：元素数 <= 32768 的梯度会被拼接成一个平坦向量
# 一次性 all-reduce，减少通信次数（小消息的 latency 开销远大于带宽开销）
PACKED_REPLICATED_GRAD_MAX_NUMEL = 1 << 15


# =============================================================================
# Optimizers — 参数分组优化器管理
# =============================================================================
# 将模型参数分为三组，分别使用不同的优化器：
#   1. 矩阵参数（QO/KV/MLP bank）→ Muon（Newton-Schulz 正交化）
#   2. 嵌入参数（tok_emb）→ AdamW（标准自适应优化）
#   3. 标量/控制参数（scale、gate、lambda 等）→ AdamW
#
# 为什么要分组？
#   - Muon 的 Newton-Schulz 正交化只适用于 2D 矩阵，对标量/1D 参数无意义
#   - 嵌入矩阵的梯度稀疏且分布特殊，AdamW 更合适
#   - 不同参数组可以使用不同的学习率和权重衰减
#
# 分布式梯度聚合策略：
#   - 矩阵参数: Muon 内部用 reduce_scatter + all_gather
#   - 复制参数（嵌入 + 标量）: all_reduce 取平均
#     - 小参数打包到一个平坦向量统一 all_reduce（减少通信次数）
#     - 大参数各自单独异步 all_reduce
# =============================================================================
class Optimizers:
    def __init__(self, h, base_model):
        # 矩阵参数：QO/KV 注意力权重库 和 MLP 上下投影权重库
        # 这些是模型的主要大型 2D 权重矩阵，由 Muon 优化
        matrix_params = [
            base_model.qo_bank,
            base_model.kv_bank,
            base_model.mlp_up_bank,
            base_model.mlp_down_bank,
        ]
        block_named_params = list(base_model.blocks.named_parameters())
        # 标量参数：从 blocks 中筛选出 1D 参数或名称匹配控制张量模式的参数
        scalar_params = [
            p
            for (name, p) in block_named_params
            if p.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        # 将各种全局控制参数也加入标量组
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if base_model.parallel_post_lambdas is not None:
            scalar_params.append(base_model.parallel_post_lambdas)
        if base_model.parallel_resid_lambdas is not None:
            scalar_params.append(base_model.parallel_resid_lambdas)
        # SmearGate 参数不在 blocks 子模块中（挂在 GPT 根级别），需手动添加
        # 两个都很小：gate_window 标量 + 1 个 lambda，用 AdamW 优化
        # SmearGate params live on GPT root (not in .blocks), so add them by hand.
        # Both are tiny (gate_window scalars + 1 lambda). Optimized via scalar Adam.
        if getattr(base_model, "smear_gate_enabled", False):
            scalar_params.append(base_model.smear_gate.weight)
            scalar_params.append(base_model.smear_lambda)
        # 嵌入学习率：如果 tie_embeddings（共享输入/输出嵌入），使用 tied_embed_lr
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [
            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
        ]
        # 嵌入优化器：AdamW（fused=True 启用 CUDA fused kernel 加速）
        self.optimizer_tok = torch.optim.AdamW(
            tok_params,
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=True,
        )
        # 矩阵优化器：Muon（动量 + Newton-Schulz 正交化）
        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        # 保存 base_lr 以供学习率调度器引用
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        # 标量优化器：AdamW，用于所有 1D 控制参数
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [
            self.optimizer_tok,
            self.optimizer_muon,
            self.optimizer_scalar,
        ]
        # 复制参数（嵌入 + 标量）需要在分布式训练中做 all_reduce
        # 按大小分为两组：小参数打包 all_reduce，大参数单独异步 all_reduce
        self.replicated_params = list(tok_params[0]["params"])
        self.replicated_params.extend(scalar_params)
        self.replicated_large_params = []
        self.replicated_packed_params = []
        for p in self.replicated_params:
            if p.numel() <= PACKED_REPLICATED_GRAD_MAX_NUMEL:
                # 小参数：稍后打包成一个平坦向量统一 all_reduce
                self.replicated_packed_params.append(p)
            else:
                # 大参数：各自单独异步 all_reduce
                self.replicated_large_params.append(p)

    def __iter__(self):
        return iter(self.optimizers)

    # 清零所有优化器的梯度（set_to_none=True 释放梯度内存，比置零更高效）
    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    # 打包 all-reduce：将所有小参数的梯度拼成一个连续向量，一次 all_reduce
    # 这避免了大量小消息的 latency 累积（NCCL 对大消息的吞吐更高）
    def _all_reduce_packed_grads(self):
        # 按 (device, dtype) 分组，确保相同设备和精度的梯度拼在一起
        grads_by_key = collections.defaultdict(list)
        for p in self.replicated_packed_params:
            if p.grad is not None:
                grads_by_key[(p.grad.device, p.grad.dtype)].append(p.grad)
        for grads in grads_by_key.values():
            # 分配平坦缓冲区，容纳所有梯度
            flat = torch.empty(
                sum(g.numel() for g in grads),
                device=grads[0].device,
                dtype=grads[0].dtype,
            )
            # 将各梯度复制到平坦缓冲区
            offset = 0
            for g in grads:
                n = g.numel()
                flat[offset : offset + n].copy_(g.contiguous().view(-1))
                offset += n
            # 一次性 all_reduce：所有 worker 求平均
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            # 将平均后的结果写回各梯度
            offset = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n

    # step: 协调所有优化器的一步更新
    # 执行顺序经过精心设计以最大化通信-计算重叠：
    #   1. 先启动 Muon 的 reduce_scatter（异步）
    #   2. 如果分布式模式，同时启动大参数的异步 all_reduce
    #   3. 同步执行小参数的打包 all_reduce
    #   4. 等待大参数 all_reduce 完成
    #   5. 执行 tok/scalar AdamW step（此时梯度已同步）
    #   6. 执行 Muon step（内部等待 reduce_scatter 并做 Newton-Schulz + all_gather）
    #   7. 清零所有梯度
    def step(self, distributed=False):
        # 先启动 Muon 的异步 reduce_scatter
        self.optimizer_muon.launch_reduce_scatters()
        if distributed:
            # 大参数各自异步 all_reduce
            reduce_handles = [
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
                for p in self.replicated_large_params
                if p.grad is not None
            ]
            # 小参数打包同步 all_reduce
            self._all_reduce_packed_grads()
            # 等待大参数 all_reduce 完成
            for handle in reduce_handles:
                handle.wait()
        # 依次执行三个优化器的 step
        self.optimizer_tok.step()
        self.optimizer_scalar.step()
        self.optimizer_muon.step()
        # 清零梯度，为下一步准备
        self.zero_grad_all()


# =============================================================================
# restore_fp32_params — 确保控制张量和权重库保持 fp32 精度
# =============================================================================
# 加载检查点或混合精度训练后，某些参数可能被意外降精度为 bf16/fp16。
# 此函数将以下参数强制恢复到 fp32：
#   - CastedLinear 模块（其权重在前向时会动态转换精度，但主权重需保持 fp32）
#   - 所有 1D/标量控制参数（scale、gate、lambda 等）
#   - 注意力和 MLP 权重库（qo_bank, kv_bank, mlp_up_bank, mlp_down_bank）
# 保持 fp32 的原因：优化器状态（如 Adam 的一阶/二阶矩）在 fp32 精度下更稳定
# =============================================================================
def restore_fp32_params(model):
    # CastedLinear 模块的权重恢复到 fp32
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    # 1D/标量控制参数恢复到 fp32
    for name, param in model.named_parameters():
        if (
            param.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ) and param.dtype != torch.float32:
            param.data = param.data.float()
    # 注意力权重库恢复到 fp32
    if hasattr(model, "qo_bank") and model.qo_bank is not None:
        model.qo_bank.data = model.qo_bank.data.float()
        model.kv_bank.data = model.kv_bank.data.float()
    # MLP 权重库恢复到 fp32
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()


# =============================================================================
# collect_hessians — 为 GPTQ 量化收集 Hessian（二阶梯度信息）
# =============================================================================
# GPTQ (Generalized Post-Training Quantization) 需要每层线性层输入的
# Hessian 矩阵 H = X^T X（其中 X 是校准数据的激活值矩阵）。
#
# 为什么需要 Hessian？
#   - 量化某一列权重时产生的误差会传播到后续列
#   - Hessian 的逆 H^{-1} 告诉我们每一列的"灵敏度"——
#     灵敏度高的列应优先量化（误差补偿效果更好）
#   - GPTQ 按 H 对角线降序排列列，优先处理灵敏度高的列
#
# 收集方式：
#   - 注册 forward hook，在每次前向传播时捕获输入激活
#   - 对注意力层：为 Q/K/V 投影收集同一个 Hessian（共享输入），
#     为输出投影 proj 收集单独的 Hessian（使用注意力输出作为输入）
#   - 对 MLP 层：为上投影 fc 和下投影 proj 分别收集 Hessian
#   - 跑 n_calibration_batches 个 batch 的前向传播，累加 X^T X
#   - 最终除以批次数得到平均 Hessian
# =============================================================================
def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []
    # 启用校准模式：让注意力和 MLP 模块保存内部激活（如 _last_proj_input）
    for i, block in enumerate(model.blocks):
        block.attn._calib = True
        block.mlp._calib = True
        block.mlp.use_fused = False    # 禁用 fused kernel 以捕获中间激活

    # 注意力层的 Hessian 收集 hook
    # 对 Q/K/V 投影共享同一个输入（attn 层的输入），分别累加 X^T X
    # 对输出投影 proj 使用 _last_proj_input（注意力输出重排后的激活）
    def make_attn_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])    # (bsz*seq, dim)
            # Q/K/V 共享输入，因此对同一个 X 累加到三个 Hessian
            for suffix in ["c_q", "c_k", "c_v"]:
                name = f"blocks.{layer_idx}.attn.{suffix}.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                # H += X^T X（外积累加）
                hessians[name].addmm_(x.T, x)
            # 输出投影的输入（注意力输出）
            y = module._last_proj_input
            if y is not None:
                y = y.float()
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                name = f"blocks.{layer_idx}.attn.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        y.shape[1], y.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(y.T, y)
        return hook_fn

    # MLP 层的 Hessian 收集 hook
    # fc（上投影）使用 MLP 输入，proj（下投影）使用激活函数后的中间特征
    def make_mlp_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            # MLP 上投影的 Hessian
            name = f"blocks.{layer_idx}.mlp.fc.weight"
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[name].addmm_(x.T, x)
            # MLP 下投影的 Hessian（使用激活函数后的中间激活）
            h_act = module._last_down_input
            if h_act is not None:
                h_act = h_act.float()
                if h_act.ndim == 3:
                    h_act = h_act.reshape(-1, h_act.shape[-1])
                name = f"blocks.{layer_idx}.mlp.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        h_act.shape[1], h_act.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(h_act.T, h_act)
        return hook_fn

    # 注册所有层的 forward hook
    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

    # 嵌入分解投影层的 Hessian hook（通用版本，收集线性层输入的 X^T X）
    # Hessian hooks for embedding factorization projection layers
    def make_linear_input_hook(weight_name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if weight_name not in hessians:
                hessians[weight_name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[weight_name].addmm_(x.T, x)
        return hook_fn

    # 当输入输出嵌入共享权重（tie_embeddings）时，
    # 需要收集 final_norm 的输出作为 tok_emb.weight 的 Hessian 输入
    # 因为 lm_head 使用 final_norm 的输出乘以 tok_emb.weight 的转置
    if model.tie_embeddings:
        hook_module = model.final_norm

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                # 注意这里用的是 out（输出），而非 inp（输入）
                # 因为 final_norm 的输出就是 lm_head 投影的输入
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            return hook_fn

        hooks.append(
            hook_module.register_forward_hook(make_output_hook("tok_emb.weight"))
        )
    # 运行校准数据收集 Hessian
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    # 清理：移除所有 hook，恢复正常模式
    for hook in hooks:
        hook.remove()
    for i, block in enumerate(model.blocks):
        block.attn._calib = False
        block.mlp._calib = False
        block.mlp.use_fused = True
    # 将 Hessian 移到 CPU 并除以批次数得到平均值
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


# =============================================================================
# gptq_quantize_weight — GPTQ 逐列量化（Hessian 引导的误差补偿）
# =============================================================================
# GPTQ 算法的核心思想：
#   - 逐列将 fp32 权重量化为 int8（或其他位宽）
#   - 每量化一列产生的误差，利用 Hessian 逆矩阵 H^{-1} 补偿到尚未量化的列
#   - 这最小化了量化后 W_q 与原始 W 在二次误差 ||WX - W_qX||^2 意义下的损失
#
# 算法步骤：
#   1. 对 Hessian 做阻尼正则化（加 0.01 * mean(diag) 到对角线），确保正定
#   2. 按 Hessian 对角线降序排列列（灵敏度高的列优先量化）
#   3. 对 H 求逆并做上三角 Cholesky 分解：Hinv = chol(H^{-1})
#   4. 分块处理（block_size=128）：
#      - 对块内每一列：量化 → 计算误差 → 用 Hinv 将误差补偿到同块后续列
#      - 块处理完后，将累积误差通过 Hinv 跨块补偿到剩余列
#   5. 缩放因子 s = clip_sigmas * row_std / clip_range（对称量化，per-row 缩放）
#
# 参数：
#   w: 要量化的权重矩阵 (rows, cols)
#   H: 对应的 Hessian 矩阵 (cols, cols)
#   clip_sigmas: 截断标准差数（默认 3.0）
#   clip_range: 量化值域 [-clip_range, clip_range]（7-bit 为 63）
#   block_size: 列分块大小（128，平衡计算效率与补偿精度）
# =============================================================================
def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    # 处理"死"列（对角线为 0 的列表示该特征在校准数据中从未激活）
    dead = torch.diag(H) == 0
    H[dead, dead] = 1            # 避免奇异矩阵
    # 阻尼正则化：在对角线加小量 damp，确保 H 正定可逆
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    # 按 Hessian 对角线降序排列列索引（灵敏度高 = 对角线值大 → 优先量化）
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)    # 逆排列，最后恢复原始列顺序
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0        # 死列权重清零
    H = H[perm][:, perm]             # 重排 Hessian
    # 计算 Hessian 逆的上三角 Cholesky 分解
    # H^{-1} = L L^T → Hinv = chol(H^{-1}, upper=True) 即上三角因子
    # 这样 Hinv[j, j] 是第 j 列的"补偿权重"，Hinv[j, j:] 用于误差传播
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    # 对称量化的 per-row 缩放因子: s = clip_sigmas * std(row) / clip_range
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    # 分块逐列量化
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]     # 当前块的 Hinv 子矩阵
        Err = torch.zeros(rows, i2 - i1)     # 累积误差矩阵
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]              # 该列的补偿权重
            # 量化：四舍五入到最近整数并截断到 [-clip_range, clip_range]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            # 量化误差除以 d，得到补偿量
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            # 块内误差补偿：将误差传播到同块后续列
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        # 跨块误差补偿：将当前块的累积误差传播到剩余所有未量化列
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    # 恢复原始列顺序并返回量化矩阵 Q 和缩放因子 s
    return Q[:, invperm], s


# =============================================================================
# _quantize_gate_int8_row — 门控张量的对称 int8 per-row 量化
# =============================================================================
# 用于小型门控张量（如 attn_gate_w），不需要 GPTQ 的 Hessian 引导。
# 简单的 per-row 对称量化：每行一个缩放因子 s = max(|row|) / 127
# 量化值 q = round(w / s)，截断到 [-127, 127]
# 存储节省：int8 + fp16 缩放因子，相比 fp16 权重节省约一半空间
# =============================================================================
def _quantize_gate_int8_row(w):
    # Symmetric int8-per-row quantization for small gate tensors. w shape
    # (R, C) -> (R,) scales in fp16, int8 values in [-127, 127]. Single scale
    # per row keeps accuracy high while halving storage vs fp16.
    W = w.float().contiguous()
    row_max = W.abs().amax(dim=1).clamp_min(1e-10)
    s = (row_max / 127.0).to(torch.float16)
    sf = s.float().view(-1, 1)
    q = torch.clamp(torch.round(W / sf), -127, 127).to(torch.int8)
    return q, s


# =============================================================================
# _lqer_pack — LQER 低秩因子的对称量化打包
# =============================================================================
# LQER (Low-rank Quantization Error Reduction) 将量化误差 E = W - W_q 做 SVD 分解：
#   E ≈ A @ B，其中 A = U[:,:r] * S[:r], B = Vh[:r,:]
# 这两个低秩因子 A、B 本身也需要量化以节省存储。
#
# _lqer_pack 对 A 和 B 做对称 per-row int8 量化：
#   - 量化范围: [-rng, rng]，其中 rng = 2^(bits-1) - 1
#   - 每行一个缩放因子 s = max(|row|) / rng
# =============================================================================
def _lqer_pack(A, B, bits):
    rng = 2 ** (bits - 1) - 1
    # A: per-row 对称量化
    sA = (A.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    # B: per-row 对称量化
    sB = (B.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float().view(-1, 1)), -rng, rng).to(torch.int8)
    qB = torch.clamp(torch.round(B / sB.float().view(-1, 1)), -rng, rng).to(torch.int8)
    return qA, sA, qB, sB


# =============================================================================
# _lqer_pack_asym — LQER 低秩因子的非对称量化打包
# =============================================================================
# 与 _lqer_pack 不同，这里 A 和 B 使用不同的量化精度和粒度：
#   - A 因子: INT2 全矩阵共享一个标量缩放因子
#     范围 [-2, 1]（注意非对称！），scale = max(|A|) / 1.5
#     INT2 极端压缩，因为 A = U*S 的值分布通常比较集中
#   - B 因子: INT4 分组量化（group size = g，默认 64）
#     范围 [-8, 7]，每组一个 fp16 缩放因子
#     分组量化在精度和存储之间取得更好平衡
#
# 非对称设计的动机：SVD 分解后 A 包含奇异值（值域大），B 是正交方向（值域小），
# 两者数值特性不同，适合用不同精度量化
# =============================================================================
def _lqer_pack_asym(A, B, g=64):
    # A: INT2 全矩阵标量量化 (signed [-2,1], scale = |A|max/1.5)
    # A: INT2 per-matrix scalar (signed [-2,1], scale = |A|max/1.5).
    sA = (A.abs().amax().clamp_min(1e-10) / 1.5).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float()), -2, 1).to(torch.int8)
    # B: INT4 分组量化 (signed [-8,7], 每 g 个元素共享一个缩放因子)
    # B: INT4 groupwise g over flattened B (signed [-8,7], per-group scale).
    Bf = B.reshape(-1, g)           # 展平后按 group size 分组
    Bmax = Bf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    sB = (Bmax / 7.5).to(torch.float16).reshape(-1)    # 每组一个 fp16 缩放因子
    qB = torch.clamp(torch.round(Bf / sB.float().reshape(-1, 1)), -8, 7).to(
        torch.int8
    ).reshape(B.shape)
    return qA, sA, qB, sB


# =============================================================================
# gptq_mixed_quantize — 混合量化策略：将不同类型的权重路由到不同量化方法
# =============================================================================
# 路由逻辑（按优先级从高到低）：
#   1. attn_gate_w → int8 per-row（小型门控张量，简单量化足够）
#   2. 非浮点或元素数 <= 65536 → passthrough（直接转 fp16 存储）
#   3. 大型矩阵 → GPTQ int8 量化（Hessian 引导的误差补偿）
#      - 不同类型矩阵使用不同的 clip_sigmas（嵌入/MLP/注意力各有最优值）
#   4. 如果开启 LQER：对量化误差最大的 top-k 层做低秩误差修正
#      - SVD 分解量化残差 E = W - W_q
#      - 取前 r 个奇异值/向量构建低秩近似 A @ B ≈ E
#      - 将 A、B 量化存储（对称或非对称模式）
#      - 反量化时 W ≈ W_q + A @ B
# =============================================================================
def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    quant_gate = bool(getattr(h, "gated_attn_quant_gate", False))
    lqer_on = bool(getattr(h, "lqer_enabled", False))
    lqer_cands = {}     # 候选 LQER 层：{name: (误差矩阵 E, ||E|| 范数)}
    for (name, tensor) in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        # --- 路由 1: 门控张量 → int8 per-row 量化 ---
        # Dedicated int8-per-row path for attn_gate_w (bypasses both GPTQ and
        # fp16 passthrough). Applied BEFORE the numel<=65536 passthrough check
        # so the gate tensor is routed here instead of to fp16.
        if (
            quant_gate
            and t.is_floating_point()
            and t.ndim == 2
            and name.endswith(".attn_gate_w")
            # Dense GatedAttn: (num_heads, dim) = (8, 512) = 4096.
            # Sparse gate: (num_heads, gate_window) = (8, 12) = 96.
            # Both need int8-per-row routing; the 1024 lower bound in stock
            # PR-1736 presumed dense-only. Widen to catch both.
            and 32 <= t.numel() <= 8192
        ):
            gq, gs = _quantize_gate_int8_row(t)
            result[name + ".gq"] = gq
            result[name + ".gs"] = gs
            meta[name] = "gate_int8_row"
            continue
        # --- 路由 2: 小张量或非浮点 → 直接 fp16 透传 ---
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        # --- 路由 3: 大型矩阵 → GPTQ 量化 ---
        # 根据权重类型选择不同的 clip_sigmas（截断标准差数）
        if "tok_emb" in name:
            cs = h.embed_clip_sigmas
        elif ".mlp." in name:
            cs = h.mlp_clip_sigmas
        elif ".attn." in name:
            cs = h.attn_clip_sigmas
        else:
            cs = h.matrix_clip_sigmas
        # 嵌入层可能使用不同位宽
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        clip_range = 2 ** (bits - 1) - 1
        ret = gptq_quantize_weight(
            t, hessians[name], clip_sigmas=cs, clip_range=clip_range
        )
        q, s = ret
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
        # --- 路由 4: 收集 LQER 候选（如果开启） ---
        # 计算量化误差 E = W_orig - W_quantized，记录范数用于排序
        if lqer_on:
            W_q = q.float() * s.float().view(-1, 1)
            E = t.float() - W_q
            lqer_cands[name] = (E, float(E.norm()))
    # --- LQER 低秩误差修正 ---
    # 选择量化误差范数最大的 top-k 层进行 SVD 低秩近似
    if lqer_on and lqer_cands:
        # 按误差范数降序排列，取前 lqer_top_k 层
        top = sorted(lqer_cands.items(), key=lambda kv: -kv[1][1])[: h.lqer_top_k]
        asym_on = bool(getattr(h, "lqer_asym_enabled", False))
        asym_g = int(getattr(h, "lqer_asym_group", 64))
        for (name, (E, _)) in top:
            # SVD 分解误差矩阵: E = U S Vh
            U, S, Vh = torch.linalg.svd(E, full_matrices=False)
            r = min(h.lqer_rank, S.numel())   # 取前 r 个奇异值
            # A = U[:,:r] * S[:r]，B = Vh[:r,:]
            # 这样 A @ B ≈ E（低秩近似）
            A = (U[:, :r] * S[:r]).contiguous()
            B = Vh[:r, :].contiguous()
            # 选择非对称或对称量化方式打包 A、B
            if asym_on and B.numel() % asym_g == 0:
                # 非对称: A 用 INT2 全矩阵缩放，B 用 INT4 分组缩放
                qA, sA, qB, sB = _lqer_pack_asym(A, B, asym_g)
                result[name + ".lqA_a"] = qA
                result[name + ".lqAs_a"] = sA
                result[name + ".lqB_a"] = qB
                result[name + ".lqBs_a"] = sB
                meta[name] = meta[name] + "+lqer_asym"
            else:
                # 对称: A、B 都用 per-row int8 量化
                qA, sA, qB, sB = _lqer_pack(A, B, h.lqer_factor_bits)
                result[name + ".lqA"] = qA
                result[name + ".lqAs"] = sA
                result[name + ".lqB"] = qB
                result[name + ".lqBs"] = sB
                meta[name] = meta[name] + "+lqer"
    # 打印量化摘要日志：按类别分组显示哪些权重使用了什么量化方式
    categories = collections.defaultdict(set)
    for (name, cat) in meta.items():
        short = re.sub("\\.\\d+$", "", re.sub("blocks\\.\\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta

# =============================================================================
# dequantize_mixed — 从量化状态字典反量化重建 fp 权重
# =============================================================================
# 根据 meta 中记录的量化类型，对每个权重执行相应的反量化操作：
#   - passthrough: 直接类型转换（fp16 → 原始 dtype）
#   - gate_int8_row: int8 * scale 还原
#   - gptq: int8 * per-row scale 还原
#   - gptq + lqer: 基础量化还原 + 低秩误差修正 (W = W_q + A @ B)
#   - gptq + lqer_asym: 同上，但 A/B 使用非对称量化的反量化
#
# 输入：
#   result: 量化状态字典（包含 .q, .scale, .lqA, .lqB 等后缀键）
#   meta: 每个权重的量化类型元数据
#   template_sd: 原始模型的 state_dict（提供目标 dtype 和权重名称列表）
# =============================================================================
def dequantize_mixed(result, meta, template_sd):
    out = {}
    for (name, orig) in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        # --- passthrough: 直接还原 ---
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        # --- gate_int8_row: q * scale 还原 ---
        if info == "gate_int8_row":
            gq = result[name + ".gq"]
            gs = result[name + ".gs"]
            out[name] = (gq.float() * gs.float().view(-1, 1)).to(orig_dtype)
            continue
        # --- GPTQ: int_val * per-row_scale 还原 ---
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            # per-row 缩放：scale 广播到 (rows, 1, ..., 1)
            W = q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
        else:
            # 标量缩放（罕见情况）
            W = q.float() * float(s.item())
        # --- LQER 非对称修正: W += dequant(A) @ dequant(B) ---
        if "lqer_asym" in info:
            qA_t = result[name + ".lqA_a"]
            sA_t = result[name + ".lqAs_a"]
            qB_t = result[name + ".lqB_a"]
            sB_t = result[name + ".lqBs_a"]
            # A: INT2 全矩阵标量缩放还原
            qA = qA_t.float() * float(sA_t)
            # B: INT4 分组缩放还原（推算 group size = 总元素 / 缩放因子数）
            g_sz = qB_t.numel() // sB_t.numel()
            qB = (qB_t.reshape(-1, g_sz).float() * sB_t.float().view(-1, 1)).reshape(
                qB_t.shape
            )
            # 加上低秩误差修正项
            W = W + qA @ qB
        # --- LQER 对称修正: W += dequant(A) @ dequant(B) ---
        elif "lqer" in info:
            # A、B 都用 per-row 缩放还原
            qA = result[name + ".lqA"].float() * result[name + ".lqAs"].float().view(-1, 1)
            qB = result[name + ".lqB"].float() * result[name + ".lqBs"].float().view(-1, 1)
            W = W + qA @ qB
        # 转换到原始 dtype 输出
        out[name] = W.to(orig_dtype)
    return out


# =====================================================================================
# 字节交错混洗 (Byte Shuffle) 与压缩/解压工具
# -------------------------------------------------------------------------------------
# 原理: 量化后的权重张量在二进制层面存在局部相关性(例如相邻字节的高位/低位模式相似)。
# 将字节按 stride 间隔交错重排 (byte shuffle) 后,相同"通道"的字节聚集在一起,
# 大幅提升 LZMA/Brotli 等通用压缩器的压缩率。
# 格式: 前4字节为魔数 "BSHF",第5字节为 stride 值,之后是交错后的数据。
# =====================================================================================
_BSHF_MAGIC = b"BSHF"


# 字节交错混洗: 将原始数据按 stride 间隔提取字节通道并顺序拼接
# 例如 stride=2 时, 偶数位置字节放前半段, 奇数位置字节放后半段
# 这样同一"比特位"的字节聚在一起, 提高压缩器对重复模式的识别率
def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off : dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


# 字节逆混洗: _byte_shuffle 的逆操作
# 从头部读取魔数和 stride, 然后将各通道数据还原到原始交错位置
# 每个通道的长度 = n // stride (+ 1 如果该通道索引 < n % stride)
def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off : src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


# 压缩: 先做字节交错混洗再用指定压缩器压缩
# 支持 lzma (preset=6, 压缩率高但较慢) 和 brotli (quality=11, 最高压缩级别)
# 这两步组合可以在量化模型上获得接近极限的压缩比
def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli

        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


# 解压: _compress 的逆操作, 先用对应解压器解压, 再做字节逆混洗还原原始数据
def _decompress(data, compressor):
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli

        raw = brotli.decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    raw = _byte_unshuffle(raw)
    return raw


# =====================================================================================
# 参数库 (Bank) 格式转换工具
# -------------------------------------------------------------------------------------
# 模型在训练时将所有层的同类权重堆叠成 "bank" 张量以提高效率:
#   - qo_bank: 形状 [2*num_layers, model_dim, model_dim], 前半为 Q 权重, 后半为 O (proj) 权重
#   - kv_bank: 形状 [2*num_layers, kv_dim, model_dim], 前半为 K 权重, 后半为 V 权重
#   - mlp_up_bank: 形状 [num_layers, hidden_dim, model_dim], MLP 上投影
#   - mlp_down_bank: 形状 [num_layers, model_dim, hidden_dim], MLP 下投影
# 量化 (GPTQ) 需要逐层处理, 因此需要 unbank 展开为 "blocks.{i}.xxx.weight" 格式。
# 反量化完成后再用 rebank 重新堆叠回 bank 格式以便加载回模型。
# =====================================================================================

# 将 bank 格式的 state_dict 展开为逐层 (per-layer) 的 state_dict
# 用于 GPTQ 量化前的权重准备, 使量化器能逐个权重矩阵独立处理
def _unbank_state_dict(state_dict, num_layers):
    sd = {}
    n = num_layers
    for k, v in state_dict.items():
        t = v.detach().cpu() if v is not None else None
        if k == "qo_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_q.weight"] = t[i]
                sd[f"blocks.{i}.attn.proj.weight"] = t[n + i]
        elif k == "kv_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_k.weight"] = t[i]
                sd[f"blocks.{i}.attn.c_v.weight"] = t[n + i]
        elif k == "mlp_up_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.fc.weight"] = t[i]
        elif k == "mlp_down_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.proj.weight"] = t[i]
        else:
            if t is not None:
                sd[k] = t
    return sd


# 将逐层 (per-layer) 的 state_dict 重新堆叠回 bank 格式
# 反量化完成后调用, 使得权重能正确加载回 GPT 模型
# 非 bank 类的参数 (如 embedding, layer norm) 直接透传
def _rebank_state_dict(flat_sd, num_layers, model_dim, kv_dim, hidden_dim):
    sd = {}
    n = num_layers
    sd["qo_bank"] = torch.zeros(2 * n, model_dim, model_dim)
    sd["kv_bank"] = torch.zeros(2 * n, kv_dim, model_dim)
    for i in range(n):
        sd["qo_bank"][i] = flat_sd[f"blocks.{i}.attn.c_q.weight"]
        sd["qo_bank"][n + i] = flat_sd[f"blocks.{i}.attn.proj.weight"]
        sd["kv_bank"][i] = flat_sd[f"blocks.{i}.attn.c_k.weight"]
        sd["kv_bank"][n + i] = flat_sd[f"blocks.{i}.attn.c_v.weight"]
    sd["mlp_up_bank"] = torch.zeros(n, hidden_dim, model_dim)
    sd["mlp_down_bank"] = torch.zeros(n, model_dim, hidden_dim)
    for i in range(n):
        sd["mlp_up_bank"][i] = flat_sd[f"blocks.{i}.mlp.fc.weight"]
        sd["mlp_down_bank"][i] = flat_sd[f"blocks.{i}.mlp.proj.weight"]
    for k, v in flat_sd.items():
        if not (
            k.startswith("blocks.")
            and any(
                p in k
                for p in [
                    ".attn.c_q.", ".attn.c_k.", ".attn.c_v.",
                    ".attn.proj.", ".mlp.fc.", ".mlp.proj.",
                ]
            )
        ):
            sd[k] = v
    return sd



# =====================================================================================
# 序列化 (Serialize) 与反序列化 (Deserialize)
# -------------------------------------------------------------------------------------
# 序列化流程: 原始模型 -> GPTQ 量化 -> torch.save -> LZMA/Brotli 压缩 -> 写入磁盘
# 反序列化流程: 读取磁盘 -> 解压 -> torch.load -> 反量化 -> rebank -> 加载回模型
# 还计算代码本身的压缩大小, 因为竞赛规则中总提交大小 = 模型文件 + 代码文件
# =====================================================================================

# 计算代码文件经 pyminify 压缩后的大小
# 先用 pyminify 去除注释/空行/文档字符串等, 再用 LZMA 压缩, 最后 base85 编码
# 返回 (原始字节数, 最终包装后字节数), 用于计算总提交大小
def _compressed_code_size(code):
    code_raw = code.encode("utf-8")
    minified = subprocess.run(
        ["pyminify", "--no-rename-locals", "--no-hoist-literals", "--remove-literal-statements", "-"],
        input=code_raw, capture_output=True, check=True,
    ).stdout
    compressed = lzma.compress(minified)
    encoded = base64.b85encode(compressed)
    wrapper = b'import lzma as L,base64 as B\nexec(L.decompress(B.b85decode("' + encoded + b'")))\n'
    return len(code_raw), len(wrapper)


# 序列化: 将训练好的模型量化并压缩保存到磁盘
# 步骤:
#   1. 先保存未量化模型并记录大小 (仅主进程)
#   2. 将 bank 格式权重展开为逐层格式 (_unbank_state_dict)
#   3. 用校准数据收集 Hessian 矩阵, 执行 GPTQ 混合精度量化
#   4. 将量化结果 (量化权重 + 元数据) 用 torch.save 序列化
#   5. 对序列化字节流做 byte_shuffle + LZMA/Brotli 压缩
#   6. 写入磁盘并报告总提交大小
def serialize(h, base_model, code):
    code_bytes_uncompressed, code_bytes = _compressed_code_size(code)
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size (uncompressed): {code_bytes_uncompressed} bytes")
        log(f"Code size (compressed): {code_bytes} bytes")
    sd_cpu = _unbank_state_dict(base_model.state_dict(), h.num_layers)
    device = torch.device("cuda", h.local_rank)
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    log("GPTQ:collecting Hessians from calibration data...")
    hessians = collect_hessians(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


# 反序列化: 从磁盘加载压缩的量化模型并还原为可推理的模型
# 步骤:
#   1. 创建空的 GPT 模型作为模板 (用于获取权重形状和 dtype)
#   2. 将模型 state_dict 展开为逐层格式 (作为反量化模板)
#   3. 从磁盘读取压缩的量化数据, 解压并 torch.load
#   4. 调用 dequantize_mixed 将量化权重还原为浮点权重
#   5. 用 _rebank_state_dict 重新堆叠回 bank 格式
#   6. 加载到评估模型中并返回
def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    flat_template = _unbank_state_dict(eval_model.state_dict(), h.num_layers)
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu"
    )
    deq_flat = dequantize_mixed(quant_state["w"], quant_state["m"], flat_template)
    head_dim = h.model_dim // h.num_heads
    kv_dim = h.num_kv_heads * head_dim
    hidden_dim = int(h.mlp_mult * h.model_dim)
    deq_state = _rebank_state_dict(deq_flat, h.num_layers, h.model_dim, kv_dim, hidden_dim)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


# =====================================================================================
# 验证评估 (Validation Evaluation)
# -------------------------------------------------------------------------------------
# BPB (Bits Per Byte) 是竞赛的核心指标: 衡量模型对文本的压缩效率
# BPB = (avg_loss / ln(2)) * (token_count / byte_count)
# 其中 avg_loss 是 token 级别的交叉熵, byte_count 是对应 UTF-8 字节数
# 需要精确统计每个 token 对应的字节数, 包括前导空格的处理
# =====================================================================================

# 从累积的 loss 和 byte 数量计算 BPB
# val_loss = loss_sum / token_count (平均 token 级交叉熵)
# val_bpb = val_loss / ln(2) * (token_count / byte_count) (转换为 bits-per-byte)
def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


# 标准验证评估: 在验证集上计算 BPB
# 特点:
#   1. 将验证 tokens 按 eval_seq_len 切分为序列, 分配给各 rank 并行处理
#   2. 使用 BOS 标记定位文档边界, 构建 cu_seqlens 以支持 flash attention 的变长序列
#   3. 支持 CaseOps (大小写操作): 如果启用, 从 sidecar 文件读取每 token 的字节预算
#      否则通过 base_bytes_lut (基础字节查找表) + has_leading_space_lut 计算字节数
#   4. 分布式场景下用 all_reduce 汇总各 rank 的 loss/token/byte 统计量
def eval_val(h, device, val_data, model, forward_logits_fn=None):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size

    # TODO: Don't truncate this.
    seq_end = seq_start + ((seq_end - seq_start) // local_batch_seqs) * local_batch_seqs

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    run_forward_logits = (
        (model.module.forward_logits if hasattr(model, "module") else model.forward_logits)
        if forward_logits_fn is None
        else forward_logits_fn
    )
    model.eval()
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1]
            y = local[1:]
            bos_pos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = _build_cu_seqlens(
                bos_pos, x.numel(), x.device, h.eval_seq_len, 64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = run_forward_logits(
                    x[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                ).detach()
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="none",
            )
            val_loss_sum += per_token_loss.to(torch.float64).sum()
            val_token_count += float(y.numel())
            prev_ids = x
            tgt_ids = y
            if val_data.caseops_enabled and val_data.val_bytes is not None:
                # CaseOps: read per-token byte budget from sidecar at the same
                # global positions as the target tokens y. raw_start/raw_end
                # span [raw_start, raw_end), x = local[:-1], y = local[1:],
                # so y is at sidecar positions [raw_start + 1, raw_end).
                sidecar_slice = val_data.val_bytes[raw_start + 1 : raw_end].to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
                val_byte_count += sidecar_slice.to(torch.float64).sum()
            else:
                token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (
                    val_data.has_leading_space_lut[tgt_ids]
                    & ~val_data.is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


# =====================================================================================
# TTT (Test-Time Training) 辅助工具
# -------------------------------------------------------------------------------------
# TTT 的核心思想: 在推理/评估时对模型做额外的微调, 让模型适应特定的验证数据分布。
# 这里使用两层 TTT 策略:
#   - Global SGD: 在验证数据的大块 (chunk) 上对全部参数做 SGD 微调
#   - Per-doc LoRA: 对每个文档独立训练低秩适配器 (LoRA), 做细粒度适应
# =====================================================================================

# 查找验证 token 序列中的文档边界
# 通过 BOS (Beginning of Sentence) 标记定位每个文档的起始位置
# 返回列表 [(start, length), ...], 每个元素表示一个文档的起始位置和长度
# 注意: 最后一个文档延伸到序列末尾, 中间文档包含下一个 BOS (+1)
def _find_docs(all_tokens):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = (
            int(bos_positions[i + 1])
            if i + 1 < len(bos_positions)
            else all_tokens.numel()
        )
        if i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


# 构建 TTT 全局批次: 将文档按长度排序后分批
# 按文档长度排序是为了让同一批次内的文档长度相近, 减少 padding 浪费
# 默认按长度降序排列 (最长文档优先), 因为长文档的 TTT 耗时更长, 优先处理可以更好地平衡负载
# ascending=True 时按升序排列, 用于调试场景 (eval_batch_set 指定特定批次)
def _build_ttt_global_batches(doc_entries, h, ascending=False):
    batch_size = h.ttt_batch_size
    global_doc_entries = sorted(doc_entries, key=lambda x: x[1][1])
    global_batches = [
        global_doc_entries[i : i + batch_size]
        for i in range(0, len(global_doc_entries), batch_size)
    ]
    indexed = list(enumerate(global_batches))
    if not ascending:
        indexed.sort(key=lambda ib: -max(dl for _, (_, dl) in ib[1]))
    return indexed


# =====================================================================================
# 基于文件的原子计数器 (File-Based Atomic Counters)
# -------------------------------------------------------------------------------------
# 用于多进程/多 GPU 分布式 TTT 的工作窃取 (work stealing) 协调机制。
# 各 worker 通过文件锁 (fcntl.LOCK_EX) 原子地领取下一个待处理批次,
# 避免了需要额外进程间通信的开销, 同时保证无冲突。
# =====================================================================================

# 初始化 4 字节的批次计数器文件, 初始值为 0
def _init_batch_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(4, "little"))


# 原子地领取下一个批次索引 (工作窃取)
# 使用文件级排他锁 (LOCK_EX) 保证多进程安全:
#   1. 读取当前计数器值 idx
#   2. 将计数器递增为 idx+1 并写回
#   3. 返回 idx 作为该 worker 要处理的批次编号
# 如果 idx >= queue_len, 表示所有批次已被领取完毕
def _claim_next_batch(counter_path, queue_len):
    try:
        with open(counter_path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            idx = int.from_bytes(f.read(4), "little")
            f.seek(0)
            f.write((idx + 1).to_bytes(4, "little"))
            f.flush()
    except FileNotFoundError:
        return queue_len
    return idx


# 计算 TTT 分块评估的滑动窗口参数
# ci: 当前 chunk 索引; pred_len: 文档的预测长度 (文档长度 - 1)
# 返回 (win_start, win_len, chunk_offset, chunk_len):
#   - win_start: 窗口在文档中的起始位置 (保证窗口不超过 eval_seq_len)
#   - win_len: 窗口长度 (即送入模型的上下文长度)
#   - chunk_offset: 当前 chunk 在窗口中的偏移量 (用于定位评估/训练区域)
#   - chunk_len: 当前 chunk 的实际长度
# 关键: 每个 chunk 的上下文窗口会向前扩展以获取更多上下文, 但不超过 eval_seq_len
def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_start = ci * chunk_size
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


# 累积 BPB 统计量: 只计算属于当前 chunk 范围内的 token 的 loss 和字节数
# 通过 chunk_offsets 和 chunk_lens 构建掩码, 排除上下文窗口中属于前一 chunk 的 token
# 这保证每个 token 只在它所属的 chunk 中被计算一次, 避免重复统计
# ptl: per-token loss 矩阵 [batch, seq_len]
# y_bytes: 如果提供 (CaseOps 模式), 直接使用 sidecar 的字节预算
def _accumulate_bpb(
    ptl,
    x,
    y,
    chunk_offsets,
    chunk_lens,
    pos_idx,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    loss_sum,
    byte_sum,
    token_count,
    y_bytes=None,
):
    pos = pos_idx[: x.size(1)].unsqueeze(0)
    mask = (
        (chunk_lens.unsqueeze(1) > 0)
        & (pos >= chunk_offsets.unsqueeze(1))
        & (pos < (chunk_offsets + chunk_lens).unsqueeze(1))
    )
    mask_f64 = mask.to(torch.float64)
    if y_bytes is not None:
        tok_bytes = y_bytes.to(torch.float64)
    else:
        tok_bytes = base_bytes_lut[y].to(torch.float64)
        tok_bytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(
            torch.float64
        )
    loss_sum += (ptl.to(torch.float64) * mask_f64).sum()
    byte_sum += (tok_bytes * mask_f64).sum()
    token_count += chunk_lens.to(torch.float64).sum()


# 从累积的 loss/token/byte 求最终 BPB (与 _loss_bpb 类似但参数顺序不同)
def _loss_bpb_from_sums(loss_sum, token_count, byte_sum):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())
    return val_loss, val_bpb


# 原子地将 delta 累加到 8 字节有符号整数计数器文件
# 用于跟踪已处理的前缀文档数量, 判断是否达到阶段边界需要暂停做 global SGD
def _add_to_counter(path, delta):
    try:
        with open(path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            cur = int.from_bytes(f.read(8), "little", signed=True)
            cur += int(delta)
            f.seek(0)
            f.write(int(cur).to_bytes(8, "little", signed=True))
            f.flush()
            return cur
    except FileNotFoundError:
        return int(delta)


# 初始化 8 字节有符号整数计数器文件, 初始值为 0
def _init_int64_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(8, "little", signed=True))


# 选择参与 TTT 的文档子集
# 如果 val_doc_fraction < 1.0, 则按固定种子随机采样指定比例的文档
# 这可以在调试或快速评估时减少 TTT 的计算量
def _select_ttt_doc_entries(docs, h):
    doc_entries = list(enumerate(docs))
    if h.val_doc_fraction < 1.0:
        sample_n = max(1, int(round(len(docs) * h.val_doc_fraction)))
        sampled_indices = sorted(
            random.Random(h.seed).sample(range(len(docs)), sample_n)
        )
        return [(i, docs[i]) for i in sampled_indices]
    return doc_entries


# =====================================================================================
# 全局 SGD TTT (Test-Time Training with Global SGD)
# -------------------------------------------------------------------------------------
# 在 TTT 的分阶段评估中, 每完成一个阶段的逐文档 LoRA 评估后,
# 收集该阶段已评估的文档 token, 对模型全部参数做 SGD 微调。
# 这样模型可以从已见过的验证数据中学习全局分布特征。
#
# 关键设计:
#   1. 将 val_tokens 按 global_ttt_chunk_tokens 分块, 最后一块只评估不训练
#   2. 学习率调度: warmup 阶段线性升温, 之后余弦退火 (cosine decay)
#   3. 分布式: 每个 rank 处理自己分配到的序列, 梯度通过 all_reduce 平均
#   4. 支持文档边界感知: 可选地在 flash attention 中使用 cu_seqlens 分隔文档
#   5. 可选的梯度裁剪 (global_ttt_grad_clip)
# =====================================================================================
def train_val_ttt_global_sgd_distributed(h, device, val_data, base_model, val_tokens, batch_seqs=None):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    seq_len = h.eval_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = h.global_ttt_chunk_tokens
    batch_seqs = h.global_ttt_batch_seqs if batch_seqs is None else batch_seqs
    # 将验证 token 分成 num_chunks 块, 每块大小约 global_ttt_chunk_tokens
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    # 对模型全部参数启用梯度, 使用 SGD 优化器 (带动量)
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        ttt_params, lr=h.global_ttt_lr, momentum=h.global_ttt_momentum
    )
    t_start = time.perf_counter()
    for ci in range(num_chunks):
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        # 最后一块只评估不训练 (避免在评估数据上过拟合)
        is_last_chunk = ci == num_chunks - 1
        if is_last_chunk or h.global_ttt_epochs <= 0:
            continue
        base_model.train()
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs <= 0:
            continue
        # 学习率调度: warmup 阶段线性从 warmup_start_lr 升到 global_ttt_lr
        warmup_chunks = max(0, min(h.global_ttt_warmup_chunks, num_chunks - 1))
        if warmup_chunks > 0 and ci < warmup_chunks:
            warmup_denom = max(warmup_chunks - 1, 1)
            warmup_t = ci / warmup_denom
            lr_now = (
                h.global_ttt_warmup_start_lr
                + (h.global_ttt_lr - h.global_ttt_warmup_start_lr) * warmup_t
            )
        else:
            # warmup 结束后用余弦退火调度学习率
            decay_steps = max(num_chunks - 1 - warmup_chunks, 1)
            decay_ci = max(ci - warmup_chunks, 0)
            lr_now = h.global_ttt_lr * 0.5 * (
                1.0 + math.cos(math.pi * decay_ci / decay_steps)
            )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        # 将当前 chunk 的序列均匀分配给各 rank (分布式数据并行)
        my_seq_s = chunk_seqs * h.rank // h.world_size
        my_seq_e = chunk_seqs * (h.rank + 1) // h.world_size
        my_chunk_seqs = my_seq_e - my_seq_s
        for _ in range(h.global_ttt_epochs):
            for bs in range(0, my_chunk_seqs, batch_seqs):
                be = min(bs + batch_seqs, my_chunk_seqs)
                actual_bs = my_seq_s + bs
                start_tok = chunk_start + actual_bs * seq_len
                end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                if end_tok > val_tokens.numel():
                    continue
                local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                x_flat = local[:-1]
                y_flat = local[1:]
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if h.global_ttt_respect_doc_boundaries:
                            bos_pos = (x_flat == BOS_ID).nonzero(as_tuple=True)[0].tolist()
                            cu_seqlens, max_seqlen = _build_cu_seqlens(
                                bos_pos, x_flat.numel(), x_flat.device, h.eval_seq_len, 64
                            )
                            loss = base_model(
                                x_flat[None],
                                y_flat[None],
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen,
                            )
                        else:
                            x = x_flat.reshape(-1, seq_len)
                            y = y_flat.reshape(-1, seq_len)
                            loss = base_model(x, y)
                loss.backward()
                # 分布式梯度同步: all_reduce 求和后除以 world_size 取平均
                if dist.is_available() and dist.is_initialized():
                    for p in ttt_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.mul_(1.0 / h.world_size)
                if h.global_ttt_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ttt_params, h.global_ttt_grad_clip)
                optimizer.step()
        base_model.eval()
        if h.rank == 0:
            elapsed = time.perf_counter() - t_start
            log(
                f"tttg: c{ci+1}/{num_chunks} lr:{lr_now:.6f} t:{elapsed:.1f}s"
            )
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()


# =====================================================================================
# 分阶段 TTT 评估 (Phased Test-Time Training Evaluation)
# -------------------------------------------------------------------------------------
# 这是 TTT 的核心评估函数, 采用两层适应策略:
#
# 第一层 - 逐文档 LoRA 适应 (Per-document LoRA):
#   - 将文档按长度分批, 各 GPU 通过工作窃取 (work stealing) 领取批次
#   - 每个批次内创建批量化的 LoRA (BatchedTTTLoRA), 对每个文档独立训练
#   - 文档被分成 chunk, 每个 chunk: 先前向推理评估, 再反向传播训练 LoRA
#   - 只有非最后 chunk 做训练, 最后 chunk 只评估 (因果性: 不能用未来数据训练)
#
# 第二层 - 全局 SGD 适应 (Global SGD between phases):
#   - 将文档分为"前缀"和"后缀"两部分, 前缀文档进一步按阶段 (phase) 划分
#   - 每完成一个阶段的前缀文档评估后, 暂停逐文档处理
#   - 收集已评估文档的 token, 对基础模型做全局 SGD 微调
#   - 微调后重建 LoRA, 继续下一阶段的逐文档评估
#   - 这让模型能从已评估的验证数据中学习全局分布
#
# 分布式协调:
#   - 使用基于文件的原子计数器实现工作窃取 (无需集中调度)
#   - 使用暂停标志文件 (pause_flag) 协调各 rank 同步进入全局 SGD 阶段
#   - all_gather_object 收集各 rank 的已评估文档列表
# =====================================================================================
def eval_val_ttt_phased(h, base_model, device, val_data, forward_ttt_train):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    # 冻结基础模型参数 (LoRA 适配器的参数会单独训练)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    all_tokens = val_data.val_tokens
    all_tokens_idx = all_tokens.to(torch.int32)
    # 查找所有文档边界并选择参与 TTT 的文档子集
    docs = _find_docs(all_tokens)
    doc_entries = _select_ttt_doc_entries(docs, h)
    # 前缀文档: 参与分阶段全局 SGD 的文档数量
    # 后缀文档: 只做 LoRA 适应不参与全局 SGD 的文档
    prefix_doc_limit = max(0, min(len(doc_entries), int(h.phased_ttt_prefix_docs)))
    # 阶段划分: 将前缀文档均分为 num_phases 个阶段
    num_phases = max(1, int(h.phased_ttt_num_phases))
    phase_boundaries = []
    for pi in range(num_phases):
        boundary = prefix_doc_limit * (pi + 1) // num_phases
        phase_boundaries.append(boundary)
    current_phase = 0
    current_phase_boundary = phase_boundaries[0]
    log(
        "ttt_phased:"
        f" total_docs:{len(doc_entries)} prefix_docs:{prefix_doc_limit} "
        f"suffix_docs:{len(doc_entries) - prefix_doc_limit}"
        f" num_phases:{num_phases} boundaries:{phase_boundaries}"
    )
    chunk_size, eval_seq_len = h.ttt_chunk_size, h.ttt_eval_seq_len
    # 可选: 仅评估指定批次 (调试用)
    eval_batch_set = None
    if h.ttt_eval_batches:
        eval_batch_set = set(int(x) for x in h.ttt_eval_batches.split(",") if x.strip())
    use_ascending = eval_batch_set is not None
    # 构建全局批次队列 (按文档长度排序)
    global_batches_sorted = _build_ttt_global_batches(
        doc_entries, h, ascending=use_ascending
    )
    queue_len = len(global_batches_sorted)
    # 初始化分布式协调用的文件路径
    # counter_path: 批次领取计数器
    # prefix_counter_path: 已处理前缀文档计数器
    # pause_flag_path: 暂停标志文件 (触发全局 SGD 阶段)
    counter_path = f"/tmp/ttt_counter_{h.run_id}"
    prefix_counter_path = f"/tmp/ttt_prefix_counter_{h.run_id}"
    pause_flag_path = f"/tmp/ttt_pause_flag_{h.run_id}"
    if h.rank == 0:
        _init_batch_counter(counter_path)
        _init_int64_counter(prefix_counter_path)
        try:
            os.remove(pause_flag_path)
        except FileNotFoundError:
            pass
    if dist.is_available() and dist.is_initialized():
        path_list = [counter_path, prefix_counter_path, pause_flag_path]
        dist.broadcast_object_list(path_list, src=0)
        counter_path, prefix_counter_path, pause_flag_path = path_list
        dist.barrier()
    # 累积统计量 (使用 float64 避免精度损失)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    t_start = time.perf_counter()
    # 预创建可复用的 LoRA 适配器和优化器, 避免每个批次重新分配显存
    reusable_lora = BatchedTTTLoRA(
        h.ttt_batch_size, base_model, h.ttt_lora_rank,
        k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
    ).to(device)

    # 构建 LoRA 优化器: 支持 SGD 或 AdamW (fused 版本)
    def _build_opt(lora):
        if h.ttt_optimizer == "sgd":
            return torch.optim.SGD(
                lora.parameters(), lr=h.ttt_lora_lr,
                momentum=h.ttt_beta1, weight_decay=h.ttt_weight_decay,
            )
        return torch.optim.AdamW(
            lora.parameters(), lr=h.ttt_lora_lr,
            betas=(h.ttt_beta1, h.ttt_beta2),
            eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True,
        )

    reusable_opt = _build_opt(reusable_lora)
    local_scored_docs = []  # 当前 rank 已处理的前缀文档记录
    global_ttt_done = prefix_doc_limit == 0  # 如果没有前缀文档, 跳过全局 SGD
    try:
      # 主循环: 通过工作窃取不断领取并处理批次
      while True:
        # 原子地领取下一个批次 (多 GPU 竞争同一个计数器)
        queue_idx = _claim_next_batch(counter_path, queue_len)
        if queue_idx >= queue_len:
            break
        orig_batch_idx, batch_entries = global_batches_sorted[queue_idx]
        batch = [doc for _, doc in batch_entries]
        bsz = len(batch)
        # 记录当前累积统计量, 用于后续计算本批次的增量 BPB
        prev_loss = loss_sum.item()
        prev_bytes = byte_sum.item()
        prev_tokens = token_count.item()
        # 如果批次大小匹配预创建的 LoRA, 直接复用 (重置参数和优化器状态)
        # 否则创建新的 LoRA 和优化器 (发生在最后一个不完整批次)
        if bsz == reusable_lora.bsz:
            reusable_lora.reset()
            for s in reusable_opt.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        v.zero_()
                    elif k == "step":
                        s[k] = 0
            cur_lora = reusable_lora
            cur_opt = reusable_opt
        else:
            cur_lora = BatchedTTTLoRA(
                bsz, base_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            cur_opt = _build_opt(cur_lora)
        # 计算每个文档的预测长度和 chunk 数量
        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        num_chunks_t = torch.tensor(num_chunks, dtype=torch.int64, device=device)
        # 逐 chunk 处理: 先评估 (所有 chunk), 再训练 LoRA (仅非最后 chunk)
        for ci in range(max_nc):
            # active[b]: 文档 b 是否还有 chunk 需要处理
            active = [ci < nc for nc in num_chunks]
            # needs_train: 是否有任何文档的当前 chunk 不是最后一个 (需要训练)
            needs_train = any(ci < nc - 1 for nc in num_chunks)
            # 为批次中每个文档计算当前 chunk 的滑动窗口参数
            tok_starts = torch.zeros(bsz, dtype=torch.int64)
            tok_wls = torch.zeros(bsz, dtype=torch.int64)
            chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
            chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
            for b in range(bsz):
                if not active[b]:
                    continue
                doc_start, doc_len = batch[b]
                win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(
                    ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len
                )
                tok_starts[b] = doc_start + win_start
                tok_wls[b] = win_len
                chunk_offsets_cpu[b] = chunk_offset
                chunk_lens_cpu[b] = chunk_len
            # 统一上下文大小: 所有文档使用相同的矩阵维度 (不活跃文档用 valid 掩码屏蔽)
            _, context_size, chunk_offset, _ = _compute_chunk_window(
                ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len
            )
            # 从全局 token 数组中收集每个文档的窗口 token
            col_idx = torch.arange(context_size + 1)
            idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
            idx.clamp_(max=all_tokens.numel() - 1)
            gathered_gpu = all_tokens_idx[idx].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            # valid 掩码: 屏蔽超出文档边界的位置
            valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(
                device, non_blocking=True
            )
            chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)
            chunk_lens = chunk_lens_cpu.to(device, non_blocking=True)
            x = torch.where(valid, gathered_gpu[:, :context_size], 0)
            y = torch.where(valid, gathered_gpu[:, 1 : context_size + 1], 0)
            ctx_pos = torch.arange(context_size, device=device, dtype=torch.int64)
            # 前向推理: 计算带 LoRA 适配器的 per-token loss
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
            # CaseOps sidecar-driven byte budget. Mirror the index pattern
            # used to build y from all_tokens: y[b, j] corresponds to the
            # token at global position tok_starts[b] + 1 + j (when valid).
            y_bytes_arg = None
            if val_data.caseops_enabled and val_data.val_bytes is not None:
                y_idx = (
                    tok_starts.unsqueeze(1)
                    + 1
                    + col_idx[:context_size].unsqueeze(0)
                )
                y_idx = y_idx.clamp_(max=val_data.val_bytes.numel() - 1)
                y_bytes_arg = val_data.val_bytes[y_idx].to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
                # Mirror the `valid` masking used for y so out-of-range tokens
                # contribute zero bytes (matches y=0 substitution above).
                y_bytes_arg = torch.where(
                    valid, y_bytes_arg, torch.zeros_like(y_bytes_arg)
                )
            # 累积当前 chunk 的 BPB 统计量 (仅计算 chunk 范围内的 token)
            with torch.no_grad():
                _accumulate_bpb(
                    per_tok_loss,
                    x,
                    y,
                    chunk_offsets,
                    chunk_lens,
                    ctx_pos,
                    val_data.base_bytes_lut,
                    val_data.has_leading_space_lut,
                    val_data.is_boundary_token_lut,
                    loss_sum,
                    byte_sum,
                    token_count,
                    y_bytes=y_bytes_arg,
                )
            # 训练 LoRA: 仅在非最后 chunk 时执行 (因果性约束)
            # activate_chunk_mask: 只对尚未到达最后 chunk 的文档计算梯度
            if needs_train:
                activate_chunk_mask = (num_chunks_t - 1 > ci).float()
                # 多步梯度更新 (ttt_grad_steps): 每步重新前向计算 loss
                for gi in range(h.ttt_grad_steps):
                    if gi > 0:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                    # 只取当前 chunk 区域的 loss 均值作为训练信号
                    per_doc = per_tok_loss[
                        :, chunk_offset : chunk_offset + chunk_size
                    ].mean(dim=-1)
                    cur_opt.zero_grad(set_to_none=True)
                    # 用 activate_chunk_mask 屏蔽已完成的文档, 避免在最后 chunk 上训练
                    (per_doc * activate_chunk_mask).sum().backward()
                    cur_opt.step()
            else:
                del per_tok_loss
        # 日志报告: 计算本批次的增量 BPB 和全局累积 BPB
        batch_num = orig_batch_idx + 1
        doc_lens = [dl for _, dl in batch]
        should_report = batch_num in eval_batch_set if eval_batch_set is not None else True
        if should_report:
            cur_tokens = token_count.item()
            cur_loss_val = loss_sum.item()
            cur_bytes_val = byte_sum.item()
            dt = cur_tokens - prev_tokens
            db = cur_bytes_val - prev_bytes
            if dt > 0 and db > 0:
                b_loss = (cur_loss_val - prev_loss) / dt
                b_bpb = b_loss / math.log(2.0) * (dt / db)
            else:
                b_loss = b_bpb = 0.0
            r_loss = cur_loss_val / max(cur_tokens, 1)
            r_bpb = r_loss / math.log(2.0) * (cur_tokens / max(cur_bytes_val, 1))
            elapsed = time.perf_counter() - t_start
            log(
                f"ttp: b{batch_num}/{queue_len} bl:{b_loss:.4f} bb:{b_bpb:.4f} "
                f"rl:{r_loss:.4f} rb:{r_bpb:.4f} dl:{min(doc_lens)}-{max(doc_lens)} "
                f"gd:{int(global_ttt_done)}"
            )
        # 阶段转换逻辑: 检查是否达到当前阶段边界, 需要暂停做全局 SGD
        if not global_ttt_done:
            # 记录已处理的前缀文档, 供全局 SGD 使用
            local_scored_docs.extend(
                (orig_batch_idx, pos, doc_start, doc_len)
                for pos, (doc_start, doc_len) in enumerate(batch)
            )
            # 原子递增前缀计数器, 检查是否达到阶段边界
            prefix_done = _add_to_counter(prefix_counter_path, len(batch_entries))
            if prefix_done >= current_phase_boundary:
                try:
                    with open(pause_flag_path, "x"):
                        pass
                except FileExistsError:
                    pass
            # 如果暂停标志文件存在, 进入全局 SGD 阶段
            should_pause = os.path.exists(pause_flag_path)
            if should_pause:
                # 所有 rank 同步等待
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                # 收集所有 rank 的已处理文档列表
                gathered_scored_docs = [None] * h.world_size
                if dist.is_available() and dist.is_initialized():
                    dist.all_gather_object(gathered_scored_docs, local_scored_docs)
                else:
                    gathered_scored_docs = [local_scored_docs]
                # 合并、排序并截断到当前阶段边界数量的文档
                scored_docs_for_global = []
                for rank_docs in gathered_scored_docs:
                    if rank_docs:
                        scored_docs_for_global.extend(rank_docs)
                scored_docs_for_global.sort(key=lambda x: (x[0], x[1]))
                scored_docs_for_global = scored_docs_for_global[:current_phase_boundary]
                # 收集这些文档的 token 拼接成全局 SGD 训练数据
                scored_token_chunks = [
                    val_data.val_tokens[doc_start : doc_start + doc_len]
                    for _, _, doc_start, doc_len in scored_docs_for_global
                ]
                if scored_token_chunks:
                    global_ttt_tokens = torch.cat(scored_token_chunks)
                else:
                    global_ttt_tokens = val_data.val_tokens[:0]
                if h.rank == 0:
                    prefix_done = 0
                    try:
                        with open(prefix_counter_path, "rb") as f:
                            prefix_done = int.from_bytes(
                                f.read(8), "little", signed=True
                            )
                    except FileNotFoundError:
                        pass
                    log(
                        f"ttpp: phase:{current_phase + 1}/{num_phases} pd:{prefix_done} "
                        f"gd:{len(scored_docs_for_global)} "
                        f"t:{time.perf_counter() - t_start:.1f}s"
                    )
                # 执行全局 SGD: 在已评估文档的 token 上对基础模型做全量参数微调
                train_val_ttt_global_sgd_distributed(
                    h, device, val_data, base_model, global_ttt_tokens
                )
                # 全局 SGD 后重新冻结基础模型, 重建 LoRA 适配器
                for p in base_model.parameters():
                    p.requires_grad_(False)
                reusable_lora = BatchedTTTLoRA(
                    h.ttt_batch_size, base_model, h.ttt_lora_rank,
                    k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
                ).to(device)
                reusable_opt = _build_opt(reusable_lora)
                # 推进到下一阶段
                current_phase += 1
                if current_phase >= num_phases:
                    # 所有阶段完成, 后续批次不再触发全局 SGD
                    global_ttt_done = True
                else:
                    current_phase_boundary = phase_boundaries[current_phase]
                    # 移除暂停标志, 允许 worker 继续领取批次
                    if h.rank == 0:
                        try:
                            os.remove(pause_flag_path)
                        except FileNotFoundError:
                            pass
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                if h.rank == 0:
                    log(f"ttpr: phase:{current_phase}/{num_phases} t:{time.perf_counter() - t_start:.1f}s")
        del cur_lora, cur_opt
    finally:
        pass
    # 汇总所有 rank 的统计量, 计算最终 BPB
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    # 恢复模型状态: 重新启用梯度, 切换回训练模式
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.train()
    return _loss_bpb_from_sums(loss_sum, token_count, byte_sum)


# 计时评估包装器: 在评估前后同步 CUDA, 精确测量评估耗时
def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(
        f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms"
    )
    return val_loss, val_bpb


# =====================================================================================
# 训练循环 (Training Loop)
# -------------------------------------------------------------------------------------
# 核心训练流程, 包含以下关键机制:
#
# 1. 编译优化: 使用 torch.compile (fullgraph=True) 编译模型和推理函数
# 2. Warmup 阶段:
#    - 先用随机初始化跑若干步, 触发 torch.compile 编译缓存 (cu_bucket warmup)
#    - 支持多种 cu_seqlens 长度的编译预热, 避免运行时重编译
#    - 如果启用 layer looping, 还需额外预热 looping 模式
#    - warmup 完成后恢复原始模型权重和优化器状态 (不保留 warmup 的参数更新)
# 3. EMA (Exponential Moving Average):
#    - 训练过程中维护参数的指数移动平均, 最终用 EMA 权重替换模型权重
#    - EMA 能平滑训练噪声, 通常比最后一步的权重泛化更好
# 4. 挂钟时间限制 (Wallclock Cap):
#    - 支持按实际运行时间停止训练 (而非固定步数)
#    - 预留 GPTQ 量化所需时间 (gptq_reserve_seconds)
#    - 分布式场景下任一 rank 达到时间上限则所有 rank 同步停止
# 5. 学习率调度:
#    - warmdown: 在训练末尾按进度线性衰减学习率
#    - Muon 动量预热: Muon 优化器的动量从低值线性增加到目标值
# 6. Layer Looping: 可在训练中途激活 (enable_looping_at), 增加有效深度
# =====================================================================================
def train_model(h, device, val_data):
    # 创建模型, 转为 bfloat16, 恢复需要 fp32 精度的参数 (如 layer norm)
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    # torch.compile 编译模型: fullgraph=True 要求完整图编译, 性能更好
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(
        base_model.forward_logits, dynamic=False, fullgraph=True
    )
    model = compiled_model
    log(f"model_params:{sum(p.numel()for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)
    train_loader = DocumentPackingLoader(h, device)
    # 计算有效训练时间上限: 总时间 - GPTQ 预留时间
    max_wallclock_ms = (
        1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    )
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        log(
            f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms"
        )

    # 计算训练进度 (0~1): 优先使用挂钟时间, 否则按步数计算
    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    # 学习率衰减倍率: 在训练末尾 (warmdown_frac 比例) 线性衰减到 min_lr
    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    # 单步训练函数: 梯度累积 + Muon 动量预热 + 学习率缩放 + 梯度裁剪 + 优化器更新
    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            x, y, cu_seqlens, _max_seqlen = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, cu_seqlens=cu_seqlens, max_seqlen=h.train_seq_len)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = (
            min(step / h.muon_momentum_warmup_steps, 1.0)
            if h.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step(distributed=h.distributed)
        return train_loss

    # ==================== Warmup (编译预热) ====================
    # 目的: 触发 torch.compile 对各种输入形状的编译, 避免训练中途重编译导致停顿
    # 关键: warmup 结束后会恢复原始权重和优化器状态, warmup 不影响最终模型
    if h.warmup_steps > 0:
        # 保存初始状态, warmup 结束后恢复
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for (name, tensor) in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        # 预计算旋转位置编码 (RoPE)
        num_tokens_local = h.train_batch_tokens // h.world_size
        for blk in base_model.blocks:
            blk.attn.rotary(num_tokens_local, device, torch.bfloat16)
        # cu_bucket warmup: 用不同的 cu_seqlens 桶大小触发编译
        # flash attention 的 cu_seqlens 长度会变化, 需要预编译各种情况
        cu_bucket_size = train_loader.cu_bucket_size
        warmup_cu_buckets = tuple(cu_bucket_size * i for i in range(1, 5))
        warmup_cu_iters = 3
        x, y, cu_seqlens, _ = train_loader.next_batch(
            h.train_batch_tokens, h.grad_accum_steps
        )
        log(f"warmup_cu_buckets:{','.join(str(b) for b in warmup_cu_buckets)} iters_each:{warmup_cu_iters}")
        def _run_cu_bucket_warmup():
            for bucket_len in warmup_cu_buckets:
                boundaries = list(range(0, x.size(1), max(h.train_seq_len, 1)))
                if boundaries[-1] != x.size(1):
                    boundaries.append(x.size(1))
                cu = torch.full((bucket_len,), x.size(1), dtype=torch.int32, device=device)
                cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
                for _ in range(warmup_cu_iters):
                    optimizers.zero_grad_all()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        wloss = model(x, y, cu_seqlens=cu, max_seqlen=h.train_seq_len)
                    (wloss / h.grad_accum_steps).backward()
            optimizers.zero_grad_all()
        _run_cu_bucket_warmup()
        # 如果启用 layer looping, 还需在 looping 模式下预热编译
        if h.num_loops > 0:
            base_model.looping_active = True
            _run_cu_bucket_warmup()
            base_model.looping_active = False
        # 正式 warmup 步: 运行完整的训练步 (包括数据加载和优化器更新)
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if (
                warmup_step <= 5
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == h.warmup_steps
            ):
                log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
        # Layer looping warmup: 在 looping 模式下再跑一轮 warmup
        if h.num_loops > 0:
            base_model.looping_active = True
            log(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if (
                    warmup_step <= 5
                    or (warmup_step + 1) % 10 == 0
                    or warmup_step + 1 == h.warmup_steps
                ):
                    log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
            base_model.looping_active = False
        # Warmup 结束: 恢复原始模型权重和优化器状态
        # 重要: warmup 只是为了编译缓存, 不应影响训练起点
        base_model.load_state_dict(initial_model_state, strict=True)
        for (opt, state) in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        # 重新创建数据加载器 (因为 warmup 消耗了部分数据)
        train_loader = DocumentPackingLoader(h, device)
    # ==================== 初始化 EMA (指数移动平均) ====================
    # 在 float32 精度下维护参数的移动平均: ema = decay * ema + (1-decay) * param
    ema_state = {
        name: t.detach().float().clone()
        for (name, t) in base_model.state_dict().items()
    }
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None  # 挂钟时间到达后设置, 延迟一步停止以完成当前评估
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # ==================== 主训练循环 ====================
    step = 0
    while True:
        # 检查是否为最后一步 (达到目标步数或挂钟时间上限)
        last_step = (
            step == h.iterations
            or stop_after_step is not None
            and step >= stop_after_step
        )
        # 验证: 在最后一步和每 val_loss_every 步时运行
        should_validate = (
            last_step or h.val_loss_every > 0 and step % h.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                h, device, val_data, model, compiled_forward_logits
            )
            log(
                f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            # 如果因挂钟时间限制提前停止, 记录日志
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}"
                )
            break
        # 计算当前训练进度和学习率缩放
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        # 在训练进行到 enable_looping_at 比例时激活 layer looping
        # Layer looping 让浅层权重被重复使用, 增加有效深度而不增加参数量
        if (
            h.num_loops > 0
            and not base_model.looping_active
            and frac >= h.enable_looping_at
        ):
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        train_loss = step_fn(step, scale)
        # 每步更新 EMA: ema = decay * ema + (1-decay) * current_param
        with torch.no_grad():
            for (name, t) in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(
                    t.detach().float(), alpha=1.0 - ema_decay
                )
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (
            step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )
        # 挂钟时间检查: 任一 rank 达到上限则所有 rank 同步停止
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if h.distributed and max_wallclock_ms is not None:
            # 用 MAX reduce 确保任一 rank 超时 -> 全部停止
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step  # 设置停止标记, 下一循环会执行最终评估后退出
    # ==================== 训练完成, 应用 EMA 权重 ====================
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB"
    )
    # 用 EMA 平均权重替换模型当前权重 (通常能提升泛化性能)
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {
        name: t.to(dtype=current_state[name].dtype) for (name, t) in ema_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model, compiled_forward_logits


# =====================================================================================
# 完整流水线 (Full Pipeline): 训练 -> 序列化 -> 反序列化 -> 评估 -> TTT 评估
# -------------------------------------------------------------------------------------
# 这是整个实验的入口函数, 按顺序执行:
#   1. 设置随机种子确保可复现性
#   2. 加载验证数据
#   3. 训练模型 (train_model) - 包含 warmup, EMA 等
#   4. 量化前诊断评估 (pre-quantization eval)
#   5. 序列化: GPTQ 量化 + 压缩 + 保存
#   6. 反序列化: 加载 + 解压 + 反量化
#   7. 量化后诊断评估 (post-quantization eval)
#   8. TTT 评估: 分阶段 LoRA 适应 + 全局 SGD
# 支持两种快捷模式:
#   - TTT_EVAL_ONLY=1: 跳过训练和 GPTQ, 直接从已保存的量化模型做 TTT 评估
#   - PREQUANT_ONLY=1: 只训练不量化, 用于快速检验训练效果
# =====================================================================================
def train_and_eval(h, device):
    # 设置所有随机种子 (Python, NumPy, PyTorch CPU/CUDA)
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    if h.artifact_dir and h.is_main_process:
        os.makedirs(h.artifact_dir, exist_ok=True)
    val_data = ValidationData(h, device)
    log(
        f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}"
    )
    log(f"val_tokens: {val_data.val_tokens.numel()-1}")
    # TTT_EVAL_ONLY: skip training + GPTQ, jump straight to TTT eval on a
    # pre-existing quantized artifact. Used to test TTT-only improvements
    # (e.g., PR-1767's alpha/warm-start/WD) without retraining.
    ttt_eval_only = os.environ.get("TTT_EVAL_ONLY", "0") == "1"
    if ttt_eval_only:
        log("TTT_EVAL_ONLY=1 — skipping training + GPTQ, loading saved artifact for TTT eval")
        log(f"ttt_lora_alpha: {BatchedLinearLoRA._ALPHA}")
        log(f"ttt_warm_start_a: {BatchedLinearLoRA._WARM_START_A}")
        log(f"ttt_weight_decay: {h.ttt_weight_decay}")
    else:
        # 第一步: 训练模型
        base_model, compiled_model, compiled_forward_logits = train_model(
            h, device, val_data
        )
        # 重置 dynamo 编译缓存, 避免与后续评估冲突
        torch._dynamo.reset()
        # 第二步: 量化前诊断评估 (EMA 后的浮点模型)
        timed_eval(
            "diagnostic pre-quantization post-ema",
            eval_val,
            h,
            device,
            val_data,
            compiled_model,
            compiled_forward_logits,
        )
        if os.environ.get("PREQUANT_ONLY", "0") == "1":
            log("PREQUANT_ONLY=1 — skipping serialize/GPTQ/post-quant eval/TTT")
            return
        # 第三步: 序列化 (GPTQ 量化 + 压缩)
        serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
        if h.distributed:
            dist.barrier()
    # 第四步: 反序列化 (加载量化模型)
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    # 第五步: 量化后诊断评估 (验证量化引起的精度损失)
    if not ttt_eval_only:
        compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
        compiled_forward_logits = torch.compile(
            eval_model.forward_logits, dynamic=False, fullgraph=True
        )
        timed_eval(
            "diagnostic quantized",
            eval_val,
            h,
            device,
            val_data,
            compiled_model,
            compiled_forward_logits,
        )
        del eval_model
    # ==================== 第六步: TTT 评估 ====================
    if h.ttt_enabled:
        # 释放之前的模型, 清理显存, 为 TTT 准备新的模型实例
        if not ttt_eval_only:
            del compiled_model
        if ttt_eval_only:
            del eval_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        # 重新反序列化一份模型用于 TTT (因为 TTT 会修改模型权重)
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        for p in ttt_model.parameters():
            p.requires_grad_(False)

        # 配置旋转位置编码 (RoPE): YaRN 扩展或标准 RoPE
        if h.rope_yarn:
            _yarn_seqlen = h.train_batch_tokens // h.grad_accum_steps
            for block in ttt_model.blocks:
                block.attn.rotary(_yarn_seqlen, device, torch.bfloat16)
        else:
            for block in ttt_model.blocks:
                block.attn.rotary._cos_cached = None
                block.attn.rotary._sin_cached = None
                block.attn.rotary._seq_len_cached = 0
                block.attn.rotary(h.ttt_eval_seq_len, device, torch.bfloat16)

        # 延迟编译 TTT 前向函数: 第一次调用时触发 torch.compile (dynamic=True)
        # dynamic=True 允许可变输入形状, 因为不同文档长度不同
        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

        _fwd_ttt_compiled_inner = None

        def _fwd_ttt(input_ids, target_ids, lora):
            nonlocal _fwd_ttt_compiled_inner
            if _fwd_ttt_compiled_inner is None:
                _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)

        fwd_ttt_compiled = _fwd_ttt
        # TTT 编译预热: 用随机 token 触发编译, 避免首次评估时编译延迟
        log(f"ttt_lora:warming up compile (random tokens, no val data)")
        global BOS_ID
        if BOS_ID is None:
            BOS_ID = 1
        t_warmup = time.perf_counter()
        # 预热两种序列长度 (ttt_chunk_size 和 ttt_eval_seq_len) 的编译路径
        warmup_bszes = [h.ttt_batch_size]
        for bsz in warmup_bszes:
            wl = BatchedTTTLoRA(
                bsz, ttt_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            wo = torch.optim.AdamW(
                wl.parameters(),
                lr=h.ttt_lora_lr,
                betas=(h.ttt_beta1, h.ttt_beta2),
                eps=1e-10,
                weight_decay=h.ttt_weight_decay,
                fused=True,
            )
            for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
                xw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                yw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = fwd_ttt_compiled(xw, yw, lora=wl)
                ptl[:, : min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
                wo.step()
                wo.zero_grad(set_to_none=True)
            del wl, wo
        torch.cuda.empty_cache()
        compile_elapsed = time.perf_counter() - t_warmup
        log(f"ttt_lora:compile warmup done ({compile_elapsed:.1f}s)")
        # 正式 TTT 评估: 精确计时 (同步 CUDA)
        log("\nbeginning TTT eval timer")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        # 执行分阶段 TTT 评估 (核心: 逐文档 LoRA + 全局 SGD)
        ttt_val_loss, ttt_val_bpb = eval_val_ttt_phased(
            h, ttt_model, device, val_data, forward_ttt_train=fwd_ttt_compiled
        )
        torch.cuda.synchronize()
        ttt_eval_elapsed = time.perf_counter() - t_ttt
        log(
            "quantized_ttt_phased "
            f"val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} "
            f"eval_time:{1e3*ttt_eval_elapsed:.0f}ms"
        )
        log(f"total_eval_time:{ttt_eval_elapsed:.1f}s")
        del ttt_model


# =====================================================================================
# 主函数 (Main Entry Point)
# -------------------------------------------------------------------------------------
# 负责:
#   1. 分布式环境初始化 (从环境变量读取 WORLD_SIZE, LOCAL_RANK, RANK)
#   2. CUDA 设备配置 (TF32, cuDNN, SDP 后端选择)
#   3. 超参数加载和日志初始化
#   4. 调用 train_and_eval 执行完整流水线
#   5. 分布式资源清理
# =====================================================================================
def main():
    # 从环境变量获取分布式配置 (torchrun/torch.distributed.launch 设置)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    # WORLD_SIZE 必须整除 8, 因为 grad_accum_steps = 8 // world_size
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # 初始化 NCCL 分布式通信后端
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    # CUDA 性能配置: 启用 TF32 以加速矩阵运算 (精度损失可忽略)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Scaled Dot Product Attention 后端选择: 只启用 Flash SDP (最快)
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    # torch.compile 配置: 禁用 DDP 优化 (使用自定义的梯度同步)
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 16
    # 加载超参数并初始化日志
    h = Hyperparameters()
    set_logging_hparams(h)
    # 仅主进程输出超参数和源代码到日志
    if h.is_main_process:
        os.makedirs(h.artifact_dir if h.artifact_dir else "logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for (k, v) in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log("Source code:", console=False)
        log("=" * 100, console=False)
        with open(__file__, "r", encoding="utf-8") as _src:
            log(_src.read(), console=False)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log("=" * 100, console=False)
    # 执行完整流水线: 训练 -> 序列化 -> 评估 -> TTT
    train_and_eval(h, device)
    # 清理分布式进程组
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
