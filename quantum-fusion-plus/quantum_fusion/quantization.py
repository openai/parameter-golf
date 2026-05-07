"""
量化模块: Hadamard旋转、AWQ显著性、分层量化、Hessian感知校准
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import math


class HadamardRotation:
    """Hadamard旋转,用于移除异常值"""
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.H = self._create_hadamard_matrix(hidden_size)
    
    def _create_hadamard_matrix(self, n: int) -> torch.Tensor:
        """递归创建Hadamard矩阵"""
        if n == 1:
            return torch.tensor([[1.0]])
        
        # 找到最接近的2的幂
        k = int(math.ceil(math.log2(n)))
        size = 2 ** k
        
        # 递归构造
        H = torch.tensor([[1.0]])
        for _ in range(k):
            H = torch.kron(H, torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        
        # 归一化
        H = H / math.sqrt(size)
        
        # 截断到所需大小
        if size > n:
            H = H[:n, :n]
        
        return H
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """应用Hadamard旋转"""
        H = self.H.to(x.device).to(x.dtype)
        x_rotated = torch.matmul(x, H.t())
        return x_rotated
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """逆旋转"""
        H = self.H.to(x.device).to(x.dtype)
        x_original = torch.matmul(x, H)
        return x_original


class AWQQuantizer:
    """显著性感知权重量化"""
    
    def __init__(self, model: nn.Module, dataloader, num_batches: int = 100):
        self.model = model
        self.dataloader = dataloader
        self.num_batches = num_batches
        self.saliency_scores = {}
    
    def compute_saliency(self):
        """计算显著性分数"""
        self.model.eval()
        
        # 注册前向钩子
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = input[0].detach()
            return hook
        
        # 为线性层注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # 通过dataloader计算激活统计
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= self.num_batches:
                    break
                
                input_ids = batch['input_ids']
                self.model(input_ids)
                
                # 计算显著性
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear) and name in activations:
                        A = activations[name]
                        W = module.weight.data
                        
                        # S = |A|^T @ |W|
                        S = torch.abs(A).mean(dim=0) @ torch.abs(W).t()
                        
                        if name not in self.saliency_scores:
                            self.saliency_scores[name] = S
                        else:
                            self.saliency_scores[name] += S
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 归一化
        for name in self.saliency_scores:
            S = self.saliency_scores[name]
            self.saliency_scores[name] = S / (S.max() + 1e-8)
    
    def get_precision_for_weight(self, name: str, channel_idx: int) -> int:
        """根据显著性获取权重的量化精度"""
        if name not in self.saliency_scores:
            return 8
        
        saliency = self.saliency_scores[name][channel_idx].item()
        
        if saliency > 0.7:
            return 8  # Int8
        elif saliency > 0.4:
            return 6  # Int6
        elif saliency > 0.2:
            return 4  # Int4
        else:
            return 2  # Int2


class LayerWiseQuantizer:
    """分层量化"""
    
    def __init__(self, config):
        self.config = config
        self.precision_map = config.quantization.layer_wise_precision
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        bits: int,
        symmetric: bool = True
    ) -> Tuple[torch.Tensor, float, int]:
        """量化张量"""
        if bits == 8:
            max_val = tensor.abs().max()
            scale = max_val / 127.0
            zero_point = 0
            quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        
        elif bits == 6:
            max_val = tensor.abs().max()
            scale = max_val / 31.0
            zero_point = 0
            quantized = torch.clamp(torch.round(tensor / scale), -32, 31).to(torch.int8)
        
        elif bits == 4:
            max_val = tensor.abs().max()
            scale = max_val / 7.0
            zero_point = 0
            quantized = torch.clamp(torch.round(tensor / scale), -8, 7).to(torch.int8)
        
        else:
            raise ValueError(f"Unsupported bits: {bits}")
        
        return quantized, scale, zero_point
    
    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: float,
        zero_point: int
    ) -> torch.Tensor:
        """反量化张量"""
        return quantized.float() * scale + zero_point


class HessianAwareCalibrator:
    """Hessian感知校准"""
    
    def __init__(self, model: nn.Module, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.fisher_info = {}
    
    def compute_fisher_information(self, num_batches: int = 50):
        """计算Fisher信息矩阵(对角线)"""
        self.model.eval()
        
        # 初始化Fisher信息
        for name, param in self.model.named_parameters():
            self.fisher_info[name] = torch.zeros_like(param.data)
        
        # 通过数据集计算Fisher信息
        num_processed = 0
        for batch in self.dataloader:
            if num_processed >= num_batches:
                break
            
            input_ids = batch['input_ids']
            labels = batch.get('labels', input_ids)
            
            # 前向传播
            logits, _ = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.model.config.model.vocab_size),
                labels.view(-1)
            )
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 累积Fisher信息
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += (param.grad ** 2)
            
            num_processed += 1
        
        # 平均化
        for name in self.fisher_info:
            self.fisher_info[name] /= max(num_processed, 1)
    
    def get_quantization_range(
        self,
        weight: torch.Tensor,
        name: str,
        bits: int
    ) -> Tuple[float, float]:
        """根据Hessian确定量化范围"""
        # 获取敏感度
        if name in self.fisher_info:
            sensitivity = torch.sqrt(self.fisher_info[name])
        else:
            sensitivity = torch.ones_like(weight)
        
        # 标准差基础范围
        std = weight.std()
        base_range = 3 * std
        
        # Hessian加权调整
        sensitivity_norm = sensitivity / (sensitivity.max() + 1e-8)
        
        # 根据敏感度调整范围
        min_val = -base_range * (1 + sensitivity_norm.mean())
        max_val = base_range * (1 + sensitivity_norm.mean())
        
        return min_val.item(), max_val.item()


class QuantizationAwareTraining(nn.Module):
    """量化感知训练"""
    
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.config = config
        self.quantizers = {}
        self._setup_quantizers()
    
    def _setup_quantizers(self):
        """设置量化器"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.quantizers[name] = LayerWiseQuantizer(self.config)
    
    def forward(self, x):
        """前向传播(带量化模拟)"""
        return self.model(x)
    
    def quantize_weights(self):
        """量化权重"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.quantizers:
                # 获取精度
                precision = self.config.quantization.layer_wise_precision.get(
                    name.split('.')[-1], 8
                )
                
                # 量化权重
                quantizer = self.quantizers[name]
                quantized, scale, zero_point = quantizer.quantize_tensor(
                    module.weight.data, precision
                )
                
                # 恢复权重(用于验证)
                module.weight.data = quantizer.dequantize_tensor(
                    quantized, scale, zero_point
                )
