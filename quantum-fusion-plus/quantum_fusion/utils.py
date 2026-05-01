"""
工具模块: 配置加载、日志、指标计算等
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any
import torch
import math


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 转换为对象以支持点号访问
    return DotDict(config)


class DotDict(dict):
    """支持点号访问的字典"""
    
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"No attribute {key}")
    
    def __setattr__(self, key, value):
        self[key] = value


def setup_logging(config, log_file: str = 'training.log'):
    """设置日志"""
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.logging.log_level))
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(getattr(logging, config.logging.log_level))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.logging.log_level))
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def calculate_model_size(model: torch.nn.Module) -> float:
    """计算模型大小(MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def calculate_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters())


def calculate_flops(model: torch.nn.Module, input_shape: tuple) -> float:
    """计算FLOPs(浮点操作数)"""
    # 这是一个简化的实现
    total_params = calculate_parameters(model)
    seq_len = input_shape[-1]
    
    # 粗略估计: 每个参数每个token需要2个FLOPs
    flops = total_params * seq_len * 2
    
    return flops


def calculate_perplexity(loss: float) -> float:
    """计算困惑度"""
    return math.exp(loss)


def calculate_bpb(loss: float, vocab_size: int = 8192) -> float:
    """计算BPB(Bits Per Byte)"""
    # BPB = loss / log2(vocab_size) * log2(e)
    bits_per_token = loss / math.log(vocab_size, 2)
    return bits_per_token


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, name: str, value: float):
        """更新指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        """获取平均值"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_latest(self, name: str) -> float:
        """获取最新值"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]
    
    def reset(self):
        """重置指标"""
        self.metrics = {}
    
    def get_summary(self) -> Dict[str, float]:
        """获取摘要"""
        summary = {}
        for name in self.metrics:
            summary[f"{name}_avg"] = self.get_average(name)
            summary[f"{name}_latest"] = self.get_latest(name)
        return summary


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
