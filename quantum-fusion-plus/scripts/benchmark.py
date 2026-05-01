#!/usr/bin/env python3
"""
性能基准测试脚本
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from quantum_fusion import (
    QuantumFusionGPT,
    InferenceEngine,
    load_config,
    setup_logging,
    calculate_model_size,
    calculate_parameters,
    calculate_bpb
)

logger = logging.getLogger(__name__)


def benchmark_inference(model, config, device, num_runs=10):
    """基准测试推理性能"""
    logger.info("开始推理性能基准测试...")
    
    model.eval()
    
    batch_size = 1
    seq_len = config.model.max_seq_length
    
    # 预热
    with torch.no_grad():
        for _ in range(2):
            input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len)).to(device)
            _ = model(input_ids)
    
    # 基准测试
    times = []
    with torch.no_grad():
        for run in range(num_runs):
            input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len)).to(device)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            _ = model(input_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    throughput = (batch_size * seq_len) / avg_time
    
    logger.info(f"平均推理时间: {avg_time*1000:.2f}ms")
    logger.info(f"吞吐量: {throughput:.0f} tokens/sec")
    
    return avg_time, throughput


def benchmark_training(model, config, device, num_steps=10):
    """基准测试训练性能"""
    logger.info("开始训练性能基准测试...")
    
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = config.training.batch_size
    seq_len = config.model.max_seq_length
    
    times = []
    for step in range(num_steps):
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len)).to(device)
        labels = input_ids.clone()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        # 前向传播
        logits, _ = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.model.vocab_size),
            labels.view(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    throughput = (batch_size * seq_len) / avg_time
    
    logger.info(f"平均训练时间: {avg_time*1000:.2f}ms")
    logger.info(f"吞吐量: {throughput:.0f} tokens/sec")
    
    return avg_time, throughput


def benchmark_memory(model, config, device):
    """基准测试内存占用"""
    logger.info("开始内存占用基准测试...")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    model.eval()
    
    batch_size = config.training.batch_size
    seq_len = config.model.max_seq_length
    
    with torch.no_grad():
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len)).to(device)
        _ = model(input_ids)
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"峰值GPU内存: {peak_memory:.2f}GB")
        return peak_memory
    else:
        logger.info("CPU模式下不测试GPU内存")
        return 0


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='性能基准测试')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    parser.add_argument('--num_runs', type=int, default=10, help='运行次数')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    logger.info("创建模型...")
    model = QuantumFusionGPT(config).to(device)
    
    # 加载检查点(如果提供)
    if args.model_path:
        logger.info(f"加载模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 模型信息
    num_params = calculate_parameters(model)
    model_size = calculate_model_size(model)
    logger.info(f"模型参数量: {num_params/1e6:.2f}M")
    logger.info(f"模型大小: {model_size:.2f}MB")
    
    # 运行基准测试
    logger.info("\n" + "="*60)
    logger.info("性能基准测试")
    logger.info("="*60)
    
    # 推理性能
    inf_time, inf_throughput = benchmark_inference(model, config, device, args.num_runs)
    
    # 训练性能
    train_time, train_throughput = benchmark_training(model, config, device, args.num_runs)
    
    # 内存占用
    peak_memory = benchmark_memory(model, config, device)
    
    # 打印总结
    logger.info("\n" + "="*60)
    logger.info("基准测试结果总结")
    logger.info("="*60)
    logger.info(f"推理时间: {inf_time*1000:.2f}ms")
    logger.info(f"推理吞吐量: {inf_throughput:.0f} tokens/sec")
    logger.info(f"训练时间: {train_time*1000:.2f}ms")
    logger.info(f"训练吞吐量: {train_throughput:.0f} tokens/sec")
    if peak_memory > 0:
        logger.info(f"峰值GPU内存: {peak_memory:.2f}GB")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
