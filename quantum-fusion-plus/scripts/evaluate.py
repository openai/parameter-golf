#!/usr/bin/env python3
"""
评估脚本: 计算性能指标
"""

import sys
import os
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from quantum_fusion import (
    QuantumFusionGPT,
    DataLoaderFactory,
    load_config,
    setup_logging,
    calculate_model_size,
    calculate_parameters,
    calculate_bpb,
    calculate_perplexity
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='评估脚本')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
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
    
    # 加载检查点
    logger.info(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 模型信息
    num_params = calculate_parameters(model)
    model_size = calculate_model_size(model)
    logger.info(f"模型参数量: {num_params/1e6:.2f}M")
    logger.info(f"模型大小: {model_size:.2f}MB")
    
    # 创建验证数据加载器
    logger.info("创建验证数据加载器...")
    val_dataloader = DataLoaderFactory.create_dataloader(
        config, split='val', batch_size=config.training.batch_size
    )
    
    # 评估
    logger.info("开始评估...")
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)
            
            # 前向传播
            logits, _ = model(input_ids)
            
            # 计算损失
            loss = nn.functional.cross_entropy(
                logits.view(-1, config.model.vocab_size),
                labels.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx+1}/{len(val_dataloader)}, Loss: {loss:.4f}")
    
    # 计算指标
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = calculate_perplexity(avg_loss)
    bpb = calculate_bpb(avg_loss, config.model.vocab_size)
    
    logger.info("\n" + "="*60)
    logger.info("评估结果:")
    logger.info("="*60)
    logger.info(f"平均损失: {avg_loss:.4f}")
    logger.info(f"困惑度: {perplexity:.4f}")
    logger.info(f"BPB: {bpb:.4f}")
    logger.info(f"模型大小: {model_size:.2f}MB")
    logger.info(f"参数量: {num_params/1e6:.2f}M")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
