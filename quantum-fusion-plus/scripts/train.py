#!/usr/bin/env python3
"""
QUANTUM-FUSION-PLUS: 完整训练脚本
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

# 导入自定义模块
from quantum_fusion import (
    QuantumFusionGPT,
    Trainer,
    DataLoaderFactory,
    load_config,
    setup_logging,
    set_seed,
    calculate_model_size,
    calculate_parameters,
    calculate_bpb,
    MetricsTracker
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='QUANTUM-FUSION-PLUS训练脚本')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    parser.add_argument('--num_epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    
    # 设置日志
    setup_logging(config)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    logger.info("创建模型...")
    model = QuantumFusionGPT(config).to(device)
    
    # 打印模型信息
    num_params = calculate_parameters(model)
    model_size = calculate_model_size(model)
    logger.info(f"模型参数量: {num_params/1e6:.2f}M")
    logger.info(f"模型大小: {model_size:.2f}MB")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_dataloader = DataLoaderFactory.create_dataloader(config, split='train')
    val_dataloader = DataLoaderFactory.create_dataloader(config, split='val', batch_size=config.training.batch_size)
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = Trainer(model, config, device=device)
    
    # 创建检查点目录
    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志目录
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 指标追踪
    metrics = MetricsTracker()
    
    # 训练循环
    logger.info("开始训练...")
    
    for epoch in range(config.training.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}")
        logger.info(f"{'='*60}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 训练步骤
            loss = trainer.train_step(batch)
            train_loss += loss
            num_batches += 1
            
            # 记录指标
            metrics.update('train_loss', loss)
            
            # 日志
            if (batch_idx + 1) % config.logging.log_freq == 0:
                avg_loss = train_loss / num_batches
                lr = trainer.scheduler.get_last_lr()[0]
                logger.info(
                    f"Batch {batch_idx+1}/{len(train_dataloader)}, "
                    f"Loss: {loss:.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"LR: {lr:.2e}, "
                    f"Step: {trainer.global_step}"
                )
            
            # 检查点保存
            if (trainer.global_step + 1) % config.checkpoint.save_freq == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{trainer.global_step}.pt"
                trainer.save_checkpoint(str(checkpoint_path))
                logger.info(f"保存检查点: {checkpoint_path}")
        
        # 验证阶段
        logger.info("开始验证...")
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                loss = trainer.eval_step(batch)
                val_loss += loss
                num_val_batches += 1
                metrics.update('val_loss', loss)
        
        avg_val_loss = val_loss / max(num_val_batches, 1)
        avg_train_loss = train_loss / num_batches
        
        # 计算BPB
        bpb = calculate_bpb(avg_val_loss, config.model.vocab_size)
        
        logger.info(f"\nEpoch {epoch+1} 完成:")
        logger.info(f"  训练损失: {avg_train_loss:.4f}")
        logger.info(f"  验证损失: {avg_val_loss:.4f}")
        logger.info(f"  BPB: {bpb:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < trainer.best_val_loss:
            trainer.best_val_loss = avg_val_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(str(best_checkpoint_path))
            logger.info(f"保存最佳模型: {best_checkpoint_path}")
        
        # 保存最后模型
        last_checkpoint_path = checkpoint_dir / "last_model.pt"
        trainer.save_checkpoint(str(last_checkpoint_path))
    
    logger.info("\n" + "="*60)
    logger.info("训练完成!")
    logger.info("="*60)
    
    # 打印最终统计
    summary = metrics.get_summary()
    logger.info("\n最终统计:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
