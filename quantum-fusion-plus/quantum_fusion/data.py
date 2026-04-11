"""
数据加载模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FineWebDataset(Dataset):
    """FineWeb数据集"""
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 1024,
        vocab_size: int = 8192,
        split: str = 'train'
    ):
        self.data_path = data_path
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.split = split
        
        # 生成虚拟数据(用于测试)
        self.num_samples = 1000
        logger.info(f"加载{split}数据集: {self.num_samples}个样本")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """获取样本"""
        # 生成虚拟token序列
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class DataLoaderFactory:
    """数据加载器工厂"""
    
    @staticmethod
    def create_dataloader(
        config,
        split: str = 'train',
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """创建数据加载器"""
        if batch_size is None:
            batch_size = config.training.batch_size
        
        # 创建数据集
        dataset = FineWebDataset(
            data_path=config.data.cache_dir,
            seq_length=config.model.max_seq_length,
            vocab_size=config.model.vocab_size,
            split=split
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            prefetch_factor=config.data.prefetch_factor
        )
        
        logger.info(f"创建{split}数据加载器: batch_size={batch_size}, num_batches={len(dataloader)}")
        
        return dataloader


def create_dummy_batch(config, batch_size: Optional[int] = None) -> dict:
    """创建虚拟批次(用于测试)"""
    if batch_size is None:
        batch_size = config.training.batch_size
    
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, config.model.max_seq_length))
    labels = input_ids.clone()
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }
