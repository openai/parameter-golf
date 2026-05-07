"""
单元测试: 模型测试
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quantum_fusion import (
    QuantumFusionGPT,
    load_config,
    calculate_parameters,
    calculate_model_size
)


class TestQuantumFusionGPT:
    """模型测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def model(self, config):
        """创建模型"""
        return QuantumFusionGPT(config)
    
    def test_model_creation(self, model):
        """测试模型创建"""
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_model_parameters(self, model, config):
        """测试模型参数"""
        num_params = calculate_parameters(model)
        assert num_params > 0
        
        # 检查参数数量是否合理
        expected_params = config.model.hidden_size * config.model.vocab_size
        assert num_params > expected_params * 0.5
    
    def test_model_size(self, model):
        """测试模型大小"""
        size_mb = calculate_model_size(model)
        assert size_mb > 0
        assert size_mb < 100  # 模型大小应小于100MB
    
    def test_forward_pass(self, model, config):
        """测试前向传播"""
        batch_size = 2
        seq_len = config.model.max_seq_length
        
        # 创建虚拟输入
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        # 前向传播
        logits, cache = model(input_ids)
        
        # 检查输出形状
        assert logits.shape == (batch_size, seq_len, config.model.vocab_size)
    
    def test_forward_pass_with_cache(self, model, config):
        """测试带缓存的前向传播"""
        batch_size = 2
        seq_len = config.model.max_seq_length
        
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        # 启用缓存
        logits, cache = model(input_ids, use_cache=True)
        
        assert logits.shape == (batch_size, seq_len, config.model.vocab_size)
        assert cache is not None
    
    def test_gradient_flow(self, model, config):
        """测试梯度流"""
        batch_size = 2
        seq_len = config.model.max_seq_length
        
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        # 前向传播
        logits, _ = model(input_ids)
        
        # 计算损失
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.model.vocab_size),
            labels.view(-1)
        )
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_device_compatibility(self, model, config):
        """测试设备兼容性"""
        if torch.cuda.is_available():
            model = model.cuda()
            
            batch_size = 2
            seq_len = config.model.max_seq_length
            input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len)).cuda()
            
            logits, _ = model(input_ids)
            assert logits.device.type == 'cuda'
        else:
            # CPU测试
            batch_size = 2
            seq_len = config.model.max_seq_length
            input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
            
            logits, _ = model(input_ids)
            assert logits.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
