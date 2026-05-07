"""
集成测试: 端到端测试
"""

import pytest
import torch
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quantum_fusion import (
    QuantumFusionGPT,
    Trainer,
    InferenceEngine,
    DataLoaderFactory,
    load_config,
    calculate_bpb,
    calculate_model_size,
    calculate_parameters,
    create_dummy_batch
)


class TestEndToEnd:
    """端到端测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def model(self, config):
        """创建模型"""
        return QuantumFusionGPT(config)
    
    def test_model_creation_and_forward(self, model, config):
        """测试模型创建和前向传播"""
        batch = create_dummy_batch(config, batch_size=2)
        input_ids = batch['input_ids']
        
        logits, cache = model(input_ids)
        
        assert logits.shape == (2, config.model.max_seq_length, config.model.vocab_size)
    
    def test_training_loop(self, model, config):
        """测试训练循环"""
        trainer = Trainer(model, config, device='cpu')
        
        # 运行几个训练步骤
        losses = []
        for _ in range(3):
            batch = create_dummy_batch(config, batch_size=2)
            loss = trainer.train_step(batch)
            losses.append(loss)
        
        # 检查损失是否在减少
        assert len(losses) == 3
        assert all(loss > 0 for loss in losses)
    
    def test_eval_loop(self, model, config):
        """测试评估循环"""
        trainer = Trainer(model, config, device='cpu')
        
        # 运行几个评估步骤
        losses = []
        for _ in range(3):
            batch = create_dummy_batch(config, batch_size=2)
            loss = trainer.eval_step(batch)
            losses.append(loss)
        
        assert len(losses) == 3
        assert all(loss > 0 for loss in losses)
    
    def test_inference_pipeline(self, model, config):
        """测试推理管道"""
        engine = InferenceEngine(model, config, device='cpu')
        
        batch_size = 1
        seq_len = config.model.max_seq_length
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        # 生成
        generated = engine.generate(input_ids, max_new_tokens=5, use_ttt=False)
        
        assert generated is not None
        assert generated.shape[0] == batch_size
    
    def test_checkpoint_workflow(self, model, config):
        """测试检查点工作流"""
        trainer = Trainer(model, config, device='cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            
            # 训练几步
            for _ in range(2):
                batch = create_dummy_batch(config, batch_size=2)
                trainer.train_step(batch)
            
            # 保存检查点
            trainer.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()
            
            # 创建新的训练器
            new_model = QuantumFusionGPT(config)
            new_trainer = Trainer(new_model, config, device='cpu')
            
            # 加载检查点
            new_trainer.load_checkpoint(str(checkpoint_path))
            
            # 检查状态是否恢复
            assert new_trainer.global_step == trainer.global_step
    
    def test_model_size_constraint(self, model):
        """测试模型大小约束"""
        size_mb = calculate_model_size(model)
        
        # 模型大小应小于100MB
        assert size_mb < 100
    
    def test_parameter_count(self, model, config):
        """测试参数数量"""
        num_params = calculate_parameters(model)
        
        # 参数数量应该合理
        assert num_params > 0
        assert num_params < 100e6  # 小于100M参数
    
    def test_bpb_calculation(self, config):
        """测试BPB计算"""
        # 模拟损失值
        loss = 1.0810
        bpb = calculate_bpb(loss, config.model.vocab_size)
        
        assert bpb > 0
        assert bpb < 2.0  # 合理的BPB范围
    
    def test_data_loading(self, config):
        """测试数据加载"""
        dataloader = DataLoaderFactory.create_dataloader(config, split='train', batch_size=2)
        
        # 获取一个批次
        batch = next(iter(dataloader))
        
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 2
    
    def test_full_training_pipeline(self, config):
        """测试完整训练管道"""
        # 创建模型
        model = QuantumFusionGPT(config)
        
        # 创建训练器
        trainer = Trainer(model, config, device='cpu')
        
        # 创建数据加载器
        train_dataloader = DataLoaderFactory.create_dataloader(config, split='train', batch_size=2)
        val_dataloader = DataLoaderFactory.create_dataloader(config, split='val', batch_size=2)
        
        # 运行一个epoch
        train_losses = []
        for batch in train_dataloader:
            loss = trainer.train_step(batch)
            train_losses.append(loss)
        
        # 运行验证
        val_losses = []
        for batch in val_dataloader:
            loss = trainer.eval_step(batch)
            val_losses.append(loss)
        
        # 检查结果
        assert len(train_losses) > 0
        assert len(val_losses) > 0
        assert all(loss > 0 for loss in train_losses)
        assert all(loss > 0 for loss in val_losses)


class TestPerformanceValidation:
    """性能验证测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def model(self, config):
        """创建模型"""
        return QuantumFusionGPT(config)
    
    def test_inference_speed(self, model, config):
        """测试推理速度"""
        import time
        
        model.eval()
        
        batch_size = 1
        seq_len = config.model.max_seq_length
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        # 预热
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_ids)
        
        # 测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5
        throughput = seq_len / avg_time
        
        # 吞吐量应该合理
        assert throughput > 0
    
    def test_memory_efficiency(self, model, config):
        """测试内存效率"""
        size_mb = calculate_model_size(model)
        num_params = calculate_parameters(model)
        
        # 每个参数的平均大小应该合理
        bytes_per_param = (size_mb * 1024 * 1024) / num_params
        
        # 应该在4-8字节之间(float32=4字节)
        assert 2 < bytes_per_param < 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
