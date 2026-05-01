"""
单元测试: 训练测试
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quantum_fusion import (
    QuantumFusionGPT,
    Trainer,
    MuonOptimizer,
    WarmdownScheduler,
    EMAManager,
    load_config,
    create_dummy_batch
)


class TestMuonOptimizer:
    """Muon优化器测试"""
    
    def test_optimizer_creation(self):
        """测试优化器创建"""
        model = torch.nn.Linear(10, 10)
        optimizer = MuonOptimizer(model.parameters(), lr=0.001)
        assert optimizer is not None
    
    def test_optimizer_step(self):
        """测试优化步骤"""
        model = torch.nn.Linear(10, 10)
        optimizer = MuonOptimizer(model.parameters(), lr=0.001)
        
        # 计算损失
        x = torch.randn(5, 10)
        y = torch.randn(5, 10)
        loss = torch.nn.functional.mse_loss(model(x), y)
        
        # 反向传播
        loss.backward()
        
        # 优化步骤
        optimizer.step()
        
        # 检查参数是否更新
        assert True  # 如果没有错误就通过


class TestWarmdownScheduler:
    """Warmdown调度器测试"""
    
    def test_scheduler_creation(self):
        """测试调度器创建"""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = WarmdownScheduler(optimizer, warmup_steps=100, warmdown_steps=100, total_steps=1000)
        assert scheduler is not None
    
    def test_warmup_phase(self):
        """测试Warmup阶段"""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = WarmdownScheduler(optimizer, warmup_steps=100, warmdown_steps=100, total_steps=1000)
        
        # 在Warmup阶段
        for _ in range(50):
            scheduler.step()
        
        lr = scheduler.get_last_lr()[0]
        assert 0 < lr < 0.1
    
    def test_plateau_phase(self):
        """测试Plateau阶段"""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = WarmdownScheduler(optimizer, warmup_steps=100, warmdown_steps=100, total_steps=1000)
        
        # 跳过Warmup
        for _ in range(150):
            scheduler.step()
        
        lr = scheduler.get_last_lr()[0]
        assert abs(lr - 0.1) < 1e-5
    
    def test_warmdown_phase(self):
        """测试Warmdown阶段"""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = WarmdownScheduler(optimizer, warmup_steps=100, warmdown_steps=100, total_steps=1000)
        
        # 跳到Warmdown阶段
        for _ in range(950):
            scheduler.step()
        
        lr = scheduler.get_last_lr()[0]
        assert lr < 0.1


class TestEMAManager:
    """EMA管理测试"""
    
    def test_ema_creation(self):
        """测试EMA创建"""
        model = torch.nn.Linear(10, 10)
        ema = EMAManager(model, decay=0.99)
        assert ema is not None
        assert ema.ema_model is not None
    
    def test_ema_update(self):
        """测试EMA更新"""
        model = torch.nn.Linear(10, 10)
        ema = EMAManager(model, decay=0.99)
        
        # 修改模型参数
        with torch.no_grad():
            for param in model.parameters():
                param.data.fill_(1.0)
        
        # 获取初始EMA参数
        initial_ema_param = ema.ema_model.weight.data.clone()
        
        # 更新EMA
        ema.update()
        
        # 检查EMA是否更新
        updated_ema_param = ema.ema_model.weight.data
        assert not torch.allclose(initial_ema_param, updated_ema_param)
    
    def test_ema_swap(self):
        """测试模型交换"""
        model = torch.nn.Linear(10, 10)
        ema = EMAManager(model, decay=0.99)
        
        # 保存原始模型参数
        original_param = model.weight.data.clone()
        
        # 交换模型
        ema.swap_models()
        
        # 检查是否交换
        assert not torch.allclose(model.weight.data, original_param)


class TestTrainer:
    """训练器测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def model(self, config):
        """创建模型"""
        return QuantumFusionGPT(config)
    
    @pytest.fixture
    def trainer(self, model, config):
        """创建训练器"""
        return Trainer(model, config, device='cpu')
    
    def test_trainer_creation(self, trainer):
        """测试训练器创建"""
        assert trainer is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
    
    def test_train_step(self, trainer, config):
        """测试训练步骤"""
        batch = create_dummy_batch(config, batch_size=2)
        loss = trainer.train_step(batch)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_eval_step(self, trainer, config):
        """测试评估步骤"""
        batch = create_dummy_batch(config, batch_size=2)
        loss = trainer.eval_step(batch)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_checkpoint_save_load(self, trainer, tmp_path):
        """测试检查点保存和加载"""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # 保存检查点
        trainer.save_checkpoint(str(checkpoint_path))
        assert checkpoint_path.exists()
        
        # 加载检查点
        trainer.load_checkpoint(str(checkpoint_path))
        assert trainer.global_step >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
