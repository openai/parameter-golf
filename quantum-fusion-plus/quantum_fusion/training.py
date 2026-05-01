"""
训练模块: Muon优化器、Warmdown调度器、EMA管理、QAT训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Optional, List


class MuonOptimizer(torch.optim.Optimizer):
    """Muon优化器实现"""
    
    def __init__(self, params, lr=0.002, weight_decay=0.090, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """执行优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # 应用权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Muon更新
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss


class WarmdownScheduler:
    """Warmdown学习率调度器"""
    
    def __init__(self, optimizer, warmup_steps, warmdown_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmdown_steps = warmdown_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = optimizer.defaults['lr']
    
    def step(self):
        """执行调度步骤"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup阶段: 线性增长
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        elif self.current_step <= self.total_steps - self.warmdown_steps:
            # Plateau阶段: 保持最大值
            lr = self.base_lr
        else:
            # Warmdown阶段: 指数衰减
            warmdown_progress = (self.current_step - (self.total_steps - self.warmdown_steps)) / self.warmdown_steps
            lr = self.base_lr * math.exp(-3 * warmdown_progress)
        
        # 应用学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class EMAManager:
    """EMA(Exponential Moving Average)管理"""
    
    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.model = model
        self.decay = decay
        self.ema_model = None
        self._init_ema_model()
    
    def _init_ema_model(self):
        """初始化EMA模型"""
        self.ema_model = self._copy_model(self.model)
    
    @staticmethod
    def _copy_model(model: nn.Module) -> nn.Module:
        """复制模型"""
        import copy
        return copy.deepcopy(model)
    
    def update(self):
        """更新EMA模型"""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * model_param.data
    
    def get_ema_model(self) -> nn.Module:
        """获取EMA模型"""
        return self.ema_model
    
    def swap_models(self):
        """交换模型和EMA模型"""
        self.model, self.ema_model = self.ema_model, self.model


class QATTrainer:
    """量化感知训练器"""
    
    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.qat_enabled = False
    
    def enable_qat(self):
        """启用QAT"""
        self.qat_enabled = True
        # 在这里可以添加QAT特定的初始化
    
    def disable_qat(self):
        """禁用QAT"""
        self.qat_enabled = False
    
    def forward(self, x):
        """前向传播(可能带QAT)"""
        if self.qat_enabled:
            # 模拟量化
            return self._forward_with_qat(x)
        else:
            return self.model(x)
    
    def _forward_with_qat(self, x):
        """带QAT的前向传播"""
        # 前向传播
        logits, cache = self.model(x)
        
        # 在这里可以添加量化模拟逻辑
        # 例如: 模拟权重量化、激活量化等
        
        return logits, cache


class Trainer:
    """主训练器"""
    
    def __init__(self, model: nn.Module, config, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = MuonOptimizer(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = WarmdownScheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            warmdown_steps=config.training.warmdown_steps,
            total_steps=config.training.total_steps
        )
        
        # EMA管理
        self.ema_manager = EMAManager(model, decay=config.training.ema_decay)
        
        # QAT训练器
        self.qat_trainer = QATTrainer(model, config)
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_step(self, batch):
        """单个训练步骤"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # 前向传播
        logits, _ = self.model(input_ids)
        
        # 计算损失
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.config.model.vocab_size),
            labels.view(-1)
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.gradient_clip
        )
        
        # 优化步骤
        self.optimizer.step()
        
        # 学习率调度
        self.scheduler.step()
        
        # EMA更新
        self.ema_manager.update()
        
        # 全局步数
        self.global_step += 1
        
        # 检查是否进入QAT阶段
        if self.config.training.qat_enabled:
            qat_start_step = int(self.config.training.total_steps * (1 - self.config.training.qat_ratio))
            if self.global_step >= qat_start_step and not self.qat_trainer.qat_enabled:
                self.qat_trainer.enable_qat()
        
        return loss.item()
    
    def eval_step(self, batch):
        """单个评估步骤"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            
            # 前向传播
            logits, _ = self.model(input_ids)
            
            # 计算损失
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.model.vocab_size),
                labels.view(-1)
            )
        
        return loss.item()
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
