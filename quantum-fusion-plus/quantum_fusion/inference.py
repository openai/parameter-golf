"""
推理模块: 合法TTT、KVLinC缓存优化、推理引擎
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class LegalTTT:
    """合法Score-First TTT (Test-Time Training)"""
    
    def __init__(self, model: nn.Module, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.inference.ttt_learning_rate
        )
        self.ttt_log = []
    
    def forward_with_ttt(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> Tuple[torch.Tensor, list]:
        """
        带TTT的前向传播
        
        Args:
            input_ids: 输入token IDs
            max_new_tokens: 最大生成token数
        
        Returns:
            生成的token IDs, TTT日志
        """
        generated = input_ids.clone()
        self.ttt_log = []
        
        for step in range(max_new_tokens):
            # 前向传播
            logits, _ = self.model(generated)
            
            # 获取最后一个token的logits
            next_logits = logits[:, -1, :]
            
            # 计算概率和分数
            probs = torch.softmax(next_logits, dim=-1)
            score = probs.max().item()
            
            # 获取下一个token
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # 条件TTT更新
            if score > self.config.inference.ttt_score_threshold:
                if len(self.ttt_log) < self.config.inference.ttt_max_updates:
                    # 计算损失
                    loss = -torch.log(probs.gather(1, next_token) + 1e-10)
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 参数更新
                    self.optimizer.step()
                    
                    # 记录
                    self.ttt_log.append({
                        'step': step,
                        'score': score,
                        'loss': loss.item(),
                        'updated': True
                    })
                else:
                    self.ttt_log.append({
                        'step': step,
                        'score': score,
                        'updated': False
                    })
            else:
                self.ttt_log.append({
                    'step': step,
                    'score': score,
                    'updated': False
                })
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated, self.ttt_log
    
    def get_ttt_log(self) -> list:
        """获取TTT日志"""
        return self.ttt_log


class KVLinCCache:
    """KVLinC: KV缓存量化优化"""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def quantize_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        bits: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """量化KV缓存"""
        # 计算缩放因子
        k_scale = k.abs().max() / (2 ** (bits - 1) - 1)
        v_scale = v.abs().max() / (2 ** (bits - 1) - 1)
        
        # 量化
        k_quantized = torch.clamp(torch.round(k / k_scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        v_quantized = torch.clamp(torch.round(v / v_scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        
        return k_quantized, v_quantized, k_scale, v_scale
    
    def dequantize_kv(
        self,
        k_quantized: torch.Tensor,
        v_quantized: torch.Tensor,
        k_scale: float,
        v_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """反量化KV缓存"""
        k = k_quantized.float() * k_scale
        v = v_quantized.float() * v_scale
        return k, v
    
    def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """存储KV缓存"""
        if self.config.inference.kvlinc_enabled:
            k_q, v_q, k_scale, v_scale = self.quantize_kv(k, v)
            self.cache[layer_idx] = {
                'k': k_q,
                'v': v_q,
                'k_scale': k_scale,
                'v_scale': v_scale
            }
        else:
            self.cache[layer_idx] = {'k': k, 'v': v}
    
    def retrieve(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索KV缓存"""
        if layer_idx not in self.cache:
            return None, None
        
        cache_item = self.cache[layer_idx]
        
        if self.config.inference.kvlinc_enabled and 'k_scale' in cache_item:
            k, v = self.dequantize_kv(
                cache_item['k'],
                cache_item['v'],
                cache_item['k_scale'],
                cache_item['v_scale']
            )
        else:
            k, v = cache_item['k'], cache_item['v']
        
        return k, v
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def get_cache_size_mb(self) -> float:
        """获取缓存大小(MB)"""
        total_size = 0
        for layer_idx, cache_item in self.cache.items():
            for key, value in cache_item.items():
                if isinstance(value, torch.Tensor):
                    total_size += value.element_size() * value.nelement()
        return total_size / (1024 * 1024)


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model: nn.Module, config, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # TTT
        self.ttt = LegalTTT(model, config)
        
        # KVLinC缓存
        self.kv_cache = KVLinCCache(config)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        use_ttt: bool = True
    ) -> torch.Tensor:
        """生成token"""
        self.model.eval()
        
        with torch.no_grad():
            if use_ttt and self.config.inference.ttt_enabled:
                generated, ttt_log = self.ttt.forward_with_ttt(input_ids, max_new_tokens)
                logger.info(f"TTT生成完成, 更新次数: {sum(1 for log in ttt_log if log['updated'])}")
            else:
                generated = input_ids.clone()
                
                for step in range(max_new_tokens):
                    # 前向传播
                    logits, _ = self.model(generated)
                    
                    # 获取下一个token
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    
                    # 添加到生成序列
                    generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_inference_stats(self) -> dict:
        """获取推理统计信息"""
        return {
            'cache_size_mb': self.kv_cache.get_cache_size_mb(),
            'ttt_updates': sum(1 for log in self.ttt.get_ttt_log() if log['updated']),
            'ttt_total_steps': len(self.ttt.get_ttt_log())
        }
