"""
单元测试: 推理测试
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quantum_fusion import (
    QuantumFusionGPT,
    InferenceEngine,
    LegalTTT,
    KVLinCCache,
    load_config
)


class TestLegalTTT:
    """合法TTT测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def model(self, config):
        """创建模型"""
        return QuantumFusionGPT(config)
    
    @pytest.fixture
    def ttt(self, model, config):
        """创建TTT"""
        return LegalTTT(model, config)
    
    def test_ttt_creation(self, ttt):
        """测试TTT创建"""
        assert ttt is not None
        assert ttt.model is not None
        assert ttt.optimizer is not None
    
    def test_ttt_forward(self, ttt, config):
        """测试TTT前向传播"""
        batch_size = 1
        seq_len = config.model.max_seq_length
        
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        generated, ttt_log = ttt.forward_with_ttt(input_ids, max_new_tokens=10)
        
        assert generated is not None
        assert ttt_log is not None
        assert len(ttt_log) > 0
    
    def test_ttt_log(self, ttt, config):
        """测试TTT日志"""
        batch_size = 1
        seq_len = config.model.max_seq_length
        
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        generated, ttt_log = ttt.forward_with_ttt(input_ids, max_new_tokens=5)
        
        # 检查日志格式
        for log in ttt_log:
            assert 'step' in log
            assert 'score' in log
            assert 'updated' in log


class TestKVLinCCache:
    """KVLinC缓存测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def cache(self, config):
        """创建缓存"""
        return KVLinCCache(config)
    
    def test_cache_creation(self, cache):
        """测试缓存创建"""
        assert cache is not None
        assert cache.cache == {}
    
    def test_quantize_kv(self, cache):
        """测试KV量化"""
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)
        
        k_q, v_q, k_scale, v_scale = cache.quantize_kv(k, v, bits=6)
        
        assert k_q.dtype == torch.int8
        assert v_q.dtype == torch.int8
        assert k_scale > 0
        assert v_scale > 0
    
    def test_dequantize_kv(self, cache):
        """测试KV反量化"""
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)
        
        k_q, v_q, k_scale, v_scale = cache.quantize_kv(k, v, bits=6)
        k_recovered, v_recovered = cache.dequantize_kv(k_q, v_q, k_scale, v_scale)
        
        # 应该接近原始值
        assert torch.allclose(k, k_recovered, atol=k_scale)
        assert torch.allclose(v, v_recovered, atol=v_scale)
    
    def test_cache_store_retrieve(self, cache):
        """测试缓存存储和检索"""
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)
        
        # 存储
        cache.store(0, k, v)
        
        # 检索
        k_retrieved, v_retrieved = cache.retrieve(0)
        
        assert k_retrieved is not None
        assert v_retrieved is not None
    
    def test_cache_size(self, cache):
        """测试缓存大小"""
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)
        
        cache.store(0, k, v)
        cache.store(1, k, v)
        
        size_mb = cache.get_cache_size_mb()
        assert size_mb > 0
    
    def test_cache_clear(self, cache):
        """测试缓存清空"""
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)
        
        cache.store(0, k, v)
        assert len(cache.cache) > 0
        
        cache.clear()
        assert len(cache.cache) == 0


class TestInferenceEngine:
    """推理引擎测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def model(self, config):
        """创建模型"""
        return QuantumFusionGPT(config)
    
    @pytest.fixture
    def engine(self, model, config):
        """创建推理引擎"""
        return InferenceEngine(model, config, device='cpu')
    
    def test_engine_creation(self, engine):
        """测试引擎创建"""
        assert engine is not None
        assert engine.model is not None
        assert engine.ttt is not None
        assert engine.kv_cache is not None
    
    def test_generate_without_ttt(self, engine, config):
        """测试不使用TTT的生成"""
        batch_size = 1
        seq_len = config.model.max_seq_length
        
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        generated = engine.generate(input_ids, max_new_tokens=5, use_ttt=False)
        
        assert generated is not None
        assert generated.shape[0] == batch_size
    
    def test_generate_with_ttt(self, engine, config):
        """测试使用TTT的生成"""
        batch_size = 1
        seq_len = config.model.max_seq_length
        
        input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
        
        generated = engine.generate(input_ids, max_new_tokens=5, use_ttt=True)
        
        assert generated is not None
        assert generated.shape[0] == batch_size
    
    def test_inference_stats(self, engine):
        """测试推理统计"""
        stats = engine.get_inference_stats()
        
        assert 'cache_size_mb' in stats
        assert 'ttt_updates' in stats
        assert 'ttt_total_steps' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
