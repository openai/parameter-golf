"""
单元测试: 量化测试
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quantum_fusion import (
    HadamardRotation,
    LayerWiseQuantizer,
    load_config
)


class TestHadamardRotation:
    """Hadamard旋转测试"""
    
    def test_hadamard_creation(self):
        """测试Hadamard矩阵创建"""
        hidden_size = 512
        rotation = HadamardRotation(hidden_size)
        assert rotation.H is not None
        assert rotation.H.shape == (hidden_size, hidden_size)
    
    def test_hadamard_orthogonality(self):
        """测试Hadamard矩阵正交性"""
        hidden_size = 256
        rotation = HadamardRotation(hidden_size)
        H = rotation.H
        
        # H @ H^T = I
        product = torch.matmul(H, H.t())
        identity = torch.eye(hidden_size)
        
        assert torch.allclose(product, identity, atol=1e-5)
    
    def test_hadamard_apply(self):
        """测试应用Hadamard旋转"""
        hidden_size = 512
        rotation = HadamardRotation(hidden_size)
        
        # 创建虚拟张量
        x = torch.randn(2, 10, hidden_size)
        
        # 应用旋转
        x_rotated = rotation.apply(x)
        
        assert x_rotated.shape == x.shape
    
    def test_hadamard_inverse(self):
        """测试Hadamard逆旋转"""
        hidden_size = 256
        rotation = HadamardRotation(hidden_size)
        
        x = torch.randn(2, 10, hidden_size)
        x_rotated = rotation.apply(x)
        x_recovered = rotation.inverse(x_rotated)
        
        # 应该恢复原始张量
        assert torch.allclose(x, x_recovered, atol=1e-5)


class TestLayerWiseQuantizer:
    """分层量化测试"""
    
    @pytest.fixture
    def config(self):
        """加载配置"""
        return load_config('configs/config.yaml')
    
    @pytest.fixture
    def quantizer(self, config):
        """创建量化器"""
        return LayerWiseQuantizer(config)
    
    def test_quantize_int8(self, quantizer):
        """测试Int8量化"""
        tensor = torch.randn(10, 512)
        quantized, scale, zero_point = quantizer.quantize_tensor(tensor, bits=8)
        
        assert quantized.dtype == torch.int8
        assert scale > 0
    
    def test_quantize_int6(self, quantizer):
        """测试Int6量化"""
        tensor = torch.randn(10, 512)
        quantized, scale, zero_point = quantizer.quantize_tensor(tensor, bits=6)
        
        assert quantized.dtype == torch.int8
        assert scale > 0
    
    def test_quantize_int4(self, quantizer):
        """测试Int4量化"""
        tensor = torch.randn(10, 512)
        quantized, scale, zero_point = quantizer.quantize_tensor(tensor, bits=4)
        
        assert quantized.dtype == torch.int8
        assert scale > 0
    
    def test_dequantize(self, quantizer):
        """测试反量化"""
        tensor = torch.randn(10, 512)
        quantized, scale, zero_point = quantizer.quantize_tensor(tensor, bits=8)
        dequantized = quantizer.dequantize_tensor(quantized, scale, zero_point)
        
        # 应该接近原始张量
        assert torch.allclose(tensor, dequantized, atol=scale)
    
    def test_quantization_error(self, quantizer):
        """测试量化误差"""
        tensor = torch.randn(100, 512)
        
        # 不同比特数的量化误差
        errors = {}
        for bits in [8, 6, 4]:
            quantized, scale, zero_point = quantizer.quantize_tensor(tensor, bits=bits)
            dequantized = quantizer.dequantize_tensor(quantized, scale, zero_point)
            error = torch.mean((tensor - dequantized) ** 2).item()
            errors[bits] = error
        
        # 更低的比特数应该有更大的误差
        assert errors[4] > errors[6] > errors[8]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
