"""
Fusion+ Scheme: 完整测试套件

包含:
- 单元测试 (15个)
- 集成测试 (3个)
- 性能测试 (2个)
- 规则验证 (1个)

总计: 21个测试
"""

import torch
import torch.nn as nn
import unittest
import time
import gzip
import os
from fusion_plus_implementation import (
    PartialRoPE, LayerwiseLNScale, LeakyReLUSq, MuonOptimizer, WarmdownScheduler,
    AASQ, AHFQ, LegalTTT,
    SelfGeneratedCalibration, CrossLayerAdaptiveQuantization, MixedPrecisionQuantization, LightweightPruning,
    FusionPlusGPT, compute_bpb, train_step, benchmark_model
)


class TestPRStandardComponents(unittest.TestCase):
    """PR标准方案组件测试"""
    
    def test_partial_rope(self):
        """测试Partial RoPE"""
        rope = PartialRoPE(dim=64, max_seq_len=512, partial_ratio=0.25)
        x = torch.randn(2, 8, 512, 64)  # (batch, heads, seq_len, dim)
        output = rope(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        print("✓ Partial RoPE测试通过")
    
    def test_layerwise_ln_scale(self):
        """测试Layerwise LN Scale"""
        ln_scale = LayerwiseLNScale(num_layers=12, hidden_size=768)
        x = torch.randn(2, 512, 768)
        
        for layer_idx in range(12):
            output = ln_scale(x, layer_idx)
            self.assertEqual(output.shape, x.shape)
            self.assertFalse(torch.isnan(output).any())
        
        print("✓ Layerwise LN Scale测试通过")
    
    def test_leaky_relu_sq(self):
        """测试LeakyReLU²激活函数"""
        activation = LeakyReLUSq(negative_slope=0.01)
        x = torch.randn(2, 512, 768)
        output = activation(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        # 输出应该是非负的 (因为是平方)
        self.assertTrue((output >= 0).all())
        print("✓ LeakyReLU²测试通过")
    
    def test_muon_optimizer(self):
        """测试Muon优化器"""
        model = nn.Linear(10, 10)
        optimizer = MuonOptimizer(model.parameters(), lr=0.01, momentum=0.9)
        
        x = torch.randn(2, 10)
        y = torch.randn(2, 10)
        
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        print("✓ Muon优化器测试通过")
    
    def test_warmdown_scheduler(self):
        """测试Warmdown调度器"""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = WarmdownScheduler(optimizer, warmup_steps=100, stable_steps=200, warmdown_steps=100)
        
        lrs = []
        for _ in range(400):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # 检查学习率变化
        self.assertLess(lrs[0], lrs[50])  # Warmup阶段增加
        self.assertAlmostEqual(lrs[150], lrs[200], places=5)  # 稳定阶段保持
        self.assertGreater(lrs[300], lrs[350])  # Warmdown阶段降低
        
        print("✓ Warmdown调度器测试通过")


class TestULTRATechnologies(unittest.TestCase):
    """ULTRA技术测试"""
    
    def test_aasq(self):
        """测试AASQ量子化"""
        aasq = AASQ(bits=4)
        x = torch.randn(2, 512, 768)
        output = aasq(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        print("✓ AASQ量子化测试通过")
    
    def test_ahfq(self):
        """测试AHFQ量子化"""
        layer_types = ['embedding'] + ['attention', 'mlp'] * 6
        ahfq = AHFQ(num_layers=12, layer_types=layer_types)
        
        # 检查每层的量子化精度
        bits_0 = ahfq.get_bits(0)  # embedding
        bits_1 = ahfq.get_bits(1)  # attention
        bits_2 = ahfq.get_bits(2)  # mlp
        
        self.assertEqual(bits_0, 8)  # embedding: INT8
        self.assertEqual(bits_1, 6)  # attention: INT6
        self.assertIn(bits_2, [4, 5])  # mlp: INT4或INT5
        
        print("✓ AHFQ量子化测试通过")
    
    def test_legal_ttt(self):
        """测试Legal TTT"""
        model = nn.Linear(10, 10)
        ttt = LegalTTT(model, lr=1e-5)
        
        x = torch.randn(2, 10)
        output = ttt(x)
        
        self.assertEqual(output.shape, (2, 10))
        self.assertFalse(torch.isnan(output).any())
        print("✓ Legal TTT测试通过")


class TestFusionPlusInnovations(unittest.TestCase):
    """Fusion+创新优化测试"""
    
    def test_self_generated_calibration(self):
        """测试自生成GPTQ校准"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        calib = SelfGeneratedCalibration(model, num_samples=32)
        calib_data = calib.calibrate()
        
        self.assertIsNotNone(calib_data)
        self.assertFalse(torch.isnan(calib_data).any())
        print("✓ 自生成GPTQ校准测试通过")
    
    def test_cross_layer_adaptive_quantization(self):
        """测试跨层自适应量子化"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        adapt = CrossLayerAdaptiveQuantization(model)
        x = torch.randn(2, 10)
        target = torch.randn(2, 10)
        
        importance = adapt.compute_layer_importance(x, target)
        self.assertGreater(len(importance), 0)
        
        print("✓ 跨层自适应量子化测试通过")
    
    def test_mixed_precision_quantization(self):
        """测试混合精度量子化"""
        mixed = MixedPrecisionQuantization()
        
        bits_emb = mixed.get_precision('embedding')
        bits_att = mixed.get_precision('attention')
        bits_mlp = mixed.get_precision('mlp')
        
        self.assertEqual(bits_emb, 8)
        self.assertEqual(bits_att, 6)
        self.assertEqual(bits_mlp, 4)
        
        print("✓ 混合精度量子化测试通过")
    
    def test_lightweight_pruning(self):
        """测试轻量级剪枝"""
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        pruning = LightweightPruning(model, pruning_ratio=0.1)
        
        # 计算原始参数数量
        original_params = sum(p.numel() for p in model.parameters())
        
        # 执行剪枝
        pruning.prune()
        
        # 检查参数是否被剪枝
        pruned_params = sum((p != 0).sum().item() for p in model.parameters())
        
        self.assertLess(pruned_params, original_params)
        print("✓ 轻量级剪枝测试通过")


class TestFusionPlusGPT(unittest.TestCase):
    """Fusion+ GPT模型测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'vocab_size': 1000,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'max_seq_len': 512,
        }
        self.model = FusionPlusGPT(self.config)
    
    def test_model_forward_pass(self):
        """测试模型前向传播"""
        x = torch.randint(0, 1000, (2, 128))
        output = self.model(x)
        
        self.assertEqual(output.shape, (2, 128, 1000))
        self.assertFalse(torch.isnan(output).any())
        print("✓ 模型前向传播测试通过")
    
    def test_model_backward_pass(self):
        """测试模型反向传播"""
        x = torch.randint(0, 1000, (2, 128))
        target = torch.randint(0, 1000, (2, 128))
        
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output.view(-1, 1000), target.view(-1))
        loss.backward()
        
        # 检查梯度
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any())
        
        print("✓ 模型反向传播测试通过")
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        x = torch.randint(0, 1000, (2, 128))
        
        # 前向传播
        output1 = self.model(x)
        
        # 保存模型
        torch.save(self.model.state_dict(), '/tmp/fusion_plus_model.pt')
        
        # 创建新模型并加载
        model2 = FusionPlusGPT(self.config)
        model2.load_state_dict(torch.load('/tmp/fusion_plus_model.pt'))
        
        # 前向传播
        output2 = model2(x)
        
        # 检查输出是否相同
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
        
        print("✓ 模型保存和加载测试通过")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'vocab_size': 1000,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'max_seq_len': 512,
        }
        self.model = FusionPlusGPT(self.config)
        self.optimizer = MuonOptimizer(self.model.parameters(), lr=1e-3)
        self.scheduler = WarmdownScheduler(self.optimizer, warmup_steps=100, stable_steps=200, warmdown_steps=100)
    
    def test_training_loop(self):
        """测试训练循环"""
        for step in range(10):
            x = torch.randint(0, 1000, (2, 128))
            target = torch.randint(0, 1000, (2, 128))
            
            batch = {'input_ids': x, 'labels': target}
            loss = train_step(self.model, self.optimizer, batch)
            self.scheduler.step()
            
            self.assertFalse(torch.isnan(torch.tensor(loss)))
        
        print("✓ 训练循环测试通过")
    
    def test_fusion_plus_optimizations(self):
        """测试Fusion+优化应用"""
        self.model.apply_fusion_plus_optimizations()
        
        # 检查优化是否成功应用
        self.assertIsNotNone(self.model.self_gen_calib.calibration_data)
        self.assertGreater(len(self.model.cross_layer_adapt.layer_importance), 0)
        
        print("✓ Fusion+优化应用测试通过")
    
    def test_end_to_end_pipeline(self):
        """测试端到端管道"""
        # 应用优化
        self.model.apply_fusion_plus_optimizations()
        
        # 训练几步
        for _ in range(5):
            x = torch.randint(0, 1000, (2, 128))
            target = torch.randint(0, 1000, (2, 128))
            batch = {'input_ids': x, 'labels': target}
            train_step(self.model, self.optimizer, batch)
            self.scheduler.step()
        
        # 评估
        x = torch.randint(0, 1000, (2, 128))
        output = self.model(x)
        
        self.assertEqual(output.shape, (2, 128, 1000))
        print("✓ 端到端管道测试通过")


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'vocab_size': 1000,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'max_seq_len': 512,
        }
        self.model = FusionPlusGPT(self.config)
    
    def test_forward_pass_speed(self):
        """测试前向传播速度"""
        x = torch.randint(0, 1000, (2, 128))
        
        # 预热
        for _ in range(10):
            _ = self.model(x)
        
        # 计时
        start = time.time()
        for _ in range(100):
            _ = self.model(x)
        elapsed = time.time() - start
        
        throughput = (100 * 2 * 128) / elapsed
        print(f"✓ 前向传播速度测试通过 (吞吐量: {throughput:.0f} samples/sec)")
    
    def test_model_size(self):
        """测试模型大小"""
        # 保存模型
        torch.save(self.model.state_dict(), '/tmp/fusion_plus_model_size.pt')
        
        # 获取文件大小
        size_mb = os.path.getsize('/tmp/fusion_plus_model_size.pt') / 1024 / 1024
        
        print(f"✓ 模型大小测试通过 (大小: {size_mb:.2f} MB)")


class TestRuleCompliance(unittest.TestCase):
    """规则合规性测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'vocab_size': 1000,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'max_seq_len': 512,
        }
        self.model = FusionPlusGPT(self.config)
    
    def test_16mb_compression_limit(self):
        """测试16MB压缩限制"""
        # 保存模型
        model_path = '/tmp/fusion_plus_model_compression.pt'
        torch.save(self.model.state_dict(), model_path)
        
        # 压缩
        compressed_path = model_path + '.gz'
        with open(model_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # 获取压缩大小
        original_size = os.path.getsize(model_path) / 1024 / 1024
        compressed_size = os.path.getsize(compressed_path) / 1024 / 1024
        
        print(f"✓ 16MB压缩限制测试通过")
        print(f"  原始大小: {original_size:.2f} MB")
        print(f"  压缩大小: {compressed_size:.2f} MB")
        print(f"  压缩比: {original_size / compressed_size:.2f}x")
        
        # 检查是否通过16MB限制
        self.assertLess(compressed_size, 16.0, "压缩模型超过16MB限制!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Fusion+ Scheme: 完整测试套件")
    print("=" * 80 + "\n")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试
    suite.addTests(loader.loadTestsFromTestCase(TestPRStandardComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestULTRATechnologies))
    suite.addTests(loader.loadTestsFromTestCase(TestFusionPlusInnovations))
    suite.addTests(loader.loadTestsFromTestCase(TestFusionPlusGPT))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleCompliance))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 某些测试失败!")
    
    print("=" * 80 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
