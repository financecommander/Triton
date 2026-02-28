"""
Integration tests for performance benchmarks.
Tests compilation speed, inference speed, memory usage, GPU utilization, and batch scaling.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import time

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import LayerDef, Param
from backend.pytorch.codegen import generate_pytorch_code
from tests.integration.test_utils import (
    measure_inference_time,
    measure_memory_usage,
    benchmark_batch_sizes,
    measure_time,
)


class TestPerformanceBenchmarks:
    """Test performance benchmarks."""
    
    def test_code_generation_speed(self, simple_layer_def):
        """Test speed of code generation."""
        times = []
        
        for _ in range(10):
            with measure_time() as get_time:
                code = generate_pytorch_code(simple_layer_def)
            times.append(get_time())
        
        avg_time = sum(times) / len(times)
        
        # Code generation should be fast (< 0.1s)
        assert avg_time < 0.1
        assert code is not None
    
    def test_compilation_speed_simple_layer(self, simple_layer_def):
        """Test compilation speed for simple layer."""
        start_time = time.perf_counter()
        
        # Generate code
        code = generate_pytorch_code(simple_layer_def)
        
        # Compile and instantiate
        namespace = {}
        exec(code, namespace)
        model = namespace["SimpleTernaryLayer"]()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should be fast (< 1 second)
        assert total_time < 1.0
        assert model is not None
    
    def test_compilation_speed_complex_layer(self, multi_layer_def):
        """Test compilation speed for complex layer."""
        start_time = time.perf_counter()
        
        code = generate_pytorch_code(multi_layer_def)
        namespace = {}
        exec(code, namespace)
        model = namespace["MultiLayerTernary"]()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should still be reasonably fast (< 2 seconds)
        assert total_time < 2.0
        assert model is not None
    
    @pytest.mark.parametrize("num_layers", [1, 5, 10])
    def test_compilation_scaling_with_layers(self, num_layers):
        """Test how compilation time scales with number of layers."""
        layer_defs = []
        
        for i in range(num_layers):
            layer_def = LayerDef(
                name=f"Layer{i}",
                params=[
                    Param(name="weights", param_type="TernaryTensor", shape=[64, 64]),
                ],
                body=[]
            )
            layer_defs.append(layer_def)
        
        start_time = time.perf_counter()
        
        for layer_def in layer_defs:
            code = generate_pytorch_code(layer_def)
            namespace = {}
            exec(code, namespace)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should scale linearly or better
        assert total_time < num_layers * 0.5
    
    def test_inference_speed_small_batch(self, compiled_simple_model):
        """Test inference speed with small batch."""
        x = torch.randn(1, 64)
        
        timing = measure_inference_time(
            compiled_simple_model,
            x,
            warmup_iterations=10,
            benchmark_iterations=100
        )
        
        # Single sample should be very fast
        assert timing['mean'] < 0.01  # < 10ms
        assert timing['std'] >= 0
    
    def test_inference_speed_large_batch(self, compiled_simple_model):
        """Test inference speed with large batch."""
        x = torch.randn(128, 64)
        
        timing = measure_inference_time(
            compiled_simple_model,
            x,
            warmup_iterations=5,
            benchmark_iterations=50
        )
        
        # Large batch should still be fast
        assert timing['mean'] < 0.1  # < 100ms
        assert timing['max'] < 0.5
    
    def test_inference_throughput(self, compiled_simple_model):
        """Test inference throughput (samples per second)."""
        batch_size = 32
        x = torch.randn(batch_size, 64)
        
        timing = measure_inference_time(
            compiled_simple_model,
            x,
            warmup_iterations=10,
            benchmark_iterations=100
        )
        
        # Calculate throughput
        throughput = batch_size / timing['mean']  # samples per second
        
        # Should achieve reasonable throughput
        assert throughput > 100  # > 100 samples/sec
    
    def test_batch_size_scaling(self, compiled_simple_model, benchmark_config):
        """Test performance scaling with batch size."""
        batch_sizes = benchmark_config['batch_sizes']
        
        results = benchmark_batch_sizes(
            compiled_simple_model,
            (64,),
            batch_sizes
        )
        
        # Check that all completed
        assert len(results) == len(batch_sizes)
        
        # Calculate throughput for each batch size
        throughputs = {}
        for batch_size in batch_sizes:
            throughput = batch_size / results[batch_size]['mean']
            throughputs[batch_size] = throughput
        
        # Larger batches should generally have better throughput
        assert throughputs[batch_sizes[-1]] >= throughputs[batch_sizes[0]] * 0.5
    
    def test_memory_usage_small_model(self, compiled_simple_model):
        """Test memory usage of small model."""
        x = torch.randn(4, 64)
        
        memory = measure_memory_usage(compiled_simple_model, x)
        
        # Small model should use minimal memory
        assert memory['total_memory_mb'] < 10
        assert memory['model_memory_mb'] > 0
    
    def test_memory_usage_scaling(self, model_factory):
        """Test memory usage scaling with model size."""
        sizes = [(32, 64), (64, 128), (128, 256), (256, 512)]
        memory_usages = []
        
        for in_features, out_features in sizes:
            model = model_factory(in_features, out_features)
            x = torch.randn(4, in_features)
            memory = measure_memory_usage(model, x)
            memory_usages.append(memory['total_memory_mb'])
        
        # Memory should scale with model size
        assert memory_usages[-1] > memory_usages[0]
        
        # Should be roughly proportional (allow variance)
        ratio = memory_usages[-1] / memory_usages[0]
        expected_ratio = (sizes[-1][0] * sizes[-1][1]) / (sizes[0][0] * sizes[0][1])
        assert ratio > expected_ratio * 0.3  # Allow overhead
    
    @pytest.mark.parametrize("input_size", [32, 64, 128, 256, 512])
    def test_inference_speed_vs_input_size(self, input_size):
        """Test how inference speed varies with input size."""
        model = nn.Linear(input_size, 128)
        x = torch.randn(8, input_size)
        
        timing = measure_inference_time(
            model,
            x,
            warmup_iterations=5,
            benchmark_iterations=50
        )
        
        # All should complete reasonably fast
        assert timing['mean'] < 0.1
    
    def test_resnet18_inference_benchmark(self):
        """Benchmark ResNet18 inference speed."""
        from models.resnet18.ternary_resnet18 import TernaryResNet, BasicBlock
        
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        model.eval()
        
        x = torch.randn(8, 3, 32, 32)
        
        timing = measure_inference_time(
            model,
            x,
            warmup_iterations=5,
            benchmark_iterations=20
        )
        
        # Should complete in reasonable time
        assert timing['mean'] < 1.0  # < 1 second per batch
        
        # Calculate FPS
        fps = 8 / timing['mean']
        assert fps > 10  # > 10 FPS
    
    def test_mobilenet_inference_benchmark(self):
        """Benchmark MobileNetV2 inference speed."""
        from models.mobilenetv2.ternary_mobilenetv2 import TernaryMobileNetV2
        
        model = TernaryMobileNetV2(num_classes=10)
        model.eval()
        
        x = torch.randn(8, 3, 32, 32)
        
        timing = measure_inference_time(
            model,
            x,
            warmup_iterations=5,
            benchmark_iterations=20
        )
        
        # MobileNet should be faster than ResNet
        assert timing['mean'] < 0.5  # < 0.5 seconds per batch
        
        # Calculate FPS
        fps = 8 / timing['mean']
        assert fps > 20  # > 20 FPS
    
    def test_memory_usage_resnet18(self):
        """Test memory usage of ResNet18."""
        from models.resnet18.ternary_resnet18 import TernaryResNet, BasicBlock
        
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        
        memory = measure_memory_usage(model, x)
        
        # Ternary ResNet18 should be memory efficient
        assert memory['total_memory_mb'] < 200
    
    def test_memory_usage_mobilenet(self):
        """Test memory usage of MobileNetV2."""
        from models.mobilenetv2.ternary_mobilenetv2 import TernaryMobileNetV2
        
        model = TernaryMobileNetV2(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        
        memory = measure_memory_usage(model, x)
        
        # MobileNet should use less memory than ResNet
        assert memory['total_memory_mb'] < 100
    
    def test_warmup_effect_on_timing(self, compiled_simple_model):
        """Test effect of warmup on timing measurements."""
        x = torch.randn(8, 64)
        
        # Without warmup
        timing_no_warmup = measure_inference_time(
            compiled_simple_model,
            x,
            warmup_iterations=0,
            benchmark_iterations=50
        )
        
        # With warmup
        timing_with_warmup = measure_inference_time(
            compiled_simple_model,
            x,
            warmup_iterations=10,
            benchmark_iterations=50
        )
        
        # Both should complete
        assert timing_no_warmup['mean'] > 0
        assert timing_with_warmup['mean'] > 0
        
        # Warmup typically reduces variance
        assert timing_with_warmup['std'] <= timing_no_warmup['std'] * 2
    
    def test_inference_consistency(self, compiled_simple_model):
        """Test consistency of inference timing."""
        x = torch.randn(8, 64)
        
        # Run multiple benchmark sessions
        timings = []
        for _ in range(3):
            timing = measure_inference_time(
                compiled_simple_model,
                x,
                warmup_iterations=10,
                benchmark_iterations=50
            )
            timings.append(timing['mean'])
        
        # Timings should be relatively consistent
        avg_timing = sum(timings) / len(timings)
        for t in timings:
            assert abs(t - avg_timing) < avg_timing * 0.5  # Within 50%
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference_speed(self, compiled_simple_model):
        """Test inference speed on GPU."""
        model = compiled_simple_model.cuda()
        x = torch.randn(32, 64).cuda()
        
        timing = measure_inference_time(
            model,
            x,
            warmup_iterations=10,
            benchmark_iterations=100
        )
        
        # GPU should be fast
        assert timing['mean'] < 0.05
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, compiled_simple_model):
        """Test GPU memory usage."""
        model = compiled_simple_model.cuda()
        x = torch.randn(32, 64).cuda()
        
        memory = measure_memory_usage(model, x)
        
        # Should report CUDA memory
        assert 'cuda_peak_memory_mb' in memory
        assert memory['cuda_peak_memory_mb'] > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_vs_gpu_speed_comparison(self, compiled_simple_model):
        """Compare CPU vs GPU inference speed."""
        x_cpu = torch.randn(32, 64)
        x_gpu = x_cpu.cuda()
        
        # CPU timing
        model_cpu = compiled_simple_model
        cpu_timing = measure_inference_time(
            model_cpu,
            x_cpu,
            warmup_iterations=5,
            benchmark_iterations=50
        )
        
        # GPU timing
        model_gpu = compiled_simple_model.cuda()
        gpu_timing = measure_inference_time(
            model_gpu,
            x_gpu,
            warmup_iterations=5,
            benchmark_iterations=50
        )
        
        # Both should complete
        assert cpu_timing['mean'] > 0
        assert gpu_timing['mean'] > 0
        
        # GPU should be faster or comparable for simple models
        # (For very small models, CPU might be faster due to overhead)
    
    def test_parallel_batch_processing(self, compiled_simple_model):
        """Test parallel processing of multiple batches."""
        batches = [torch.randn(8, 64) for _ in range(10)]
        
        start_time = time.perf_counter()
        
        model = compiled_simple_model
        model.eval()
        
        with torch.no_grad():
            outputs = [model(batch) for batch in batches]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should process all batches quickly
        assert total_time < 1.0
        assert len(outputs) == 10
    
    def test_memory_leak_detection(self, compiled_simple_model):
        """Test for memory leaks during repeated inference."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        x = torch.randn(8, 64)
        model = compiled_simple_model
        model.eval()
        
        # Run many inferences
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (< 50MB)
        assert memory_increase < 50
    
    def test_inference_under_different_loads(self, compiled_simple_model):
        """Test inference performance under different computational loads."""
        batch_sizes = [1, 8, 32, 128]
        results = {}
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 64)
            timing = measure_inference_time(
                compiled_simple_model,
                x,
                warmup_iterations=5,
                benchmark_iterations=20
            )
            results[batch_size] = timing['mean']
        
        # All should complete
        for batch_size, time_taken in results.items():
            assert time_taken > 0
            assert time_taken < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
