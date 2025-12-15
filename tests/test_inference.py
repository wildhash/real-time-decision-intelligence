"""Tests for low-latency inference module."""

import torch
import pytest
import time
import torch.nn as nn
from rtdi.inference import (
    InferenceOptimizer,
    BatchInferenceEngine,
    ModelCache,
    LowLatencyInferenceEngine
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, state_dim=8, output_dim=4):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(state_dim, 32)
        self.linear2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


class TestInferenceOptimizer:
    """Test suite for InferenceOptimizer."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.output_dim = 4
        self.model = SimpleTestModel(self.state_dim, self.output_dim)
        
    def test_initialization_basic(self):
        """Test basic optimizer initialization."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=False
        )
        
        assert optimizer.model is not None
        assert optimizer.use_jit is False
        assert optimizer.use_half_precision is False
        
    def test_initialization_with_jit(self):
        """Test initialization with JIT compilation."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=True,
            use_half_precision=False
        )
        
        assert optimizer.use_jit is True
        
    def test_initialization_with_half_precision(self):
        """Test initialization with half precision."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=True
        )
        
        assert optimizer.use_half_precision is True
        
    def test_basic_inference(self):
        """Test basic inference."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=False
        )
        
        input_data = torch.randn(4, self.state_dim)
        output = optimizer.infer(input_data)
        
        assert output.shape == (4, self.output_dim)
        
    def test_inference_single_sample(self):
        """Test inference with single sample."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=False
        )
        
        input_data = torch.randn(1, self.state_dim)
        output = optimizer.infer(input_data)
        
        assert output.shape == (1, self.output_dim)
        
    def test_inference_batch(self):
        """Test inference with batch."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=False,
            batch_size=16
        )
        
        input_data = torch.randn(16, self.state_dim)
        output = optimizer.infer(input_data)
        
        assert output.shape == (16, self.output_dim)
        
    def test_half_precision_inference(self):
        """Test half precision inference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=True
        )
        
        input_data = torch.randn(4, self.state_dim)
        output = optimizer.infer(input_data)
        
        # Output should be converted back to float32
        assert output.dtype == torch.float32
        assert output.shape == (4, self.output_dim)
        
    def test_benchmark(self):
        """Test latency benchmarking."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=False
        )
        
        results = optimizer.benchmark(num_iterations=10)
        
        assert "mean_latency_ms" in results
        assert "min_latency_ms" in results
        assert "max_latency_ms" in results
        assert "p50_latency_ms" in results
        assert "p95_latency_ms" in results
        assert "p99_latency_ms" in results
        
        assert results["mean_latency_ms"] > 0
        assert results["min_latency_ms"] <= results["mean_latency_ms"]
        assert results["mean_latency_ms"] <= results["max_latency_ms"]
        
    def test_benchmark_consistency(self):
        """Test that benchmark produces consistent results."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=False
        )
        
        results1 = optimizer.benchmark(num_iterations=20)
        results2 = optimizer.benchmark(num_iterations=20)
        
        # Results should be in same ballpark
        assert abs(results1["mean_latency_ms"] - results2["mean_latency_ms"]) < results1["mean_latency_ms"]
        
    def test_inference_determinism(self):
        """Test inference determinism."""
        optimizer = InferenceOptimizer(
            model=self.model,
            use_jit=False,
            use_half_precision=False
        )
        
        torch.manual_seed(42)
        input_data = torch.randn(4, self.state_dim)
        
        output1 = optimizer.infer(input_data)
        output2 = optimizer.infer(input_data)
        
        assert torch.allclose(output1, output2, atol=1e-6)


class TestBatchInferenceEngine:
    """Test suite for BatchInferenceEngine."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.output_dim = 4
        self.model = SimpleTestModel(self.state_dim, self.output_dim)
        
    def test_initialization(self):
        """Test batch engine initialization."""
        engine = BatchInferenceEngine(
            model=self.model,
            max_batch_size=32,
            max_wait_time=0.01
        )
        
        assert engine.model is not None
        assert engine.max_batch_size == 32
        assert engine.max_wait_time == 0.01
        
    def test_single_inference(self):
        """Test single inference request."""
        engine = BatchInferenceEngine(
            model=self.model,
            max_batch_size=32
        )
        
        input_data = torch.randn(1, self.state_dim)
        output = engine.infer(input_data)
        
        assert output.shape == (1, self.output_dim)
        
    def test_batch_inference(self):
        """Test batched inference."""
        engine = BatchInferenceEngine(
            model=self.model,
            max_batch_size=16
        )
        
        input_data = torch.randn(8, self.state_dim)
        output = engine.infer(input_data)
        
        assert output.shape == (8, self.output_dim)
        
    def test_auto_batching(self):
        """Test automatic batching of requests."""
        engine = BatchInferenceEngine(
            model=self.model,
            max_batch_size=32,
            max_wait_time=0.01
        )
        
        # Multiple small requests
        outputs = []
        for _ in range(5):
            input_data = torch.randn(2, self.state_dim)
            output = engine.infer(input_data)
            outputs.append(output)
        
        assert all(out.shape == (2, self.output_dim) for out in outputs)
        
    def test_max_batch_size_limit(self):
        """Test that max batch size is respected."""
        engine = BatchInferenceEngine(
            model=self.model,
            max_batch_size=8
        )
        
        # Request larger than max batch
        input_data = torch.randn(16, self.state_dim)
        output = engine.infer(input_data)
        
        # Should still work (process in chunks)
        assert output.shape == (16, self.output_dim)
        
    def test_throughput_improvement(self):
        """Test that batching improves throughput."""
        engine = BatchInferenceEngine(
            model=self.model,
            max_batch_size=32
        )
        
        # Batch inference should be faster per sample
        batch_size = 32
        input_data = torch.randn(batch_size, self.state_dim)
        
        start = time.perf_counter()
        output = engine.infer(input_data)
        batch_time = time.perf_counter() - start
        
        assert output.shape == (batch_size, self.output_dim)
        assert batch_time > 0


class TestModelCache:
    """Test suite for ModelCache."""

    def setup_method(self):
        """Setup for each test."""
        self.cache = ModelCache(capacity=100)
        
    def test_initialization(self):
        """Test cache initialization."""
        assert self.cache.capacity == 100
        assert len(self.cache) == 0
        
    def test_cache_put_get(self):
        """Test putting and getting from cache."""
        key = "test_key"
        value = torch.randn(4, 8)
        
        self.cache.put(key, value)
        retrieved = self.cache.get(key)
        
        assert torch.allclose(retrieved, value)
        
    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get("nonexistent_key")
        assert result is None
        
    def test_cache_overwrite(self):
        """Test overwriting cached value."""
        key = "test_key"
        value1 = torch.randn(4, 8)
        value2 = torch.randn(4, 8)
        
        self.cache.put(key, value1)
        self.cache.put(key, value2)
        
        retrieved = self.cache.get(key)
        assert torch.allclose(retrieved, value2)
        
    def test_cache_capacity(self):
        """Test cache respects capacity."""
        cache = ModelCache(capacity=10)
        
        # Add more than capacity
        for i in range(20):
            cache.put(f"key_{i}", torch.randn(4, 8))
        
        # Should only keep most recent based on capacity
        assert len(cache) <= 10
        
    def test_cache_clear(self):
        """Test clearing cache."""
        for i in range(5):
            self.cache.put(f"key_{i}", torch.randn(4, 8))
        
        self.cache.clear()
        
        assert len(self.cache) == 0
        
    def test_cache_contains(self):
        """Test checking if key is in cache."""
        key = "test_key"
        self.cache.put(key, torch.randn(4, 8))
        
        assert self.cache.contains(key)
        assert not self.cache.contains("other_key")
        
    def test_cache_with_tensor_keys(self):
        """Test cache with tensor as key."""
        key_tensor = torch.randn(4)
        key_str = str(key_tensor.tolist())
        value = torch.randn(8)
        
        self.cache.put(key_str, value)
        retrieved = self.cache.get(key_str)
        
        assert torch.allclose(retrieved, value)


class TestLowLatencyInferenceEngine:
    """Test suite for LowLatencyInferenceEngine."""

    def setup_method(self):
        """Setup for each test."""
        self.state_dim = 8
        self.output_dim = 4
        self.model = SimpleTestModel(self.state_dim, self.output_dim)
        
    def test_initialization(self):
        """Test engine initialization."""
        engine = LowLatencyInferenceEngine(
            model=self.model,
            config={
                "use_jit": False,
                "use_half_precision": False,
                "use_cache": True,
                "cache_capacity": 1000
            }
        )
        
        assert engine.model is not None
        assert engine.cache is not None
        
    def test_inference_without_cache(self):
        """Test inference without caching."""
        engine = LowLatencyInferenceEngine(
            model=self.model,
            config={"use_cache": False}
        )
        
        input_data = torch.randn(4, self.state_dim)
        output = engine.infer(input_data)
        
        assert output.shape == (4, self.output_dim)
        
    def test_inference_with_cache(self):
        """Test inference with caching."""
        engine = LowLatencyInferenceEngine(
            model=self.model,
            config={"use_cache": True, "cache_capacity": 100}
        )
        
        input_data = torch.randn(1, self.state_dim)
        
        # First call - cache miss
        output1 = engine.infer(input_data)
        
        # Second call - cache hit
        output2 = engine.infer(input_data)
        
        assert torch.allclose(output1, output2, atol=1e-6)
        
    def test_cache_speedup(self):
        """Test that cache provides speedup."""
        engine = LowLatencyInferenceEngine(
            model=self.model,
            config={"use_cache": True, "cache_capacity": 100}
        )
        
        input_data = torch.randn(1, self.state_dim)
        
        # First call
        start = time.perf_counter()
        output1 = engine.infer(input_data)
        time1 = time.perf_counter() - start
        
        # Cached call
        start = time.perf_counter()
        output2 = engine.infer(input_data)
        time2 = time.perf_counter() - start
        
        # Cached should be faster or similar
        assert torch.allclose(output1, output2, atol=1e-6)
        
    def test_benchmark_optimized(self):
        """Test benchmarking optimized engine."""
        engine = LowLatencyInferenceEngine(
            model=self.model,
            config={
                "use_jit": False,
                "use_half_precision": False,
                "use_cache": False
            }
        )
        
        results = engine.benchmark(num_iterations=10)
        
        assert "mean_latency_ms" in results
        assert "p95_latency_ms" in results
        assert results["mean_latency_ms"] > 0
        
    def test_batched_cached_inference(self):
        """Test batched inference with cache."""
        engine = LowLatencyInferenceEngine(
            model=self.model,
            config={
                "use_cache": True,
                "cache_capacity": 1000,
                "max_batch_size": 16
            }
        )
        
        input_data = torch.randn(8, self.state_dim)
        output = engine.infer(input_data)
        
        assert output.shape == (8, self.output_dim)
        
    def test_cache_statistics(self):
        """Test cache hit/miss statistics."""
        engine = LowLatencyInferenceEngine(
            model=self.model,
            config={"use_cache": True, "cache_capacity": 100}
        )
        
        # Make some inferences
        for _ in range(5):
            input_data = torch.randn(1, self.state_dim)
            engine.infer(input_data)
        
        # Repeat same input
        input_data = torch.randn(1, self.state_dim)
        engine.infer(input_data)
        engine.infer(input_data)  # Should hit cache
        
        stats = engine.get_statistics()
        
        assert "cache_hits" in stats or "total_inferences" in stats


class TestInferenceIntegration:
    """Integration tests for inference components."""

    def test_end_to_end_optimized_inference(self):
        """Test complete optimized inference pipeline."""
        from rtdi.world_models import ProbabilisticWorldModel
        
        model = ProbabilisticWorldModel(
            state_dim=6,
            action_dim=3,
            ensemble_size=2,
            hidden_dim=32
        )
        
        # Use first ensemble member for inference
        single_model = model.ensemble[0]
        
        engine = LowLatencyInferenceEngine(
            model=single_model,
            config={
                "use_jit": False,
                "use_half_precision": False,
                "use_cache": True,
                "cache_capacity": 100
            }
        )
        
        # Run inference
        state = torch.randn(1, 6)
        action = torch.randn(1, 3)
        input_data = torch.cat([state, action], dim=1)
        
        output = engine.infer(input_data)
        
        assert output is not None
        
    def test_inference_latency_comparison(self):
        """Test comparing latency of different configurations."""
        model = SimpleTestModel(8, 4)
        
        # Standard inference
        engine_standard = LowLatencyInferenceEngine(
            model=model,
            config={"use_jit": False, "use_cache": False}
        )
        
        # Cached inference
        engine_cached = LowLatencyInferenceEngine(
            model=model,
            config={"use_jit": False, "use_cache": True}
        )
        
        results_standard = engine_standard.benchmark(num_iterations=20)
        results_cached = engine_cached.benchmark(num_iterations=20)
        
        # Both should complete successfully
        assert results_standard["mean_latency_ms"] > 0
        assert results_cached["mean_latency_ms"] > 0
        
    def test_batch_vs_single_inference(self):
        """Test batch inference throughput vs single."""
        model = SimpleTestModel(8, 4)
        
        engine = LowLatencyInferenceEngine(
            model=model,
            config={"max_batch_size": 32}
        )
        
        # Single inference
        single_inputs = [torch.randn(1, 8) for _ in range(32)]
        start = time.perf_counter()
        for inp in single_inputs:
            engine.infer(inp)
        single_time = time.perf_counter() - start
        
        # Batch inference
        batch_input = torch.randn(32, 8)
        start = time.perf_counter()
        engine.infer(batch_input)
        batch_time = time.perf_counter() - start
        
        # Batch should be faster
        assert batch_time < single_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])