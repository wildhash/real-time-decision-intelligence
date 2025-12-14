"""Low-latency inference optimizations for real-time decision making."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import time
from contextlib import contextmanager


class InferenceOptimizer:
    """Optimizer for low-latency inference."""

    def __init__(
        self,
        model: nn.Module,
        use_jit: bool = True,
        use_half_precision: bool = False,
        batch_size: int = 1
    ):
        """Initialize inference optimizer.
        
        Args:
            model: Model to optimize
            use_jit: Whether to use TorchScript JIT
            use_half_precision: Whether to use FP16
            batch_size: Default batch size
        """
        self.model = model
        self.use_jit = use_jit
        self.use_half_precision = use_half_precision
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Apply optimizations
        self.optimized_model = self._optimize_model()

    def _optimize_model(self) -> nn.Module:
        """Apply optimizations to model.
        
        Returns:
            Optimized model
        """
        model = self.model
        
        # Set to eval mode
        model.eval()
        
        # JIT compilation
        if self.use_jit and torch.cuda.is_available():
            try:
                # Create example input
                example_input = torch.randn(self.batch_size, self.model.state_dim).to(self.device)
                
                # Trace model
                with torch.no_grad():
                    model = torch.jit.trace(model, example_input)
            except Exception as e:
                print(f"JIT compilation failed: {e}, using original model")
        
        # Half precision
        if self.use_half_precision and torch.cuda.is_available():
            model = model.half()
        
        return model

    @torch.no_grad()
    def infer(self, input_data: torch.Tensor) -> torch.Tensor:
        """Fast inference with optimizations.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Model output
        """
        input_data = input_data.to(self.device)
        
        if self.use_half_precision:
            input_data = input_data.half()
        
        output = self.optimized_model(input_data)
        
        if self.use_half_precision:
            output = output.float()
        
        return output

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference latency.
        
        Args:
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark results
        """
        # Warmup
        example_input = torch.randn(self.batch_size, self.model.state_dim).to(self.device)
        for _ in range(10):
            self.infer(example_input)
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.infer(example_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
        
        return {
            "mean_latency_ms": sum(latencies) / len(latencies) * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] * 1000,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000,
        }


class BatchInferenceEngine:
    """Batched inference engine for improved throughput."""

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01
    ):
        """Initialize batch inference engine.
        
        Args:
            model: Model for inference
            max_batch_size: Maximum batch size
            max_wait_time: Maximum time to wait for batching (seconds)
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.pending_requests = []
        self.pending_start_time = None

    @torch.no_grad()
    def infer_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Batch inference.
        
        Args:
            inputs: Batched input tensor
            
        Returns:
            Batched output tensor
        """
        inputs = inputs.to(self.device)
        return self.model(inputs)

    def add_request(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """Add inference request (returns result if batch is full).
        
        Args:
            input_data: Input data
            
        Returns:
            Output if batch ready, None otherwise
        """
        self.pending_requests.append(input_data)
        
        if self.pending_start_time is None:
            self.pending_start_time = time.time()
        
        # Check if should process batch
        should_process = (
            len(self.pending_requests) >= self.max_batch_size or
            (time.time() - self.pending_start_time) >= self.max_wait_time
        )
        
        if should_process:
            return self.process_batch()
        
        return None

    def process_batch(self) -> torch.Tensor:
        """Process pending batch.
        
        Returns:
            Batched outputs
        """
        if not self.pending_requests:
            return None
        
        # Stack inputs
        batched_input = torch.stack(self.pending_requests)
        
        # Inference
        output = self.infer_batch(batched_input)
        
        # Reset
        self.pending_requests = []
        self.pending_start_time = None
        
        return output


@contextmanager
def inference_mode():
    """Context manager for inference mode with optimizations."""
    with torch.no_grad():
        with torch.inference_mode():
            yield


class ModelCache:
    """Cache for model outputs to avoid redundant computation."""

    def __init__(self, capacity: int = 1000):
        """Initialize model cache.
        
        Args:
            capacity: Cache capacity
        """
        self.capacity = capacity
        self.cache = {}
        self.access_count = {}

    def _key(self, input_data: torch.Tensor) -> str:
        """Generate cache key from input.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Cache key
        """
        # Use hash of tensor data
        return str(hash(input_data.cpu().numpy().tobytes()))

    def get(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached output.
        
        Args:
            input_data: Input data
            
        Returns:
            Cached output or None
        """
        key = self._key(input_data)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def put(self, input_data: torch.Tensor, output: torch.Tensor):
        """Cache output.
        
        Args:
            input_data: Input data
            output: Output to cache
        """
        if len(self.cache) >= self.capacity:
            # Evict least recently used
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        key = self._key(input_data)
        self.cache[key] = output.clone()
        self.access_count[key] = 0

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()


class LowLatencyInferenceEngine:
    """Complete low-latency inference engine with all optimizations."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize low-latency inference engine.
        
        Args:
            model: Model for inference
            config: Configuration dictionary
        """
        config = config or {}
        
        self.optimizer = InferenceOptimizer(
            model,
            use_jit=config.get("use_jit", True),
            use_half_precision=config.get("use_half_precision", False)
        )
        
        self.cache = ModelCache(capacity=config.get("cache_capacity", 1000))
        self.use_cache = config.get("use_cache", False)
        
        self.stats = {
            "total_inferences": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @torch.no_grad()
    def infer(self, input_data: torch.Tensor) -> torch.Tensor:
        """Fast inference with caching.
        
        Args:
            input_data: Input tensor
            
        Returns:
            Output tensor
        """
        # Check cache
        if self.use_cache:
            cached = self.cache.get(input_data)
            if cached is not None:
                self.stats["cache_hits"] += 1
                self.stats["total_inferences"] += 1
                return cached
            self.stats["cache_misses"] += 1
        
        # Inference
        output = self.optimizer.infer(input_data)
        
        # Cache result
        if self.use_cache:
            self.cache.put(input_data, output)
        
        self.stats["total_inferences"] += 1
        
        return output

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            num_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        results = self.optimizer.benchmark(num_iterations)
        results["cache_hit_rate"] = (
            self.stats["cache_hits"] / self.stats["total_inferences"]
            if self.stats["total_inferences"] > 0 else 0.0
        )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.stats.copy()
