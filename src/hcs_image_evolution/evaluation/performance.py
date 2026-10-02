"""Inference performance, memory, and latency benchmarking."""

import time
from collections.abc import Callable
from typing import Any

import psutil
import torch

from hcs_image_evolution.utils.logging import log_event, logger


class PerformanceBenchmark:
    """Profiles execution latency, throughput, and system/GPU memory consumption."""

    @staticmethod
    def measure(
        fn: Callable[[], Any],
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
    ) -> dict[str, float]:
        """Runs warmup and benchmark passes, returning latency and memory metrics."""
        for _ in range(warmup_runs):
            fn()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        latencies = []
        for _ in range(benchmark_runs):
            start = time.time()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.time() - start)

        avg_latency = float(sum(latencies) / len(latencies))
        fps = 1.0 / max(avg_latency, 1e-5)

        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

        mem = psutil.virtual_memory()
        results = {
            "mean_latency_seconds": round(avg_latency, 3),
            "min_latency_seconds": round(min(latencies), 3),
            "max_latency_seconds": round(max(latencies), 3),
            "throughput_fps": round(fps, 2),
            "peak_vram_gb": peak_vram_gb,
            "ram_used_gb": round((mem.total - mem.available) / (1024**3), 2),
        }

        log_event("performance_benchmark", results)
        logger.info("Performance Benchmark: %s", results)
        return results
