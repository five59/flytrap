"""
Memory monitoring and cleanup module.
"""

import time
import gc
import torch
import sys
from typing import Dict, Any


class MemoryManager:
    """Handles memory monitoring and cleanup operations."""

    def __init__(self):
        self.memory_history = []
        self.last_memory_cleanup = time.time()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except ImportError:
            return 0.0

    def estimate_object_memory(self) -> Dict[str, Any]:
        """Estimate memory usage of major components."""
        memory_breakdown = {}

        try:
            # This would need access to the main objects
            # For now, return basic info
            memory_breakdown['current_memory_mb'] = self.get_memory_usage()
            return memory_breakdown
        except Exception as e:
            return {'error': str(e)}

    def aggressive_memory_cleanup(self, frame_count: int):
        """Perform aggressive memory cleanup."""
        try:
            initial_memory = self.get_memory_usage()

            # Force multiple garbage collection cycles
            collected = 0
            for _ in range(3):
                collected += gc.collect()

            if collected > 0 and frame_count % 200 == 0:
                print(f"GC: Collected {collected} objects")

            # Clear GPU/CPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            elif torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            else:
                gc.collect()

            # Track memory usage
            final_memory = self.get_memory_usage()
            self.memory_history.append((time.time(), final_memory))
            if len(self.memory_history) > 20:
                self.memory_history.pop(0)

            # Check for memory leak trend
            self._check_memory_leak_trend()

        except Exception as e:
            print(f"Memory cleanup error: {e}")

    def _check_memory_leak_trend(self):
        """Check if memory usage is trending upward."""
        if len(self.memory_history) < 10:
            return

        recent_readings = self.memory_history[-10:]
        times, memories = zip(*recent_readings)

        time_span = times[-1] - times[0]
        memory_span = memories[-1] - memories[0]
        slope = memory_span / time_span if time_span > 0 else 0

        current_memory = self.memory_history[-1][1] if self.memory_history else 0
        if current_memory > 2000 and slope > (20.0 / 60.0):
            print(f"⚠️  Memory leak detected! Trend: +{slope*60:.2f}MB/min")
            self.emergency_memory_cleanup()

    def emergency_memory_cleanup(self):
        """Emergency memory cleanup when leak is detected."""
        print("🚨 Emergency memory cleanup initiated...")

        try:
            for _ in range(5):
                gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            final_memory = self.get_memory_usage()
            print(f"Emergency cleanup complete. Memory now: {final_memory:.1f}MB")

        except Exception as e:
            print(f"Emergency cleanup error: {e}")

    def deep_cleanup(self):
        """Perform deep cleanup every 200 frames."""
        for _ in range(5):
            gc.collect()

        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()