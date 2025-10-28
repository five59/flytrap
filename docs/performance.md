---
layout: default
title: "Performance Optimization"
description: "Performance optimization techniques, benchmarking, and tuning strategies"
nav_order: 7
---

# Performance Optimization Guide

This guide covers performance optimization techniques, benchmarking, and tuning strategies for Flytrap deployments.

## Performance Metrics

### Key Performance Indicators (KPIs)

| Metric | Target | Description |
|--------|--------|-------------|
| **FPS** | 6-15 FPS | Processing frame rate |
| **Latency** | <200ms | End-to-end processing time |
| **Memory Usage** | <2GB RAM | System memory consumption |
| **GPU Memory** | <4GB VRAM | GPU memory usage |
| **CPU Usage** | <70% | CPU utilization |
| **Accuracy** | >80% mAP | Detection accuracy |

### Monitoring Performance

```python
import time
import psutil
import torch

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.process = psutil.Process()

    def update(self, frame_processed=True):
        if frame_processed:
            self.frame_count += 1

    def get_stats(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024

        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024

        return {
            'fps': fps,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'gpu_memory_mb': gpu_memory,
            'uptime_seconds': elapsed
        }

# Usage
monitor = PerformanceMonitor()
# ... processing loop ...
stats = monitor.get_stats()
print(f"FPS: {stats['fps']:.1f}, CPU: {stats['cpu_percent']:.1f}%, "
      f"RAM: {stats['memory_mb']:.1f}MB, GPU: {stats['gpu_memory_mb']:.1f}MB")
```

## Core Optimizations

### 1. Frame Processing Optimizations

#### Frame Skipping Strategy

Flytrap processes every 5th frame from 30 FPS input to achieve 6 FPS processing:

```python
class OptimizedFrameProcessor:
    def __init__(self, target_fps=6.0):
        self.frame_interval = int(30.0 / target_fps)  # 30 FPS input
        self.frame_count = 0

    def should_process_frame(self) -> bool:
        self.frame_count += 1
        return self.frame_count % self.frame_interval == 0

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        if not self.should_process_frame():
            return []  # Skip frame

        # Process frame with optimizations
        return self._detect_objects(frame)
```

**Performance Impact:**
- Reduces processing load by 80%
- Maintains real-time responsiveness
- Preserves detection accuracy for moving objects

#### Motion-Based Processing

Skip YOLO inference when no motion is detected:

```python
class MotionDetector:
    def __init__(self, threshold=5000):
        self.prev_frame = None
        self.threshold = threshold

    def detect_motion(self, frame: np.ndarray) -> bool:
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(current_gray, self.prev_frame)
        motion_score = np.sum(frame_diff)

        self.prev_frame = current_gray
        return motion_score > self.threshold

# Usage in processing pipeline
motion_detector = MotionDetector()

def process_with_motion_detection(frame):
    if not motion_detector.detect_motion(frame):
        return []  # Skip expensive YOLO inference

    return yolov_model(frame)
```

**Performance Impact:**
- Reduces inference by 60-80% in static scenes
- Maintains full accuracy for moving objects
- Minimal computational overhead

### 2. Memory Management

#### Aggressive Cleanup Strategy

Automatic memory cleanup every 20 frames:

```python
class MemoryManager:
    def __init__(self, cleanup_interval=20):
        self.cleanup_interval = cleanup_interval
        self.frame_count = 0

    def should_cleanup(self) -> bool:
        self.frame_count += 1
        return self.frame_count % self.cleanup_interval == 0

    def cleanup(self):
        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear frame buffers
        self.clear_frame_buffers()

        # Log memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory cleanup completed. Usage: {memory_mb:.1f} MB")

    def clear_frame_buffers(self):
        # Clear any cached frames
        pass
```

#### GPU Memory Optimization

```python
# PyTorch memory optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Memory-efficient inference
with torch.no_grad():
    predictions = model(frame)

# Gradient accumulation disabled
torch.set_grad_enabled(False)

# Memory pinning for faster transfers
frame_tensor = torch.from_numpy(frame).pin_memory()
```

### 3. Model Optimizations

#### Model Selection Guide

| Model | Size | FPS | Accuracy | Use Case |
|-------|------|-----|----------|----------|
| YOLO11n | 5MB | 20-30 | 70% mAP | Mobile/Edge |
| YOLO11s | 20MB | 15-25 | 75% mAP | Real-time |
| YOLO11m | 40MB | 10-20 | 80% mAP | **Balanced** |
| YOLO11l | 100MB | 5-15 | 85% mAP | High Accuracy |
| YOLO11x | 250MB | 2-10 | 88% mAP | Maximum Accuracy |

#### Dynamic Model Switching

Switch models based on system load:

```python
class AdaptiveModelManager:
    def __init__(self):
        self.models = {
            'fast': YOLO('yolo11n.pt'),
            'balanced': YOLO('yolo11m.pt'),
            'accurate': YOLO('yolo11l.pt')
        }
        self.current_model = 'balanced'
        self.performance_monitor = PerformanceMonitor()

    def select_model(self) -> str:
        stats = self.performance_monitor.get_stats()

        if stats['fps'] < 5.0:
            return 'fast'  # Switch to faster model
        elif stats['fps'] > 15.0 and stats['gpu_memory_mb'] < 2000:
            return 'accurate'  # Switch to more accurate model
        else:
            return 'balanced'  # Keep balanced model

    def predict(self, frame):
        model_key = self.select_model()
        if model_key != self.current_model:
            logger.info(f"Switching model from {self.current_model} to {model_key}")
            self.current_model = model_key

        return self.models[self.current_model](frame)
```

### 4. Hardware Acceleration

#### GPU Optimization

```python
# CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Multi-GPU support
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# GPU memory pre-allocation
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
```

#### CPU Optimization

```python
# CPU threading optimizations
torch.set_num_threads(psutil.cpu_count())
torch.set_num_interop_threads(1)  # Minimize GIL contention

# OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # Use all available cores
```

## Benchmarking

### Performance Benchmark Script

```python
import time
import numpy as np
from flytrap import ObjectDetector

def benchmark_detector(configs, test_frames=1000):
    """Benchmark different detector configurations."""

    results = []

    for config in configs:
        print(f"\nBenchmarking: {config['name']}")

        # Create detector with config
        detector = ObjectDetector(**config)

        # Warm up
        for _ in range(10):
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            detector.frame_processor.process(frame)

        # Benchmark
        start_time = time.time()
        detections = 0

        for i in range(test_frames):
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            result = detector.frame_processor.process(frame)
            detections += len(result)

        end_time = time.time()
        elapsed = end_time - start_time

        fps = test_frames / elapsed
        avg_detections = detections / test_frames

        results.append({
            'config': config['name'],
            'fps': fps,
            'avg_detections': avg_detections,
            'elapsed': elapsed
        })

        print(".2f")
        print(".1f")

    return results

# Benchmark configurations
configs = [
    {'name': 'YOLO11n Fast', 'model_path': 'yolo11n.pt', 'confidence': 0.4},
    {'name': 'YOLO11m Balanced', 'model_path': 'yolo11m.pt', 'confidence': 0.4},
    {'name': 'YOLO11l Accurate', 'model_path': 'yolo11l.pt', 'confidence': 0.4},
    {'name': 'High Confidence', 'model_path': 'yolo11m.pt', 'confidence': 0.8},
]

results = benchmark_detector(configs)
```

### System Resource Monitoring

```bash
# Monitor system resources during benchmarking
#!/bin/bash

echo "Timestamp,CPU%,Memory_MB,GPU_Memory_MB,FPS" > performance_log.csv

while true; do
    timestamp=$(date +%s)
    cpu=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    memory=$(free | grep Mem | awk '{printf "%.0f", $3/1024}')
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    # Get FPS from application logs or API
    fps=$(tail -n 10 flytrap.log | grep "FPS:" | tail -1 | sed 's/.*FPS: \([0-9.]*\).*/\1' || echo "0")

    echo "$timestamp,$cpu,$memory,$gpu_memory,$fps" >> performance_log.csv
    sleep 1
done
```

## Tuning Strategies

### 1. FPS Optimization

#### Target FPS Based on Use Case

```python
def get_optimal_fps(use_case: str) -> float:
    """Get optimal FPS for different use cases."""

    fps_targets = {
        'real_time_tracking': 15.0,    # Live monitoring
        'traffic_analysis': 6.0,       # Standard analytics
        'security_surveillance': 4.0,  # Long-term recording
        'batch_processing': 2.0,       # Offline analysis
        'low_power': 1.0               # Battery/mobile
    }

    return fps_targets.get(use_case, 6.0)
```

#### Dynamic FPS Adjustment

```python
class AdaptiveFPSController:
    def __init__(self, target_fps=6.0, adjustment_threshold=0.1):
        self.target_fps = target_fps
        self.current_fps = target_fps
        self.adjustment_threshold = adjustment_threshold
        self.measurements = []

    def measure_performance(self, actual_fps: float):
        self.measurements.append(actual_fps)
        if len(self.measurements) > 10:
            self.measurements.pop(0)

    def get_adjusted_fps(self) -> float:
        if len(self.measurements) < 5:
            return self.current_fps

        avg_fps = sum(self.measurements) / len(self.measurements)
        fps_ratio = avg_fps / self.target_fps

        if abs(fps_ratio - 1.0) > self.adjustment_threshold:
            # Adjust FPS towards target
            adjustment = (self.target_fps - avg_fps) * 0.1
            self.current_fps = max(0.5, min(30.0, self.current_fps + adjustment))

        return self.current_fps
```

### 2. Memory Optimization

#### Buffer Size Tuning

```python
def calculate_optimal_buffer_size(system_memory_gb: float, target_fps: float) -> int:
    """Calculate optimal frame buffer size based on system resources."""

    # Base buffer for smooth playback
    base_buffer = 30  # 1 second at 30 FPS

    # Adjust based on system memory
    memory_factor = min(2.0, system_memory_gb / 8.0)  # Scale up to 16GB

    # Adjust based on processing speed
    speed_factor = max(0.5, 6.0 / target_fps)  # Slower processing needs larger buffer

    optimal_size = int(base_buffer * memory_factor * speed_factor)

    return max(10, min(200, optimal_size))  # Clamp between 10-200
```

### 3. Accuracy vs Speed Trade-offs

#### Confidence Threshold Tuning

```python
def find_optimal_confidence(model, test_frames, target_precision=0.8):
    """Find optimal confidence threshold for target precision."""

    confidences = np.linspace(0.1, 0.9, 17)  # 0.1 to 0.9 in 0.05 steps
    results = []

    for conf in confidences:
        true_positives = 0
        false_positives = 0
        total_predictions = 0

        for frame in test_frames:
            predictions = model(frame, conf_threshold=conf)
            # Evaluate predictions against ground truth
            # ... evaluation logic ...

        precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0
        results.append({'confidence': conf, 'precision': precision, 'predictions': total_predictions})

    # Find confidence that meets target precision with minimal predictions
    optimal = min(
        (r for r in results if r['precision'] >= target_precision),
        key=lambda x: x['predictions'],
        default=results[0]
    )

    return optimal['confidence']
```

## Production Deployment

### Multi-Instance Scaling

```python
# Deploy multiple instances for different cameras
cameras = [
    {'id': 'cam1', 'uri': 'srt://cam1:4201', 'gpu': 0},
    {'id': 'cam2', 'uri': 'srt://cam2:4201', 'gpu': 1},
    {'id': 'cam3', 'uri': 'srt://cam3:4201', 'gpu': 0},
    {'id': 'cam4', 'uri': 'srt://cam4:4201', 'gpu': 1},
]

processes = []
for camera in cameras:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(camera['gpu'])

    process = subprocess.Popen([
        'uv', 'run', 'python', 'main.py', camera['uri'],
        '--model', 'yolo11m.pt',
        '--influx-bucket', f"detections_{camera['id']}"
    ], env=env)

    processes.append(process)

# Monitor all processes
for p in processes:
    p.wait()
```

### Load Balancing

```python
class LoadBalancer:
    def __init__(self, instances):
        self.instances = instances
        self.performance_stats = {}

    def get_best_instance(self, camera_id):
        """Get the best instance for a camera based on current load."""

        available_instances = [
            inst for inst in self.instances
            if self.get_instance_load(inst) < 0.8  # Less than 80% load
        ]

        if not available_instances:
            return self.instances[0]  # Fallback to first instance

        # Choose instance with lowest load
        return min(available_instances, key=self.get_instance_load)

    def get_instance_load(self, instance):
        """Get current load of an instance."""
        # Query instance metrics from InfluxDB or internal monitoring
        return self.performance_stats.get(instance['id'], 0.0)
```

## Monitoring & Alerting

### Performance Alerts

```python
def setup_performance_alerts():
    """Set up alerts for performance degradation."""

    alerts = [
        {
            'name': 'Low FPS Alert',
            'query': 'from(bucket: "detections") |> range(start: -5m) |> filter(fn: (r) => r._measurement == "processing") |> filter(fn: (r) => r._field == "fps") |> mean() < 5.0',
            'message': 'Processing FPS dropped below 5.0'
        },
        {
            'name': 'High Memory Usage',
            'query': 'from(bucket: "detections") |> range(start: -5m) |> filter(fn: (r) => r._measurement == "system") |> filter(fn: (r) => r._field == "memory_mb") |> mean() > 3000',
            'message': 'Memory usage exceeded 3GB'
        },
        {
            'name': 'GPU Memory Alert',
            'query': 'from(bucket: "detections") |> range(start: -5m) |> filter(fn: (r) => r._measurement == "system") |> filter(fn: (r) => r._field == "gpu_memory_mb") |> mean() > 6000',
            'message': 'GPU memory usage exceeded 6GB'
        }
    ]

    return alerts
```

### Automated Optimization

```python
class AutoOptimizer:
    def __init__(self, detector):
        self.detector = detector
        self.performance_history = []
        self.optimization_attempts = 0

    def monitor_and_optimize(self):
        """Monitor performance and apply optimizations automatically."""

        stats = self.detector.get_performance_stats()
        self.performance_history.append(stats)

        # Keep only last 100 measurements
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        # Check if optimization needed
        if self._should_optimize():
            self._apply_optimization()

    def _should_optimize(self) -> bool:
        """Determine if optimization is needed."""

        if len(self.performance_history) < 10:
            return False

        recent_fps = [s['fps'] for s in self.performance_history[-10:]]
        avg_fps = sum(recent_fps) / len(recent_fps)

        # Optimize if FPS is consistently low
        return avg_fps < 4.0 and self.optimization_attempts < 3

    def _apply_optimization(self):
        """Apply performance optimization."""

        current_conf = self.detector.confidence
        current_fps = self.detector.detection_fps

        # Try different optimization strategies
        if current_conf > 0.3:
            # Increase confidence threshold
            self.detector.confidence = max(0.3, current_conf - 0.1)
            logger.info(f"Increased confidence threshold to {self.detector.confidence}")

        elif current_fps > 2.0:
            # Reduce FPS
            self.detector.detection_fps = max(2.0, current_fps - 1.0)
            logger.info(f"Reduced detection FPS to {self.detector.detection_fps}")

        else:
            # Switch to faster model
            if 'yolo11m' in self.detector.model_path:
                self.detector.model_path = 'yolo11n.pt'
                logger.info("Switched to faster YOLO11n model")

        self.optimization_attempts += 1
```

This comprehensive performance optimization guide provides the tools and strategies needed to achieve optimal Flytrap performance across different hardware configurations and use cases.