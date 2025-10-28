---
layout: default
title: "Configuration Guide"
description: "Complete configuration options for Flytrap including environment variables and parameters"
nav_order: 4
---

# Configuration Guide

Flytrap offers extensive configuration options to adapt to different environments, hardware capabilities, and use cases. This guide covers all configuration methods and parameters.

## Configuration Methods

### 1. Environment Variables (.env file)

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
nano .env
```

### 2. Command Line Arguments

Pass configuration directly when running:

```bash
# Basic usage with SRT URI
uv run python main.py srt://192.168.1.100:4201

# With custom detection FPS
uv run python main.py srt://192.168.1.100:4201 12.0
```

### 3. Programmatic Configuration

Configure via Python API:

```python
from flytrap import ObjectDetector

detector = ObjectDetector(
    srt_uri="srt://192.168.1.195:4201",
    model_path='yolo11m.pt',
    confidence=0.4,
    road_width_feet=52,
    detection_fps=6.0,
    roi_box=(0, 200, 1920, 1030),
    enable_influx=True,
    headless=False
)
```

## Environment Variables

### InfluxDB Configuration

```bash
# InfluxDB connection settings
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=flytrap-super-secret-token-change-in-production
INFLUXDB_ORG=flytrap
INFLUXDB_BUCKET=detections

# Optional: InfluxDB authentication
INFLUXDB_USERNAME=admin
INFLUXDB_PASSWORD=flytrap-admin-password
```

### Application Settings

```bash
# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=vehicle_tracking.log

# Performance settings
DETECTION_FPS=6.0
MAX_QUEUE_SIZE=100
MEMORY_CLEANUP_INTERVAL=20

# Hardware settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## ObjectDetector Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `srt_uri` | `str` | Required | SRT stream URI (e.g., `srt://192.168.1.100:4201`) |
| `model_path` | `str` | `'yolo11m.pt'` | Path to YOLO model weights |
| `confidence` | `float` | `0.4` | Detection confidence threshold (0.0-1.0) |
| `detection_fps` | `float` | `6.0` | Target processing frame rate |
| `road_width_feet` | `int` | `52` | Road width for speed calculations |

### Region of Interest (ROI)

```python
# Define region of interest as (x, y, width, height)
roi_box = (0, 200, 1920, 1030)  # Focus on road area, exclude sky

# Disable ROI (process entire frame)
roi_box = None
```

**ROI Configuration Tips:**
- Use ROI to focus processing on relevant areas
- Exclude sky, buildings, or irrelevant background
- Improves performance by reducing processing area
- Coordinates are (left, top, right, bottom) pixels

### Hardware Acceleration

```python
# Auto-detect (recommended)
detector = ObjectDetector(srt_uri="...")  # Uses CUDA if available

# Force CPU
detector = ObjectDetector(srt_uri="...", device='cpu')

# Force specific GPU
detector = ObjectDetector(srt_uri="...", device='cuda:1')

# Apple Silicon (MPS)
detector = ObjectDetector(srt_uri="...", device='mps')
```

### Memory Management

```python
# Aggressive cleanup (recommended for long-running)
memory_manager = MemoryManager(cleanup_interval=20)

# Less aggressive (for debugging)
memory_manager = MemoryManager(cleanup_interval=100)

# Disable cleanup (not recommended)
memory_manager = None
```

## Command Line Usage

### Basic Commands

```bash
# Run with SRT stream
uv run python main.py srt://192.168.1.100:4201

# Run with custom FPS
uv run python main.py srt://192.168.1.100:4201 12.0

# Run without URI (will prompt)
uv run python main.py
```

### Advanced Options

```bash
# Custom model and confidence
uv run python main.py srt://192.168.1.100:4201 --model yolo11l.pt --confidence 0.6

# Custom road width for speed calculation
uv run python main.py srt://192.168.1.100:4201 --road-width 60

# Region of interest
uv run python main.py srt://192.168.1.100:4201 --roi 0,200,1920,1030

# Disable InfluxDB logging
uv run python main.py srt://192.168.1.100:4201 --no-influx

# Force headless mode
uv run python main.py srt://192.168.1.100:4201 --headless
```

## Configuration Files

### .env.example Template

```bash
# InfluxDB Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=flytrap-super-secret-token-change-in-production
INFLUXDB_ORG=flytrap
INFLUXDB_BUCKET=detections

# Application Configuration
LOG_LEVEL=INFO
LOG_FILE=vehicle_tracking.log

# Performance Tuning
DETECTION_FPS=6.0
MAX_QUEUE_SIZE=100
MEMORY_CLEANUP_INTERVAL=20

# Hardware Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### JSON Configuration (Advanced)

For complex deployments, use JSON configuration:

```json
{
  "influxdb": {
    "url": "http://localhost:8086",
    "token": "flytrap-super-secret-token-change-in-production",
    "org": "flytrap",
    "bucket": "detections"
  },
  "detection": {
    "model_path": "yolo11m.pt",
    "confidence": 0.4,
    "fps": 6.0,
    "roi_box": [0, 200, 1920, 1030]
  },
  "analytics": {
    "road_width_feet": 52,
    "enable_screenshots": true,
    "screenshot_direction": "right-to-left"
  },
  "system": {
    "memory_cleanup_interval": 20,
    "max_queue_size": 100,
    "log_level": "INFO"
  }
}
```

## Performance Tuning

### FPS Configuration

```python
# High performance (requires powerful GPU)
detector = ObjectDetector(srt_uri="...", detection_fps=15.0)

# Balanced performance (recommended)
detector = ObjectDetector(srt_uri="...", detection_fps=6.0)

# Low power/conservative
detector = ObjectDetector(srt_uri="...", detection_fps=2.0)
```

### Memory Optimization

```python
# For systems with limited RAM
detector = ObjectDetector(
    srt_uri="...",
    max_queue_size=50,  # Smaller buffer
    memory_cleanup_interval=10  # More frequent cleanup
)

# For high-memory systems
detector = ObjectDetector(
    srt_uri="...",
    max_queue_size=200,  # Larger buffer
    memory_cleanup_interval=50  # Less frequent cleanup
)
```

### GPU Memory Management

```bash
# Limit GPU memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use specific GPU
export CUDA_VISIBLE_DEVICES=1

# Enable CUDA caching
export CUDA_CACHE_DISABLE=0
```

## Environment-Specific Configuration

### Development Environment

```bash
# .env file for development
INFLUXDB_URL=http://localhost:8086
LOG_LEVEL=DEBUG
DETECTION_FPS=2.0  # Slower for debugging
MEMORY_CLEANUP_INTERVAL=5  # Frequent cleanup
```

### Production Environment

```bash
# .env file for production
INFLUXDB_URL=http://influxdb.production.company.com:8086
INFLUXDB_TOKEN=production-token-here
LOG_LEVEL=WARNING
DETECTION_FPS=10.0  # Higher performance
MEMORY_CLEANUP_INTERVAL=30  # Less frequent
```

### Docker Environment

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV INFLUXDB_URL=http://host.docker.internal:8086
ENV DETECTION_FPS=6.0
ENV LOG_LEVEL=INFO

# Copy configuration
COPY .env.production .env

# Run application
CMD ["uv", "run", "python", "main.py", "srt://stream-source:4201"]
```

### Multi-GPU Setup

```python
# Use specific GPU
detector1 = ObjectDetector(srt_uri="srt://cam1:4201", device='cuda:0')
detector2 = ObjectDetector(srt_uri="srt://cam2:4201", device='cuda:1')

# Load balancing
detectors = [
    ObjectDetector(srt_uri=f"srt://cam{i}:4201", device=f'cuda:{i%2}')
    for i in range(4)
]
```

## Validation and Testing

### Configuration Validation

```python
# Validate configuration before running
from flytrap.config import validate_config

config = {
    'srt_uri': 'srt://192.168.1.100:4201',
    'confidence': 0.4,
    'detection_fps': 6.0
}

if validate_config(config):
    detector = ObjectDetector(**config)
else:
    print("Invalid configuration")
```

### Performance Benchmarking

```python
# Test configuration performance
import time
from flytrap import ObjectDetector

detector = ObjectDetector(srt_uri="...", detection_fps=6.0)
start_time = time.time()

# Process 1000 frames
for _ in range(1000):
    frame = detector.stream_handler.get_frame()
    detections = detector.frame_processor.process(frame)

end_time = time.time()
fps = 1000 / (end_time - start_time)
print(f"Achieved FPS: {fps:.2f}")
```

## Troubleshooting Configuration

### Common Issues

**Environment variables not loaded:**
```bash
# Ensure .env file exists
ls -la .env

# Install python-dotenv
uv pip install python-dotenv

# Load manually
from dotenv import load_dotenv
load_dotenv()
```

**CUDA not available:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**InfluxDB connection failed:**
```bash
# Test connection
curl http://localhost:8086/health

# Check credentials
uv run python -c "from flytrap.influx_client import InfluxClient; c = InfluxClient(); c.test_connection()"
```

**Memory issues:**
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

This comprehensive configuration system allows Flytrap to adapt to diverse deployment scenarios while maintaining optimal performance and reliability.