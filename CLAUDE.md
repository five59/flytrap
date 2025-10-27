# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eyeball is a Python-based real-time object detection system that integrates SRT (Secure Reliable Transport) video streaming with YOLO11 computer vision. The system receives live video feeds via SRT protocol, performs real-time object detection with tracking, calculates speed and direction of moving objects, automatically captures screenshots, and logs comprehensive metrics to InfluxDB for visualization in Grafana.

## Development Environment

- **Python Version**: 3.12+
- **Package Manager**: uv
- **Hardware Acceleration**: Auto-detected (CUDA > MPS > CPU)
- **Streaming Methods**: GStreamer (primary), OpenCV (secondary), FFmpeg (fallback)
- **Display Modes**: GUI mode (default) or headless mode (auto-detected for WSL/SSH)

## Common Commands

```bash
# Install dependencies
uv sync

# Install dev dependencies (includes JupyterLab and ipykernel)
uv sync --group dev

# Start InfluxDB and Grafana services
docker-compose up -d

# Stop services
docker-compose down

# Run application (uses default SRT URI: srt://192.168.1.195:4201)
uv run python main.py

# Run with custom SRT URI
uv run python main.py srt://your-host:port

# Test InfluxDB connection
uv run python -m eyeball.influx_client

# Start JupyterLab for experimentation
uv run jupyter lab

# Install PyTorch (platform-specific, see pyproject.toml comments)
# Mac M4 (Apple Silicon with MPS):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Windows/Linux with NVIDIA CUDA 12.1:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Windows/Linux with NVIDIA CUDA 11.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Architecture

### Video Streaming Pipeline

The system uses a three-tier fallback strategy for SRT stream reception:

1. **GStreamer** (primary): Best performance, uses `srtsrc` element with automatic format conversion to BGR
2. **OpenCV** (secondary): Falls back if GStreamer unavailable, uses `cv2.VideoCapture` with FFMPEG backend
3. **FFmpeg subprocess** (fallback): Direct pipe from ffmpeg process when OpenCV fails

Frame processing flow: SRT stream → Frame queue (GStreamer) or direct capture → Motion detection → YOLO inference → Tracking → Annotation → Display/logging

### Core Components

#### ObjectDetector ([eyeball/detector.py](eyeball/detector.py))

Main class handling the entire detection pipeline:

- **SRT Stream Connection**: Multi-method connection with automatic fallback (GStreamer → OpenCV → FFmpeg)
- **Motion Detection**: Uses MOG2 background subtraction + frame differencing to skip YOLO when no motion
- **Object Tracking**: YOLO11's built-in tracking maintains unique IDs across frames
- **Speed Calculation**: Tracks object positions over time, calculates speed in mph based on configurable road width
- **Direction Detection**: Monitors midpoint crossing to classify movement as left-to-right or right-to-left
- **Screenshot Capture**: Automatically saves annotated frames for right-to-left movement
- **Memory Management**: Aggressive memory cleanup every 20 frames, emergency cleanup at 800MB+ usage
- **Metric Overlay**: Real-time on-screen display of frame count, queue depth, processing time, memory usage
- **Headless Mode**: Auto-detects absence of DISPLAY environment variable for WSL/SSH environments

Tracked classes (COCO dataset): person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)

#### DetectionLogger ([eyeball/influx_client.py](eyeball/influx_client.py))

InfluxDB 2.7 client for metrics storage:

- **Three measurement types**:
  - `frame`: Per-frame aggregates (detection count, processing time, motion pixels, queue depth, memory usage)
  - `detection`: Individual object data (class, confidence, bounding box coordinates)
  - `direction`: Object movement tracking (class, direction, speed in mph, track ID)
- **Configuration**: Reads from environment variables (`.env` file)
- **Context Manager Support**: Automatic connection cleanup

#### Main Entry Point ([main.py](main.py))

- Auto-detects headless mode based on DISPLAY environment variable
- Accepts SRT URI as command-line argument (defaults to `srt://192.168.1.195:4201`)
- Adds connection timeout parameter to SRT URI if not present

### Key Dependencies

- **ultralytics**: YOLO11 model and inference engine with built-in tracking
- **opencv-python**: Video capture, frame manipulation, display, and background subtraction
- **numpy**: Array operations for frame data processing
- **influxdb-client**: Python client for InfluxDB time-series database
- **torch/torchvision**: PyTorch for ML model execution (platform-specific installation)
- **gi (PyGObject)**: GStreamer Python bindings for SRT streaming
- **pycairo**: Required for PyGObject
- **ffmpeg-python**: FFmpeg Python wrapper
- **python-dotenv**: Environment variable management

## InfluxDB and Grafana Setup

### Initial Setup

1. **Start services**:
   ```bash
   docker-compose up -d
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   ```

3. **Verify InfluxDB connection**:
   ```bash
   uv run python -m eyeball.influx_client
   ```

### Access

- **InfluxDB UI**: http://localhost:8086
  - Username: `admin`
  - Password: `eyeball-admin-password`

- **Grafana UI**: http://localhost:3000
  - Username: `admin`
  - Password: `admin`
  - Datasource: Pre-configured InfluxDB connection

### Configuration

Environment variables (`.env` file):
- `INFLUXDB_URL`: Server URL (default: `http://localhost:8086`)
- `INFLUXDB_TOKEN`: Auth token (default: `eyeball-super-secret-token-change-in-production`)
- `INFLUXDB_ORG`: Organization (default: `eyeball`)
- `INFLUXDB_BUCKET`: Bucket name (default: `detections`)

Docker Compose also provisions Grafana with automatic dashboard configuration from `grafana/provisioning/`.

### Data Schema

**Measurements**:
- `frame`: Per-frame metrics
  - Tags: `source`
  - Fields: `frame_number`, `detection_count`, `processing_time_ms`, `motion_pixels`, `queue_depth`, `gstreamer_buffers`, `memory_usage_mb`

- `detection`: Individual object detections
  - Tags: `source`, `class`
  - Fields: `confidence`, `frame_number`, `bbox_x1_f`, `bbox_y1_f`, `bbox_x2_f`, `bbox_y2_f`

- `direction`: Object movement tracking
  - Tags: `source`, `class`, `direction`
  - Fields: `speed_mph`, `track_id`

### Example Flux Queries

```flux
// Average detections per minute
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "frame")
  |> filter(fn: (r) => r._field == "detection_count")
  |> aggregateWindow(every: 1m, fn: mean)

// Memory usage trend
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "frame")
  |> filter(fn: (r) => r._field == "memory_usage_mb")

// Vehicles by direction
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "direction")
  |> group(columns: ["direction", "class"])
  |> count()

// Average vehicle speeds
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "direction")
  |> filter(fn: (r) => r._field == "speed_mph")
  |> group(columns: ["class"])
  |> mean()
```

## Output and Logging

### Log File Format

Vehicle tracking events are logged to `vehicle_tracking.log`:
```
2025-10-27 14:23:45.123 | Track ID: 1 | Type: car | Direction: left-to-right | Speed: 28.5 mph
2025-10-27 14:23:52.456 | Track ID: 3 | Type: truck | Direction: right-to-left | Speed: 22.3 mph | Screenshot: screenshots/track_3_20251027_142352_456.jpg
```

### Screenshots

Automatically saved to `screenshots/` directory for objects moving right-to-left, with annotated bounding boxes and labels. Filename format: `track_{id}_{timestamp}.jpg`

## Memory Management

The system implements aggressive memory management to prevent leaks during long-running sessions:

- **Regular cleanup**: Every 20 frames (~3.3 seconds at 6 FPS)
  - Multiple garbage collection passes
  - GPU cache clearing (CUDA)
  - Frame queue trimming (keep max 10 frames)
  - Stale object removal (>15 seconds old)
  - Position history limiting (keep last 10 positions)

- **Memory leak detection**: Monitors trend over last 10 readings
  - Triggers emergency cleanup if increasing >1MB/min

- **Emergency cleanup**: Triggered at 800MB+ usage
  - Clears all tracked objects
  - Empties frame queue completely
  - Resets background subtractor
  - Forces system-level cache clearing

## Performance Optimizations

- **Frame skipping**: Processes every 5th frame from GStreamer (reduces ~30 FPS to ~6 FPS)
- **Motion-based inference**: Skips YOLO when no significant motion detected
- **Queue management**: Max 50 frames in queue, drops frames when full
- **Batch InfluxDB writes**: All detections written in single batch per frame
