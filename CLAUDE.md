# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eyeball is a Python-based real-time object detection system that integrates SRT (Secure Reliable Transport) video streaming with YOLO11 computer vision. The project receives live video feeds via SRT protocol and performs real-time object detection using the YOLO11 model.

## Development Environment

- **Python Version**: 3.12+
- **Package Manager**: uv
- **Hardware Acceleration**: Configured for Apple Silicon (MPS) for ML inference

## Project Structure

```
eyeball/
├── eyeball/              # Main package
│   ├── __init__.py      # Package exports
│   ├── detector.py      # ObjectDetector class (SRT-based)
│   └── influx_client.py # DetectionLogger for InfluxDB
├── main.py              # Application entry point
├── docker-compose.yml   # InfluxDB service
├── .env.example         # Environment variables template
└── pyproject.toml       # Project dependencies
```

## Common Commands

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --group dev

# Start InfluxDB (time-series database for metrics)
docker-compose up -d

# Stop InfluxDB
docker-compose down

# Run the main application
uv run python main.py

# Run with custom SRT URI
uv run python main.py srt://192.168.1.195:4201

# Start JupyterLab for experimentation
uv run jupyter lab

# Test InfluxDB connection
uv run python -m eyeball.influx_client
```

## Architecture

### Core Components

1. **ObjectDetector** ([eyeball/detector.py](eyeball/detector.py))
    - Main class for real-time object detection on SRT video streams
    - Uses OpenCV VideoCapture for SRT stream reception
    - Connects directly to SRT URI (no discovery needed)
    - Receives video in BGR format via OpenCV
    - Tracks objects across frames with unique IDs
    - Calculates speed and direction of moving objects
    - Logs detections to file and InfluxDB

2. **YOLO11 Integration**
   - Model: YOLO11 medium (`yolo11m.pt`)
   - Device: Auto-detected (CUDA > MPS > CPU)
   - Confidence threshold: 0.4 (configurable)
   - Real-time inference with object tracking
   - Detects: person, bicycle, car, motorcycle, bus, truck

3. **Video Processing Pipeline**
    - SRT receiver → BGR frame data → YOLO inference → Annotated display
    - Frame format: BGR (OpenCV/SRT) → YOLO processing
    - OpenCV used for SRT capture, visualization and window management
    - Real-time display with bounding boxes and labels

4. **DetectionLogger** ([eyeball/influx_client.py](eyeball/influx_client.py))
   - InfluxDB 2.7 client for storing detection metrics
   - Runs in Docker container (see [docker-compose.yml](docker-compose.yml))
   - Logs per-frame detection counts, processing times
   - Individual detection data: class, confidence, bounding boxes
   - Queryable for analytics and visualization (Grafana integration possible)

### Key Dependencies

- **ultralytics**: YOLO11 model and inference engine
- **opencv-python**: SRT video capture, frame manipulation and display
- **numpy**: Array operations for frame data processing
- **influxdb-client**: Python client for InfluxDB time-series database
- **torch**: PyTorch for ML model execution

## InfluxDB Setup

### Initial Setup

1. **Start InfluxDB**:
   ```bash
   docker-compose up -d
   ```

2. **Create environment file** (copy from example):
   ```bash
   cp .env.example .env
   ```

3. **Verify connection**:
   ```bash
   uv run python influx_client.py
   ```

### Configuration

Environment variables (set in `.env` file):
- `INFLUXDB_URL`: InfluxDB server URL (default: `http://localhost:8086`)
- `INFLUXDB_TOKEN`: Authentication token (default: `eyeball-super-secret-token-change-in-production`)
- `INFLUXDB_ORG`: Organization name (default: `eyeball`)
- `INFLUXDB_BUCKET`: Data bucket name (default: `detections`)

**Note**: Change the default token in production!

### Accessing InfluxDB UI

1. Navigate to http://localhost:8086
2. Login with credentials from docker-compose.yml:
   - Username: `admin`
   - Password: `eyeball-admin-password`

### Data Schema

**Measurements**:
- `frame`: Per-frame metrics (detection count, processing time, frame number)
- `detection`: Individual object detections (class, confidence, bounding box)
- `class_count`: Aggregated class counts over time

**Tags** (for filtering):
- `source`: NDI source name
- `class`: Object class name (person, car, truck, etc.)

**Fields** (measured values):
- `frame_number`: Video frame sequence number
- `detection_count`: Number of objects detected in frame
- `processing_time_ms`: YOLO inference time in milliseconds
- `confidence`: Detection confidence score (0.0-1.0)
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: Bounding box coordinates

### Example Queries

```flux
// Average detections per minute
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "frame")
  |> filter(fn: (r) => r._field == "detection_count")
  |> aggregateWindow(every: 1m, fn: mean)

// Person detections over time
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "detection")
  |> filter(fn: (r) => r.class == "person")
  |> count()

// Average processing time
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "frame")
  |> filter(fn: (r) => r._field == "processing_time_ms")
  |> mean()
```

### Known Issues

The Jupyter notebook ([Untitled.ipynb](Untitled.ipynb)) shows an `AttributeError` when accessing `video_frame.data`. The `VideoFrameSync` object may require a different API for accessing frame buffer data. Check cyndilib documentation for proper frame data access methods.
