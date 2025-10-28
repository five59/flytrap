---
title: "Usage"
description: "How to run Flytrap and interpret detection results"
permalink: /usage.html
nav_order: 3
---

# Usage Guide

This guide covers how to run Flytrap, interpret its output, and understand the various operational modes and features.

## Quick Start

### Basic Usage

```bash
# Run with SRT stream
uv run python main.py srt://192.168.1.100:4201

# Run with custom detection FPS
uv run python main.py srt://192.168.1.100:4201 12.0

# Run without URI (will prompt for input)
uv run python main.py
```

### Command Line Options

```bash
uv run python main.py [OPTIONS] [SRT_URI] [DETECTION_FPS]

Options:
  --model PATH           YOLO model path (default: yolo11m.pt)
  --confidence FLOAT     Detection confidence threshold (default: 0.4)
  --road-width INT       Road width in feet for speed calc (default: 52)
  --roi X,Y,W,H          Region of interest (default: none)
  --no-influx            Disable InfluxDB logging
  --headless             Force headless mode
  --help                 Show help message
```

## Running Modes

### Interactive Mode

When no SRT URI is provided, Flytrap enters interactive mode:

```bash
uv run python main.py

# Output:
# Enter SRT URI (or 'q' to quit): srt://192.168.1.100:4201
# Enter detection FPS (default 6.0): 12.0
# Starting Flytrap with srt://192.168.1.100:4201 at 12.0 FPS...
```

### Direct Mode

Provide URI directly for automated/scripted runs:

```bash
# Basic SRT stream
uv run python main.py srt://192.168.1.100:4201

# Custom FPS
uv run python main.py srt://192.168.1.100:4201 10.0

# With additional options
uv run python main.py srt://192.168.1.100:4201 8.0 --model yolo11l.pt --confidence 0.6
```

### Programmatic Usage

```python
from flytrap import ObjectDetector

# Basic usage
detector = ObjectDetector(srt_uri="srt://192.168.1.100:4201")
detector.run()

# Advanced configuration
detector = ObjectDetector(
    srt_uri="srt://192.168.1.100:4201",
    detection_fps=12.0,
    confidence=0.5,
    road_width_feet=60,
    roi_box=(0, 200, 1920, 1080),
    enable_influx=True
)
detector.run()
```

## Output Formats

### Console Output

Flytrap provides real-time console feedback:

```
Flytrap v1.0.0 - Real-time Object Detection
===========================================
Stream: srt://192.168.1.100:4201
Model: yolo11m.pt (confidence: 0.4)
FPS: 6.0 (target)
ROI: (0, 200, 1920, 1030)

[2025-10-27 14:23:45] Starting detection pipeline...
[2025-10-27 14:23:46] GStreamer initialized successfully
[2025-10-27 14:23:47] YOLO model loaded (CUDA available)
[2025-10-27 14:23:48] InfluxDB connection established
[2025-10-27 14:23:49] Processing started...

Frame 1: 3 detections (car: 2, truck: 1) - 45.2ms
Frame 2: 1 detection (car: 1) - 38.7ms
Frame 3: 0 detections - 25.1ms (motion skip)
Frame 4: 2 detections (person: 1, bicycle: 1) - 42.3ms
```

### Log Files

Structured logging to `vehicle_tracking.log`:

```
2025-10-27 14:23:45.123 | Track ID: 1 | Type: car | Direction: left-to-right | Speed: 28.5 mph
2025-10-27 14:23:52.456 | Track ID: 3 | Type: truck | Direction: right-to-left | Speed: 22.3 mph | Screenshot: screenshots/track_3_20251027_142352_456.jpg
2025-10-27 14:24:01.789 | Track ID: 1 | Type: car | Direction: left-to-right | Speed: 31.2 mph
2025-10-27 14:24:15.234 | Track ID: 5 | Type: person | Direction: unknown | Speed: N/A
```

**Log Format:**
```
TIMESTAMP | Track ID: ID | Type: CLASS | Direction: DIRECTION | Speed: SPEED mph [| Screenshot: PATH]
```

### Screenshots

Automatic screenshots are captured for right-to-left movement:

```
screenshots/
├── track_3_20251027_142352_456.jpg
├── track_7_20251027_143015_789.jpg
└── track_12_20251027_144201_123.jpg
```

**Screenshot Naming:** `track_{track_id}_{timestamp}.jpg`

Screenshots include:
- Original frame with bounding boxes
- Track ID and metadata overlay
- Timestamp and detection information
- Speed and direction annotations

## Real-time Visualization

### GUI Dashboard

When display is available, Flytrap shows a real-time dashboard:

**Features:**
- Live video feed with detection overlays
- Bounding boxes with confidence scores
- Track IDs and trails
- Performance metrics (FPS, memory usage)
- Detection statistics

**Controls:**
- `q` or `ESC`: Quit application
- `p`: Pause/resume processing
- `s`: Take manual screenshot
- `r`: Reset tracking

### Headless Mode

Auto-detection for WSL/SSH environments:

```bash
# Force headless mode
uv run python main.py srt://192.168.1.100:4201 --headless

# Auto-detection (no GUI in SSH sessions)
ssh user@server "uv run python main.py srt://192.168.1.100:4201"
```

## Detection Results

### Object Classes

Flytrap detects and tracks:

- **Vehicles:** car, truck, bus, motorcycle
- **People:** person
- **Two-wheelers:** bicycle, motorcycle
- **Other:** traffic signs, animals (configurable)

### Detection Data Structure

```python
# Individual detection
{
    'bbox': (x1, y1, x2, y2),      # Bounding box coordinates
    'confidence': 0.87,             # Detection confidence (0-1)
    'class_id': 2,                  # COCO class ID
    'class_name': 'car',            # Human-readable class name
    'track_id': 15                  # Unique tracking ID
}

# Track with analytics
{
    'track_id': 15,
    'class_name': 'car',
    'bbox': (450, 300, 520, 380),
    'confidence': 0.87,
    'direction': 'left-to-right',
    'speed_mph': 32.5,
    'first_seen': '2025-10-27T14:23:45.123Z',
    'last_seen': '2025-10-27T14:24:15.456Z',
    'screenshot_path': None
}
```

### Direction Detection

**Algorithm:**
1. Track object centroid movement across frames
2. Calculate trajectory vector
3. Classify as left-to-right, right-to-left, or unknown
4. Require minimum movement distance for classification

**Direction States:**
- `left-to-right`: Object moving from left side to right side
- `right-to-left`: Object moving from right side to left side
- `unknown`: Insufficient movement data or ambiguous trajectory

### Speed Calculation

**Formula:**
```
speed_mph = (pixels_per_frame * fps * road_width_feet * 0.6818) / pixels_in_track
```

**Factors:**
- `road_width_feet`: Configurable road width (default: 52 feet)
- `fps`: Detection frame rate
- `pixels_per_frame`: Object movement in pixels
- `0.6818`: Conversion factor (feet/second to mph)

**Accuracy Considerations:**
- Calibrated for perpendicular camera angles
- Requires accurate road width measurement
- Affected by camera height and lens distortion

## Performance Monitoring

### Real-time Metrics

```bash
# Console metrics (every 10 seconds)
FPS: 5.8/6.0 (actual/target)
Memory: 1.2GB RAM, 2.1GB GPU
Queue: 3/100 frames
Detections: 12 active tracks
```

### InfluxDB Metrics

**Frame Processing:**
- `processing_fps`: Actual processing frame rate
- `processing_time_ms`: Time per frame
- `queue_depth`: Frame buffer size
- `memory_usage_mb`: RAM usage

**Detection Statistics:**
- `detections_total`: Objects detected by class
- `tracks_active`: Currently tracked objects
- `direction_distribution`: Movement patterns
- `speed_histogram`: Speed measurements

### Performance Tuning

```bash
# High performance setup
uv run python main.py srt://192.168.1.100:4201 15.0 --model yolo11l.pt

# Conservative setup (low power)
uv run python main.py srt://192.168.1.100:4201 2.0 --model yolo11n.pt

# Balanced setup (recommended)
uv run python main.py srt://192.168.1.100:4201 6.0 --roi 0,200,1920,1030
```

## Error Handling

### Automatic Recovery

Flytrap includes robust error handling:

**Stream Recovery:**
```
[2025-10-27 14:25:00] SRT stream disconnected, attempting reconnection...
[2025-10-27 14:25:05] SRT stream reconnected successfully
[2025-10-27 14:25:05] Resuming detection pipeline...
```

**Fallback Mechanisms:**
- GStreamer → OpenCV → FFmpeg for video input
- GPU → CPU processing fallback
- InfluxDB → File-only logging when database unavailable

### Error Messages

**Common Errors:**
```
ERROR: SRT stream connection failed
CAUSE: Network unreachable or invalid URI
SOLUTION: Check network connectivity and SRT URI format

WARNING: GPU memory low, falling back to CPU
CAUSE: Insufficient GPU memory for model
SOLUTION: Reduce batch size or use smaller model

ERROR: InfluxDB connection timeout
CAUSE: Database unavailable
SOLUTION: Check InfluxDB status, will continue with file logging
```

## Advanced Usage

### Multi-Camera Setup

```bash
# Terminal 1: Camera 1
uv run python main.py srt://cam1:4201 --model yolo11m.pt

# Terminal 2: Camera 2
uv run python main.py srt://cam2:4201 --model yolo11m.pt

# Terminal 3: Camera 3
uv run python main.py srt://cam3:4201 --model yolo11m.pt
```

### Batch Processing

```bash
# Process video file
uv run python main.py /path/to/video.mp4

# Process RTSP stream
uv run python main.py rtsp://camera-ip/live

# Process HTTP stream
uv run python main.py http://camera-ip/video.m3u8
```

### Custom Model Integration

```python
from flytrap import ObjectDetector
from ultralytics import YOLO

# Load custom trained model
custom_model = YOLO('path/to/custom-model.pt')

detector = ObjectDetector(
    srt_uri="srt://192.168.1.100:4201",
    model=custom_model,  # Use custom model
    confidence=0.3       # Lower threshold for custom model
)
```

### API Integration

```python
import asyncio
from flytrap import ObjectDetector

async def process_detections():
    detector = ObjectDetector(srt_uri="srt://192.168.1.100:4201")

    async for detection in detector.detection_stream():
        print(f"Detected: {detection['class_name']} at {detection['bbox']}")

        # Custom processing
        if detection['class_name'] == 'car' and detection['speed_mph'] > 50:
            await send_speed_alert(detection)

asyncio.run(process_detections())
```

## Troubleshooting

### Common Issues

**No video stream:**
```bash
# Test SRT connection
gst-launch-1.0 srtsrc uri="srt://192.168.1.100:4201" ! fakesink

# Check network
ping 192.168.1.100
telnet 192.168.1.100 4201
```

**Low FPS:**
```bash
# Check GPU usage
nvidia-smi

# Monitor system resources
top -p $(pgrep -f flytrap)

# Try CPU-only mode
export CUDA_VISIBLE_DEVICES=""
uv run python main.py srt://192.168.1.100:4201
```

**Memory issues:**
```bash
# Monitor memory usage
free -h

# Check for leaks
uv run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

**Detection problems:**
```bash
# Test model loading
uv run python -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
print('Model loaded successfully')
"

# Check confidence threshold
uv run python main.py srt://192.168.1.100:4201 --confidence 0.1
```

This comprehensive usage guide covers all aspects of running and operating Flytrap for real-time object detection and tracking.