# Eyeball - Real-time Object Detection with SRT Streams

Real-time object detection and tracking using YOLO11 with SRT (Secure Reliable Transport) video streams. Tracks vehicles, people, and bicycles with direction detection, speed calculation, and automatic screenshot capture. Includes comprehensive time-series metrics storage with InfluxDB and Grafana visualization.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- SRT video stream source
- GPU recommended (CUDA/MPS/CPU fallback)

### System Dependencies (Ubuntu/Debian)

Install required system packages for GStreamer, SRT streaming, and GUI support:

```bash
# Update package list
sudo apt update

# GStreamer core and plugins (for SRT streaming)
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

# SRT protocol support
sudo apt install -y libsrt-openssl-dev gstreamer1.0-plugins-bad

# GTK and GObject introspection (for PyGObject GUI support)
sudo apt install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
    gir1.2-gstreamer-1.0

# Qt6 development packages (alternative GUI support)
sudo apt install -y qt6-base-dev python3-pyqt6

# OpenCV development packages
sudo apt install -y python3-opencv libopencv-dev

# Additional development tools
sudo apt install -y build-essential pkg-config
```

**Note**: Some packages may require additional repositories. For Ubuntu 22.04+, the `gstreamer1.0-plugins-bad` package includes SRT support. If SRT plugins are not available in your distribution's repositories, you may need to build GStreamer from source or use a PPA.

### Verify Installation

After installing system dependencies, verify everything works:

```bash
# Test GStreamer SRT support
gst-inspect-1.0 srtsrc

# Test GTK/GObject introspection
python3 -c "import gi; gi.require_version('Gtk', '3.0'); from gi.repository import Gtk; print('GTK available')"

# Test OpenCV
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Test PyTorch installation
uv run python -c "import torch; print(f'PyTorch available: {torch.cuda.is_available()}')"

# Test full application import
uv run python -c "from eyeball import ObjectDetector; print('Eyeball import successful')"
```

### 1. Install Dependencies
```bash
# Install Python dependencies
uv sync

# Install PyTorch (platform-specific - see pyproject.toml comments)
# Mac M4: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. (Optional) Start InfluxDB & Grafana
```bash
# Copy environment template
cp .env.example .env

# Start services with pre-configured dashboards
docker-compose up -d

# Verify InfluxDB connection
uv run python -m eyeball.influx_client
```

### 3. Run the Application
```bash
# Provide SRT URI directly
uv run python main.py srt://192.168.1.100:4201

# Or run without URI to be prompted for one
uv run python main.py

# Optional: specify detection FPS
uv run python main.py srt://192.168.1.100:4201 12.0
```

## ✨ Features

- **Multi-method SRT streaming** with automatic fallback (GStreamer → OpenCV → FFmpeg)
- **Real-time YOLO11 detection** with built-in object tracking
- **Motion-based inference** to skip processing when no movement detected
- **Direction detection** (left-to-right/right-to-left) with midpoint crossing
- **Speed calculation** in mph based on configurable road width
- **Automatic screenshots** for right-to-left movement with annotations
- **Region of Interest (ROI)** support to focus on road areas
- **Memory management** with aggressive cleanup and leak detection
- **Hardware acceleration** (CUDA/MPS/CPU) with auto-detection
- **Headless mode** auto-detection for WSL/SSH environments
- **Comprehensive metrics** stored in InfluxDB with Grafana dashboards
- **Configurable detection FPS** for performance tuning

## 🎯 Architecture

The system uses a sophisticated pipeline:

1. **Stream Reception**: SRT stream → GStreamer/OpenCV/FFmpeg fallback
2. **Frame Processing**: Motion detection → YOLO inference → Tracking
3. **Analytics**: Direction/speed calculation → Screenshot capture
4. **Storage**: File logging + InfluxDB metrics + Grafana visualization

### Core Components
- `ObjectDetector`: Main orchestrator coordinating all components
- `StreamHandler`: Multi-method SRT stream reception with fallback
- `FrameProcessor`: Motion detection and YOLO inference
- `ObjectTracker`: YOLO tracking with speed/direction analytics
- `GUIDashboard`: Real-time visualization (when display available)
- `MemoryManager`: Aggressive memory cleanup and leak prevention
- `DetectionLogger`: InfluxDB client for time-series metrics

## 📊 Monitoring & Visualization

### Access Points
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **InfluxDB UI**: http://localhost:8086 (admin/eyeball-admin-password)

### Metrics Collected
- **Frame metrics**: Detection count, processing time, memory usage, queue depth
- **Object detections**: Class, confidence, bounding boxes per frame
- **Movement tracking**: Direction, speed (mph), track IDs over time

## ⚙️ Configuration

### Environment Variables (.env)
```bash
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=eyeball-super-secret-token-change-in-production
INFLUXDB_ORG=eyeball
INFLUXDB_BUCKET=detections
```

### Command Line Options
```bash
uv run python main.py <srt_uri> [detection_fps]

# Examples:
uv run python main.py srt://10.0.0.1:4201       # Custom stream at 6 FPS
uv run python main.py srt://10.0.0.1:4201 12.0 # Custom stream at 12 FPS

# If no SRT URI is provided, you will be prompted to enter one
```

### ObjectDetector Parameters
```python
from eyeball import ObjectDetector

detector = ObjectDetector(
    srt_uri="srt://192.168.1.195:4201",  # SRT stream URI
    model_path='yolo11m.pt',             # YOLO model weights
    confidence=0.4,                      # Detection threshold
    road_width_feet=52,                  # Road width for speed calc
    detection_fps=6.0,                   # Processing frame rate
    roi_box=(0, 200, 1920, 1030),       # Region of interest
    enable_influx=True,                  # Enable metrics storage
    headless=False                       # Force headless mode
)
```

## 📁 Output

### Log Files
Detections logged to `vehicle_tracking.log`:
```
2025-10-27 14:23:45.123 | Track ID: 1 | Type: car | Direction: left-to-right | Speed: 28.5 mph
2025-10-27 14:23:52.456 | Track ID: 3 | Type: truck | Direction: right-to-left | Speed: 22.3 mph | Screenshot: screenshots/track_3_20251027_142352_456.jpg
```

### Screenshots
Automatically captured for right-to-left movement in `screenshots/` with annotated frames.

## 🏗️ Development

### Project Structure
```
eyeball/
├── eyeball/                    # Main package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Configuration constants
│   ├── detector.py            # Main ObjectDetector class
│   ├── stream_handler.py      # SRT stream reception
│   ├── frame_processor.py     # Motion detection & YOLO
│   ├── object_tracker.py      # Tracking & analytics
│   ├── gui_dashboard.py       # Real-time visualization
│   ├── memory_manager.py      # Memory cleanup & monitoring
│   ├── influx_client.py       # InfluxDB metrics client
│   └── ...
├── grafana/                   # Grafana dashboard provisioning
├── main.py                    # Application entry point
├── pyproject.toml             # Dependencies & project config
├── docker-compose.yml         # InfluxDB & Grafana services
├── .env.example               # Environment template
└── AGENTS.md                  # Development guidelines
```

### Key Dependencies
- **ultralytics**: YOLO11 model and inference
- **opencv-python**: Video processing and display
- **torch/torchvision**: PyTorch ML framework
- **influxdb-client**: Time-series database client
- **pygobject/pycairo**: GStreamer SRT streaming
- **ffmpeg-python**: FFmpeg subprocess fallback

## 🔧 Troubleshooting

### Common Issues
- **No video stream**: Verify SRT URI and network connectivity
- **High memory usage**: System implements automatic cleanup every 20 frames
- **Slow performance**: Reduce detection_fps or check GPU utilization
- **GStreamer errors**: Falls back to OpenCV, then FFmpeg automatically
- **Import errors with PyGObject/GStreamer**: Install system packages (see System Dependencies above)
- **GUI not working**: Ensure GTK/Qt6 development packages are installed
- **SRT streaming not available**: Install `gstreamer1.0-plugins-bad` and SRT libraries
- **OpenCV errors**: Install `python3-opencv` and `libopencv-dev`

### Performance Tuning
- Adjust `detection_fps` (default 6.0) based on hardware
- Configure `roi_box` to focus processing on relevant areas
- Monitor memory usage in Grafana dashboard

## 📈 Performance Optimizations

- **Frame skipping**: Processes every 5th frame from GStreamer (~30 FPS → 6 FPS)
- **Motion detection**: Skips YOLO inference when no significant movement
- **Queue management**: Maintains optimal frame buffer size
- **Batch operations**: Efficient InfluxDB metric writes
- **Memory monitoring**: Automatic cleanup and leak detection
