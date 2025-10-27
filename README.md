# Eyeball - Real-time Object Detection with NDI

Real-time object detection and tracking using YOLO11 with NDI video streams. Tracks vehicles, people, and bicycles with direction detection, speed calculation, and automatic screenshot capture. Includes time-series metrics storage with InfluxDB.

## Features

- **Real-time object detection** using YOLO11 medium model
- **NDI video stream input** support
- **Direction detection** (left-to-right / right-to-left)
- **Speed calculation** in mph
- **Automatic screenshot capture** for right-to-left movement
- **Midpoint crossing detection** for accurate classification
- **Hardware acceleration** support (CUDA, MPS, or CPU)
- **Time-series metrics** storage with InfluxDB
- **Headless mode** for WSL/SSH environments (auto-detected)

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. (Optional) Start InfluxDB

```bash
# Copy environment template
cp .env.example .env

# Start InfluxDB container
docker-compose up -d
```

### 3. Run the Application

```bash
uv run python main.py
```

The application will:
1. Auto-detect display capability (GUI vs headless mode)
2. Search for NDI sources on the network
3. Connect to the first available source
4. Begin real-time detection and tracking
5. Log detections to `vehicle_tracking.log`
6. Save screenshots to `screenshots/` folder
7. (If configured) Store metrics in InfluxDB

**Controls:**
- Press `q` to quit (GUI mode)
- Press `Ctrl+C` to stop (headless mode)

## Headless Mode (WSL/SSH)

The application automatically detects when running without a display (e.g., WSL, SSH) and switches to headless mode. All detections are still logged to file and InfluxDB, but no visual window is shown.

To force headless mode:
```python
from eyeball import ObjectDetector
detector = ObjectDetector(headless=True)
detector.run()
```

## Output

### Log File Format
```
2025-10-27 14:23:45.123 | Track ID: 1 | Type: car | Direction: left-to-right | Speed: 28.5 mph
2025-10-27 14:23:52.456 | Track ID: 3 | Type: truck | Direction: right-to-left | Speed: 22.3 mph | Screenshot: screenshots/track_3_20251027_142352_456.jpg
```

### Screenshots
Screenshots are automatically saved for objects moving right-to-left in the `screenshots/` directory with annotated bounding boxes and labels.

## Configuration

The `ObjectDetector` class accepts the following parameters:

```python
from eyeball import ObjectDetector

detector = ObjectDetector(
    model_path='yolo11m.pt',          # YOLO model weights
    confidence=0.4,                    # Detection threshold
    road_width_feet=32,                # Road width for speed calc
    log_file='vehicle_tracking.log',  # Log file path
    screenshots_dir='screenshots',     # Screenshot directory
    enable_influx=True,                # Enable InfluxDB logging
    headless=False                     # Force headless mode
)
```

### InfluxDB Configuration

Create a `.env` file:
```bash
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=eyeball-super-secret-token-change-in-production
INFLUXDB_ORG=eyeball
INFLUXDB_BUCKET=detections
```

**Access InfluxDB UI:**
- URL: http://localhost:8086
- Username: `admin`
- Password: `eyeball-admin-password`

## Model

The application uses **YOLO11m** (medium) model which provides a good balance between speed and accuracy. The model will be automatically downloaded on first run (~40MB).

## Project Structure

```
eyeball/
├── eyeball/              # Main package
│   ├── __init__.py      # Package exports
│   ├── detector.py      # ObjectDetector class
│   └── influx_client.py # DetectionLogger for InfluxDB
├── main.py              # Application entry point
├── docker-compose.yml   # InfluxDB service
├── .env.example         # Environment variables template
└── pyproject.toml       # Project dependencies
```

## Requirements

- Python 3.12+
- NDI-compatible video source on the network
- GPU recommended for real-time performance (CUDA or MPS)
- Docker (optional, for InfluxDB)
