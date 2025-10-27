# Eyeball - Real-time Object Detection with NDI

Real-time object detection and tracking using YOLO11 with NDI video streams. Tracks vehicles, people, and bicycles with direction detection, speed calculation, and automatic screenshot capture.

## Features

- Real-time object detection using YOLO11 medium model
- NDI video stream input support
- Direction detection (left-to-right / right-to-left)
- Speed calculation in mph
- Automatic screenshot capture for right-to-left movement
- Midpoint crossing detection for accurate classification
- Hardware acceleration support (CUDA, MPS, or CPU)

## Setup

### 1. Install Dependencies

```bash
# Install base dependencies
uv sync
```

### 2. Install PyTorch (Platform-Specific)

#### For Mac M4 (Apple Silicon with MPS):
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### For Windows/Linux with NVIDIA CUDA 12.1:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### For Windows/Linux with NVIDIA CUDA 11.8:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Configuration

The script will automatically detect and use the best available hardware:
- **CUDA**: Nvidia GPU acceleration
- **MPS**: Apple Silicon acceleration
- **CPU**: Fallback if no GPU available

## Usage

```bash
uv run python ndi-to-yolo.py
```

The script will:
1. Search for NDI sources on the network
2. Connect to the first available source
3. Begin real-time detection and tracking
4. Log detected objects to `vehicle_tracking.log`
5. Save screenshots to `screenshots/` folder for right-to-left movement

Press `q` to quit.

## Configuration

Edit these constants in [ndi-to-yolo.py](ndi-to-yolo.py) to customize:

- `ROAD_WIDTH_FEET` (line 22): Approximate width of road in frame for speed calculation (default: 27.5 ft)
- `VEHICLE_CLASSES` (line 34): Object classes to track (default: person, bicycle, car, motorcycle, bus, truck)
- `LOG_FILE` (line 26): Path to log file
- `SCREENSHOTS_DIR` (line 27): Path to screenshots directory

## Output

### Log File Format
```
2025-10-27 14:23:45.123 | Track ID: 1 | Type: car | Direction: left-to-right | Speed: 28.5 mph
2025-10-27 14:23:52.456 | Track ID: 3 | Type: truck | Direction: right-to-left | Speed: 22.3 mph | Screenshot: screenshots/track_3_20251027_142352_456.jpg
```

### Screenshots
Screenshots are automatically saved for objects moving right-to-left in the `screenshots/` directory with annotated bounding boxes and labels.

## Model

The script uses **YOLO11m** (medium) model which provides a good balance between speed and accuracy. The model will be automatically downloaded on first run.

## Requirements

- Python 3.12+
- NDI-compatible video source on the network
- GPU recommended for real-time performance (CUDA or MPS)
