# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eyeball is a Python-based real-time object detection system that integrates NDI (Network Device Interface) video streaming with YOLOv8 computer vision. The project receives live video feeds via NDI protocol and performs real-time object detection using the YOLOv8 model.

## Development Environment

- **Python Version**: 3.12+
- **Package Manager**: uv
- **Hardware Acceleration**: Configured for Apple Silicon (MPS) for ML inference

## Common Commands

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --group dev

# Run the main application (placeholder)
uv run python main.py

# Run the NDI-to-YOLO detection script
uv run python ndi-to-yolo.py

# Start JupyterLab for experimentation
uv run jupyter lab
```

## Architecture

### Core Components

1. **NDI Video Capture** ([ndi-to-yolo.py](ndi-to-yolo.py))
   - Uses `cyndilib` library for NDI source discovery and video reception
   - Implements finder/callback pattern for dynamic NDI source detection
   - Receives video in RGBA format via `RecvColorFormat.RGBX_RGBA`
   - Frame synchronization through `VideoFrameSync`

2. **YOLOv8 Integration**
   - Model: YOLOv8 nano (`yolov8n.pt`)
   - Device: MPS (Metal Performance Shaders) for Mac M4 acceleration
   - Confidence threshold: 0.4
   - Real-time inference on video frames

3. **Video Processing Pipeline**
   - NDI receiver → RGBA frame data → NumPy array → BGR conversion → YOLO inference → Annotated display
   - Frame format conversions: RGBA (NDI) → BGR (OpenCV/YOLO)
   - OpenCV used for visualization and window management

### Key Dependencies

- **cyndilib**: NDI protocol implementation for receiving video streams
- **ultralytics**: YOLOv8 model and inference engine
- **opencv-python**: Video frame manipulation and display
- **numpy**: Array operations for frame data conversion

### Known Issues

The Jupyter notebook ([Untitled.ipynb](Untitled.ipynb)) shows an `AttributeError` when accessing `video_frame.data`. The `VideoFrameSync` object may require a different API for accessing frame buffer data. Check cyndilib documentation for proper frame data access methods.
