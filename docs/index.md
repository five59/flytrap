---
layout: default
title: "Flytrap Documentation"
description: "Real-time object detection and tracking using YOLO11 with SRT streams"
permalink: /
---

# Flytrap Documentation

Real-time object detection and tracking using YOLO11 with SRT (Secure Reliable Transport) video streams. Tracks vehicles, people, and bicycles with direction detection, speed calculation, and automatic screenshot capture. Includes comprehensive time-series metrics storage with InfluxDB and Grafana visualization.

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=alert_status&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=reliability_rating&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=security_rating&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=sqale_rating&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=vulnerabilities&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=code_smells&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)

![Screenshot]({{ '/assets/images/screenshot.jpg' | relative_url }})

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Documentation Sections](#documentation-sections)
- [System Overview](#system-overview)
- [Monitoring & Visualization](#monitoring--visualization)
- [Development](#development)

## 🚀 Quick Start

Get up and running quickly with Flytrap:

1. **[Installation Guide](installation.md)** - Prerequisites, system dependencies, and setup
2. **[Usage Guide](usage.md)** - Running the application and understanding output
3. **[Configuration](configuration.md)** - Environment variables and parameters

## ✨ Key Features

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

## 📚 Documentation Sections

### Getting Started
- **[Installation](installation.md)** - Complete setup instructions
- **[Usage](usage.md)** - How to run Flytrap and interpret results
- **[Configuration](configuration.md)** - All configuration options

### Architecture & Development
- **[Architecture](architecture.md)** - System design and core components
- **[Computer Vision Model Selection](cv-model-selection.md)** - Why YOLO11m was chosen
- **[Development](development.md)** - Project structure, testing, and contributing

### Operations & Monitoring
- **[Monitoring](monitoring.md)** - Grafana dashboards and InfluxDB metrics
- **[Performance](performance.md)** - Optimizations and tuning guides
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🎯 System Architecture

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
- **InfluxDB UI**: http://localhost:8086 (admin/flytrap-admin-password)

### Metrics Collected
- **Frame metrics**: Detection count, processing time, memory usage, queue depth
- **Object detections**: Class, confidence, bounding boxes per frame
- **Movement tracking**: Direction, speed (mph), track IDs over time

## 🏗️ Development

### Project Structure
```
flytrap/
├── flytrap/                    # Main package
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

### Testing
```bash
# Run tests
uv run python -m pytest tests/

# Run tests with coverage
uv run python -m pytest --cov=flytrap --cov-report=xml --cov-report=term tests/
```

## 📈 Performance Optimizations

- **Frame skipping**: Processes every 5th frame from GStreamer (~30 FPS → 6 FPS)
- **Motion detection**: Skips YOLO inference when no significant movement
- **Queue management**: Maintains optimal frame buffer size
- **Batch operations**: Efficient InfluxDB metric writes
- **Memory monitoring**: Automatic cleanup and leak detection

## 🔧 Support

For issues, questions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/five59/flytrap/issues)
- **Discussions**: [GitHub Discussions](https://github.com/five59/flytrap/discussions)
- **Documentation**: This docs folder contains comprehensive guides

---

*Flytrap is open source software released under the MIT License.*