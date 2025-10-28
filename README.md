# Flytrap - Real-time Object Detection with SRT Streams

Real-time object detection and tracking using YOLO11 with SRT (Secure Reliable Transport) video streams. Tracks vehicles, people, and bicycles with direction detection, speed calculation, and automatic screenshot capture. Includes comprehensive time-series metrics storage with InfluxDB and Grafana visualization.

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=alert_status&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=reliability_rating&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=security_rating&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=sqale_rating&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=vulnerabilities&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=five59_flytrap&metric=code_smells&token=483a73bd75336b574d29619019467791e30a8a18)](https://sonarcloud.io/summary/new_code?id=five59_flytrap)

![Screenshot](screenshot.jpg)

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

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run with SRT stream
uv run python main.py srt://192.168.1.100:4201
```
## 📚 Documentation

Complete documentation is available in the [`docs/`](./docs/) directory:

- **[Installation Guide](docs/installation.md)** - Complete setup instructions
- **[Usage Guide](docs/usage.md)** - How to run Flytrap and interpret results
- **[Configuration](docs/configuration.md)** - All configuration options
- **[Architecture](docs/architecture.md)** - System design and components
- **[Monitoring](docs/monitoring.md)** - Grafana dashboards and InfluxDB metrics
- **[Performance](docs/performance.md)** - Optimizations and tuning guides
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Development](docs/development.md)** - Project structure, testing, and contributing

## 🎯 System Overview

The system uses a sophisticated pipeline:

1. **Stream Reception**: SRT stream → GStreamer/OpenCV/FFmpeg fallback
2. **Frame Processing**: Motion detection → YOLO inference → Tracking
3. **Analytics**: Direction/speed calculation → Screenshot capture
4. **Storage**: File logging + InfluxDB metrics + Grafana visualization

## 📊 Monitoring & Visualization

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **InfluxDB UI**: http://localhost:8086 (admin/flytrap-admin-password)

## 🏗️ Development

```bash
# Run tests
uv run python -m pytest tests/

# Run with coverage
uv run python -m pytest --cov=flytrap --cov-report=term tests/
```

---

*Flytrap is open source software released under the MIT License.*
