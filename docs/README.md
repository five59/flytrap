# Flytrap Documentation

Welcome to the comprehensive documentation for Flytrap - a real-time object detection and tracking system using YOLO11 with SRT video streams.

## 🚀 Quick Start

Get up and running quickly:

1. **[Installation Guide](installation.md)** - Prerequisites, system dependencies, and setup
2. **[Usage Guide](usage.md)** - Running the application and understanding output
3. **[Configuration](configuration.md)** - Environment variables and parameters

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

## 📖 Contributing

We welcome contributions! Please see the [Development Guide](development.md) for information on:

- Setting up a development environment
- Code style and standards
- Testing procedures
- Submitting pull requests

## 📄 License

Flytrap is open source software released under the MIT License.

---

*For the latest information, visit the [main repository](https://github.com/five59/flytrap).*"