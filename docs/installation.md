---
layout: default
title: "Installation Guide"
description: "Complete setup instructions for Flytrap including system dependencies and configuration"
nav_order: 2
---

# Installation Guide

This guide covers the complete installation and setup process for Flytrap, including system dependencies, Python packages, and optional monitoring components.

## Prerequisites

Before installing Flytrap, ensure your system meets these requirements:

- **Python 3.12+**
- **SRT video stream source** (for production use)
- **GPU recommended** (CUDA/MPS/CPU fallback supported)

## System Dependencies (Ubuntu/Debian)

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

### Installation Notes

**Note**: Some packages may require additional repositories. For Ubuntu 22.04+, the `gstreamer1.0-plugins-bad` package includes SRT support. If SRT plugins are not available in your distribution's repositories, you may need to build GStreamer from source or use a PPA.

### Alternative Package Managers

#### Using conda/mamba:
```bash
# Create environment
conda create -n flytrap python=3.12
conda activate flytrap

# Install system dependencies via conda
conda install -c conda-forge gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav libsrt
```

#### Using Homebrew (macOS):
```bash
# Install GStreamer and SRT
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav srt

# Install additional dependencies
brew install opencv gtk+3 pygobject3
```

## Verify Installation

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
uv run python -c "from flytrap import ObjectDetector; print('Flytrap import successful')"
```

## Python Dependencies

### 1. Install uv (if not already installed)

uv is a fast Python package installer and resolver:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 2. Install Python Dependencies

```bash
# Install Python dependencies
uv sync

# Install dev dependencies (for development)
uv sync --group dev
```

### 3. Install PyTorch (Platform-Specific)

PyTorch installation varies by platform. Choose the appropriate command:

```bash
# NVIDIA CUDA 12.1+
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# NVIDIA CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (M1/M2/M3/M4)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CPU-only (fallback)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Note**: Check your CUDA version with `nvidia-smi` and PyTorch compatibility at https://pytorch.org/get-started/locally/

## Optional: InfluxDB & Grafana Setup

For metrics collection and visualization, set up InfluxDB and Grafana:

### 1. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env
```

### 2. Start Services

```bash
# Start InfluxDB and Grafana with docker-compose
docker-compose up -d

# Verify InfluxDB connection
uv run python -m flytrap.influx_client
```

### 3. Access Dashboards

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **InfluxDB UI**: http://localhost:8086 (admin/flytrap-admin-password)

## Testing Installation

Run a comprehensive test to ensure everything is working:

```bash
# Run the test suite
uv run python -m pytest tests/

# Test with coverage
uv run python -m pytest --cov=flytrap --cov-report=term tests/

# Test basic import
uv run python -c "from flytrap import ObjectDetector; print('✓ Import successful')"

# Test InfluxDB connection (if configured)
uv run python -m flytrap.influx_client
```

## Troubleshooting Installation

### Common Issues

**GStreamer SRT not available:**
- Ensure `gstreamer1.0-plugins-bad` is installed
- Check if SRT libraries are available: `apt list --installed | grep srt`

**PyTorch CUDA issues:**
- Verify CUDA installation: `nvcc --version`
- Check GPU compatibility with PyTorch version
- Try CPU-only installation as fallback

**OpenCV import errors:**
- Install system packages: `sudo apt install python3-opencv libopencv-dev`
- Rebuild OpenCV: `pip uninstall opencv-python && pip install opencv-python`

**GTK/GUI issues:**
- Install GTK packages: `sudo apt install python3-gi gir1.2-gtk-3.0`
- For headless environments, Flytrap auto-detects and runs in headless mode

### Environment-Specific Setup

**WSL/Windows Subsystem for Linux:**
```bash
# Install WSL-specific packages
sudo apt install -y x11-apps mesa-utils

# Set DISPLAY variable
export DISPLAY=:0
```

**Docker Container:**
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libsrt-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync
```

## Next Steps

Once installation is complete:

1. **[Configure Flytrap](configuration.md)** - Set up environment variables and parameters
2. **[Run the Application](usage.md)** - Start processing video streams
3. **[Monitor Performance](monitoring.md)** - Set up dashboards and alerts

For development setup, see the [Development Guide](development.md).