---
title: "Troubleshooting"
description: "Common issues, causes, and solutions for Flytrap deployment and operation"
permalink: /troubleshooting.html
nav_order: 8
---

# Troubleshooting Guide

This guide covers common issues, their causes, and solutions for Flytrap deployment and operation.

## Quick Diagnosis

### System Health Check

Run this comprehensive diagnostic script:

```bash
#!/bin/bash
echo "=== Flytrap System Health Check ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo ""

echo "=== Python Environment ==="
python3 --version
which python3
echo ""

echo "=== System Resources ==="
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Disk: $(df -h . | tail -1 | awk '{print $4}') available"
echo ""

echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "NVIDIA GPU not detected or nvidia-smi not available"
fi
echo ""

echo "=== Network Connectivity ==="
echo "Default gateway: $(ip route | grep default | awk '{print $3}')"
echo "DNS: $(grep nameserver /etc/resolv.conf | head -1 | awk '{print $2}')"
echo ""

echo "=== Required Packages ==="
check_package() {
    if dpkg -l | grep -q "^ii  $1"; then
        echo "✓ $1 installed"
    else
        echo "✗ $1 missing"
    fi
}

check_package "gstreamer1.0-tools"
check_package "gstreamer1.0-plugins-base"
check_package "gstreamer1.0-plugins-good"
check_package "gstreamer1.0-plugins-bad"
check_package "libsrt-openssl-dev"
check_package "python3-opencv"
echo ""

echo "=== Python Dependencies ==="
python3 -c "
try:
    import cv2
    print('✓ OpenCV', cv2.__version__)
except ImportError as e:
    print('✗ OpenCV missing:', e)

try:
    import torch
    print('✓ PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())
except ImportError as e:
    print('✗ PyTorch missing:', e)

try:
    from ultralytics import YOLO
    print('✓ Ultralytics YOLO available')
except ImportError as e:
    print('✗ Ultralytics missing:', e)
"
```

## Common Issues

### Video Stream Issues

#### SRT Stream Connection Failed

**Symptoms:**
```
ERROR: SRT stream connection failed
ERROR: gst-stream-error-quark: Connection timed out (1)
```

**Causes:**
- Network connectivity issues
- Invalid SRT URI format
- Firewall blocking SRT port (default: 4201)
- SRT server not running

**Solutions:**

1. **Check Network Connectivity:**
```bash
# Test basic connectivity
ping <srt-server-ip>

# Test SRT port
telnet <srt-server-ip> 4201

# Check firewall rules
sudo ufw status
sudo iptables -L
```

2. **Verify SRT URI Format:**
```bash
# Correct formats:
srt://192.168.1.100:4201
srt://192.168.1.100:4201?mode=caller
srt://192.168.1.100:4201?latency=100

# Test with gst-launch
gst-launch-1.0 srtsrc uri="srt://192.168.1.100:4201" ! fakesink
```

3. **Check SRT Server:**
```bash
# If running on same machine, check SRT server status
ps aux | grep srt
netstat -tlnp | grep 4201
```

#### GStreamer SRT Plugin Missing

**Symptoms:**
```
ERROR: GStreamer SRT plugin not found
WARNING: no SRT plugin found, falling back to OpenCV
```

**Causes:**
- Missing `gstreamer1.0-plugins-bad` package
- SRT libraries not installed
- Plugin not registered

**Solutions:**

```bash
# Install SRT plugins
sudo apt update
sudo apt install -y gstreamer1.0-plugins-bad libsrt-openssl-dev

# Verify plugin installation
gst-inspect-1.0 srtsrc

# Rebuild plugin cache
gst-plugin-scanner
```

#### Video Stream Freezes or Drops Frames

**Symptoms:**
- Video playback stops
- Low FPS or frame drops
- Buffer overflow warnings

**Causes:**
- Network congestion
- Insufficient bandwidth
- High latency
- Buffer size issues

**Solutions:**

1. **Check Network Performance:**
```bash
# Test bandwidth
iperf3 -c <server-ip>

# Check latency
ping -c 10 <server-ip>

# Monitor network usage
iftop -i <interface>
```

2. **Adjust Buffer Settings:**
```python
# Increase buffer size in configuration
detector = ObjectDetector(
    srt_uri="srt://192.168.1.100:4201?latency=200",
    max_queue_size=200  # Increase buffer
)
```

3. **Optimize SRT Parameters:**
```bash
# Use URI parameters for better performance
srt://192.168.1.100:4201?latency=100&rcvbuf=1000000&sndbuf=1000000
```

### Detection Issues

#### No Objects Detected

**Symptoms:**
- Console shows "0 detections"
- Log file empty of detection entries
- GUI shows no bounding boxes

**Causes:**
- Confidence threshold too high
- Model not loaded correctly
- Video quality issues
- ROI configuration problems

**Solutions:**

1. **Lower Confidence Threshold:**
```bash
# Test with lower confidence
uv run python main.py srt://192.168.1.100:4201 --confidence 0.1
```

2. **Verify Model Loading:**
```bash
# Test model loading
uv run python -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
print('Model loaded successfully')
print('Model info:', model.info())
"
```

3. **Check Video Quality:**
```bash
# Test video stream quality
gst-launch-1.0 srtsrc uri="srt://192.168.1.100:4201" ! videoconvert ! autovideosink
```

4. **Adjust ROI:**
```bash
# Disable ROI to test full frame
uv run python main.py srt://192.168.1.100:4201 --roi ""
```

#### False Positives/Negatives

**Symptoms:**
- Detecting wrong objects
- Missing valid detections
- Inconsistent results

**Causes:**
- Inappropriate confidence threshold
- Wrong model for use case
- Lighting conditions
- Camera angle issues

**Solutions:**

1. **Tune Confidence Threshold:**
```python
# Experiment with different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for conf in thresholds:
    detector = ObjectDetector(srt_uri="...", confidence=conf)
    # Test and evaluate results
```

2. **Use Appropriate Model:**
```bash
# For high accuracy (slower)
uv run python main.py srt://... --model yolo11l.pt --confidence 0.3

# For speed (lower accuracy)
uv run python main.py srt://... --model yolo11n.pt --confidence 0.5
```

3. **Adjust Processing Parameters:**
```python
detector = ObjectDetector(
    srt_uri="...",
    detection_fps=3.0,  # Slower processing for better accuracy
    roi_box=(0, 200, 1920, 1030)  # Focus on relevant area
)
```

#### Memory Issues

**Symptoms:**
- Out of memory errors
- System slowdown
- Application crashes
- GPU memory errors

**Causes:**
- Large model with insufficient RAM/GPU memory
- Memory leaks
- Large frame buffers
- High resolution video

**Solutions:**

1. **Monitor Memory Usage:**
```bash
# Check system memory
free -h

# Check GPU memory
nvidia-smi

# Monitor process memory
top -p $(pgrep -f flytrap)
```

2. **Reduce Memory Usage:**
```python
# Use smaller model
detector = ObjectDetector(
    srt_uri="...",
    model_path='yolo11n.pt',  # Smaller model
    max_queue_size=50,        # Smaller buffer
    detection_fps=3.0         # Lower FPS
)
```

3. **Enable Memory Cleanup:**
```python
# Aggressive cleanup
detector = ObjectDetector(
    srt_uri="...",
    memory_cleanup_interval=10  # More frequent cleanup
)
```

4. **GPU Memory Optimization:**
```bash
# Limit GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use CPU if GPU memory insufficient
export CUDA_VISIBLE_DEVICES=""
```

### Performance Issues

#### Low FPS

**Symptoms:**
- FPS below target
- Frame drops
- Processing backlog

**Causes:**
- Insufficient hardware
- Large model
- High resolution video
- Inefficient settings

**Solutions:**

1. **Profile Performance:**
```python
# Check hardware utilization
nvidia-smi -l 1  # GPU monitoring
top -d 1         # CPU monitoring
```

2. **Optimize Settings:**
```python
# Performance tuning
detector = ObjectDetector(
    srt_uri="...",
    model_path='yolo11n.pt',  # Fastest model
    detection_fps=6.0,        # Reasonable FPS
    roi_box=(0, 200, 1920, 1030),  # Limit processing area
    confidence=0.4            # Balance accuracy/speed
)
```

3. **Hardware Acceleration:**
```bash
# Ensure CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Use MPS on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### High CPU/GPU Usage

**Symptoms:**
- System overheating
- High power consumption
- Other applications slowed

**Causes:**
- Continuous processing
- Large batch sizes
- Inefficient algorithms

**Solutions:**

1. **Reduce Processing Load:**
```python
detector = ObjectDetector(
    srt_uri="...",
    detection_fps=2.0,        # Lower FPS
    max_queue_size=30,        # Smaller buffer
    confidence=0.6            # Higher threshold
)
```

2. **Enable Motion Detection:**
```python
# Motion detection reduces processing by 60-80%
detector = ObjectDetector(
    srt_uri="...",
    enable_motion_detection=True  # Default enabled
)
```

### Database Issues

#### InfluxDB Connection Failed

**Symptoms:**
```
ERROR: Failed to connect to InfluxDB
WARNING: InfluxDB unavailable, using file logging only
```

**Causes:**
- InfluxDB not running
- Wrong connection parameters
- Network issues
- Authentication problems

**Solutions:**

1. **Check InfluxDB Status:**
```bash
# Check if running
docker-compose ps influxdb

# Check logs
docker-compose logs influxdb

# Test connection
curl http://localhost:8086/health
```

2. **Verify Configuration:**
```bash
# Check environment variables
cat .env | grep INFLUX

# Test connection manually
uv run python -c "
from flytrap.influx_client import InfluxClient
client = InfluxClient()
client.test_connection()
"
```

3. **Fix Connection Parameters:**
```bash
# Update .env file
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token-here
INFLUXDB_ORG=flytrap
INFLUXDB_BUCKET=detections
```

#### Grafana Dashboard Issues

**Symptoms:**
- Dashboards not loading
- No data displayed
- Authentication errors

**Causes:**
- Grafana not running
- Data source misconfiguration
- Permission issues

**Solutions:**

1. **Check Grafana Status:**
```bash
# Check if running
docker-compose ps grafana

# Check logs
docker-compose logs grafana

# Access dashboard
curl http://localhost:3000/api/health
```

2. **Verify Data Source:**
```bash
# Check data source configuration
curl -u admin:admin http://localhost:3000/api/datasources

# Test query
curl -X POST http://localhost:3000/api/ds/query \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{"queries":[{"refId":"A","datasource":{"type":"influxdb","uid":"influxdb"},"rawQuery":true,"query":"from(bucket: \"detections\") |> range(start: -1h)"}]}'
```

### GUI Issues

#### GUI Not Displaying

**Symptoms:**
- No GUI window appears
- Application runs in headless mode unexpectedly

**Causes:**
- Display not available (SSH, WSL)
- GTK libraries missing
- X11 forwarding issues

**Solutions:**

1. **Check Display Availability:**
```bash
# Check display variable
echo $DISPLAY

# Test X11
xeyes  # Should open a window

# Check WSL configuration
cat /proc/version
```

2. **Install GUI Libraries:**
```bash
# Ubuntu/Debian
sudo apt install -y python3-gi gir1.2-gtk-3.0 x11-apps

# Test GTK
python3 -c "import gi; gi.require_version('Gtk', '3.0'); from gi.repository import Gtk; print('GTK OK')"
```

3. **Force GUI Mode:**
```bash
# Override auto-detection
uv run python main.py srt://... --no-headless
```

4. **SSH X11 Forwarding:**
```bash
# Connect with X11 forwarding
ssh -X user@server

# Or use VNC
sudo apt install tightvncserver
vncserver :1
```

### Installation Issues

#### Import Errors

**Symptoms:**
```
ImportError: No module named 'flytrap'
ModuleNotFoundError: No module named 'cv2'
```

**Causes:**
- Dependencies not installed
- Virtual environment issues
- Python path problems

**Solutions:**

1. **Install Dependencies:**
```bash
# Ensure uv is used
uv sync

# Check Python path
uv run python -c "import sys; print(sys.path)"
```

2. **Virtual Environment Issues:**
```bash
# Activate environment
source .venv/bin/activate  # or uv shell

# Check packages
uv pip list | grep -E "(opencv|torch|ultralytics)"
```

3. **Reinstall Problematic Packages:**
```bash
# Reinstall OpenCV
uv pip uninstall opencv-python
uv pip install opencv-python

# Reinstall PyTorch
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Advanced Troubleshooting

### Debug Logging

Enable detailed logging for diagnosis:

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('flytrap_debug.log'),
        logging.StreamHandler()
    ]
)

# Run with debug output
uv run python main.py srt://192.168.1.100:4201
```

### Performance Profiling

Profile application performance:

```python
import cProfile
import pstats

# Profile main execution
cProfile.run('main()', 'profile_output.prof')

# Analyze results
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
```

### Memory Leak Detection

Monitor for memory leaks:

```python
import tracemalloc
import gc

# Start tracing
tracemalloc.start()

# Run application for a while
# ... your code ...

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")

# Get top memory consumers
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

### Network Debugging

Debug network issues:

```bash
# Capture SRT traffic
sudo tcpdump -i any port 4201 -w srt_traffic.pcap

# Analyze with Wireshark
wireshark srt_traffic.pcap

# Check SRT statistics
gst-launch-1.0 srtsrc uri="srt://192.168.1.100:4201" ! srtjitterbuffer ! fakesink
```

## Getting Help

### Information to Provide

When seeking help, include:

1. **System Information:**
   - OS and version
   - Python version
   - Hardware specs (CPU, GPU, RAM)

2. **Error Messages:**
   - Full error output
   - Stack traces
   - Log files

3. **Configuration:**
   - Command used to run Flytrap
   - `.env` file (redact secrets)
   - Hardware configuration

4. **Steps to Reproduce:**
   - Minimal example that reproduces the issue
   - Expected vs actual behavior

### Support Channels

- **GitHub Issues**: https://github.com/five59/flytrap/issues
- **Discussions**: https://github.com/five59/flytrap/discussions
- **Documentation**: Check this troubleshooting guide first

### Emergency Procedures

For production issues:

1. **Stop the Application:**
```bash
pkill -f flytrap
```

2. **Check System Resources:**
```bash
# Monitor system health
top
df -h
free -h
```

3. **Restart Services:**
```bash
# Restart InfluxDB/Grafana if needed
docker-compose restart

# Restart Flytrap with conservative settings
uv run python main.py srt://... --model yolo11n.pt --fps 2.0
```

4. **Collect Diagnostics:**
```bash
# Gather system information
uname -a > system_info.txt
python --version >> system_info.txt
nvidia-smi >> system_info.txt

# Archive logs
tar czf flytrap_logs.tar.gz *.log
```

This comprehensive troubleshooting guide should resolve most common issues with Flytrap. For persistent problems, please provide detailed information when seeking help.