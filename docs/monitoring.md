---
layout: default
title: "Monitoring & Visualization"
description: "InfluxDB and Grafana setup for comprehensive metrics collection and visualization"
nav_order: 6
---

# Monitoring & Visualization

Flytrap includes comprehensive monitoring capabilities with InfluxDB time-series storage and Grafana dashboards for real-time visualization and alerting. This guide covers setup, configuration, and available metrics.

## Quick Start

### 1. Start Monitoring Stack

```bash
# Copy environment template
cp .env.example .env

# Start InfluxDB and Grafana
docker-compose up -d

# Verify InfluxDB connection
uv run python -m flytrap.influx_client
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **InfluxDB UI**: http://localhost:8086 (admin/flytrap-admin-password)

### 3. Run Flytrap with Monitoring

```bash
# Flytrap will automatically send metrics to InfluxDB
uv run python main.py srt://192.168.1.100:4201
```

## Architecture

```
Flytrap → InfluxDB → Grafana
    ↓         ↓         ↓
Metrics → Storage → Visualization
    ↑         ↑         ↑
Queries ← Dashboards ← Alerts
```

## InfluxDB Setup

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2
      - ./influxdb/config:/etc/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=flytrap-admin-password
      - DOCKER_INFLUXDB_INIT_ORG=flytrap
      - DOCKER_INFLUXDB_INIT_BUCKET=detections
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=flytrap-super-secret-token-change-in-production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8086/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - influxdb
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  influxdb_data:
  grafana_data:
```

### Manual Installation

```bash
# Install InfluxDB
wget -qO- https://repos.influxdata.com/influxdata-archive_compat.key | sudo apt-key add -
echo "deb https://repos.influxdata.com/debian stable main" | sudo tee /etc/apt/sources.list.d/influxdb.list
sudo apt update && sudo apt install influxdb2

# Start InfluxDB
sudo systemctl start influxdb
sudo systemctl enable influxdb

# Install Grafana
sudo apt install -y apt-transport-https
sudo apt install -y software-properties-common wget
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt update && sudo apt install grafana

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

## Grafana Configuration

### Data Source Setup

```yaml
# grafana/provisioning/datasources/influxdb.yml
apiVersion: 1

datasources:
  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    jsonData:
      version: Flux
      organization: flytrap
      defaultBucket: detections
      tlsSkipVerify: true
    secureJsonData:
      token: flytrap-super-secret-token-change-in-production
```

### Dashboard Provisioning

```yaml
# grafana/provisioning/dashboards/dashboard.yml
apiVersion: 1

providers:
  - name: 'flytrap'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

## Metrics Overview

### Frame Metrics

| Metric | Description | Type | Tags |
|--------|-------------|------|------|
| `frames_processed` | Total frames processed | Counter | `camera_id`, `model` |
| `processing_time_ms` | Time to process each frame | Histogram | `camera_id`, `stage` |
| `queue_depth` | Current frame queue size | Gauge | `camera_id` |
| `fps_actual` | Actual processing FPS | Gauge | `camera_id` |

### Detection Metrics

| Metric | Description | Type | Tags |
|--------|-------------|------|------|
| `detections_total` | Total objects detected | Counter | `camera_id`, `class` |
| `detection_confidence` | Detection confidence scores | Histogram | `camera_id`, `class` |
| `objects_tracked` | Currently tracked objects | Gauge | `camera_id`, `class` |
| `track_duration` | How long objects are tracked | Histogram | `camera_id`, `class` |

### Movement Analytics

| Metric | Description | Type | Tags |
|--------|-------------|------|------|
| `direction_changes` | Direction changes detected | Counter | `camera_id`, `direction` |
| `speed_measurements` | Speed calculations | Histogram | `camera_id`, `direction` |
| `screenshots_captured` | Automatic screenshots taken | Counter | `camera_id`, `reason` |

### System Metrics

| Metric | Description | Type | Tags |
|--------|-------------|------|------|
| `memory_usage_mb` | Memory usage | Gauge | `component` |
| `gpu_memory_mb` | GPU memory usage | Gauge | `device` |
| `cpu_usage_percent` | CPU utilization | Gauge | `core` |
| `errors_total` | Error count by type | Counter | `component`, `error_type` |

## Sample Queries

### Flux Queries for InfluxDB

```flux
// Recent detections by class
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "detections")
  |> filter(fn: (r) => r._field == "count")
  |> group(columns: ["class"])
  |> aggregateWindow(every: 5m, fn: sum)

// Processing performance
from(bucket: "detections")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "processing")
  |> filter(fn: (r) => r._field == "fps")
  |> aggregateWindow(every: 1m, fn: mean)

// Memory usage trends
from(bucket: "detections")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "system")
  |> filter(fn: (r) => r._field == "memory_mb")
  |> aggregateWindow(every: 10m, fn: mean)
```

### Grafana Dashboard Panels

#### Real-time FPS Monitor
```json
{
  "title": "Processing FPS",
  "type": "timeseries",
  "targets": [
    {
      "query": "from(bucket: \"detections\") |> range(start: -5m) |> filter(fn: (r) => r._measurement == \"processing\") |> filter(fn: (r) => r._field == \"fps\")",
      "refId": "A"
    }
  ]
}
```

#### Detection Count by Class
```json
{
  "title": "Detections by Class",
  "type": "barchart",
  "targets": [
    {
      "query": "from(bucket: \"detections\") |> range(start: -1h) |> filter(fn: (r) => r._measurement == \"detections\") |> filter(fn: (r) => r._field == \"count\") |> group(columns: [\"class\"]) |> sum()",
      "refId": "A"
    }
  ]
}
```

## Dashboard Examples

### Main Dashboard

The pre-configured dashboard includes:

1. **System Overview**
   - Current FPS and processing status
   - Memory usage (RAM/GPU)
   - Queue depth and system health

2. **Detection Analytics**
   - Objects detected by class (car, truck, person, etc.)
   - Detection confidence distribution
   - Tracking statistics

3. **Movement Analysis**
   - Direction distribution (left-to-right vs right-to-left)
   - Speed measurements histogram
   - Screenshot capture events

4. **Performance Metrics**
   - Processing time per frame
   - GPU utilization
   - Error rates and recovery events

### Custom Dashboards

#### Multi-Camera Dashboard

```json
{
  "dashboard": {
    "title": "Multi-Camera Monitoring",
    "tags": ["flytrap", "multi-camera"],
    "panels": [
      {
        "title": "Camera Status",
        "type": "stat",
        "targets": [
          {
            "query": "from(bucket: \"detections\") |> range(start: -5m) |> filter(fn: (r) => r._measurement == \"system\") |> filter(fn: (r) => r._field == \"status\") |> group(columns: [\"camera_id\"]) |> last()",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

#### Alert Dashboard

```json
{
  "dashboard": {
    "title": "Flytrap Alerts",
    "tags": ["flytrap", "alerts"],
    "panels": [
      {
        "title": "Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "query": "from(bucket: \"detections\") |> range(start: -1h) |> filter(fn: (r) => r._measurement == \"errors\") |> filter(fn: (r) => r._field == \"count\") |> aggregateWindow(every: 5m, fn: sum)",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

## Alerting

### Grafana Alert Rules

#### High Memory Usage Alert
```yaml
alert: High Memory Usage
expr: memory_usage_mb{component="detector"} > 80
for: 5m
labels:
  severity: warning
annotations:
  summary: "High memory usage detected"
  description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"
```

#### Processing FPS Drop Alert
```yaml
alert: FPS Drop
expr: rate(processing_fps[5m]) < 0.5
for: 2m
labels:
  severity: critical
annotations:
  summary: "Processing FPS dropped significantly"
  description: "FPS dropped to {{ $value }} on {{ $labels.camera_id }}"
```

#### Detection Failure Alert
```yaml
alert: No Detections
expr: sum(rate(detections_total[10m])) == 0
for: 10m
labels:
  severity: warning
annotations:
  summary: "No detections in last 10 minutes"
  description: "Camera {{ $labels.camera_id }} has stopped detecting objects"
```

## Data Retention

### InfluxDB Retention Policies

```bash
# Create retention policy (90 days for detailed metrics)
influx bucket create --name detections --retention 2160h

# Create retention policy (1 year for summary metrics)
influx bucket create --name detections_summary --retention 8760h
```

### Automated Downsampling

```flux
// Downsample hourly data to daily summaries
from(bucket: "detections")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "detections")
  |> aggregateWindow(every: 1h, fn: sum)
  |> to(bucket: "detections_summary")
```

## Troubleshooting Monitoring

### Common Issues

**InfluxDB connection failed:**
```bash
# Check InfluxDB status
docker-compose ps influxdb

# Test connection
curl http://localhost:8086/health

# Check logs
docker-compose logs influxdb
```

**Grafana dashboards not loading:**
```bash
# Check Grafana status
docker-compose ps grafana

# Verify data source
curl -u admin:admin http://localhost:3000/api/datasources

# Check provisioning
docker-compose logs grafana
```

**Metrics not appearing:**
```bash
# Test metric writing
uv run python -c "
from flytrap.influx_client import InfluxClient
client = InfluxClient()
client.write_point('test', {'value': 1.0}, {'test': 'connection'})
print('Test metric written')
"

# Query test metric
curl -X POST http://localhost:8086/api/v2/query \
  -H "Authorization: Token flytrap-super-secret-token-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"query": "from(bucket: \"detections\") |> range(start: -1h) |> filter(fn: (r) => r._measurement == \"test\")"}'
```

**Performance issues:**
```bash
# Check InfluxDB performance
docker stats influxdb

# Monitor query performance
influx query 'SHOW QUERIES'

# Optimize retention policies
influx bucket list
```

## Advanced Configuration

### High Availability Setup

```yaml
# docker-compose HA setup
version: '3.8'
services:
  influxdb-1:
    image: influxdb:2.7
    # InfluxDB clustering configuration
  influxdb-2:
    image: influxdb:2.7
    # Secondary node
  grafana:
    image: grafana/grafana:10.2.0
    # Load balancer configuration
```

### Custom Metrics

```python
# Add custom metrics to Flytrap
from flytrap.influx_client import InfluxClient

class CustomMetrics:
    def __init__(self):
        self.client = InfluxClient()

    def record_custom_metric(self, name, value, tags=None):
        self.client.write_point(
            measurement=name,
            fields={'value': value},
            tags=tags or {}
        )

# Usage in detector
metrics = CustomMetrics()
metrics.record_custom_metric('custom_processing_time', 0.45, {'camera': 'cam1'})
```

This monitoring system provides comprehensive observability for Flytrap deployments, enabling proactive issue detection and performance optimization.