#!/usr/bin/env python3
"""
Create Flytrap dashboard in Grafana via API
"""

import requests

# Grafana API details
GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin")

# Dashboard JSON
dashboard = {
    "dashboard": {
        "title": "Flytrap Object Detection",
        "tags": ["flytrap", "object-detection"],
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "title": "Detection Count Over Time",
                "type": "graph",
                "targets": [
                    {
                        "query": 'from(bucket: "detections") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "frame") |> filter(fn: (r) => r._field == "detection_count")',
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            },
            {
                "id": 2,
                "title": "Processing Time (ms)",
                "type": "graph",
                "targets": [
                    {
                        "query": 'from(bucket: "detections") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "frame") |> filter(fn: (r) => r._field == "processing_time_ms")',
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            },
            {
                "id": 3,
                "title": "Object Classes Detected",
                "type": "table",
                "targets": [
                    {
                        "query": 'from(bucket: "detections") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "detection") |> group(columns: ["class"]) |> count()',
                        "refId": "A",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            },
        ],
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "5s",
    }
}

# Create dashboard
response = requests.post(
    f"{GRAFANA_URL}/api/dashboards/db",
    auth=AUTH,
    json=dashboard,
    headers={"Content-Type": "application/json"},
)

if response.status_code == 200:
    print("✅ Dashboard created successfully!")
    print("Open: http://localhost:3000")
else:
    print(f"❌ Failed to create dashboard: {response.status_code}")
    print(response.text)
