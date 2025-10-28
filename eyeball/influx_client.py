"""
InfluxDB client wrapper for storing object detection metrics.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.domain.write_precision import WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DetectionLogger:
    """Handles logging of object detection data to InfluxDB."""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        """
        Initialize the InfluxDB client.

        Args:
            url: InfluxDB URL (defaults to env var INFLUXDB_URL)
            token: InfluxDB token (defaults to env var INFLUXDB_TOKEN)
            org: InfluxDB organization (defaults to env var INFLUXDB_ORG)
            bucket: InfluxDB bucket (defaults to env var INFLUXDB_BUCKET)
        """
        self.url = url or os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = token or os.getenv("INFLUXDB_TOKEN")
        self.org = org or os.getenv("INFLUXDB_ORG", "eyeball")
        self.bucket = bucket or os.getenv("INFLUXDB_BUCKET", "detections")

        if not self.token:
            raise ValueError(
                "InfluxDB token must be provided via parameter or INFLUXDB_TOKEN env var"
            )

        if not self.org:
            raise ValueError(
                "InfluxDB org must be provided via parameter or INFLUXDB_ORG env var"
            )

        if not self.bucket:
            raise ValueError(
                "InfluxDB bucket must be provided via parameter or INFLUXDB_BUCKET env var"
            )



        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def log_detections(
        self,
        detections: List[Dict],
        source_name: str = "default",
        frame_number: int = 0,
        processing_time_ms: Optional[float] = None,
        motion_pixels: Optional[int] = None,
        queue_depth: Optional[int] = None,
        gstreamer_buffers: Optional[int] = None,
        detection_backlog: Optional[int] = None,
        memory_usage_mb: Optional[float] = None,
    ) -> None:
        """
        Log detected objects to InfluxDB.

        Args:
            detections: List of detection dicts with keys: class_name, confidence, bbox
            source_name: Name of the video source (e.g., NDI source name)
            frame_number: Frame number from the video stream
            processing_time_ms: Time taken to process the frame in milliseconds
            motion_pixels: Number of motion pixels detected
            queue_depth: Current frame queue depth (for detecting processing backlog)
            gstreamer_buffers: Number of buffers in GStreamer pipeline
            detection_backlog: Number of frames waiting for detection processing
            memory_usage_mb: Current memory usage in MB
        """
        timestamp = datetime.now(timezone.utc)

        # Log frame-level metrics
        frame_point = (
            Point("frame")
            .tag("source", source_name)
            .field("frame_number", frame_number)
            .field("detection_count", len(detections))
            .time(timestamp, WritePrecision.NS)
        )

        if processing_time_ms is not None:
            frame_point.field("processing_time_ms", processing_time_ms)

        if motion_pixels is not None:
            frame_point.field("motion_pixels", motion_pixels)

        if queue_depth is not None:
            frame_point.field("queue_depth", queue_depth)

        if gstreamer_buffers is not None:
            frame_point.field("gstreamer_buffers", gstreamer_buffers)

        if detection_backlog is not None:
            frame_point.field("detection_backlog", detection_backlog)

        if memory_usage_mb is not None:
            frame_point.field("memory_usage_mb", memory_usage_mb)

        points = [frame_point]

        # Log individual detections
        for detection in detections:
            detection_point = (
                Point("detection")
                .tag("source", source_name)
                .tag("class", detection.get("class_name", "unknown"))
                .field("confidence", detection.get("confidence", 0.0))
                .field("frame_number", frame_number)
            )

            # Add bounding box coordinates if available
            if "bbox" in detection:
                bbox = detection["bbox"]
                detection_point.field("bbox_x1_f", float(bbox.get("x1", 0)))
                detection_point.field("bbox_y1_f", float(bbox.get("y1", 0)))
                detection_point.field("bbox_x2_f", float(bbox.get("x2", 0)))
                detection_point.field("bbox_y2_f", float(bbox.get("y2", 0)))

            detection_point.time(timestamp, WritePrecision.NS)
            points.append(detection_point)

        # Write all points in batch
        self.write_api.write(bucket=self.bucket, org=self.org, record=points)  # type: ignore

    def log_class_counts(
        self, class_counts: Dict[str, int], source_name: str = "default"
    ) -> None:
        """
        Log aggregated class counts.

        Args:
            class_counts: Dictionary mapping class names to counts
            source_name: Name of the video source
        """
        timestamp = datetime.now(timezone.utc)

        for class_name, count in class_counts.items():
            point = (
                Point("class_count")
                .tag("source", source_name)
                .tag("class", class_name)
                .field("count", count)
                .time(timestamp, WritePrecision.NS)
            )
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)  # type: ignore

    def log_directions(
        self, directions: List[Dict], source_name: str = "default"
    ) -> None:
        """
        Log object movement directions to InfluxDB.

        Args:
            directions: List of direction dicts with keys: class_name, direction, speed_mph, track_id
            source_name: Name of the video source
        """
        timestamp = datetime.now(timezone.utc)

        points = []
        for direction in directions:
            point = (
                Point("direction")
                .tag("source", source_name)
                .tag("class", direction.get("class_name", "unknown"))
                .tag("direction", direction.get("direction", "unknown"))
                .field("speed_mph", direction.get("speed_mph", 0.0))
                .field("track_id", direction.get("track_id", 0))
                .time(timestamp, WritePrecision.NS)
            )
            points.append(point)

        # Write all points in batch
        self.write_api.write(bucket=self.bucket, org=self.org, record=points)  # type: ignore

    def close(self) -> None:
        """Close the InfluxDB client connection."""
        self.write_api.close()
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Example usage
    print("Testing InfluxDB connection...")

    try:
        with DetectionLogger() as logger:
            # Test logging some sample detections
            sample_detections = [
                {
                    "class_name": "person",
                    "confidence": 0.92,
                    "bbox": {"x1": 100.5, "y1": 200.5, "x2": 300.5, "y2": 500.5},
                },
                {
                    "class_name": "car",
                    "confidence": 0.87,
                    "bbox": {"x1": 400.5, "y1": 300.5, "x2": 600.5, "y2": 450.5},
                },
            ]

            logger.log_detections(
                detections=sample_detections,
                source_name="test_source",
                frame_number=1,
                processing_time_ms=45.2,
            )

            # Test direction logging
            sample_directions = [
                {
                    "class_name": "car",
                    "direction": "left-to-right",
                    "speed_mph": 35.5,
                    "track_id": 123,
                },
                {
                    "class_name": "truck",
                    "direction": "right-to-left",
                    "speed_mph": 28.2,
                    "track_id": 456,
                },
            ]

            logger.log_directions(
                directions=sample_directions, source_name="test_source"
            )

            print("✓ Successfully logged test detections and directions to InfluxDB")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("1. InfluxDB is running: docker-compose up -d")
        print("2. Environment variables are set (check .env file)")
