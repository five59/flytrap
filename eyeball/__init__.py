"""
Eyeball - Real-time object detection with NDI and YOLO.
"""

from eyeball.detector import ObjectDetector
from eyeball.influx_client import DetectionLogger

__version__ = "0.1.0"

__all__ = ["ObjectDetector", "DetectionLogger"]
