"""
Flytrap - Real-time object detection with NDI and YOLO.
"""

import logging

from flytrap.detector import ObjectDetector
from flytrap.influx_client import DetectionLogger

__version__ = "0.1.0"

__all__ = ["ObjectDetector", "DetectionLogger"]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
