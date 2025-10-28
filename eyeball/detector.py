"""
Real-time object detection module for SRT video streams using YOLO.
Main orchestrator class that coordinates stream handling, processing, tracking, and display.
"""

import logging
import cv2
import time
import os
import queue
import numpy as np
from typing import Optional, Tuple
from eyeball.config import (
    DEFAULT_DETECTION_FPS,
    FRAME_SKIP_INTERVAL_BASE,
    FRAME_QUEUE_MAX_SIZE,
    MEMORY_HIGH_USAGE_THRESHOLD_MB,
    MEMORY_CLEANUP_INTERVAL_FRAMES,
    MEMORY_DEEP_CLEANUP_INTERVAL_FRAMES,
    WINDOW_NAME,
)
from eyeball.stream_handler import StreamHandler
from eyeball.frame_processor import FrameProcessor
from eyeball.object_tracker import ObjectTracker
from eyeball.gui_dashboard import GUIDashboard
from eyeball.memory_manager import MemoryManager
from eyeball.influx_client import DetectionLogger


class ObjectDetector:
    """Handles real-time object detection on SRT video streams."""

    def __init__(
        self,
        srt_uri: str,
        model_path: str = "yolo11m.pt",
        confidence: float = 0.4,
        road_width_feet: float = 52,
        log_file: str = "vehicle_tracking.log",
        screenshots_dir: str = "screenshots",
        enable_influx: bool = True,
        headless: bool = False,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        detection_fps: float = DEFAULT_DETECTION_FPS,
    ):
        """
        Initialize the object detector.

        Args:
            srt_uri: SRT stream URI (e.g., 'srt://192.168.1.100:1234')
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold (0.0-1.0)
            road_width_feet: Width of road in feet for speed calculation
            log_file: Path to log file for vehicle tracking
            screenshots_dir: Directory to save screenshots
            enable_influx: Whether to enable InfluxDB logging
            headless: Run without GUI display (for WSL/headless servers)
            roi_box: Region of interest bounding box [x1, y1, x2, y2] in original frame coordinates.
                      Crop to this box before resizing. Set to None to process entire frame
            detection_fps: Target detection frame rate (FPS). Assumes ~30 FPS input stream.
                            Lower values reduce CPU usage but may miss fast-moving objects.
        """
        self.logger = logging.getLogger(__name__)
        self.srt_uri = srt_uri
        self.headless = headless
        self.detection_fps = detection_fps
        self.frame_skip_interval = max(1, int(FRAME_SKIP_INTERVAL_BASE / detection_fps))

        # Create screenshots directory
        os.makedirs(screenshots_dir, exist_ok=True)

        # Initialize device
        self.device = self._get_device()

        # Initialize InfluxDB logger
        self.influx_logger = None
        if enable_influx:
            try:
                self.influx_logger = DetectionLogger()
                self.logger.info("InfluxDB logging enabled")
            except Exception as e:
                self.logger.warning(f"InfluxDB logging disabled: {e}")
                self.logger.info("Continuing without time-series logging")

        # Initialize components
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)
        self.stream_handler = StreamHandler(
            srt_uri, self.frame_queue, self.frame_skip_interval
        )
        self.frame_processor = FrameProcessor(
            model_path, confidence, self.device, roi_box
        )
        self.object_tracker = ObjectTracker(
            road_width_feet, log_file, screenshots_dir, self.influx_logger
        )
        self.gui_dashboard = GUIDashboard()
        self.memory_manager = MemoryManager()

        # Frame processing state
        self.frame_count = 0
        self.frame_skip_counter = 0

    def _get_device(self) -> str:
        """Detect and return the best available device for inference."""
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon MPS acceleration")
        else:
            device = "cpu"
            print("Using CPU (no GPU acceleration available)")
        return device

    def run(self, window_size: tuple = (1600, 900)):
        """
        Start the main detection loop.

        Args:
            window_size: Fixed window size as (width, height) (ignored in headless mode)
        """
        self._setup_display_window(window_size)

        if not self._setup_stream():
            return

        try:
            self._run_detection_loop()
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()

    def _setup_display_window(self, window_size: tuple) -> None:
        """Set up the display window if not in headless mode."""
        if self.headless:
            print("Running in headless mode (no display window)")
            return

        try:
            self._create_display_window(window_size)
            print(
                f"✓ Display window created ({window_size[0]}x{window_size[1]})"
            )
            print("  2-column dashboard: Video frames | Metrics dashboard")
        except Exception as e:
            self._handle_window_creation_error(e)

    def _create_display_window(self, window_size: tuple) -> None:
        """Create and initialize the display window."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, window_size[0], window_size[1])

        # Show initial blank frame
        blank_frame = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        cv2.putText(
            blank_frame,
            "Waiting for SRT stream...",
            (50, window_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, blank_frame)
        cv2.waitKey(1)  # Force window creation

    def _handle_window_creation_error(self, error: Exception) -> None:
        """Handle display window creation failure by switching to headless mode."""
        print(f"⚠ Could not create display window: {error}")
        print(
            "  This may be due to missing GUI libraries or running in a headless environment"
        )
        print("  Switching to headless mode")
        self.headless = True

    def _setup_stream(self) -> bool:
        """Set up the SRT stream connection."""
        print(f"Connecting to SRT stream: {self.srt_uri}")
        if not self.stream_handler.setup_stream():
            print("Failed to connect to SRT stream")
            return False

        print("✓ SRT stream connected successfully")
        print("Press Ctrl+C to stop")
        return True

    def _run_detection_loop(self) -> None:
        """Run the main detection processing loop."""
        while True:
            frame_bgr = self.stream_handler.get_frame(
                self.frame_skip_counter, self.frame_skip_interval
            )
            if frame_bgr is None:
                continue

            self.frame_skip_counter += 1
            self.frame_count += 1

            self._process_single_frame(frame_bgr)
            self._handle_display()
            self._perform_memory_management()

            if self._should_exit():
                break

    def _process_single_frame(self, frame_bgr: np.ndarray) -> None:
        """Process a single frame through the detection pipeline."""
        self._initialize_frame_info_if_needed()
        processing_results = self._process_frame_with_yolo(frame_bgr)
        self._update_object_tracks(processing_results["detections"])
        self._cleanup_old_tracks_if_needed()
        self._store_results_for_display(processing_results)
        self._cleanup_frame_references(frame_bgr, processing_results)

    def _initialize_frame_info_if_needed(self) -> None:
        """Initialize frame info for tracker if not already set."""
        if self.stream_handler.frame_width and not hasattr(
            self.object_tracker, "_frame_info_set"
        ):
            self.object_tracker.set_frame_info(self.stream_handler.frame_width)
            self.object_tracker._frame_info_set = True

    def _process_frame_with_yolo(self, frame_bgr: np.ndarray) -> dict:
        """Process frame through YOLO and return results."""
        (
            annotated_frame,
            fg_mask,
            _,  # has_motion (unused)
            detections,
            processing_time_ms,
            motion_pixels,
        ) = self.frame_processor.process_frame(
            frame_bgr, self.frame_count
        )

        return {
            "annotated_frame": annotated_frame,
            "fg_mask": fg_mask,
            "detections": detections,
            "processing_time_ms": processing_time_ms,
            "motion_pixels": motion_pixels,
        }

    def _update_object_tracks(self, detections: list) -> None:
        """Update object tracks with new detections."""
        current_time = time.time()
        self.object_tracker.update_tracks(detections, current_time)

    def _cleanup_old_tracks_if_needed(self) -> None:
        """Clean up old tracks periodically."""
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.object_tracker.cleanup_old_tracks(current_time)

    def _store_results_for_display(self, processing_results: dict) -> None:
        """Store processing results for display if not in headless mode."""
        if not self.headless:
            self._last_processing_results = {
                "annotated_frame": processing_results["annotated_frame"],
                "fg_mask": processing_results["fg_mask"],
                "processing_time_ms": processing_results["processing_time_ms"],
                "motion_pixels": processing_results["motion_pixels"],
            }

    def _cleanup_frame_references(self, frame_bgr: np.ndarray, processing_results: dict) -> None:
        """Clean up frame references to free memory."""
        del processing_results["annotated_frame"], processing_results["fg_mask"], processing_results["detections"]
        del frame_bgr

    def _handle_display(self) -> None:
        """Handle display updates if not in headless mode."""
        if self.headless:
            return

        dashboard_data = self._collect_dashboard_data()
        dashboard = self.gui_dashboard.create_dashboard(**dashboard_data)
        cv2.imshow(WINDOW_NAME, dashboard)

    def _collect_dashboard_data(self) -> dict:
        """Collect all data needed for the dashboard display."""
        processing_time_ms = self._last_processing_results.get("processing_time_ms", 0.0)
        motion_pixels = self._last_processing_results.get("motion_pixels", 0.0)

        return {
            "processing_time_ms": processing_time_ms,
            "motion_pixels": motion_pixels,
            "fg_mask": self._last_processing_results.get("fg_mask"),
            "annotated_frame": self._last_processing_results.get("annotated_frame"),
            "queue_depth": self.frame_queue.qsize(),
            "queue_max": self.frame_queue.maxsize,
            "memory_usage_mb": self.memory_manager.get_memory_usage(),
            "frame_count": self.frame_count,
            "tracked_objects_count": self.object_tracker.get_tracked_count(),
            "device_type": "GPU" if "cuda" in self.device or "mps" in self.device else "CPU",
            "inference_device": self.device,
            "influx_log_lines": self.object_tracker.influx_log_lines,
        }

    def _perform_memory_management(self) -> None:
        """Perform periodic memory management tasks."""
        # Regular cleanup
        if self.frame_count % MEMORY_CLEANUP_INTERVAL_FRAMES == 0:
            self.memory_manager.aggressive_memory_cleanup(self.frame_count)

        # Emergency cleanup for high memory usage
        current_memory = self.memory_manager.get_memory_usage()
        if current_memory > MEMORY_HIGH_USAGE_THRESHOLD_MB:
            self.logger.warning(f"High memory usage detected: {current_memory:.1f}MB")
            self.memory_manager.emergency_memory_cleanup()

        # Deep cleanup
        if self.frame_count % MEMORY_DEEP_CLEANUP_INTERVAL_FRAMES == 0:
            self.memory_manager.deep_cleanup()

    def _should_exit(self) -> bool:
        """Check if the application should exit."""
        if not self.headless:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True
        else:
            # Small sleep in headless mode to prevent tight loop
            time.sleep(0.001)
        return False

    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        self.stream_handler.cleanup()
        if self.influx_logger:
            self.influx_logger.close()
            print("InfluxDB connection closed")


if __name__ == "__main__":
    # Import and run main from main.py
    from main import main

    main()
