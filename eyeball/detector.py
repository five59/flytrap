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
    MEMORY_DEEP_CLEANUP_INTERVAL_FRAMES
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
        model_path: str = 'yolo11m.pt',
        confidence: float = 0.4,
        road_width_feet: float = 52,
        log_file: str = 'vehicle_tracking.log',
        screenshots_dir: str = 'screenshots',
        enable_influx: bool = True,
        headless: bool = False,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        detection_fps: float = DEFAULT_DETECTION_FPS
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
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)
        self.stream_handler = StreamHandler(srt_uri, self.frame_queue, self.frame_skip_interval)
        self.frame_processor = FrameProcessor(model_path, confidence, self.device, roi_box)
        self.object_tracker = ObjectTracker(road_width_feet, log_file, screenshots_dir, self.influx_logger)
        self.gui_dashboard = GUIDashboard()
        self.memory_manager = MemoryManager()

        # Frame processing state
        self.frame_count = 0
        self.frame_skip_counter = 0

    def _get_device(self) -> str:
        """Detect and return the best available device for inference."""
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("Using Apple Silicon MPS acceleration")
        else:
            device = 'cpu'
            print("Using CPU (no GPU acceleration available)")
        return device



    def run(self, window_name: str = "Traffic Detector", window_size: tuple = (1600, 900)):
        """
        Start the main detection loop.

        Args:
            window_name: Name of the display window (ignored in headless mode)
            window_size: Fixed window size as (width, height) (ignored in headless mode)
        """
        # Create display window only if not in headless mode
        if not self.headless:
            try:
                # Create fixed-size, non-resizable window
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow(window_name, window_size[0], window_size[1])

                # Show initial blank frame
                blank_frame = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Waiting for SRT stream...", (50, window_size[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, blank_frame)
                # Force a waitKey to ensure window is created and displayed
                cv2.waitKey(1)
                print(f"✓ Display window created: {window_name} ({window_size[0]}x{window_size[1]})")
                print("  2-column dashboard: Video frames | Metrics dashboard")
            except Exception as e:
                print(f"⚠ Could not create display window: {e}")
                print("  This may be due to missing GUI libraries or running in a headless environment")
                print("  Switching to headless mode")
                self.headless = True
        else:
            print("Running in headless mode (no display window)")

        # Set up SRT stream
        print(f"Connecting to SRT stream: {self.srt_uri}")
        connection_successful = self.stream_handler.setup_stream()
        if not connection_successful:
            print("Failed to connect to SRT stream")
            return

        print("✓ SRT stream connected successfully")
        print("Press Ctrl+C to stop")

        try:
            while True:
                # Get frame from stream
                frame_bgr = self.stream_handler.get_frame(self.frame_skip_counter, self.frame_skip_interval)
                if frame_bgr is None:
                    continue

                self.frame_skip_counter += 1
                self.frame_count += 1

                # Set frame info for tracker if not set
                if self.stream_handler.frame_width and not hasattr(self.object_tracker, '_frame_info_set'):
                    self.object_tracker.set_frame_info(self.stream_handler.frame_width)
                    self.object_tracker._frame_info_set = True

                # Process frame
                annotated_frame, fg_mask, has_motion, detections, processing_time_ms, motion_pixels = self.frame_processor.process_frame(
                    frame_bgr, self.frame_count, self.frame_skip_interval
                )

                # Update object tracks
                current_time = time.time()
                self.object_tracker.update_tracks(detections, current_time)

                # Clean up old tracks
                if self.frame_count % 30 == 0:
                    self.object_tracker.cleanup_old_tracks(current_time)

                # Display result only if not in headless mode
                if not self.headless:
                    # Get metrics for dashboard
                    processing_time_ms = 0.0  # Would need to track this from frame_processor
                    motion_pixels = 0.0  # Would need to track this from frame_processor
                    queue_depth = self.frame_queue.qsize()
                    queue_max = self.frame_queue.maxsize
                    memory_usage_mb = self.memory_manager.get_memory_usage()
                    tracked_objects_count = self.object_tracker.get_tracked_count()
                    device_type = "GPU" if "cuda" in self.device or "mps" in self.device else "CPU"
                    inference_device = self.device

                    # Create the 2-column dashboard
                    dashboard = self.gui_dashboard.create_dashboard(
                        processing_time_ms, motion_pixels, fg_mask, annotated_frame,
                        queue_depth, queue_max, memory_usage_mb, self.frame_count,
                        tracked_objects_count, device_type, inference_device,
                        self.object_tracker.influx_log_lines
                    )

                    cv2.imshow(window_name, dashboard)

                # Clean up frame references immediately after use
                del annotated_frame, fg_mask, detections
                del frame_bgr

                # Memory management
                if self.frame_count % MEMORY_CLEANUP_INTERVAL_FRAMES == 0:
                    self.memory_manager.aggressive_memory_cleanup(self.frame_count)

                # Check memory usage and trigger emergency cleanup if needed
                current_memory = self.memory_manager.get_memory_usage()
                if current_memory > MEMORY_HIGH_USAGE_THRESHOLD_MB:
                    self.logger.warning(f"High memory usage detected: {current_memory:.1f}MB")
                    self.memory_manager.emergency_memory_cleanup()

                # Additional cleanup every N frames
                if self.frame_count % MEMORY_DEEP_CLEANUP_INTERVAL_FRAMES == 0:
                    self.memory_manager.deep_cleanup()

                # Exit on 'q' key (only in GUI mode)
                if not self.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif self.headless:
                    # Small sleep in headless mode to prevent tight loop
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()

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
