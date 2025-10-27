"""
Real-time object detection module for SRT video streams using YOLO.
"""

import cv2
import time
import os
import torch
import numpy as np
import queue
import threading
import select
import platform
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List
from ultralytics import YOLO
from eyeball.influx_client import DetectionLogger

# GStreamer imports
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    GSTREAMER_AVAILABLE = True
except Exception as e:
    GSTREAMER_AVAILABLE = False
    print(f"GStreamer not available: {e}, falling back to FFmpeg")


class ObjectDetector:
    """Handles real-time object detection on NDI video streams."""

    # COCO dataset class IDs for tracked objects
    VEHICLE_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(
        self,
        srt_uri: str,
        model_path: str = 'yolo11m.pt',
        confidence: float = 0.4,
        road_width_feet: float = 52,
        log_file: str = 'vehicle_tracking.log',
        screenshots_dir: str = 'screenshots',
        enable_influx: bool = True,
        headless: bool = False
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
        """
        self.srt_uri = srt_uri
        self.model_path = model_path
        self.confidence = confidence
        self.road_width_feet = road_width_feet
        self.log_file = log_file
        self.screenshots_dir = screenshots_dir
        self.headless = headless

        # Create screenshots directory
        os.makedirs(self.screenshots_dir, exist_ok=True)

        # Initialize device and model
        self.device = self._get_device()
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Initialize InfluxDB logger
        self.influx_logger = None
        if enable_influx:
            try:
                self.influx_logger = DetectionLogger()
                print("✓ InfluxDB logging enabled")
            except Exception as e:
                print(f"⚠ InfluxDB logging disabled: {e}")
                print("  (Continuing without time-series logging)")

        # Initialize SRT video capture
        self.cap = None
        self.use_gstreamer = GSTREAMER_AVAILABLE
        self.use_opencv = not self.use_gstreamer  # Use OpenCV as secondary option if GStreamer not available

        # GStreamer components
        self.pipeline = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.gstreamer_thread = None

        # Motion detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
        self.motion_threshold = 500  # Minimum pixels for motion detection

        # Tracking state
        self.tracked_objects: Dict = {}
        self.frame_width: Optional[int] = None
        self.frame_midpoint_x: Optional[float] = None
        self.frame_count: int = 0
        self.prev_frame = None

    def _get_device(self) -> str:
        """Detect and return the best available device for inference."""
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

    def _setup_gstreamer(self):
        """Set up GStreamer pipeline for SRT streaming."""
        Gst.init(None)

        # Pipeline for SRT stream with auto-detection
        pipeline_str = (
            f'srtsrc uri="{self.srt_uri}" ! '
            'decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink'
        )

        self.pipeline = Gst.parse_launch(pipeline_str)

        # Get appsink
        appsink = self.pipeline.get_by_name('sink')
        appsink.set_property('emit-signals', True)
        appsink.connect('new-sample', self._on_new_sample)

        # Set up bus for messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_message)

        # Set playing
        self.pipeline.set_state(Gst.State.PLAYING)

        # Start GLib loop in thread
        self.gstreamer_thread = threading.Thread(target=self._run_gst_loop, daemon=True)
        self.gstreamer_thread.start()

    def _on_new_sample(self, sink):
        """Callback for new video sample from GStreamer."""
        sample = sink.emit('pull-sample')
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        if self.frame_width is None:
            self.frame_width = width
            self.frame_midpoint_x = self.frame_width / 2
            print(f"Frame size: {self.frame_width}x{height}")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            frame_bgr = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
            try:
                self.frame_queue.put(frame_bgr.copy(), timeout=0.1)
            except queue.Full:
                pass  # Discard if queue full
        buffer.unmap(map_info)
        return Gst.FlowReturn.OK

    def _on_message(self, bus, message):
        """Handle GStreamer bus messages."""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer error: {err}, debug: {debug}")
            self.pipeline.set_state(Gst.State.NULL)
        elif message.type == Gst.MessageType.EOS:
            print("GStreamer: End of stream")
            self.pipeline.set_state(Gst.State.NULL)

    def _run_gst_loop(self):
        """Run GLib main loop for GStreamer."""
        loop = GLib.MainLoop()
        loop.run()

    def _setup_opencv(self):
        """Set up OpenCV VideoCapture for SRT streaming."""
        self.cap = cv2.VideoCapture(self.srt_uri, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open SRT stream with OpenCV")

        # Get frame size
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_midpoint_x = self.frame_width / 2
        print(f"OpenCV SRT stream opened, frame size: {self.frame_width}x{frame_height}")

    def _setup_ffmpeg(self):
        """Set up FFmpeg subprocess for SRT streaming (fallback)."""
        # Use ffmpeg to decode SRT and pipe to stdout
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', self.srt_uri,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',  # Disable audio
            '-sn',  # Disable subtitles
            '-dn',  # Disable data
            'pipe:1'  # Output to stdout
        ]

        try:
            self.ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            # Give ffmpeg a moment to start
            time.sleep(2)
            if self.ffmpeg_proc.poll() is not None:
                raise RuntimeError("ffmpeg process failed to start")
            print("SRT stream connected via ffmpeg pipe")
        except Exception as e:
            raise RuntimeError(f"Failed to start ffmpeg for SRT stream: {e}")



    def _process_frame(self, frame_bgr):
        """
        Process a single video frame with YOLO detection.

        Args:
            frame_bgr: BGR-format frame from OpenCV

        Returns:
            Annotated frame with detections drawn
        """
        self.frame_count += 1

        # Motion detection using background subtraction
        fg_mask = self.back_sub.apply(frame_bgr)
        motion_pixels = cv2.countNonZero(fg_mask)
        has_motion = motion_pixels > self.motion_threshold

        # Convert to grayscale for frame differencing
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            motion_diff = cv2.countNonZero(thresh)
            has_motion = has_motion or (motion_diff > self.motion_threshold)
        self.prev_frame = gray

        # Skip YOLO if no significant motion (to save CPU when nothing is moving)
        if not has_motion and self.frame_count > 10:  # Allow initial frames
            # Still log frame metrics to InfluxDB
            if self.influx_logger:
                try:
                    source_name = "srt_stream"
                    self.influx_logger.log_detections(
                        detections=[],
                        source_name=source_name,
                        frame_number=self.frame_count,
                        processing_time_ms=0.0,
                        motion_pixels=motion_pixels
                    )
                except Exception as e:
                    print(f"Error logging to InfluxDB: {e}")
            return frame_bgr  # Return original frame

        # Run YOLO inference with tracking
        inference_start = time.time()
        results = self.model.track(
            frame_bgr,
            device=self.device,
            conf=self.confidence,
            verbose=False,
            persist=True
        )
        inference_time_ms = (time.time() - inference_start) * 1000

        # Annotate frame with detections
        annotated_frame = results[0].plot()

        # Prepare InfluxDB detections list
        influx_detections = []

        # Process tracking results
        screenshots_to_save = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                # Only track vehicles
                if cls not in self.VEHICLE_CLASSES:
                    continue

                # Get center point of bounding box
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_time = time.time()

                # Add to InfluxDB detections
                influx_detections.append({
                    "class_name": self.CLASS_NAMES.get(int(cls), "unknown"),
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })

                # Initialize or update tracking data
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        'positions': [(center_x, center_y, current_time)],
                        'class': cls,
                        'logged': False,
                        'crossed_midpoint': False,
                        'midpoint_cross_position': None
                    }
                else:
                    # Update class in case it was misidentified initially
                    self.tracked_objects[track_id]['class'] = cls
                    self.tracked_objects[track_id]['positions'].append((center_x, center_y, current_time))

                    # Keep only last 30 positions for direction calculation
                    if len(self.tracked_objects[track_id]['positions']) > 30:
                        self.tracked_objects[track_id]['positions'].pop(0)

                    # Check if object has crossed the midpoint
                    if not self.tracked_objects[track_id]['crossed_midpoint'] and len(self.tracked_objects[track_id]['positions']) >= 2:
                        prev_x = self.tracked_objects[track_id]['positions'][-2][0]
                        curr_x = center_x

                        # Check if crossed from either direction
                        if (prev_x < self.frame_midpoint_x <= curr_x) or (prev_x > self.frame_midpoint_x >= curr_x):
                            self.tracked_objects[track_id]['crossed_midpoint'] = True
                            self.tracked_objects[track_id]['midpoint_cross_position'] = len(self.tracked_objects[track_id]['positions']) - 1

                    # Log when object crosses midpoint
                    if self.tracked_objects[track_id]['crossed_midpoint'] and not self.tracked_objects[track_id]['logged']:
                        screenshot_path = self._log_tracked_object(track_id, cls)
                        if screenshot_path:
                            screenshots_to_save.append(screenshot_path)

        # Log to InfluxDB if available
        if self.influx_logger and influx_detections:
            try:
                source_name = "srt_stream"
                self.influx_logger.log_detections(
                    detections=influx_detections,
                    source_name=source_name,
                    frame_number=self.frame_count,
                    processing_time_ms=inference_time_ms,
                    motion_pixels=motion_pixels
                )
            except Exception as e:
                print(f"Error logging to InfluxDB: {e}")

        # Save screenshots for right-to-left vehicles
        for screenshot_path in screenshots_to_save:
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Screenshot saved: {screenshot_path}")

        return annotated_frame

    def _log_tracked_object(self, track_id: int, cls: int) -> Optional[str]:
        """
        Log a tracked object that has crossed the midpoint.

        Args:
            track_id: Unique tracking ID
            cls: Object class ID

        Returns:
            Screenshot path if saved, None otherwise
        """
        positions = self.tracked_objects[track_id]['positions']
        if len(positions) < 10:  # Need enough points for reliable calculation
            return None

        start_x, start_y, start_time = positions[0]
        end_x, end_y, end_time = positions[-1]
        displacement_pixels = end_x - start_x
        time_elapsed = end_time - start_time

        # Determine direction based on displacement
        if abs(displacement_pixels) <= 50 or time_elapsed <= 0:
            return None

        direction = "left-to-right" if displacement_pixels > 0 else "right-to-left"

        # Calculate speed
        pixels_per_foot = self.frame_width / self.road_width_feet
        distance_feet = abs(displacement_pixels) / pixels_per_foot
        speed_fps = distance_feet / time_elapsed
        speed_mph = speed_fps * 3600 / 5280

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        vehicle_type = self.CLASS_NAMES.get(cls, "object")

        # Prepare screenshot if moving right-to-left
        screenshot_filename = None
        if direction == "right-to-left":
            screenshot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            screenshot_filename = f"{self.screenshots_dir}/track_{track_id}_{screenshot_timestamp}.jpg"

        # Log to file
        with open(self.log_file, 'a') as f:
            log_entry = f"{timestamp} | Track ID: {track_id} | Type: {vehicle_type} | Direction: {direction} | Speed: {speed_mph:.1f} mph"
            if screenshot_filename:
                log_entry += f" | Screenshot: {screenshot_filename}"
            f.write(log_entry + "\n")

        self.tracked_objects[track_id]['logged'] = True
        print(f"Logged: {vehicle_type} {direction} at {speed_mph:.1f} mph - {timestamp}")

        # Log direction to InfluxDB
        if self.influx_logger:
            try:
                direction_data = [{
                    "class_name": vehicle_type,
                    "direction": direction,
                    "speed_mph": speed_mph,
                    "track_id": track_id
                }]
                self.influx_logger.log_directions(direction_data, source_name="srt_stream")
            except Exception as e:
                print(f"Failed to log direction to InfluxDB: {e}")

        return screenshot_filename

    def run(self, window_name: str = "YOLO Object Detection", window_size: tuple = (1280, 720)):
        """
        Start the main detection loop.

        Args:
            window_name: Name of the display window (ignored in headless mode)
            window_size: Window size as (width, height) (ignored in headless mode)
        """
        # Create display window only if not in headless mode
        if not self.headless:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, *window_size)
                # Show initial blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Waiting for SRT stream...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, blank_frame)
                print(f"Display window created: {window_name}")
            except Exception as e:
                print(f"⚠ Could not create display window: {e}")
                print("  Switching to headless mode")
                self.headless = True
        else:
            print("Running in headless mode (no display window)")

        # Open SRT stream
        if self.use_gstreamer:
            print(f"Connecting to SRT stream using GStreamer: {self.srt_uri}")
            self._setup_gstreamer()
            print("GStreamer pipeline started")
        elif self.use_opencv:
            print(f"Connecting to SRT stream using OpenCV: {self.srt_uri}")
            try:
                self._setup_opencv()
            except Exception as e:
                print(f"OpenCV SRT setup failed: {e}, falling back to FFmpeg")
                self.use_opencv = False
                self._setup_ffmpeg()
        else:
            print(f"Connecting to SRT stream using FFmpeg: {self.srt_uri}")
            self._setup_ffmpeg()

        print("SRT stream connected successfully")
        print("Press Ctrl+C to stop")

        try:
            while True:
                # Get frame from queue, OpenCV, or pipe
                if self.use_gstreamer:
                    try:
                        frame_bgr = self.frame_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                elif self.use_opencv:
                    ret, frame_bgr = self.cap.read()
                    if not ret:
                        print("Failed to read frame from OpenCV SRT stream")
                        time.sleep(1)
                        continue
                else:
                    # Handle ffmpeg process
                    if self.ffmpeg_proc.poll() is not None:
                        print("ffmpeg process exited, attempting to restart...")
                        try:
                            self._setup_ffmpeg()
                        except Exception as e:
                            print(f"Failed to restart ffmpeg: {e}")
                            time.sleep(5)  # Wait before retrying
                            continue

                    # Check if data is available with timeout
                    ready, _, _ = select.select([self.ffmpeg_proc.stdout], [], [], 1.0)
                    if not ready:
                        continue  # No data available, continue loop

                    # Read raw BGR frame from ffmpeg pipe
                    frame_width = 1920  # Assumed
                    frame_height = 1080
                    frame_size = frame_width * frame_height * 3  # BGR24 = 3 bytes per pixel
                    frame_data = self.ffmpeg_proc.stdout.read(frame_size)

                    if len(frame_data) != frame_size:
                        print("Failed to read complete frame from SRT stream")
                        time.sleep(0.1)
                        continue

                    # Convert bytes to numpy array and reshape
                    frame_bgr = np.frombuffer(frame_data, dtype=np.uint8).reshape((frame_height, frame_width, 3))

                # Process frame with YOLO
                annotated_frame = self._process_frame(frame_bgr)

                # Display result only if not in headless mode
                if not self.headless:
                    cv2.imshow(window_name, annotated_frame)

                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count} frames")

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
        if self.use_gstreamer:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            # Thread is daemon, will stop
        elif self.use_opencv:
            if self.cap:
                self.cap.release()
        else:
            if hasattr(self, 'ffmpeg_proc') and self.ffmpeg_proc:
                self.ffmpeg_proc.terminate()
                self.ffmpeg_proc.wait()
        if self.influx_logger:
            self.influx_logger.close()
            print("InfluxDB connection closed")


if __name__ == "__main__":
    # Import and run main from main.py
    from main import main
    main()
