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
import subprocess
import gc
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
        headless: bool = False,
        inference_size: tuple = (1280, 720),
        roi_mask: Optional[np.ndarray] = None
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
            inference_size: Resolution for YOLO inference (width, height).
                           Smaller = faster & less memory. Recommended:
                           - (1280, 720): Good balance, 55% less memory
                           - (960, 540): Aggressive, 75% less memory
                           - (640, 360): Maximum speed, 88% less memory
                           Set to None to use original frame size
            roi_mask: Region of interest mask (numpy array, same size as inference_size).
                     White (255) = detect, Black (0) = ignore.
                     Use to exclude parked cars, buildings, etc.
                     Set to None to process entire frame
        """
        self.srt_uri = srt_uri
        self.model_path = model_path
        self.confidence = confidence
        self.road_width_feet = road_width_feet
        self.log_file = log_file
        self.screenshots_dir = screenshots_dir
        self.headless = headless
        self.inference_size = inference_size
        self.roi_mask = roi_mask

        # Create screenshots directory
        os.makedirs(self.screenshots_dir, exist_ok=True)

        # Initialize device and model
        self.device = self._get_device()
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Log inference resolution configuration
        if self.inference_size:
            print(f"Inference resolution: {self.inference_size[0]}x{self.inference_size[1]}")
            frame_mb = (self.inference_size[0] * self.inference_size[1] * 3) / (1024 * 1024)
            print(f"  Per-frame memory: ~{frame_mb:.1f}MB (vs ~6MB at 1080p)")
            print(f"  Queue memory (10 frames): ~{frame_mb * 10:.1f}MB")
        else:
            print("Inference resolution: Original frame size (no downscaling)")

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
        # Reduced queue size from 50 to 10 to reduce memory footprint
        # At 1920x1080 BGR, each frame is ~6MB, so 10 frames = ~60MB vs 300MB
        self.frame_queue = queue.Queue(maxsize=10)
        self.gstreamer_thread = None

        # Motion detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
        # Increased from 500 to 2000 pixels to reduce false positives from stream jitter
        # At 1280x720 (921,600 pixels), 2000 pixels = 0.22% of frame
        self.motion_threshold = 2000  # Minimum pixels for motion detection

        # Tracking state
        self.tracked_objects: Dict = {}
        self.frame_width: Optional[int] = None
        self.frame_midpoint_x: Optional[float] = None
        self.frame_count: int = 0
        self.prev_frame = None
        self.frame_skip_counter: int = 0

        # Memory monitoring
        self.memory_history = []
        self.last_memory_cleanup = time.time()

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

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback if psutil not available
            return 0.0

    def _estimate_object_memory(self) -> dict:
        """Estimate memory usage of major components."""
        import sys
        memory_breakdown = {}

        try:
            # Estimate frame queue memory
            if hasattr(self, 'frame_queue'):
                queue_size = self.frame_queue.qsize()
                # Calculate actual frame size based on current frame width
                if hasattr(self, 'frame_width') and self.frame_width:
                    # Estimate height as 9/16 of width (16:9 aspect ratio)
                    estimated_height = int(self.frame_width * 9 / 16)
                    frame_mb = (self.frame_width * estimated_height * 3) / (1024 * 1024)
                else:
                    frame_mb = 6.0  # Default assumption for 1920x1080
                memory_breakdown['frame_queue_mb'] = queue_size * frame_mb
                memory_breakdown['queue_size'] = queue_size

            # Estimate tracked objects memory
            if hasattr(self, 'tracked_objects'):
                # Rough estimate: each position is ~50 bytes, plus overhead
                total_positions = sum(len(obj['positions']) for obj in self.tracked_objects.values())
                memory_breakdown['tracked_objects_mb'] = (total_positions * 50) / (1024 * 1024)
                memory_breakdown['num_tracked_objects'] = len(self.tracked_objects)

            # Estimate YOLO model memory (typically 40-80MB for yolo11m)
            if hasattr(self, 'model') and self.model is not None:
                memory_breakdown['model_estimated_mb'] = 60  # Rough estimate for yolo11m

            # Background subtractor (stores history of 100 frames internally)
            if hasattr(self, 'back_sub'):
                # Rough estimate based on history size
                memory_breakdown['back_sub_estimated_mb'] = 20  # Conservative estimate

            return memory_breakdown
        except Exception as e:
            return {'error': str(e)}

    def _get_detailed_memory_info(self):
        """Get detailed memory information for debugging."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'num_fds': getattr(process, 'num_fds', lambda: 0)()
            }
        except Exception as e:
            return {'error': str(e)}

    def _get_gstreamer_buffer_info(self) -> int:
        """Get GStreamer pipeline buffer/queue information."""
        if not GSTREAMER_AVAILABLE or not self.pipeline:
            return 0

        try:
            # Try to get buffer information from the pipeline
            # This is a simplified approach - in a real implementation,
            # you might want to query specific elements like queue elements
            bus = self.pipeline.get_bus()
            if bus:
                # Get the number of messages in the bus (rough indicator of activity)
                # This is not perfect but gives some insight into pipeline health
                return 0  # Placeholder - would need more complex GStreamer introspection
            return 0
        except Exception:
            return 0

    def _wait_for_stream_connection(self, timeout: float = 10.0) -> bool:
        """Wait for SRT stream connection to be established and frames to arrive."""
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to get a frame based on current method
                if self.use_gstreamer and hasattr(self, 'frame_queue'):
                    # For GStreamer, check if frames are being added to queue
                    if self.frame_queue.qsize() > 0:
                        print("✓ GStreamer connection established - frames received")
                        return True
                elif self.use_opencv and hasattr(self, 'cap') and self.cap.isOpened():
                    # For OpenCV, try to read a frame
                    ret, _ = self.cap.read()
                    if ret:
                        print("✓ OpenCV SRT connection established - frames received")
                        return True
                elif hasattr(self, 'ffmpeg_proc') and self.ffmpeg_proc and self.ffmpeg_proc.poll() is None:
                    # For FFmpeg, check if process is still running and try to read
                    import select
                    if hasattr(self, 'ffmpeg_proc'):
                        ready, _, _ = select.select([self.ffmpeg_proc.stdout], [], [], 0.1)
                        if ready:
                            # Try to read a small amount of data
                            try:
                                data = self.ffmpeg_proc.stdout.read(1024)
                                if data:
                                    print("✓ FFmpeg SRT connection established - data received")
                                    return True
                            except:
                                pass

                # Check for any error messages from GStreamer
                if self.use_gstreamer and hasattr(self, 'pipeline'):
                    # This would need more complex GStreamer bus message checking
                    pass

            except Exception as e:
                print(f"Connection check error: {e}")

            time.sleep(0.5)  # Check every 500ms

        return False

    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup to prevent memory leaks."""
        try:
            initial_memory = self._get_memory_usage()

            # Force multiple garbage collection cycles
            collected = 0
            for _ in range(3):
                collected += gc.collect()

            # Only log on deep cleanup cycles (every 200 frames)
            if collected > 0 and self.frame_count % 200 == 0:
                print(f"GC: Collected {collected} objects")

            # Clear GPU/CPU cache aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force memory deallocation
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            elif torch.backends.mps.is_available():
                # MPS (Apple Silicon) cache clearing
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            else:
                # CPU - force garbage collection of tensors
                # This helps when using CPU inference
                # Note: gc already imported at module level, no need to import again
                gc.collect()

            # Clear frame queue more aggressively
            if hasattr(self, 'frame_queue'):
                queue_size = self.frame_queue.qsize()
                if queue_size > 5:  # Keep only 5 frames max (maxsize is now 10)
                    excess_frames = queue_size - 5
                    for _ in range(excess_frames):
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    # Only log on deep cleanup cycles
                    if self.frame_count % 200 == 0:
                        print(f"Queue cleanup: Removed {excess_frames} frames, kept {self.frame_queue.qsize()}")

            # Very aggressive cleanup of tracked objects
            current_time = time.time()
            to_remove = []
            for track_id, data in self.tracked_objects.items():
                # Remove objects that haven't been updated in 15 seconds (very aggressive)
                if data['positions'] and current_time - data['positions'][-1][2] > 15:
                    to_remove.append(track_id)
                # Limit position history to prevent memory bloat
                elif len(data['positions']) > 20:
                    data['positions'] = data['positions'][-10:]  # Keep only last 10 positions

            for track_id in to_remove:
                del self.tracked_objects[track_id]

            # Only log significant cleanups on deep cleanup cycles
            if to_remove and len(to_remove) > 5 and self.frame_count % 200 == 0:
                print(f"Tracking cleanup: Removed {len(to_remove)} objects, {len(self.tracked_objects)} remaining")

            # Clear YOLO model tracking cache (critical for memory management)
            if hasattr(self, 'model') and self.model is not None:
                try:
                    # Reset the tracker to clear internal state
                    # YOLO stores tracking state in model.predictor
                    if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                        # Clear tracker history but keep the tracker objects intact
                        # Don't set to empty list as YOLO needs trackers to exist
                        if hasattr(self.model.predictor, 'trackers') and self.model.predictor.trackers:
                            # Clear each tracker's internal state instead of removing them
                            for tracker in self.model.predictor.trackers:
                                if hasattr(tracker, 'reset'):
                                    tracker.reset()
                                # Clear tracked objects in the tracker
                                if hasattr(tracker, 'tracks'):
                                    tracker.tracks = []
                        # Also clear any cached results
                        if hasattr(self.model.predictor, 'results'):
                            self.model.predictor.results = None
                    # Only log on deep cleanup (every 200 frames), not every 50
                    if self.frame_count % 200 == 0:
                        print("Cleared YOLO tracking state")
                except Exception as e:
                    print(f"Error clearing YOLO cache: {e}")

            # Reset background subtractor to clear accumulated history
            if hasattr(self, 'back_sub'):
                self.back_sub = cv2.createBackgroundSubtractorMOG2(
                    history=100, varThreshold=50, detectShadows=True
                )
                # Removed verbose logging - happens too often

            # NOTE: cv2.destroyAllWindows() removed from regular cleanup
            # It was causing window handle leaks by destroying/recreating windows every 20 frames
            # Only call this in final cleanup() method

            # Force cleanup of any remaining references
            gc.collect()

            final_memory = self._get_memory_usage()
            memory_delta = final_memory - initial_memory
            # Only report significant changes AND only on deep cleanup cycles
            if abs(memory_delta) > 5.0 and self.frame_count % 200 == 0:
                print(f"Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB ({memory_delta:+.1f}MB)")

            # Track memory usage for trend analysis
            self.memory_history.append((time.time(), final_memory))
            # Keep only last 20 memory readings
            if len(self.memory_history) > 20:
                self.memory_history.pop(0)

            # Check for memory leak trend
            self._check_memory_leak_trend()

        except Exception as e:
            print(f"Memory cleanup error: {e}")
            import traceback
            traceback.print_exc()

    def _check_memory_leak_trend(self):
        """Check if memory usage is trending upward and trigger emergency cleanup."""
        if len(self.memory_history) < 10:
            return  # Need more data points

        # Calculate memory trend over last 10 readings
        recent_readings = self.memory_history[-10:]
        times, memories = zip(*recent_readings)

        # Simple linear regression to detect trend
        n = len(times)
        if n < 2:
            return

        # Calculate slope (memory increase per second)
        time_span = times[-1] - times[0]
        memory_span = memories[-1] - memories[0]
        slope = memory_span / time_span if time_span > 0 else 0

        # If memory is increasing by more than 5MB per minute, trigger emergency cleanup
        # Increased from 1MB/min to 5MB/min - less sensitive to normal fluctuations
        if slope > (5.0 / 60.0):  # 5MB per minute threshold
            print(f"⚠️  Memory leak detected! Trend: +{slope*60:.2f}MB/min")
            self._emergency_memory_cleanup()

    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup when leak is detected."""
        print("🚨 Emergency memory cleanup initiated...")

        try:
            # Most aggressive cleanup possible
            for _ in range(5):
                gc.collect()

            # Clear all tracked objects
            old_count = len(self.tracked_objects)
            self.tracked_objects.clear()
            print(f"Cleared {old_count} tracked objects")

            # Clear frame queue completely
            if hasattr(self, 'frame_queue'):
                queue_size = self.frame_queue.qsize()
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                print(f"Cleared {queue_size} frames from queue")

            # Clear background subtractor
            if hasattr(self, 'back_sub'):
                self.back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
                print("Reset background subtractor")

            # Force system cleanup
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()

            # Final garbage collection
            for _ in range(3):
                gc.collect()

            final_memory = self._get_memory_usage()
            print(f"Emergency cleanup complete. Memory now: {final_memory:.1f}MB")

        except Exception as e:
            print(f"Emergency cleanup error: {e}")
            import traceback
            traceback.print_exc()

    def _add_metric_overlay(self, frame: np.ndarray, processing_time_ms: float, motion_pixels: int) -> np.ndarray:
        """Add monitoring metrics overlay to the frame."""
        if self.headless:
            print("DEBUG: Skipping overlay - headless mode")
            return frame

        overlay_frame = frame.copy()

        # Get current metrics
        queue_depth = self.frame_queue.qsize() if hasattr(self, 'frame_queue') else 0
        memory_usage_mb = self._get_memory_usage()
        gstreamer_buffers = self._get_gstreamer_buffer_info()

        # Log memory usage for monitoring trends
        if self.frame_count % 50 == 0:  # More frequent logging
            trend = ""
            if len(self.memory_history) >= 5:
                recent = self.memory_history[-5:]
                if len(recent) >= 2:
                    start_mem = recent[0][1]
                    end_mem = recent[-1][1]
                    delta = end_mem - start_mem
                    trend = f" ({delta:+.1f}MB trend)"
            print(f"Memory: {memory_usage_mb:.1f}MB{trend}, Queue: {queue_depth}, Objects: {len(self.tracked_objects)}")

        # Define overlay position (top-left corner)
        x, y = 10, 30
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # Bright green text for better visibility
        thickness = 2

        # Add background rectangle for better readability
        metrics_text = [
            f"Frame: {self.frame_count}",
            f"Queue: {queue_depth}/50",
            f"Processing: {processing_time_ms:.1f} ms",
            f"Memory: {memory_usage_mb:.1f} MB",
            f"Motion: {motion_pixels}",
            f"GStreamer: {gstreamer_buffers}"
        ]

        # Calculate background rectangle size
        max_text_width = max(cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text in metrics_text)
        bg_width = max_text_width + 20
        bg_height = len(metrics_text) * line_height + 20

        # Draw more opaque background for better visibility
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (x-5, y-25), (x + bg_width, y + bg_height - 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay_frame, 0.5, overlay, 0.5, 0, overlay_frame)

        # Draw colored border around overlay area
        cv2.rectangle(overlay_frame, (x-5, y-25), (x + bg_width, y + bg_height - 25), (0, 255, 0), 2)

        # Draw metrics text
        for i, text in enumerate(metrics_text):
            cv2.putText(overlay_frame, text, (x, y + i * line_height),
                       font, font_scale, color, thickness)

        return overlay_frame

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
        self.frame_skip_counter += 1
        # Process every 5th frame to reduce FPS from ~30 to ~6
        if self.frame_skip_counter % 5 != 0:
            # Still need to pull sample to clear pipeline buffer
            sample = sink.emit('pull-sample')
            del sample
            return Gst.FlowReturn.OK

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
            finally:
                buffer.unmap(map_info)
                # Explicitly delete references to allow GStreamer to free memory
                del frame_bgr
                del map_info
        else:
            buffer.unmap(map_info)

        # Critical: Delete references to GStreamer objects to prevent memory leak
        del buffer
        del sample
        del caps
        del structure

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

        # Store original frame dimensions for scaling detections back
        original_height, original_width = frame_bgr.shape[:2]

        # Downscale frame for inference if inference_size is specified
        if self.inference_size is not None:
            inference_frame = cv2.resize(frame_bgr, self.inference_size, interpolation=cv2.INTER_LINEAR)
            scale_x = original_width / self.inference_size[0]
            scale_y = original_height / self.inference_size[1]
        else:
            inference_frame = frame_bgr
            scale_x = 1.0
            scale_y = 1.0

        # Motion detection using background subtraction (on inference-sized frame)
        fg_mask = self.back_sub.apply(inference_frame)

        # Apply morphological operations to remove noise from compression artifacts
        # This helps eliminate false positives from video jitter in parked cars
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps

        # Apply ROI mask if provided (exclude areas like parked cars)
        if self.roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self.roi_mask)

        motion_pixels = cv2.countNonZero(fg_mask)
        has_motion = motion_pixels > self.motion_threshold

        # Convert to grayscale for frame differencing (on inference-sized frame)
        gray = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            # Increased threshold from 25 to 40 to reduce sensitivity to compression artifacts
            _, thresh = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)

            # Apply morphological operations to frame diff as well
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Apply ROI mask to frame differencing as well
            if self.roi_mask is not None:
                thresh = cv2.bitwise_and(thresh, thresh, mask=self.roi_mask)

            motion_diff = cv2.countNonZero(thresh)
            has_motion = has_motion or (motion_diff > self.motion_threshold)
        self.prev_frame = gray

        # Skip YOLO if no significant motion (to save CPU when nothing is moving)
        if not has_motion and self.frame_count > 10:  # Allow initial frames
            # Still log frame metrics to InfluxDB
            if self.influx_logger:
                try:
                    source_name = "srt_stream"
                    queue_depth = self.frame_queue.qsize() if hasattr(self, 'frame_queue') else 0
                    memory_usage_mb = self._get_memory_usage()
                    gstreamer_buffers = self._get_gstreamer_buffer_info()
                    self.influx_logger.log_detections(
                        detections=[],
                        source_name=source_name,
                        frame_number=self.frame_count,
                        processing_time_ms=0.0,
                        motion_pixels=motion_pixels,
                        queue_depth=queue_depth,
                        gstreamer_buffers=gstreamer_buffers,
                        memory_usage_mb=memory_usage_mb
                    )
                except Exception as e:
                    print(f"Error logging to InfluxDB: {e}")
            # Add metric overlay even when skipping YOLO processing
            frame_bgr = self._add_metric_overlay(frame_bgr, 0.0, motion_pixels)

            # Clean up intermediate objects before returning
            del fg_mask
            if 'frame_diff' in locals():
                del frame_diff
            if 'thresh' in locals():
                del thresh

            return frame_bgr  # Return original frame

        # Run YOLO inference with tracking
        inference_start = time.time()

        # Check if model exists (might be None after emergency cleanup)
        if self.model is None:
            print("⚠️  YOLO model not available, skipping detection")
            # Return original frame with overlay
            annotated_frame = self._add_metric_overlay(frame_bgr, 0.0, motion_pixels)
            return annotated_frame

        # Run inference on downscaled frame
        results = self.model.track(
            inference_frame,
            device=self.device,
            conf=self.confidence,
            verbose=False,
            persist=True
        )
        inference_time_ms = (time.time() - inference_start) * 1000

        # Annotate frame with detections (on inference-sized frame first)
        annotated_frame = results[0].plot()

        # Scale annotated frame back to original size for display
        if self.inference_size is not None:
            annotated_frame = cv2.resize(annotated_frame, (original_width, original_height),
                                        interpolation=cv2.INTER_LINEAR)

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

                # Get center point of bounding box (in inference resolution)
                x1, y1, x2, y2 = box

                # Filter out detections in masked ROI areas
                if self.roi_mask is not None:
                    # Check if detection center is in masked area (black = 0 = ignore)
                    center_x_inf = int((x1 + x2) / 2)
                    center_y_inf = int((y1 + y2) / 2)
                    # Ensure coordinates are within bounds
                    if (0 <= center_y_inf < self.roi_mask.shape[0] and
                        0 <= center_x_inf < self.roi_mask.shape[1]):
                        # If center is in masked area (black/0), skip this detection
                        if self.roi_mask[center_y_inf, center_x_inf] == 0:
                            continue

                # Scale coordinates back to original frame resolution
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                x2_orig = x2 * scale_x
                y2_orig = y2 * scale_y
                center_x = (x1_orig + x2_orig) / 2
                center_y = (y1_orig + y2_orig) / 2
                current_time = time.time()

                # Add to InfluxDB detections (using original resolution coords)
                influx_detections.append({
                    "class_name": self.CLASS_NAMES.get(int(cls), "unknown"),
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x1_orig),
                        "y1": float(y1_orig),
                        "x2": float(x2_orig),
                        "y2": float(y2_orig)
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
        if self.influx_logger:
            try:
                source_name = "srt_stream"
                queue_depth = self.frame_queue.qsize() if hasattr(self, 'frame_queue') else 0
                memory_usage_mb = self._get_memory_usage()
                gstreamer_buffers = self._get_gstreamer_buffer_info()
                self.influx_logger.log_detections(
                    detections=influx_detections,
                    source_name=source_name,
                    frame_number=self.frame_count,
                    processing_time_ms=inference_time_ms if influx_detections else 0.0,
                    motion_pixels=motion_pixels,
                    queue_depth=queue_depth,
                    gstreamer_buffers=gstreamer_buffers,
                    memory_usage_mb=memory_usage_mb
                )
            except Exception as e:
                print(f"Error logging to InfluxDB: {e}")

        # Save screenshots for right-to-left vehicles
        for screenshot_path in screenshots_to_save:
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Screenshot saved: {screenshot_path}")

        # Add metric overlay to the frame
        annotated_frame = self._add_metric_overlay(annotated_frame, inference_time_ms, motion_pixels)

        # Visually black out the masked ROI area on display
        if self.roi_mask is not None:
            # Scale mask to original frame size if needed
            if self.inference_size is not None:
                display_mask = cv2.resize(self.roi_mask, (original_width, original_height),
                                        interpolation=cv2.INTER_NEAREST)
            else:
                display_mask = self.roi_mask

            # Black out ignored areas (where mask is 0)
            annotated_frame[display_mask == 0] = 0

        # Explicitly delete large objects to help garbage collector
        # This is critical when processing frames at high rates
        del results
        del fg_mask
        del inference_frame  # Delete the downscaled frame
        if 'frame_diff' in locals():
            del frame_diff
        if 'thresh' in locals():
            del thresh
        if 'boxes' in locals():
            del boxes
        if 'track_ids' in locals():
            del track_ids
        if 'classes' in locals():
            del classes
        if 'confidences' in locals():
            del confidences

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
                # Force a waitKey to ensure window is created and displayed
                cv2.waitKey(1)
                print(f"✓ Display window created successfully: {window_name}")
                print("  Metric overlay will be visible in top-left corner")
            except Exception as e:
                print(f"⚠ Could not create display window: {e}")
                print("  This may be due to missing GUI libraries or running in a headless environment")
                print("  Switching to headless mode")
                self.headless = True
        else:
            print("Running in headless mode (no display window)")

        # Open SRT stream
        connection_successful = False
        if self.use_gstreamer:
            print(f"Connecting to SRT stream using GStreamer: {self.srt_uri}")
            try:
                self._setup_gstreamer()
                print("GStreamer pipeline started - waiting for connection...")
                # Wait up to 10 seconds for first frame
                connection_successful = self._wait_for_stream_connection(timeout=10.0)
            except Exception as e:
                print(f"GStreamer SRT setup failed: {e}, falling back to OpenCV")
                self.use_gstreamer = False

        if not connection_successful and self.use_opencv:
            print(f"Connecting to SRT stream using OpenCV: {self.srt_uri}")
            try:
                self._setup_opencv()
                print("OpenCV SRT setup completed - waiting for connection...")
                connection_successful = self._wait_for_stream_connection(timeout=10.0)
            except Exception as e:
                print(f"OpenCV SRT setup failed: {e}, falling back to FFmpeg")
                self.use_opencv = False

        if not connection_successful:
            print(f"Connecting to SRT stream using FFmpeg: {self.srt_uri}")
            try:
                self._setup_ffmpeg()
                print("FFmpeg SRT setup completed - waiting for connection...")
                connection_successful = self._wait_for_stream_connection(timeout=10.0)
            except Exception as e:
                print(f"All SRT connection methods failed. Last error: {e}")
                print("Please check:")
                print("1. SRT stream is running and accessible")
                print("2. Network connectivity to the SRT server")
                print("3. Correct SRT URI format: srt://host:port")
                print("4. Firewall settings allow SRT traffic (default port 4201)")
                return

        if connection_successful:
            print("✓ SRT stream connected successfully")
            print("Press Ctrl+C to stop")
        else:
            print("⚠ SRT stream connection timeout - no frames received")
            print("The stream may be running but not sending data")
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

                # Clean up frame references immediately after use
                del annotated_frame
                if 'frame_bgr' in locals():
                    del frame_bgr

                # Memory management and progress reporting
                # Reduced from every 20 frames to every 50 frames (~8 seconds at 6 FPS)
                # Less aggressive now that memory leaks are fixed
                if self.frame_count % 50 == 0:
                    print(f"Processed {self.frame_count} frames")
                    self._aggressive_memory_cleanup()

                # Additional: Reset background subtractor every 100 frames to prevent history accumulation
                # This is in addition to the reset in _aggressive_memory_cleanup
                if self.frame_count % 100 == 0:
                    self.back_sub = cv2.createBackgroundSubtractorMOG2(
                        history=100, varThreshold=50, detectShadows=True
                    )
                    # Removed verbose logging - already happens in regular cleanup

                # Check memory usage and trigger emergency cleanup if needed
                current_memory = self._get_memory_usage()
                # Increased threshold from 800MB to 2400MB after optimizations
                # Normal baseline is now ~1200-1400MB with 1280x720 inference
                if current_memory > 2400:  # If over 2400MB, trigger emergency cleanup
                    print(f"⚠️  High memory usage detected: {current_memory:.1f}MB")
                    self._emergency_memory_cleanup()

                # Additional cleanup every 200 frames (~3.5 minutes at 6 FPS)
                if self.frame_count % 200 == 0:
                    print("Performing deep memory cleanup...")
                    # Print memory breakdown for diagnostics
                    memory_breakdown = self._estimate_object_memory()
                    print(f"Memory breakdown: {memory_breakdown}")
                    # Force Python garbage collection multiple times
                    for _ in range(5):
                        gc.collect()
                    # Clear any cached compiled functions
                    import sys
                    if hasattr(sys, '_clear_type_cache'):
                        sys._clear_type_cache()
                    print("Deep cleanup completed")

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
