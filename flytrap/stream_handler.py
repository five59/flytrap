"""
Stream handling module for SRT video streams.
"""

import logging
import cv2
import time
import queue
import threading
import select
import subprocess
import numpy as np

# GStreamer imports
try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib

    GSTREAMER_AVAILABLE = True
except Exception:
    GSTREAMER_AVAILABLE = False


class StreamHandler:
    """Handles SRT stream connection and frame acquisition."""

    def __init__(
        self, srt_uri: str, frame_queue: queue.Queue, frame_skip_interval: int = 1
    ):
        self.logger = logging.getLogger(__name__)
        self.srt_uri = srt_uri
        self.frame_queue = frame_queue
        self.frame_skip_interval = frame_skip_interval
        self.use_gstreamer = GSTREAMER_AVAILABLE
        self.use_opencv = not self.use_gstreamer

        # GStreamer components
        self.pipeline = None
        self.gstreamer_thread = None
        self.frame_skip_counter = 0

        # OpenCV components
        self.cap = None

        # FFmpeg components
        self.ffmpeg_proc = None

        # Frame info
        self.frame_width = None
        self.frame_midpoint_x = None

    def setup_stream(self):
        """Set up the SRT stream using available methods."""
        if self.use_gstreamer:
            try:
                self._setup_gstreamer()
                return True
            except Exception as e:
                self.logger.warning(
                    f"GStreamer SRT setup failed: {e}, falling back to OpenCV"
                )
                self.use_gstreamer = False

        if self.use_opencv:
            try:
                self._setup_opencv()
                return True
            except Exception as e:
                print(f"OpenCV SRT setup failed: {e}, falling back to FFmpeg")
                self.use_opencv = False

        try:
            self._setup_ffmpeg()
            return True
        except Exception as e:
            print(f"All SRT connection methods failed. Last error: {e}")
            return False

    def _setup_gstreamer(self):
        """Set up GStreamer pipeline for SRT streaming."""
        Gst.init(None)

        # Pipeline for SRT stream with auto-detection
        pipeline_str = (
            f'srtsrc uri="{self.srt_uri}" ! '
            "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink"
        )

        self.pipeline = Gst.parse_launch(pipeline_str)

        # Get appsink
        appsink = self.pipeline.get_by_name("sink")
        appsink.set_property("emit-signals", True)
        appsink.connect("new-sample", self._on_new_sample)

        # Set up bus for messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)

        # Set playing
        self.pipeline.set_state(Gst.State.PLAYING)

        # Start GLib loop in thread
        self.gstreamer_thread = threading.Thread(target=self._run_gst_loop, daemon=True)
        self.gstreamer_thread.start()

    def _on_new_sample(self, sink):
        """Callback for new video sample from GStreamer."""
        self.frame_skip_counter += 1

        # Skip frames to achieve target detection FPS
        if self.frame_skip_counter % self.frame_skip_interval != 0:
            self._consume_sample(sink)
            return Gst.FlowReturn.OK

        sample = sink.emit("pull-sample")
        self._process_gstreamer_sample(sample)
        return Gst.FlowReturn.OK

    def _consume_sample(self, sink):
        """Consume sample without processing to clear pipeline buffer."""
        sample = sink.emit("pull-sample")
        del sample

    def _process_gstreamer_sample(self, sample):
        """Process a GStreamer sample into a frame."""
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        self._update_frame_dimensions(width)

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            self._convert_and_queue_frame(map_info, height, width)
            buffer.unmap(map_info)

        # Clean up
        del buffer, sample, caps, structure

    def _update_frame_dimensions(self, width):
        """Update frame dimensions if not already set."""
        if self.frame_width is None:
            self.frame_width = width
            self.frame_midpoint_x = self.frame_width / 2

    def _convert_and_queue_frame(self, map_info, height, width):
        """Convert buffer data to frame and add to queue."""
        frame_bgr = np.frombuffer(map_info.data, dtype=np.uint8).reshape(
            (height, width, 3)
        )
        try:
            self.frame_queue.put(frame_bgr.copy(), timeout=0.1)
        except queue.Full:
            pass  # Discard if queue full
        finally:
            del frame_bgr

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
        self.frame_midpoint_x = self.frame_width / 2

    def _setup_ffmpeg(self):
        """Set up FFmpeg subprocess for SRT streaming (fallback)."""
        # Use ffmpeg to decode SRT and pipe to stdout
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            self.srt_uri,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",  # Disable audio
            "-sn",  # Disable subtitles
            "-dn",  # Disable data
            "pipe:1",  # Output to stdout
        ]

        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )
        # Give ffmpeg a moment to start
        time.sleep(2)
        if self.ffmpeg_proc.poll() is not None:
            raise RuntimeError("ffmpeg process failed to start")

    def get_frame(self, frame_skip_counter: int, frame_skip_interval: int):
        """Get next frame from the stream."""
        if self.use_gstreamer:
            return self._get_frame_gstreamer()
        elif self.use_opencv:
            return self._get_frame_opencv(frame_skip_counter, frame_skip_interval)
        else:
            return self._get_frame_ffmpeg(frame_skip_counter, frame_skip_interval)

    def _get_frame_gstreamer(self):
        """Get frame from GStreamer pipeline."""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def _get_frame_opencv(self, frame_skip_counter: int, frame_skip_interval: int):
        """Get frame from OpenCV VideoCapture."""
        assert self.cap is not None, "OpenCV capture not initialized"
        ret, frame_bgr = self.cap.read()
        if not ret:
            print("Failed to read frame from OpenCV SRT stream")
            time.sleep(1)
            return None

        # Skip frames to achieve target detection FPS
        if frame_skip_counter % frame_skip_interval != 0:
            return None
        return frame_bgr

    def _get_frame_ffmpeg(self, frame_skip_counter: int, frame_skip_interval: int):
        """Get frame from FFmpeg subprocess."""
        if not self._ensure_ffmpeg_running():
            return None

        frame_data = self._read_ffmpeg_frame()
        if frame_data is None:
            return None

        frame_bgr = self._process_ffmpeg_frame_data(frame_data)

        # Skip frames to achieve target detection FPS
        if frame_skip_counter % frame_skip_interval != 0:
            return None

        return frame_bgr

    def _ensure_ffmpeg_running(self):
        """Ensure FFmpeg process is running, restart if needed."""
        if self.ffmpeg_proc.poll() is not None:
            print("ffmpeg process exited, attempting to restart...")
            try:
                self._setup_ffmpeg()
            except Exception as e:
                print(f"Failed to restart ffmpeg: {e}")
                time.sleep(5)
                return False
        return True

    def _read_ffmpeg_frame(self):
        """Read raw frame data from FFmpeg stdout."""
        # Check if data is available with timeout
        ready, _, _ = select.select([self.ffmpeg_proc.stdout], [], [], 1.0)
        if not ready:
            return None  # No data available

        # Read raw BGR frame from ffmpeg pipe
        frame_width = 1920  # Assumed
        frame_height = 1080
        frame_size = frame_width * frame_height * 3  # BGR24 = 3 bytes per pixel
        frame_data = self.ffmpeg_proc.stdout.read(frame_size)

        if len(frame_data) != frame_size:
            print("Failed to read complete frame from SRT stream")
            time.sleep(0.1)
            return None

        return frame_data

    def _process_ffmpeg_frame_data(self, frame_data):
        """Convert FFmpeg frame data to numpy array."""
        frame_height = 1080
        frame_width = 1920
        return np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (frame_height, frame_width, 3)
        )

    def cleanup(self):
        """Clean up stream resources."""
        if self.use_gstreamer:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            # Thread is daemon, will stop
        elif self.use_opencv:
            if self.cap:
                self.cap.release()
        else:
            if self.ffmpeg_proc:
                self.ffmpeg_proc.terminate()
                self.ffmpeg_proc.wait()
