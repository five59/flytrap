"""
Frame processing module for motion detection and YOLO inference.
"""

import logging
import cv2
import time
import numpy as np
from typing import Optional, Tuple, List, Dict
from ultralytics import YOLO
from eyeball.config import (
    MOTION_THRESHOLD_PIXELS,
    MOTION_PIXEL_PERCENTAGE_MULTIPLIER,
    BACKGROUND_SUBTRACTOR_HISTORY,
    BACKGROUND_SUBTRACTOR_VAR_THRESHOLD
)


class FrameProcessor:
    """Handles frame preprocessing, motion detection, and YOLO inference."""

    def __init__(self, model_path: str, confidence: float, device: str, roi_box: Optional[Tuple[int, int, int, int]]):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.roi_box = roi_box

        # Initialize model
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Motion detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=BACKGROUND_SUBTRACTOR_HISTORY,
            varThreshold=BACKGROUND_SUBTRACTOR_VAR_THRESHOLD,
            detectShadows=True
        )
        self.motion_threshold = MOTION_THRESHOLD_PIXELS

        # Frame state
        self.prev_frame = None
        self.inference_size = None

    def process_frame(self, frame_bgr: np.ndarray, frame_count: int, frame_skip_interval: int) -> Tuple[np.ndarray, np.ndarray, bool, List[Dict], float, float]:
        """
        Process a single video frame with motion detection and optional YOLO inference.

        Args:
            frame_bgr: Input BGR frame
            frame_count: Current frame count
            frame_skip_interval: Frame skipping interval for FPS control

        Returns:
            Tuple of (annotated_frame, fg_mask, has_motion, detections, processing_time_ms, motion_pixels_percent)
        """
        # Store original frame dimensions for scaling detections back
        original_height, original_width = frame_bgr.shape[:2]

        # Crop to ROI box if provided
        if self.roi_box is not None:
            x1, y1, x2, y2 = self.roi_box
            x1, x2 = max(0, x1), min(original_width, x2)
            y1, y2 = max(0, y1), min(original_height, y2)
            frame_bgr = frame_bgr[y1:y2, x1:x2]
            original_height, original_width = frame_bgr.shape[:2]

        # Resize to 640px height maintaining aspect ratio
        target_height = 640
        current_height, current_width = frame_bgr.shape[:2]
        if current_height != target_height:
            aspect_ratio = current_width / current_height
            target_width = int(target_height * aspect_ratio)
            inference_frame = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            self.inference_size = (target_width, target_height)
        else:
            inference_frame = frame_bgr
            self.inference_size = (current_width, current_height)

        # Calculate scale factors
        scale_x = original_width / self.inference_size[0]
        scale_y = original_height / self.inference_size[1]

        # Motion detection
        fg_mask = self.back_sub.apply(inference_frame)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        motion_pixels_raw = cv2.countNonZero(fg_mask)
        total_pixels = inference_frame.shape[0] * inference_frame.shape[1]
        motion_pixels = (motion_pixels_raw / total_pixels) * MOTION_PIXEL_PERCENTAGE_MULTIPLIER if total_pixels > 0 else 0
        has_motion = motion_pixels_raw > self.motion_threshold

        # Frame differencing
        gray = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            _, thresh = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            motion_diff = cv2.countNonZero(thresh)
            has_motion = has_motion or (motion_diff > self.motion_threshold)
        self.prev_frame = gray

        # Skip YOLO if no significant motion
        if not has_motion and frame_count > 10:
            # Return resized frame and motion mask
            annotated_frame = cv2.resize(frame_bgr, self.inference_size, interpolation=cv2.INTER_LINEAR)
            return annotated_frame, fg_mask, has_motion, [], 0.0, motion_pixels

        # Run YOLO inference
        inference_start = time.time()
        results = self.model.track(
            inference_frame,
            device=self.device,
            conf=self.confidence,
            verbose=False,
            persist=True
        )
        inference_time_ms = (time.time() - inference_start) * 1000

        # Annotate frame
        annotated_frame = results[0].plot()

        # Process detections
        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                x1, y1, x2, y2 = box
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                x2_orig = x2 * scale_x
                y2_orig = y2 * scale_y

                detections.append({
                    "track_id": track_id,
                    "class_id": cls,
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x1_orig),
                        "y1": float(y1_orig),
                        "x2": float(x2_orig),
                        "y2": float(y2_orig)
                    }
                })

        return annotated_frame, fg_mask, has_motion, detections, inference_time_ms, motion_pixels

    def reset_background_subtractor(self):
        """Reset the background subtractor to clear accumulated history."""
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)