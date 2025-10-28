"""
GUI dashboard creation module.
"""

import logging
import cv2
import numpy as np
from typing import List
from eyeball.config import (
    DASHBOARD_WIDTH,
    DASHBOARD_HEIGHT,
    VIDEO_FRAME_HEIGHT,
    METRICS_COLUMN_WIDTH,
    VIDEO_COLUMN_WIDTH
)


class GUIDashboard:
    """Handles creation of the GUI dashboard display."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def create_dashboard(self, processing_time_ms: float, motion_pixels: float,
                         fg_mask: np.ndarray, annotated_frame: np.ndarray,
                         queue_depth: int, queue_max: int, memory_usage_mb: float,
                         frame_count: int, tracked_objects_count: int,
                         device_type: str, inference_device: str,
                         influx_log_lines: List[str]) -> np.ndarray:
        """Create a 2-column dashboard layout with video frames and metrics."""

        # Dashboard dimensions
        dashboard_width = DASHBOARD_WIDTH
        dashboard_height = DASHBOARD_HEIGHT
        left_column_width = VIDEO_COLUMN_WIDTH  # 2/3 width for video frames
        right_column_width = METRICS_COLUMN_WIDTH  # 1/3 width for metrics

        # Create dashboard canvas
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)  # Dark background

        # Left column: Video frames (stacked vertically)
        video_frame_height = VIDEO_FRAME_HEIGHT  # Half the height for each frame

        # Resize annotated frame to fit
        if annotated_frame.shape[1] != left_column_width:
            aspect_ratio = annotated_frame.shape[0] / annotated_frame.shape[1]
            resized_height = int(left_column_width * aspect_ratio)
            if resized_height > video_frame_height:
                scale = video_frame_height / annotated_frame.shape[0]
                new_width = int(annotated_frame.shape[1] * scale)
                annotated_frame = cv2.resize(annotated_frame, (new_width, video_frame_height))
                x_offset = (left_column_width - new_width) // 2
                dashboard[0:video_frame_height, x_offset:x_offset+new_width] = annotated_frame
            else:
                annotated_frame = cv2.resize(annotated_frame, (left_column_width, resized_height))
                y_offset = (video_frame_height - resized_height) // 2
                dashboard[y_offset:y_offset+resized_height, 0:left_column_width] = annotated_frame

        # Resize and display fg_mask (subtractor frame)
        if fg_mask is not None:
            fg_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            fg_display[fg_mask > 0] = [0, 255, 0]  # Green for motion

            if fg_display.shape[1] != left_column_width:
                aspect_ratio = fg_display.shape[0] / fg_display.shape[1]
                resized_height = int(left_column_width * aspect_ratio)
                if resized_height > video_frame_height:
                    scale = video_frame_height / fg_display.shape[0]
                    new_width = int(fg_display.shape[1] * scale)
                    fg_display = cv2.resize(fg_display, (new_width, video_frame_height))
                    x_offset = (left_column_width - new_width) // 2
                    dashboard[video_frame_height:video_frame_height*2, x_offset:x_offset+new_width] = fg_display
                else:
                    fg_display = cv2.resize(fg_display, (left_column_width, resized_height))
                    y_offset = video_frame_height + (video_frame_height - resized_height) // 2
                    dashboard[y_offset:y_offset+resized_height, 0:left_column_width] = fg_display

        # Right column: Dashboard metrics
        metrics_x = left_column_width + 20
        metrics_y_start = 50

        # Check if POI (Person of Interest) detected
        poi_detected = tracked_objects_count > 0

        # Dashboard metrics with labels and values
        metrics = [
            ("POI Detected", "YES" if poi_detected else "NO", (0, 255, 0) if poi_detected else (0, 0, 255)),
            ("Processing Time", f"{processing_time_ms:.1f} ms", (255, 255, 255)),
            ("Frame Count", str(frame_count), (255, 255, 255)),
            ("Objects Detected", str(tracked_objects_count), (255, 255, 255)),
            ("Motion Detected", f"{motion_pixels:.1f}%", (255, 255, 255)),
            ("Device", device_type, (255, 255, 255)),
            ("Inference Device", inference_device.upper(), (255, 255, 255)),
            ("Memory Usage", f"{memory_usage_mb:.1f} MB", (255, 255, 255)),
        ]

        # Split metrics into two columns (4 each)
        left_metrics = metrics[:4]
        right_metrics = metrics[4:]

        metrics_x_left = left_column_width + 20
        metrics_x_right = metrics_x_left + 250
        y_pos = metrics_y_start

        # Draw left column
        for label, value, color in left_metrics:
            cv2.putText(dashboard, label, (metrics_x_left, y_pos), self.font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            y_pos += 25
            cv2.putText(dashboard, value, (metrics_x_left, y_pos), self.font, 0.8, color, 2, cv2.LINE_AA)
            y_pos += 60

        # Reset y_pos for right column
        y_pos = metrics_y_start

        # Draw right column
        for label, value, color in right_metrics:
            cv2.putText(dashboard, label, (metrics_x_right, y_pos), self.font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            y_pos += 25
            cv2.putText(dashboard, value, (metrics_x_right, y_pos), self.font, 0.8, color, 2, cv2.LINE_AA)
            y_pos += 60

        # Draw queue bar graph
        queue_label_y = y_pos + 20
        cv2.putText(dashboard, "Queue Depth", (metrics_x, queue_label_y), self.font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        bar_x = metrics_x
        bar_y = queue_label_y + 10
        bar_width = 300
        bar_height = 30

        cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        fill_width = int((queue_depth / queue_max) * bar_width) if queue_max > 0 else 0
        cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

        cv2.putText(dashboard, f"{queue_depth}/{queue_max}", (bar_x + bar_width + 10, bar_y + 20), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Influx log display below queue
        influx_y_start = bar_y + bar_height + 20
        available_height = dashboard_height - influx_y_start - 10
        line_height = 20
        max_lines = available_height // line_height

        cv2.rectangle(dashboard, (metrics_x - 5, influx_y_start - 5), (dashboard_width - 10, dashboard_height - 5), (100, 100, 100), 2)

        for i, line in enumerate(influx_log_lines[:max_lines]):
            y = influx_y_start + i * line_height + 15
            display_line = line[:60] + "..." if len(line) > 60 else line
            cv2.putText(dashboard, display_line, (metrics_x, y), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return dashboard