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
    VIDEO_COLUMN_WIDTH,
)


class GUIDashboard:
    """Handles creation of the GUI dashboard display."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def create_dashboard(
        self,
        processing_time_ms: float,
        motion_pixels: float,
        fg_mask: np.ndarray,
        annotated_frame: np.ndarray,
        queue_depth: int,
        queue_max: int,
        memory_usage_mb: float,
        frame_count: int,
        tracked_objects_count: int,
        device_type: str,
        inference_device: str,
        influx_log_lines: List[str],
    ) -> np.ndarray:
        """Create a 2-column dashboard layout with video frames and metrics."""
        dashboard = self._setup_dashboard_canvas()
        self._add_video_frames(dashboard, annotated_frame, fg_mask)
        self._add_metrics_display(
            dashboard,
            processing_time_ms,
            motion_pixels,
            frame_count,
            tracked_objects_count,
            device_type,
            inference_device,
            memory_usage_mb,
        )
        self._add_queue_bar(dashboard, queue_depth, queue_max)
        self._add_influx_logs(dashboard, influx_log_lines)
        return dashboard

    def _setup_dashboard_canvas(self) -> np.ndarray:
        """Create the base dashboard canvas with dark background."""
        dashboard = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)  # Dark background
        return dashboard

    def _add_video_frames(
        self, dashboard: np.ndarray, annotated_frame: np.ndarray, fg_mask: np.ndarray
    ) -> None:
        """Add video frames to the left column of the dashboard."""
        video_frame_height = VIDEO_FRAME_HEIGHT

        # Add annotated frame to top half
        self._place_resized_frame(
            dashboard, annotated_frame, VIDEO_COLUMN_WIDTH, 0, video_frame_height
        )

        # Add foreground mask to bottom half if available
        if fg_mask is not None:
            fg_display = self._prepare_fg_mask_display(fg_mask)
            self._place_resized_frame(
                dashboard,
                fg_display,
                VIDEO_COLUMN_WIDTH,
                video_frame_height,
                video_frame_height * 2,
            )

    def _prepare_fg_mask_display(self, fg_mask: np.ndarray) -> np.ndarray:
        """Convert foreground mask to displayable BGR format with green motion highlighting."""
        fg_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        fg_display[fg_mask > 0] = [0, 255, 0]  # Green for motion
        return fg_display

    def _place_resized_frame(
        self,
        dashboard: np.ndarray,
        frame: np.ndarray,
        target_width: int,
        y_start: int,
        y_end: int,
    ) -> None:
        """Resize and place a frame in the dashboard maintaining aspect ratio."""
        if frame.shape[1] == target_width:
            dashboard[y_start:y_end, 0:target_width] = frame
            return

        target_height = y_end - y_start
        aspect_ratio = frame.shape[0] / frame.shape[1]
        resized_height = int(target_width * aspect_ratio)

        if resized_height > target_height:
            self._fit_to_height(dashboard, frame, target_width, target_height, y_start, y_end)
        else:
            self._fit_to_width(dashboard, frame, target_width, resized_height, y_start, target_height)

    def _fit_to_height(
        self,
        dashboard: np.ndarray,
        frame: np.ndarray,
        target_width: int,
        target_height: int,
        y_start: int,
        y_end: int,
    ) -> None:
        """Fit frame to target height and center horizontally."""
        scale = target_height / frame.shape[0]
        new_width = int(frame.shape[1] * scale)
        resized_frame = cv2.resize(frame, (new_width, target_height))
        x_offset = (target_width - new_width) // 2
        dashboard[y_start:y_end, x_offset : x_offset + new_width] = resized_frame

    def _fit_to_width(
        self,
        dashboard: np.ndarray,
        frame: np.ndarray,
        target_width: int,
        resized_height: int,
        y_start: int,
        target_height: int,
    ) -> None:
        """Fit frame to target width and center vertically."""
        resized_frame = cv2.resize(frame, (target_width, resized_height))
        y_offset = y_start + (target_height - resized_height) // 2
        dashboard[y_offset : y_offset + resized_height, 0:target_width] = resized_frame

    def _add_metrics_display(
        self,
        dashboard: np.ndarray,
        processing_time_ms: float,
        motion_pixels: float,
        frame_count: int,
        tracked_objects_count: int,
        device_type: str,
        inference_device: str,
        memory_usage_mb: float,
    ) -> None:
        """Add metrics display to the right column of the dashboard."""
        metrics = self._prepare_metrics_data(
            processing_time_ms,
            motion_pixels,
            frame_count,
            tracked_objects_count,
            device_type,
            inference_device,
            memory_usage_mb,
        )

        # Draw both columns and return the final Y position
        final_y_pos = self._draw_metrics_columns(dashboard, metrics)
        # Store the position for queue bar to use
        self._metrics_bottom_y = final_y_pos

    def _draw_metrics_columns(self, dashboard: np.ndarray, metrics: List[tuple]) -> int:
        """Draw both columns of metrics and return the bottom Y position."""
        left_column_x = VIDEO_COLUMN_WIDTH + 20
        right_column_x = VIDEO_COLUMN_WIDTH + 270

        # Draw left column
        left_bottom_y = self._draw_metrics_column(dashboard, metrics[:4], left_column_x)

        # Draw right column
        right_bottom_y = self._draw_metrics_column(dashboard, metrics[4:], right_column_x)

        # Return the maximum bottom position
        return max(left_bottom_y, right_bottom_y)

    def _draw_metrics_column(
        self, dashboard: np.ndarray, metrics: List[tuple], x_pos: int
    ) -> int:
        """Draw a column of metrics on the dashboard and return bottom Y position."""
        y_pos = 50
        for label, value, color in metrics:
            self._draw_metric_pair(dashboard, label, value, color, x_pos, y_pos)
            y_pos += 85  # Space for label + value + gap
        return y_pos

    def _prepare_metrics_data(
        self,
        processing_time_ms: float,
        motion_pixels: float,
        frame_count: int,
        tracked_objects_count: int,
        device_type: str,
        inference_device: str,
        memory_usage_mb: float,
    ) -> List[tuple]:
        """Prepare metrics data with labels, values, and colors."""
        poi_detected = tracked_objects_count > 0
        return [
            (
                "POI Detected",
                "YES" if poi_detected else "NO",
                (0, 255, 0) if poi_detected else (0, 0, 255),
            ),
            ("Processing Time", f"{processing_time_ms:.1f} ms", (255, 255, 255)),
            ("Frame Count", str(frame_count), (255, 255, 255)),
            ("Objects Detected", str(tracked_objects_count), (255, 255, 255)),
            ("Motion Detected", f"{motion_pixels:.1f}%", (255, 255, 255)),
            ("Device", device_type, (255, 255, 255)),
            ("Inference Device", inference_device.upper(), (255, 255, 255)),
            ("Memory Usage", f"{memory_usage_mb:.1f} MB", (255, 255, 255)),
        ]

    def _draw_metric_pair(
        self,
        dashboard: np.ndarray,
        label: str,
        value: str,
        color: tuple,
        x_pos: int,
        y_pos: int,
    ) -> None:
        """Draw a label-value pair for metrics display."""
        cv2.putText(
            dashboard,
            label,
            (x_pos, y_pos),
            self.font,
            0.5,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            dashboard, value, (x_pos, y_pos + 25), self.font, 0.8, color, 2, cv2.LINE_AA
        )

    def _add_queue_bar(
        self, dashboard: np.ndarray, queue_depth: int, queue_max: int
    ) -> None:
        """Add queue depth visualization bar to the dashboard."""
        metrics_x = VIDEO_COLUMN_WIDTH + 20
        queue_label_y = getattr(self, '_metrics_bottom_y', 50 + (8 * 85)) + 20  # After metrics + spacing

        self._draw_queue_label(dashboard, metrics_x, queue_label_y)
        self._draw_queue_visualization(dashboard, metrics_x, queue_label_y, queue_depth, queue_max)

    def _draw_queue_label(self, dashboard: np.ndarray, x: int, y: int) -> None:
        """Draw the queue depth label."""
        cv2.putText(
            dashboard, "Queue Depth", (x, y), self.font, 0.5, (150, 150, 150), 1, cv2.LINE_AA
        )

    def _draw_queue_visualization(
        self, dashboard: np.ndarray, x: int, label_y: int, queue_depth: int, queue_max: int
    ) -> None:
        """Draw the queue depth bar and text."""
        bar_x, bar_y = x, label_y + 10
        bar_width, bar_height = 300, 30

        # Draw background and fill
        self._draw_bar_background(dashboard, bar_x, bar_y, bar_width, bar_height)
        self._draw_bar_fill(dashboard, bar_x, bar_y, bar_width, bar_height, queue_depth, queue_max)
        self._draw_bar_border(dashboard, bar_x, bar_y, bar_width, bar_height)

        # Draw text label
        cv2.putText(
            dashboard,
            f"{queue_depth}/{queue_max}",
            (bar_x + bar_width + 10, bar_y + 20),
            self.font,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_bar_background(
        self, dashboard: np.ndarray, x: int, y: int, width: int, height: int
    ) -> None:
        """Draw the background of the progress bar."""
        cv2.rectangle(dashboard, (x, y), (x + width, y + height), (50, 50, 50), -1)

    def _draw_bar_fill(
        self,
        dashboard: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        queue_depth: int,
        queue_max: int,
    ) -> None:
        """Draw the filled portion of the progress bar."""
        fill_width = int((queue_depth / queue_max) * width) if queue_max > 0 else 0
        cv2.rectangle(dashboard, (x, y), (x + fill_width, y + height), (0, 255, 0), -1)

    def _draw_bar_border(
        self, dashboard: np.ndarray, x: int, y: int, width: int, height: int
    ) -> None:
        """Draw the border of the progress bar."""
        cv2.rectangle(dashboard, (x, y), (x + width, y + height), (255, 255, 255), 1)

    def _add_influx_logs(
        self, dashboard: np.ndarray, influx_log_lines: List[str]
    ) -> None:
        """Add InfluxDB log display to the bottom of the dashboard."""
        metrics_x = VIDEO_COLUMN_WIDTH + 20
        # Position after queue bar (which is after metrics)
        queue_bar_bottom = getattr(self, '_metrics_bottom_y', 50 + (8 * 85)) + 20 + 10 + 30
        influx_y_start = queue_bar_bottom + 20

        available_height = DASHBOARD_HEIGHT - influx_y_start - 10
        line_height = 20
        max_lines = available_height // line_height

        # Draw border
        cv2.rectangle(
            dashboard,
            (metrics_x - 5, influx_y_start - 5),
            (DASHBOARD_WIDTH - 10, DASHBOARD_HEIGHT - 5),
            (100, 100, 100),
            2,
        )

        # Draw log lines
        for i, line in enumerate(influx_log_lines[:max_lines]):
            y = influx_y_start + i * line_height + 15
            display_line = line[:60] + "..." if len(line) > 60 else line
            cv2.putText(
                dashboard,
                display_line,
                (metrics_x, y),
                self.font,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
