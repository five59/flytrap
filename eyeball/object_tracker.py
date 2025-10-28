"""
Object tracking and logging module.
"""

import cv2
import time
import os
from datetime import datetime
from typing import Dict, Optional, Tuple


class ObjectTracker:
    """Handles object tracking, direction detection, and logging."""

    VEHICLE_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(self, road_width_feet: float, log_file: str, screenshots_dir: str, influx_logger=None):
        self.road_width_feet = road_width_feet
        self.log_file = log_file
        self.screenshots_dir = screenshots_dir
        self.influx_logger = influx_logger

        # Tracking state
        self.tracked_objects: Dict = {}
        self.frame_width = None
        self.frame_midpoint_x = None

        # Influx log lines for GUI
        self.influx_log_lines = []

    def set_frame_info(self, frame_width: int):
        """Set frame dimensions for tracking calculations."""
        self.frame_width = frame_width
        self.frame_midpoint_x = self.frame_width / 2

    def update_tracks(self, detections: list, current_time: float):
        """Update object tracks with new detections."""
        for detection in detections:
            track_id = detection["track_id"]
            cls = detection["class_id"]

            # Only track vehicles
            if cls not in self.VEHICLE_CLASSES:
                continue

            bbox = detection["bbox"]
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2

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
                self.tracked_objects[track_id]['class'] = cls
                self.tracked_objects[track_id]['positions'].append((center_x, center_y, current_time))

                # Keep only last 30 positions
                if len(self.tracked_objects[track_id]['positions']) > 30:
                    self.tracked_objects[track_id]['positions'].pop(0)

                # Check if object has crossed the midpoint
                if not self.tracked_objects[track_id]['crossed_midpoint'] and len(self.tracked_objects[track_id]['positions']) >= 2:
                    prev_x = self.tracked_objects[track_id]['positions'][-2][0]
                    curr_x = center_x

                    if (prev_x < self.frame_midpoint_x <= curr_x) or (prev_x > self.frame_midpoint_x >= curr_x):
                        self.tracked_objects[track_id]['crossed_midpoint'] = True
                        self.tracked_objects[track_id]['midpoint_cross_position'] = len(self.tracked_objects[track_id]['positions']) - 1

                # Log when object crosses midpoint
                if self.tracked_objects[track_id]['crossed_midpoint'] and not self.tracked_objects[track_id]['logged']:
                    self._log_tracked_object(track_id, cls)

    def _log_tracked_object(self, track_id: int, cls: int) -> Optional[str]:
        """Log a tracked object that has crossed the midpoint."""
        positions = self.tracked_objects[track_id]['positions']
        if len(positions) < 10:
            return None

        start_x, start_y, start_time = positions[0]
        end_x, end_y, end_time = positions[-1]
        displacement_pixels = end_x - start_x
        time_elapsed = end_time - start_time

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

        # Prepare screenshot
        time_str = datetime.now().strftime("%m%d_%H%M")
        screenshot_filename = f"{self.screenshots_dir}/{time_str}-{vehicle_type}-{track_id}.jpg"

        # Log to file
        with open(self.log_file, 'a') as f:
            log_entry = f"{timestamp} | Track ID: {track_id} | Type: {vehicle_type} | Direction: {direction} | Speed: {speed_mph:.1f} mph"
            if screenshot_filename:
                log_entry += f" | Screenshot: {screenshot_filename}"
            f.write(log_entry + "\n")

        self.tracked_objects[track_id]['logged'] = True

        # Add to GUI log
        log_timestamp = datetime.now().strftime("%H:%M:%S")
        self.influx_log_lines.insert(0, f"{log_timestamp}: Detected {vehicle_type}. Motion: {direction}")
        if len(self.influx_log_lines) > 150:
            self.influx_log_lines.pop()

        print(f"Logged: {vehicle_type} {direction} at {speed_mph:.1f} mph - {timestamp}")

        # Log to InfluxDB
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

    def get_tracked_count(self) -> int:
        """Get the number of currently tracked objects."""
        return len(self.tracked_objects)

    def cleanup_old_tracks(self, current_time: float):
        """Remove tracks that haven't been updated recently."""
        to_remove = []
        for track_id, data in self.tracked_objects.items():
            if data['positions'] and current_time - data['positions'][-1][2] > 15:
                to_remove.append(track_id)
            elif len(data['positions']) > 20:
                data['positions'] = data['positions'][-10:]

        for track_id in to_remove:
            del self.tracked_objects[track_id]

        return len(to_remove)