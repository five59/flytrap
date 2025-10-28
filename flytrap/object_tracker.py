"""
Object tracking and logging module.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
from flytrap.config import (
    TRACKING_POSITION_HISTORY_SIZE,
    TRACKING_CLEANUP_TIME_SECONDS,
    TRACKING_MAX_POSITIONS,
    TRACKING_MIN_POSITIONS_FOR_LOGGING,
    TRACKING_MIDPOINT_DISPLACEMENT_THRESHOLD,
    INFLUX_LOG_MAX_ENTRIES,
    FEET_PER_MILE,
    SECONDS_PER_HOUR,
)


class ObjectTracker:
    """Handles object tracking, direction detection, and logging."""

    VEHICLE_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck
    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        road_width_feet: float,
        log_file: str,
        screenshots_dir: str,
        influx_logger=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.road_width_feet = road_width_feet
        self.log_file = log_file
        self.screenshots_dir = screenshots_dir
        self.influx_logger = influx_logger

        # Tracking state
        self.tracked_objects: Dict = {}
        self.frame_width: Optional[int] = None
        self.frame_midpoint_x: Optional[float] = None

        # Influx log lines for GUI
        self.influx_log_lines: list[str] = []

    def set_frame_info(self, frame_width: int):
        """Set frame dimensions for tracking calculations."""
        self.frame_width = frame_width
        self.frame_midpoint_x = self.frame_width / 2

    def _should_track_detection(self, detection: dict) -> bool:
        """Check if a detection should be tracked (only vehicles)."""
        return detection["class_id"] in self.VEHICLE_CLASSES

    def _calculate_center(self, bbox: dict) -> tuple:
        """Calculate the center point of a bounding box."""
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2
        return center_x, center_y

    def _initialize_track(
        self,
        track_id: int,
        cls: int,
        center_x: float,
        center_y: float,
        current_time: float,
    ):
        """Initialize a new track for an object."""
        self.tracked_objects[track_id] = {
            "positions": [(center_x, center_y, current_time)],
            "class": cls,
            "logged": False,
            "crossed_midpoint": False,
            "midpoint_cross_position": None,
        }

    def _update_existing_track(
        self,
        track_id: int,
        cls: int,
        center_x: float,
        center_y: float,
        current_time: float,
    ):
        """Update an existing track with new position data."""
        self.tracked_objects[track_id]["class"] = cls
        self.tracked_objects[track_id]["positions"].append(
            (center_x, center_y, current_time)
        )

        # Keep only last N positions
        if (
            len(self.tracked_objects[track_id]["positions"])
            > TRACKING_POSITION_HISTORY_SIZE
        ):
            self.tracked_objects[track_id]["positions"].pop(0)

        # Check for midpoint crossing and log if needed
        if self._check_midpoint_crossing(track_id, center_x):
            self._log_tracked_object(track_id, cls)

    def _check_midpoint_crossing(self, track_id: int, center_x: float) -> bool:
        """Check if an object has crossed the frame midpoint."""
        if self.frame_midpoint_x is None:
            return False

        track_data = self.tracked_objects[track_id]

        if track_data["crossed_midpoint"] or len(track_data["positions"]) < 2:
            return False

        prev_x = track_data["positions"][-2][0]

        if (prev_x < self.frame_midpoint_x <= center_x) or (
            prev_x > self.frame_midpoint_x >= center_x
        ):
            track_data["crossed_midpoint"] = True
            track_data["midpoint_cross_position"] = len(track_data["positions"]) - 1
            return True

        return False

    def update_tracks(self, detections: list, current_time: float):
        """Update object tracks with new detections."""
        for detection in detections:
            if not self._should_track_detection(detection):
                continue

            track_id = detection["track_id"]
            cls = detection["class_id"]
            center_x, center_y = self._calculate_center(detection["bbox"])

            if track_id not in self.tracked_objects:
                self._initialize_track(track_id, cls, center_x, center_y, current_time)
            else:
                self._update_existing_track(
                    track_id, cls, center_x, center_y, current_time
                )

    def _log_tracked_object(self, track_id: int, cls: int) -> Optional[str]:
        """Log a tracked object that has crossed the midpoint."""
        positions = self.tracked_objects[track_id]["positions"]
        if len(positions) < TRACKING_MIN_POSITIONS_FOR_LOGGING:
            return None

        # Calculate movement data
        movement_data = self._calculate_movement_data(positions)
        if not movement_data:
            return None

        # Prepare logging data
        logging_data = self._prepare_logging_data(track_id, cls, movement_data)

        # Perform all logging operations
        self._perform_logging(track_id, logging_data)

        return logging_data["screenshot_filename"]

    def _calculate_movement_data(self, positions: list) -> Optional[dict]:
        """Calculate movement data from position history."""
        if self.frame_width is None:
            return None

        start_x, _, start_time = positions[0]
        end_x, _, end_time = positions[-1]
        displacement_pixels = end_x - start_x
        time_elapsed = end_time - start_time

        if (
            abs(displacement_pixels) <= TRACKING_MIDPOINT_DISPLACEMENT_THRESHOLD
            or time_elapsed <= 0
        ):
            return None

        direction = "left-to-right" if displacement_pixels > 0 else "right-to-left"

        # Calculate speed
        pixels_per_foot = self.frame_width / self.road_width_feet
        distance_feet = abs(displacement_pixels) / pixels_per_foot
        speed_fps = distance_feet / time_elapsed
        speed_mph = speed_fps * SECONDS_PER_HOUR / FEET_PER_MILE

        return {
            "direction": direction,
            "speed_mph": speed_mph,
            "displacement_pixels": displacement_pixels,
            "time_elapsed": time_elapsed,
        }

    def _prepare_logging_data(
        self, track_id: int, cls: int, movement_data: dict
    ) -> dict:
        """Prepare all data needed for logging."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        vehicle_type = self.CLASS_NAMES.get(cls, "object")

        time_str = datetime.now().strftime("%m%d_%H%M")
        screenshot_filename = (
            f"{self.screenshots_dir}/{time_str}-{vehicle_type}-{track_id}.jpg"
        )

        return {
            "timestamp": timestamp,
            "vehicle_type": vehicle_type,
            "track_id": track_id,
            "screenshot_filename": screenshot_filename,
            **movement_data,
        }

    def _perform_logging(self, track_id: int, logging_data: dict):
        """Perform all logging operations (file, GUI, InfluxDB)."""
        self._log_to_file(logging_data)
        self._log_to_gui(logging_data)
        self._log_to_influx(logging_data)
        self.tracked_objects[track_id]["logged"] = True

        print(
            f"Logged: {logging_data['vehicle_type']} {logging_data['direction']} "
            f"at {logging_data['speed_mph']:.1f} mph - {logging_data['timestamp']}"
        )

    def _log_to_file(self, logging_data: dict):
        """Log tracking data to file."""
        with open(self.log_file, "a") as f:
            log_entry = (
                f"{logging_data['timestamp']} | Track ID: {logging_data['track_id']} | "
                f"Type: {logging_data['vehicle_type']} | Direction: {logging_data['direction']} | "
                f"Speed: {logging_data['speed_mph']:.1f} mph"
            )
            if logging_data["screenshot_filename"]:
                log_entry += f" | Screenshot: {logging_data['screenshot_filename']}"
            f.write(log_entry + "\n")

    def _log_to_gui(self, logging_data: dict):
        """Add log entry to GUI display."""
        log_timestamp = datetime.now().strftime("%H:%M:%S")
        self.influx_log_lines.insert(
            0,
            f"{log_timestamp}: Detected {logging_data['vehicle_type']}. Motion: {logging_data['direction']}",
        )
        if len(self.influx_log_lines) > INFLUX_LOG_MAX_ENTRIES:
            self.influx_log_lines.pop()

    def _log_to_influx(self, logging_data: dict):
        """Log direction data to InfluxDB."""
        if not self.influx_logger:
            return

        try:
            direction_data = [
                {
                    "class_name": logging_data["vehicle_type"],
                    "direction": logging_data["direction"],
                    "speed_mph": logging_data["speed_mph"],
                    "track_id": logging_data["track_id"],
                }
            ]
            self.influx_logger.log_directions(direction_data, source_name="srt_stream")
        except Exception as e:
            print(f"Failed to log direction to InfluxDB: {e}")

    def get_tracked_count(self) -> int:
        """Get the number of currently tracked objects."""
        return len(self.tracked_objects)

    def cleanup_old_tracks(self, current_time: float):
        """Remove tracks that haven't been updated recently."""
        to_remove = []
        for track_id, data in self.tracked_objects.items():
            if (
                data["positions"]
                and current_time - data["positions"][-1][2]
                > TRACKING_CLEANUP_TIME_SECONDS
            ):
                to_remove.append(track_id)
            elif len(data["positions"]) > TRACKING_MAX_POSITIONS:
                data["positions"] = data["positions"][-10:]

        for track_id in to_remove:
            del self.tracked_objects[track_id]

        return len(to_remove)
