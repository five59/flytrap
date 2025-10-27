import cv2
import time
import os
import torch
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync


# Detect available hardware acceleration
def get_device():
    """Detect and return the best available device for inference"""
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

# Load YOLO11 medium model and set device
device = get_device()
model = YOLO('yolo11m.pt')
model.to(device)

# Tracked object class IDs in COCO dataset (YOLOv8 uses COCO classes)
VEHICLE_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck

# Calibration: frame shows ~25-30 ft of road width
# Using midpoint of 27.5 ft for calculations
ROAD_WIDTH_FEET = 32

# Tracking data
tracked_objects = {}  # track_id: {'positions': [(x, y, timestamp), ...], 'class': class_id, 'logged': False, 'crossed_midpoint': False}
LOG_FILE = 'vehicle_tracking.log'
SCREENSHOTS_DIR = 'screenshots'
frame_width = None  # Will be set after first frame
frame_midpoint_x = None  # Will be set after first frame

# Create screenshots directory if it doesn't exist
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Initialize NDI components
finder = Finder()
receiver = Receiver(
    color_format=RecvColorFormat.RGBX_RGBA,  # Compatible with OpenCV
    bandwidth=RecvBandwidth.highest,
)
video_frame = VideoFrameSync()
receiver.frame_sync.set_video_frame(video_frame)

source = None

def on_finder_change():
    """Callback when NDI sources are discovered"""
    global source
    if finder is None or source is not None:
        return
    
    ndi_source_names = finder.get_source_names()
    if len(ndi_source_names) == 0:
        return
    
    print(f"Found NDI sources: {ndi_source_names}")
    print(f"Connecting to: {ndi_source_names[0]}")
    
    with finder.notify:
        source = finder.get_source(ndi_source_names[0])
        receiver.set_source(source)


# Set up finder callback and start discovery
finder.set_change_callback(on_finder_change)
finder.open()

# Create display window
cv2.namedWindow("YOLOv8 NDI Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 NDI Detection", 1280, 720)

print("Waiting for NDI sources...")

while True:
    if receiver.is_connected():
        # Capture video frame from NDI
        receiver.frame_sync.capture_video()
        
        # Check if we have valid frame data
        if min(video_frame.xres, video_frame.yres) != 0:
            # Convert NDI frame to numpy array
            # CyndiLib provides frames in RGBA format as flat array
            frame_data = video_frame.get_array()

            # Ensure frame is contiguous and in correct format
            if frame_data is not None and frame_data.size > 0:
                # Reshape flat array to (height, width, channels)
                frame_rgba = frame_data.reshape((video_frame.yres, video_frame.xres, 4))

                # Set frame width and midpoint for pixel-to-feet conversion (once)
                if frame_width is None:
                    frame_width = video_frame.xres
                    frame_midpoint_x = frame_width / 2

                # Convert RGBA to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            else:
                continue
            
            # Run YOLOv8 inference with tracking
            results = model.track(
                frame_bgr,
                device=device,
                conf=0.4,
                verbose=False,
                persist=True  # Keep track IDs across frames
            )

            # Annotate frame with detections first
            annotated_frame = results[0].plot()

            # Process tracking results
            screenshots_to_save = []  # Store info for screenshots to save after annotation
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    # Only track vehicles
                    if cls not in VEHICLE_CLASSES:
                        continue

                    # Get center point of bounding box
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    current_time = time.time()

                    # Initialize or update tracking data
                    if track_id not in tracked_objects:
                        tracked_objects[track_id] = {
                            'positions': [(center_x, center_y, current_time)],
                            'class': cls,
                            'logged': False,
                            'crossed_midpoint': False,
                            'midpoint_cross_position': None
                        }
                    else:
                        # Update class in case it was misidentified initially
                        tracked_objects[track_id]['class'] = cls
                        tracked_objects[track_id]['positions'].append((center_x, center_y, current_time))

                        # Keep only last 30 positions for direction calculation
                        if len(tracked_objects[track_id]['positions']) > 30:
                            tracked_objects[track_id]['positions'].pop(0)

                        # Check if object has crossed the midpoint
                        if not tracked_objects[track_id]['crossed_midpoint'] and len(tracked_objects[track_id]['positions']) >= 2:
                            prev_x = tracked_objects[track_id]['positions'][-2][0]
                            curr_x = center_x

                            # Check if crossed from either direction
                            if (prev_x < frame_midpoint_x <= curr_x) or (prev_x > frame_midpoint_x >= curr_x):
                                tracked_objects[track_id]['crossed_midpoint'] = True
                                tracked_objects[track_id]['midpoint_cross_position'] = len(tracked_objects[track_id]['positions']) - 1

                        # Log when object crosses midpoint (not before)
                        if tracked_objects[track_id]['crossed_midpoint'] and not tracked_objects[track_id]['logged']:
                            positions = tracked_objects[track_id]['positions']
                            if len(positions) >= 10:  # Need enough points for reliable direction and speed
                                start_x, start_y, start_time = positions[0]
                                end_x, end_y, end_time = positions[-1]
                                displacement_pixels = end_x - start_x
                                time_elapsed = end_time - start_time

                                # Determine direction based on displacement
                                if abs(displacement_pixels) > 50 and time_elapsed > 0:  # Lower threshold since we're at midpoint
                                    direction = "left-to-right" if displacement_pixels > 0 else "right-to-left"

                                    # Calculate speed
                                    # Convert pixels to feet
                                    pixels_per_foot = frame_width / ROAD_WIDTH_FEET
                                    distance_feet = abs(displacement_pixels) / pixels_per_foot

                                    # Convert to mph: (feet/second) * (3600 sec/hour) / (5280 feet/mile)
                                    speed_fps = distance_feet / time_elapsed
                                    speed_mph = speed_fps * 3600 / 5280

                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                    class_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
                                    vehicle_type = class_names.get(cls, "object")

                                    # Prepare screenshot if moving right-to-left
                                    screenshot_filename = None
                                    if direction == "right-to-left":
                                        screenshot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                        screenshot_filename = f"{SCREENSHOTS_DIR}/track_{track_id}_{screenshot_timestamp}.jpg"
                                        screenshots_to_save.append(screenshot_filename)

                                    # Log to file
                                    with open(LOG_FILE, 'a') as f:
                                        log_entry = f"{timestamp} | Track ID: {track_id} | Type: {vehicle_type} | Direction: {direction} | Speed: {speed_mph:.1f} mph"
                                        if screenshot_filename:
                                            log_entry += f" | Screenshot: {screenshot_filename}"
                                        f.write(log_entry + "\n")
                                    tracked_objects[track_id]['logged'] = True
                                    print(f"Logged: {vehicle_type} {direction} at {speed_mph:.1f} mph - {timestamp}")

            # Save screenshots for right-to-left vehicles
            for screenshot_path in screenshots_to_save:
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")

            # Display result
            cv2.imshow('YOLOv8 NDI Detection', annotated_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Cleanup
cv2.destroyAllWindows()
finder.close()
