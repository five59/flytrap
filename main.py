"""
Eyeball - Real-time object detection with NDI and YOLO.

Main entry point for the application.
"""

import os
import platform
from eyeball import ObjectDetector


def main():
    """Run the object detector."""
    import sys

    # Auto-detect headless mode (no DISPLAY or GUI not available)
    display_env = os.environ.get('DISPLAY')
    headless = not display_env

    if headless:
        print("No DISPLAY detected - running in headless mode")
    else:
        print(f"DISPLAY detected: {display_env} - attempting GUI mode")

    # Get SRT URI from command line or use default
    if len(sys.argv) > 1:
        srt_uri = sys.argv[1]
    else:
        # Default SRT URI
        srt_uri = "srt://192.168.1.195:4201"
        print(f"Using default SRT URI: {srt_uri}")

    # Get detection FPS from command line or use default
    detection_fps = 6.0
    if len(sys.argv) > 2:
        try:
            detection_fps = float(sys.argv[2])
            print(f"Using detection FPS: {detection_fps}")
        except ValueError:
            print(f"Invalid detection FPS: {sys.argv[2]}, using default {detection_fps}")

    print("Usage: python main.py [srt://your-ip:port] [detection_fps]")
    print("Examples:")
    print("  python main.py srt://192.168.1.100:4201 6.0")
    print("  python main.py srt://192.168.1.100:4201 12.0")

    # Add SRT connection timeout to prevent hanging
    from eyeball.config import SRT_CONNECTION_TIMEOUT_MS
    timeout_param = f'timeout={SRT_CONNECTION_TIMEOUT_MS}'
    if '?' not in srt_uri:
        srt_uri += f'?{timeout_param}'
    else:
        srt_uri += f'&{timeout_param}'

    # Define ROI bounding box in original 1920x1080 frame coordinates
    # Exclude top 200px and bottom 50px to focus on road area
    # [x1, y1, x2, y2] = [left, top, right, bottom]
    roi_box = (0, 200, 1920, 1030)  # Full width, exclude top/bottom

    # Additional areas can be masked if needed (in inference coordinates):
    # roi_mask[200:400, 0:200] = 0      # Custom area

    detector = ObjectDetector(
        srt_uri=srt_uri,
        headless=headless,
        roi_box=roi_box,
        detection_fps=detection_fps
    )
    detector.run()


if __name__ == "__main__":
    main()
