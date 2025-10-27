"""
Eyeball - Real-time object detection with NDI and YOLO.

Main entry point for the application.
"""

import os
import platform
import numpy as np
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
        print("Pass your SRT URI as command line argument: python main.py srt://your-ip:port")

    # Add SRT connection timeout (5 seconds) to prevent hanging
    if '?' not in srt_uri:
        srt_uri += '?timeout=5000000'
    else:
        srt_uri += '&timeout=5000000'

    # Configure inference resolution for memory optimization
    # Default: 1280x720 (55% less memory than 1080p, good detection quality)
    # Options: (1280, 720), (960, 540), (640, 360), or None for original size
    inference_size = (1280, 720)

    # Create ROI mask to ignore specific areas
    # IMPORTANT: Mask coordinates are in INFERENCE resolution (1280x720)
    # To mask top 200 pixels and bottom 50 pixels of ORIGINAL 1080p frame, we need to scale:
    # 200 pixels at 1080p → (200 / 1080) * 720 = 133 pixels at 720p
    # 50 pixels at 1080p → (50 / 1080) * 720 = 33 pixels at 720p
    roi_mask = np.ones((inference_size[1], inference_size[0]), dtype=np.uint8) * 255

    # Calculate pixels to mask in inference resolution
    pixels_to_mask_at_top = int((200 / 1080) * inference_size[1])
    pixels_to_mask_at_bottom = int((50 / 1080) * inference_size[1])

    # Mask out top 200 pixels of original 1920x1080 frame
    roi_mask[0:pixels_to_mask_at_top, :] = 0

    # Mask out bottom 50 pixels of original 1920x1080 frame
    bottom_start = inference_size[1] - pixels_to_mask_at_bottom
    roi_mask[bottom_start:, :] = 0

    print(f"Masking top {pixels_to_mask_at_top} pixels (= 200px at 1080p)")
    print(f"Masking bottom {pixels_to_mask_at_bottom} pixels (= 50px at 1080p)")

    # Additional areas can be masked if needed (in inference coordinates):
    # roi_mask[200:400, 0:200] = 0      # Custom area

    detector = ObjectDetector(
        srt_uri=srt_uri,
        headless=headless,
        inference_size=inference_size,
        roi_mask=roi_mask
    )
    detector.run()


if __name__ == "__main__":
    main()
