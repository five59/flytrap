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
    # To mask top 280 pixels of ORIGINAL 1080p frame, we need to scale:
    # 280 pixels at 1080p → (280 / 1080) * 720 = 187 pixels at 720p
    roi_mask = np.ones((inference_size[1], inference_size[0]), dtype=np.uint8) * 255

    # Mask out top 280 pixels of original 1920x1080 frame
    # This equals ~187 pixels in the 1280x720 inference frame
    pixels_to_mask_at_inference = int((280 / 1080) * inference_size[1])
    roi_mask[0:pixels_to_mask_at_inference, :] = 0
    print(f"Masking top {pixels_to_mask_at_inference} pixels at inference resolution (= 280px at 1080p)")

    # Additional areas can be masked if needed (in inference coordinates):
    # roi_mask[550:720, 0:300] = 0      # Bottom-left corner (parked car)
    # roi_mask[550:720, 980:1280] = 0   # Bottom-right corner

    detector = ObjectDetector(
        srt_uri=srt_uri,
        headless=headless,
        inference_size=inference_size,
        roi_mask=roi_mask
    )
    detector.run()


if __name__ == "__main__":
    main()
