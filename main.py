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

    # Auto-detect headless mode (no DISPLAY)
    headless = not os.environ.get('DISPLAY')

    if headless:
        print("No DISPLAY detected - running in headless mode")

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

    detector = ObjectDetector(srt_uri=srt_uri, headless=headless)
    detector.run()


if __name__ == "__main__":
    main()
