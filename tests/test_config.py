"""Tests for config module."""

import pytest
from eyeball.config import (
    DEFAULT_DETECTION_FPS,
    FRAME_SKIP_INTERVAL_BASE,
    FRAME_QUEUE_MAX_SIZE,
    WINDOW_NAME,
)


class TestConfig:
    """Test configuration constants."""

    def test_default_detection_fps(self):
        """Test default detection FPS value."""
        assert isinstance(DEFAULT_DETECTION_FPS, float)
        assert DEFAULT_DETECTION_FPS > 0

    def test_frame_skip_interval_base(self):
        """Test frame skip interval base value."""
        assert isinstance(FRAME_SKIP_INTERVAL_BASE, int)
        assert FRAME_SKIP_INTERVAL_BASE > 0

    def test_frame_queue_max_size(self):
        """Test frame queue max size value."""
        assert isinstance(FRAME_QUEUE_MAX_SIZE, int)
        assert FRAME_QUEUE_MAX_SIZE > 0

    def test_window_name(self):
        """Test window name constant."""
        assert isinstance(WINDOW_NAME, str)
        assert len(WINDOW_NAME) > 0