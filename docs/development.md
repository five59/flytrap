---
layout: default
title: "Development Guide"
description: "Development environment, project structure, testing, and contribution guidelines"
nav_order: 9
---

# Development Guide

This guide covers Flytrap's development environment, project structure, testing, and contribution guidelines.

## Development Environment

### Prerequisites

- Python 3.12+
- uv package manager
- Git
- GPU recommended (CUDA/MPS/CPU fallback)

### Setup

```bash
# Clone repository
git clone https://github.com/five59/flytrap.git
cd flytrap

# Install dependencies
uv sync

# Install development dependencies
uv sync --group dev

# Verify installation
uv run python -c "from flytrap import ObjectDetector; print('✓ Development environment ready')"
```

## Project Structure

```
flytrap/
├── flytrap/                    # Main package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Configuration constants
│   ├── detector.py            # Main ObjectDetector class
│   ├── stream_handler.py      # SRT stream reception
│   ├── frame_processor.py     # Motion detection & YOLO
│   ├── object_tracker.py      # Tracking & analytics
│   ├── gui_dashboard.py       # Real-time visualization
│   ├── memory_manager.py      # Memory cleanup & monitoring
│   ├── influx_client.py       # InfluxDB metrics client
│   └── ...
├── grafana/                   # Grafana dashboard provisioning
│   ├── provisioning/
│   │   ├── dashboards/
│   │   │   └── dashboard.yml
│   │   └── datasources/
│   │       └── influxdb.yml
│   └── ...
├── docs/                      # Documentation
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_memory_manager.py
│   └── ...
├── main.py                    # Application entry point
├── pyproject.toml             # Dependencies & project config
├── docker-compose.yml         # InfluxDB & Grafana services
├── .env.example               # Environment template
├── AGENTS.md                  # Development guidelines
├── uv.lock                    # Dependency lock file
└── README.md                  # Project documentation
```

## Development Commands

### Build/Lint/Test Commands

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --group dev

# Run linting
uv run ruff check flytrap/

# Run type checking
uv run python -m mypy flytrap/ --ignore-missing-imports

# Run tests
uv run python -m pytest tests/

# Run tests with coverage
uv run python -m pytest --cov=flytrap --cov-report=xml --cov-report=term tests/

# Run application
uv run python main.py

# Run with custom SRT URI
uv run python main.py srt://your-ip:port

# Test InfluxDB connection
uv run python -m flytrap.influx_client

# Start InfluxDB
docker-compose up -d

# Stop InfluxDB
docker-compose down
```

## Code Style Guidelines

### Imports

```python
# Standard library first
import os
import sys
from typing import Optional, List, Dict

# Third-party libraries
import cv2
import torch
import numpy as np

# Local imports (absolute)
from flytrap.config import Config
from flytrap.stream_handler import StreamHandler
```

**Rules:**
- Group imports with blank lines
- Use absolute imports for local modules
- Sort imports alphabetically within groups

### Types

```python
# Use type hints for all parameters and returns
def process_frame(frame: np.ndarray, confidence: float = 0.4) -> List[Detection]:
    pass

# Use Optional for nullable types
def get_track(track_id: int) -> Optional[Track]:
    pass

# Use Union for multiple types
def load_model(path: Union[str, Path]) -> YOLO:
    pass
```

### Naming Conventions

```python
# Classes: PascalCase
class ObjectDetector:
    pass

class StreamHandler:
    pass

# Functions/methods: snake_case
def process_frame(frame: np.ndarray) -> List[Detection]:
    pass

def get_detection_confidence() -> float:
    pass

# Constants: UPPER_CASE
MAX_QUEUE_SIZE = 100
DEFAULT_CONFIDENCE = 0.4

# Variables: snake_case
frame_count = 0
detection_results = []
```

### Documentation

```python
def detect_objects(
    frame: np.ndarray,
    confidence: float = 0.4,
    classes: Optional[List[str]] = None
) -> List[Detection]:
    """Detect objects in a video frame using YOLO model.

    Args:
        frame: Input video frame as numpy array (H, W, 3)
        confidence: Detection confidence threshold (0.0-1.0)
        classes: Optional list of class names to detect

    Returns:
        List of Detection objects with bounding boxes and metadata

    Raises:
        ModelLoadError: If YOLO model fails to load
        InferenceError: If detection inference fails

    Example:
        >>> detections = detect_objects(frame, confidence=0.5)
        >>> for det in detections:
        ...     print(f"Found {det.class_name} at {det.bbox}")
    """
    pass
```

**Docstring Format:**
- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Add examples for complex functions
- Keep descriptions concise but informative

### Error Handling

```python
# Use specific exceptions
class StreamConnectionError(Exception):
    """Raised when video stream connection fails."""
    pass

class ModelLoadError(Exception):
    """Raised when YOLO model loading fails."""
    pass

# Handle errors gracefully with logging
try:
    stream = StreamHandler(uri)
    stream.connect()
except StreamConnectionError as e:
    logger.error(f"Failed to connect to stream: {e}")
    # Attempt fallback or graceful degradation
    raise
```

### Best Practices

```python
# Use pathlib for paths
from pathlib import Path

def save_screenshot(frame: np.ndarray, track_id: int) -> Path:
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"track_{track_id}_{timestamp}.jpg"
    filepath = screenshot_dir / filename

    cv2.imwrite(str(filepath), frame)
    return filepath

# Use f-strings for formatting
def format_detection(detection: Detection) -> str:
    return f"Track {detection.track_id}: {detection.class_name} " \
           f"({detection.confidence:.2f}) at {detection.bbox}"

# Use context managers
with open(log_file, 'a') as f:
    f.write(f"{timestamp} | {message}\n")

# Prefer dataclasses for data structures
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
├── test_config.py          # Configuration tests
├── test_memory_manager.py  # Memory management tests
├── test_stream_handler.py  # Stream handling tests
├── test_frame_processor.py # Frame processing tests
├── test_object_tracker.py  # Object tracking tests
└── test_detector.py        # Main detector tests
```

### Writing Tests

```python
import pytest
import numpy as np
from flytrap.frame_processor import FrameProcessor

class TestFrameProcessor:
    @pytest.fixture
    def processor(self):
        """Create FrameProcessor instance for testing."""
        return FrameProcessor(model_path='yolo11n.pt')

    def test_initialization(self, processor):
        """Test FrameProcessor initializes correctly."""
        assert processor.model is not None
        assert processor.confidence == 0.4

    def test_process_empty_frame(self, processor):
        """Test processing empty frame returns empty results."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = processor.process(empty_frame)
        assert len(detections) == 0

    @pytest.mark.parametrize("confidence", [0.1, 0.5, 0.9])
    def test_confidence_threshold(self, processor, confidence):
        """Test confidence threshold filtering."""
        processor.confidence = confidence
        # Test with mock detections
        mock_detections = [
            {'confidence': 0.8, 'class_name': 'car'},
            {'confidence': 0.3, 'class_name': 'person'}
        ]

        filtered = [
            d for d in mock_detections
            if d['confidence'] >= confidence
        ]

        assert len(filtered) == (2 if confidence <= 0.3 else 1)
```

### Running Tests

```bash
# Run all tests
uv run python -m pytest tests/

# Run specific test file
uv run python -m pytest tests/test_frame_processor.py

# Run specific test
uv run python -m pytest tests/test_frame_processor.py::TestFrameProcessor::test_process_empty_frame

# Run with coverage
uv run python -m pytest --cov=flytrap --cov-report=html tests/

# Run tests in verbose mode
uv run python -m pytest -v tests/

# Run tests with debugging
uv run python -m pytest --pdb tests/
```

### Test Configuration

```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_frame():
    """Create a sample video frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_detection():
    """Create mock detection data."""
    return {
        'bbox': (100, 100, 200, 200),
        'confidence': 0.85,
        'class_id': 2,
        'class_name': 'car',
        'track_id': 1
    }

@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Create temporary directory for test files."""
    return tmp_path_factory.mktemp("flytrap_test")
```

## Dependencies

### Core Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "ultralytics>=8.0.0",        # YOLO models and inference
    "opencv-python>=4.8.0",      # Video processing
    "torch>=2.0.0",              # PyTorch ML framework
    "torchvision>=0.15.0",       # Computer vision utilities
    "influxdb-client>=1.36.0",   # Time-series database client
    "pygobject>=3.42.0",         # GStreamer SRT streaming
    "numpy>=1.24.0",             # Numerical computing
    "pillow>=10.0.0",            # Image processing
    "python-dotenv>=1.0.0",      # Environment variable management
]
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",             # Testing framework
    "pytest-cov>=4.0.0",         # Coverage reporting
    "ruff>=0.1.0",               # Linting and formatting
    "mypy>=1.0.0",               # Type checking
    "black>=23.0.0",             # Code formatting
    "pre-commit>=3.0.0",         # Git hooks
    "ipykernel>=6.0.0",          # Jupyter notebook support
]
```

### Dependency Management

```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --group dev package-name

# Update dependencies
uv sync

# Update lock file
uv lock

# Check for outdated packages
uv outdated

# Remove dependency
uv remove package-name
```

## Git Workflow

### Branching Strategy

```bash
# Create feature branch
git checkout -b feature/add-new-feature

# Create bug fix branch
git checkout -b bugfix/fix-memory-leak

# Create documentation branch
git checkout -b docs/update-api-docs
```

### Commit Guidelines

```bash
# Use conventional commit format
git commit -m "feat: add multi-camera support"
git commit -m "fix: resolve memory leak in frame processor"
git commit -m "docs: update installation guide"
git commit -m "refactor: simplify object tracking logic"

# Commit types:
# feat: New feature
# fix: Bug fix
# docs: Documentation
# style: Code style changes
# refactor: Code refactoring
# test: Testing
# chore: Maintenance
```

### Pull Request Process

1. **Create Branch**: `git checkout -b feature/your-feature`
2. **Make Changes**: Implement your feature/fix
3. **Run Tests**: `uv run python -m pytest tests/`
4. **Run Linting**: `uv run ruff check flytrap/`
5. **Update Docs**: Update documentation if needed
6. **Commit Changes**: `git commit -m "feat: your feature description"`
7. **Push Branch**: `git push origin feature/your-feature`
8. **Create PR**: Open pull request with description
9. **Code Review**: Address review feedback
10. **Merge**: Squash and merge when approved

## Debugging

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.debug("Processing frame with shape: %s", frame.shape)
logger.info("Detected %d objects", len(detections))
logger.warning("High memory usage: %.1f MB", memory_mb)
logger.error("Failed to connect to stream: %s", str(e))
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_function():
    detector = ObjectDetector(srt_uri="srt://test:4201")
    # Run profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code here
    for _ in range(100):
        frame = detector.stream_handler.get_frame()
        detections = detector.frame_processor.process(frame)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)  # Top 20 functions

if __name__ == "__main__":
    profile_function()
```

### Memory Debugging

```python
import tracemalloc
import gc

# Start tracing
tracemalloc.start()

# Your code here
detector = ObjectDetector(srt_uri="srt://test:4201")
# ... processing ...

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

# Get top memory consumers
stats = tracemalloc.take_snapshot().statistics('lineno')
for stat in stats[:10]:
    print(stat)

tracemalloc.stop()
```

## Contributing

### Code of Conduct

- Be respectful and inclusive
- Follow the established code style
- Write tests for new features
- Update documentation
- Keep commits focused and descriptive

### Issue Reporting

When reporting bugs:
1. Use the bug report template
2. Include system information (OS, Python version, GPU)
3. Provide minimal reproduction steps
4. Include relevant log output
5. Attach screenshots if applicable

### Feature Requests

When requesting features:
1. Check if the feature already exists
2. Describe the use case clearly
3. Explain why it's needed
4. Consider implementation complexity
5. Suggest a design if possible

## Release Process

### Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Run full test suite
- [ ] Update version in `pyproject.toml`
- [ ] Update changelog
- [ ] Update documentation
- [ ] Create git tag
- [ ] Build and test distribution
- [ ] Publish to PyPI
- [ ] Create GitHub release

This development guide ensures consistent, high-quality contributions to the Flytrap project while maintaining code quality and project standards.