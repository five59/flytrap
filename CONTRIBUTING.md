# Contributing to Flytrap

Thank you for your interest in contributing to Flytrap! 🎉

We welcome contributions from everyone, whether you're fixing bugs, adding features, improving documentation, or helping with testing. This document provides guidelines and information to help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- **Be respectful** and inclusive in all interactions
- **Focus on constructive feedback** and collaborative problem-solving
- **Accept responsibility** for mistakes and learn from them
- **Show empathy** towards other contributors and users
- **Help create a positive community** environment

Unacceptable behavior includes harassment, discrimination, or any form of disrespectful conduct. Violations may result in temporary or permanent exclusion from the community.

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.12+** installed
- **uv package manager** ([installation guide](https://github.com/astral-sh/uv))
- **Git** for version control
- **GPU recommended** (CUDA/MPS/CPU fallback supported)

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/flytrap.git
   cd flytrap
   ```

3. **Set up the development environment**:
   ```bash
   # Install dependencies
   uv sync

   # Install development dependencies
   uv sync --group dev

   # Verify installation
   uv run python -c "from flytrap import ObjectDetector; print('✓ Development environment ready')"
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 🐛 **Bug Fixes**
- Identify and fix bugs in the codebase
- Improve error handling and edge cases
- Fix performance issues or memory leaks

#### ✨ **New Features**
- Implement new functionality
- Add support for new video formats or protocols
- Enhance analytics and tracking capabilities

#### 📚 **Documentation**
- Improve existing documentation
- Add tutorials and examples
- Translate documentation to other languages

#### 🧪 **Testing**
- Write unit tests and integration tests
- Improve test coverage
- Add performance benchmarks

#### 🎨 **UI/UX Improvements**
- Enhance the GUI dashboard
- Improve user interface design
- Add accessibility features

#### 🔧 **Infrastructure**
- Improve CI/CD pipelines
- Update dependencies and security patches
- Optimize build processes

## Development Workflow

### Git Workflow

We use a feature branch workflow:

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes:
   git checkout -b bugfix/issue-description
   # or for documentation:
   git checkout -b docs/update-guide
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   # Run tests
   uv run python -m pytest tests/

   # Run linting
   uv run ruff check flytrap/

   # Run type checking
   uv run python -m mypy flytrap/ --ignore-missing-imports
   ```

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add multi-camera support

   - Implement concurrent camera processing
   - Add camera configuration validation
   - Update documentation with examples"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

### Commit Message Guidelines

We follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Testing related changes
- `chore`: Maintenance tasks

**Examples:**
```bash
feat: add support for RTSP streams
fix: resolve memory leak in frame processor
docs: update installation guide for Ubuntu 22.04
test: add unit tests for object tracking
```

## Coding Standards

### Code Style

We use modern Python practices with strict style guidelines:

#### Imports
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

#### Type Hints
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

#### Naming Conventions
```python
# Classes: PascalCase
class ObjectDetector:
    pass

# Functions/methods: snake_case
def process_frame(frame: np.ndarray) -> List[Detection]:
    pass

# Constants: UPPER_CASE
MAX_QUEUE_SIZE = 100
DEFAULT_CONFIDENCE = 0.4

# Private methods: leading underscore
def _validate_config(self, config: dict) -> bool:
    pass
```

#### Documentation
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

### Linting and Formatting

We use automated tools to maintain code quality:

```bash
# Run linting (ruff)
uv run ruff check flytrap/

# Auto-fix linting issues
uv run ruff check flytrap/ --fix

# Run type checking (mypy)
uv run python -m mypy flytrap/ --ignore-missing-imports

# Format code (if using black)
uv run black flytrap/
```

**Pre-commit hooks** are recommended to run these checks automatically before commits.

## Testing

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── test_config.py          # Configuration tests
├── test_memory_manager.py  # Memory management tests
├── test_stream_handler.py  # Stream handling tests
├── test_frame_processor.py # Frame processing tests
├── test_object_tracker.py  # Object tracking tests
└── test_detector.py        # Main detector tests
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

### Test Coverage

We aim for high test coverage. Check coverage reports:

```bash
# Generate coverage report
uv run python -m pytest --cov=flytrap --cov-report=html tests/

# View HTML report
open htmlcov/index.html
```

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run all checks**:
   ```bash
   # Tests pass
   uv run python -m pytest tests/

   # Linting passes
   uv run ruff check flytrap/

   # Type checking passes
   uv run python -m mypy flytrap/ --ignore-missing-imports
   ```

3. **Update documentation** if needed:
   - Update relevant docs in `docs/`
   - Update docstrings for public APIs
   - Add examples for new features

4. **Create a Pull Request**:
   - Use a descriptive title
   - Fill out the PR template
   - Reference any related issues
   - Add screenshots for UI changes

5. **Address review feedback**:
   - Make requested changes
   - Re-run tests and checks
   - Update PR with new commits

### PR Guidelines

**Title Format:**
```
type: Brief description of changes
```

**Description Template:**
```markdown
## Description
Brief description of what this PR does.

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Documentation updated

## Screenshots (if applicable)
<!-- Add screenshots for UI changes -->

## Related Issues
Closes #123
```

## Reporting Issues

### Bug Reports

When reporting bugs, please provide:

1. **Clear title** describing the issue
2. **Detailed description** of the problem
3. **Steps to reproduce**:
   ```bash
   # Commands or code that reproduce the issue
   uv run python main.py srt://example.com:4201
   ```
4. **Expected vs actual behavior**
5. **Environment information**:
   - OS and version
   - Python version
   - GPU/CPU configuration
   - Flytrap version/commit
6. **Logs and error messages**
7. **Screenshots** if applicable

### Feature Requests

For new features, please provide:

1. **Clear description** of the proposed feature
2. **Use case** - why is this feature needed?
3. **Proposed implementation** (optional)
4. **Alternatives considered** (optional)
5. **Mockups or examples** if applicable

### Issue Labels

We use labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `question`: Questions or discussions
- `help wanted`: Good for newcomers
- `good first issue`: Beginner-friendly issues

## Documentation

### Contributing to Documentation

Documentation lives in the `docs/` directory and is built with Jekyll for GitHub Pages.

1. **Edit existing docs** or create new ones in `docs/`
2. **Add Jekyll front matter**:
   ```yaml
   ---
   layout: default
   title: "Your Page Title"
   description: "Brief description for SEO"
   nav_order: 10
   ---
   ```
3. **Test locally** (requires Ruby/Jekyll):
   ```bash
   cd docs
   bundle install
   bundle exec jekyll serve
   ```
4. **Follow documentation guidelines**:
   - Use clear, concise language
   - Include code examples
   - Add table of contents for long pages
   - Test all links and commands

### Documentation Standards

- Use **Markdown** for all documentation
- Include **code examples** for technical content
- Add **table of contents** to pages longer than 1000 words
- Use **relative links** for internal references
- Include **screenshots** for UI-related content
- Keep language **inclusive and accessible**

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the docs first for common questions

### Communication Guidelines

- **Be respectful** and constructive in all interactions
- **Use clear language** and provide context
- **Search existing issues** before creating new ones
- **Stay on topic** in discussions
- **Help others** when you can

### Recognition

Contributors are recognized through:
- **GitHub contributor statistics**
- **Mention in release notes** for significant contributions
- **Community recognition** in discussions

## Recognition

We appreciate all contributions, big and small! Contributors may be:

- Listed in `CONTRIBUTORS.md` (for significant contributions)
- Mentioned in release notes
- Featured in community spotlights
- Invited to become maintainers (for sustained contributions)

Thank you for contributing to Flytrap! Your efforts help make real-time object detection more accessible and powerful for everyone. 🚀