# AGENTS.md - Development Guidelines for Eyeball

## Build/Lint/Test Commands
- **Install dependencies**: `uv sync`
- **Install dev dependencies**: `uv sync --group dev`
- **Run application**: `uv run python main.py`
- **Run with custom SRT URI**: `uv run python main.py srt://your-ip:port`
- **Test InfluxDB connection**: `uv run python -m eyeball.influx_client`
- **Start InfluxDB**: `docker-compose up -d`
- **Stop InfluxDB**: `docker-compose down`

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party, then local imports
- Use absolute imports for local modules
- Group imports with blank lines between groups

### Formatting & Types
- Use type hints for function parameters and return values
- Use `Optional[T]` for nullable types
- Use `Union[T1, T2]` for multiple possible types

### Naming Conventions
- **Classes**: PascalCase (e.g., `ObjectDetector`, `DetectionLogger`)
- **Functions/Methods**: snake_case (e.g., `log_detections`, `_get_device`)
- **Constants**: UPPER_CASE (e.g., `VEHICLE_CLASSES`, `CLASS_NAMES`)
- **Variables**: snake_case (e.g., `frame_count`, `model_path`)

### Documentation
- Use triple-quoted docstrings for modules, classes, and public methods
- Follow Google-style docstrings with Args/Returns sections
- Include type information in docstrings when not using type hints

### Error Handling
- Use try/except blocks for external operations (file I/O, network calls)
- Provide informative error messages
- Log warnings but continue execution when possible

### Best Practices
- Use f-strings for string formatting
- Use pathlib for path operations
- Follow single responsibility principle for classes and methods
- Use descriptive variable names over abbreviations