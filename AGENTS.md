# AGENTS.md - Development Guidelines for Eyeball

## Build/Lint/Test Commands
- **Install dependencies**: `uv sync`
- **Install dev dependencies**: `uv sync --group dev`
- **Run linting**: `uv run ruff check eyeball/`
- **Run type checking**: `uv run python -m mypy eyeball/ --ignore-missing-imports`
- **Run tests**: `uv run python -m pytest tests/`
- **Run tests with coverage**: `uv run python -m pytest --cov=eyeball --cov-report=xml --cov-report=term tests/`
- **Run application**: `uv run python main.py`
- **Run with custom SRT URI**: `uv run python main.py srt://your-ip:port`
- **Test InfluxDB connection**: `uv run python -m eyeball.influx_client`
- **Start InfluxDB**: `docker-compose up -d`
- **Stop InfluxDB**: `docker-compose down`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local. Use absolute imports. Group with blank lines.
- **Types**: Use type hints for all parameters/returns. `Optional[T]` for nullable, `Union[T1, T2]` for multiple types.
- **Naming**: Classes=PascalCase, functions/methods=snake_case, constants=UPPER_CASE, variables=snake_case.
- **Documentation**: Google-style docstrings with Args/Returns sections for modules, classes, and public methods.
- **Error Handling**: try/except for external ops (I/O, network). Informative messages. Log warnings, continue execution.
- **Best Practices**: f-strings for formatting, pathlib for paths, single responsibility principle, descriptive names.