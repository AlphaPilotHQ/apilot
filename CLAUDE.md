# APilot Development Guide

## Build & Test Commands
```
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/

# Run a single test file
python -m unittest tests/test_bar_generator.py
# OR
python -m pytest tests/test_bar_generator.py

# Run a specific test within a file
python -m unittest tests.test_bar_generator.TestBarGenerator.test_x_minute_window

# Linting
ruff check apilot/

# Formatting
ruff format apilot/

# Type checking
mypy apilot/
```

## Coding Style
- **Naming**: PascalCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- **Imports**: Standard library first, third-party second, local modules last; group by source
- **Formatting**: 88 character line limit, double quotes, 4-space indentation
- **Type hints**: Required for all functions and methods with consistent annotations
- **Docstrings**: Triple double-quotes with Google-style format for parameters and returns
- **Error handling**: Use try/except with specific exceptions, log errors properly
- **Logging**: Use module-specific loggers with `get_logger("ModuleName")`
- **Organization**: Clear separation between initialization, public and private methods
- **Comments**: Use inline comments to explain complex logic, TODO comments for future work