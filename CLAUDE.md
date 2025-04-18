# APilot Development Guide

## Build & Test Commands
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_bar_generator.py

# Run a specific test
python -m pytest tests/test_bar_generator.py::TestBarGenerator::test_x_minute_window

# Linting
ruff check apilot/

# Formatting
ruff format apilot/

# Type checking
mypy apilot/
```

## Coding Style
- **Naming**: PascalCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- **Imports**: Standard library → third-party → local modules; group by source
- **Formatting**: 88 character line limit, double quotes, 4-space indentation
- **Type hints**: Required for all functions/methods with consistent annotations
- **Docstrings**: Triple double-quotes with Google-style format
- **Error handling**: Use specific exceptions with proper logging
- **Logging**: Use module-specific loggers with `get_logger("ModuleName")`
- **Organization**: Clear separation between initialization, public and private methods
- **Comments**: Minimize comments, focus on complex logic and TODOs

## Project Focus
- Primary: Crypto exchanges (Binance, future: Bybit/OKX)
- Secondary: U.S. stock markets
- Architecture: Event-driven for strategy development