## Build, Lint, and Test

- **Dependency Management**: Use `uv`. Run `uv add <package>` to add a dependency and `uv sync` to install.
- **Formatting**: `ruff format`
- **Linting**: `pylint main.py test_main.py`
- **Type Checking**: `mypy main.py`
- **Testing**:
  - Run all tests: `python -m unittest test_main.py`
  - Run a single test: `python -m unittest test_main.TestSlackThreadBot.test_handle_tz_no_codes`

## Code Style and Development Guidelines

- **Formatting**: Follow PEP 8 guidelines. Use `ruff format` to enforce it.
- **Imports**: Use standard library imports first, then third-party imports. Keep them organized.
- **Types**: Use type hints for all function signatures and run `mypy` for static checking.
- **Naming Conventions**:
  - Use `snake_case` for variables and functions.
  - Use `PascalCase` for classes.
  - Use `UPPER_CASE` for constants.
- **Docstrings**: Write clear and concise docstrings for all public modules, classes, and functions.
- **Error Handling**:
  - Use try-except blocks for operations that can fail (e.g., API calls, file I/O).
  - Log errors with detailed information.
- **Testing**:
    - Write unit tests for all new features and bug fixes.
    - Avoid testing the bot directly if possible.
- **Commits**: Keep commits small and focused with descriptive messages.
- **Documentation**: Always update the README if you change things or add new features.
