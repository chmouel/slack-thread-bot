# Gemini Development Guidelines

1. Use `uv` for dependency management and formatting. if you add a dependency,
   run `uv add <package>` to add it to the project. make sure to run `uv sync`
to synchronize dependencies after adding or removing packages.
2. Always check the Python code with `pylint` and format it with `ruff format`.
3. Always update the README if you change things or add new features.
4. Write unit tests for all new features and bug fixes.
5. Follow PEP 8 style guidelines for Python code.
6. Use type hints and run `mypy` for static type checking.
7. Document all public functions and classes with docstrings.
9. Review and test your code before submitting a pull request.
10. Keep commits small and focused; use descriptive commit messages.
11. don't test the bot directly. If you need to ask the user to run it but
    usually avoid so.
