# Contributing to VBPCApy

Thank you for your interest in contributing to VBPCApy! We welcome contributions from the community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/VBPCApy.git
   cd VBPCApy
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```
4. **Install just** (command runner):
   ```bash
   # macOS
   brew install just
   
   # Linux
   curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
   
   # Or via cargo
   cargo install just
   ```

## Development Workflow

### Before Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
   or
   ```bash
   git checkout -b fix/issue-123
   ```

### Making Changes

1. **Write clear, focused code** that solves one problem at a time
2. **Follow the existing code style** (enforced by `ruff`)
3. **Add tests** for new functionality
4. **Update documentation** as needed

### Code Quality

Run these commands before committing:

```bash
# Lint your code
just lint

# Type check
just typecheck

# Run tests
just test

# Or run all checks at once
just ci
```

All checks must pass before your pull request can be merged.

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in the present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Add details in the body if needed

Example:
```
Add support for sparse matrix input

- Implement CSR matrix handling in VBPCA.fit()
- Add tests for sparse input
- Update documentation
```

## Testing

### Running Tests

```bash
# Run all tests
just test

# Run tests with coverage
just test-cov

# Run specific test file
pytest tests/test_estimators.py

# Run specific test
pytest tests/test_estimators.py::test_vbpca_fit_transform_shapes
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive names that explain what is being tested
- Follow the existing test structure and style

### Public API and internals

- Public API should be exported from `vbpca_py` (package root).
- Internal modules/symbols are prefixed with `_` and may change without deprecation.
- If a change adds new user-facing functionality, ensure it is intentionally exported and documented in README.

## Pull Request Process

1. **Update your branch** with the latest changes from main:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Push your changes** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

3. **Open a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues (e.g., "Fixes #123")
   - Explain what changes you made and why
   - Include any breaking changes or migration notes

4. **Respond to feedback**:
   - Address review comments
   - Make requested changes
   - Keep the discussion focused and professional

5. **Wait for approval**:
   - At least one maintainer must approve your PR
   - All CI checks must pass
   - Resolve any merge conflicts

## Code Style

We use `ruff` for linting and formatting. The configuration is in `pyproject.toml`.

Key style guidelines:
- Use Google-style docstrings
- Type hints for all public APIs
- Maximum line length: 88 characters (Black default)
- Import order: standard library, third-party, local

## Documentation

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def my_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    More detailed description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.
    """
```

### README Updates

If your changes affect the public API or usage:
- Update the relevant sections in README.md
- Add examples if appropriate
- Update the API reference section

### Benchmark and manuscript reproducibility updates

If your changes affect benchmark methodology, model-selection behavior, runtime
controls, outputs, or paper figures/tables:

- Update both `README.md` and `scripts/benchmark_study.md` in the same PR.
- Keep `justfile` benchmark recipes and documentation in sync.
- Document the exact reproducible command sequence:
   - `just bench-study-full`
   - `just bench-study-summary`
   - `just bench-study-paper`
- Do not commit generated benchmark artifacts (`results/*.csv`, `results/paper/*.png`, etc.);
  reproducibility should rely on deterministic commands and seeds.
- Clearly state whether changes alter locked publication settings or only pilot
   settings.

## Reporting Issues

### Bug Reports

Include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Python version and platform
- Minimal code example that demonstrates the issue

### Feature Requests

Include:
- A clear description of the feature
- The motivation and use case
- Examples of how it would be used
- Any alternatives you've considered

## Questions?

- Open an issue with the "question" label
- Check existing issues and discussions first

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Other conduct that would be considered inappropriate in a professional setting

## License

By contributing to VBPCApy, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in the project documentation and release notes.

Thank you for contributing to VBPCApy!
