# Repository Guidelines

## Project Structure & Module Organization
- `src/vbpca_py/` contains the Python implementation and C++ extension sources (`errpca_pt.cpp`, `subtract_mu_from_sparse.cpp`).
- `test/` holds datasets and reference artifacts used by legacy tests and scripts.
- `setup.py` and `pyproject.toml` define build metadata and dependencies.
- `__init__.py` at repo root exists but the installable package lives under `src/`.

## Build, Test, and Development Commands
- `python -m pip install -e .` — installs the package in editable mode and builds the C++ extensions.
- `python setup.py build_ext --inplace` — builds C++ extensions in-place for local development.
- `python -m pytest` — runs tests if you add pytest-based tests (see Testing Guidelines below).

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Formatting and linting: Ruff is configured in `pyproject.toml` with Google-style docstrings.
- C++: keep headers minimal and prefer explicit include paths; use `-std=c++14` or `-std=c++11` as defined in `setup.py`.

## Testing Guidelines
- Pytest is configured in `pyproject.toml`, but the current repository uses `test/` for data and scripts rather than `tests/`.
- If adding pytest tests, place them under `tests/` and use `test_*.py` filenames.
- When adding new datasets, document their origin and expected usage in the test folder.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and lower-case (e.g., “fix pca_full”, “update gitignore”).
- Keep commits focused and describe the change in one line; add a short body if the change is non-trivial.
- Pull requests should include: a short summary, testing notes, and any data/output changes or new files.

## Environment & Build Notes
- The C++ build expects Eigen headers. Set `EIGEN_INCLUDE_DIR` if Eigen is not in a standard path.
- The project targets Python >= 3.11; ensure your environment matches before building.
- On macOS, `setup.py` auto-detects the SDK path to find libc++ headers during extension builds.
