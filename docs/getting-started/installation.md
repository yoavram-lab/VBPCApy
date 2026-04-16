# Installation

## Requirements

- **Python**: >= 3.11, < 3.15
- **C++ Compiler** *(source builds only)*: C++14 compatible (gcc, clang, MSVC)
- **Eigen** *(source builds only)*: Linear algebra library (version 3.x)
- **Matplotlib** *(optional)*: install via `pip install vbpca_py[plot]`

## From PyPI (recommended)

Pre-built wheels are available for Python 3.11–3.14 on Linux, macOS, and Windows:

```bash
pip install vbpca-py
```

With plotting support:

```bash
pip install vbpca-py[plot]
```

## From source

Building from source requires Eigen and a C++14-compatible compiler.

### Install Eigen

=== "Ubuntu / Debian"

    ```bash
    sudo apt-get install libeigen3-dev
    ```

=== "macOS (Homebrew)"

    ```bash
    brew install eigen
    ```

=== "Conda / Mamba"

    ```bash
    conda install -c conda-forge eigen
    ```

=== "Manual"

    Download from [eigen.tuxfamily.org](https://eigen.tuxfamily.org/) and set:
    ```bash
    export EIGEN_INCLUDE_DIR=/path/to/eigen3
    ```

Eigen is located automatically via `EIGEN_INCLUDE_DIR`, `$CONDA_PREFIX/include/eigen3`,
`/opt/homebrew/include/eigen3`, `/usr/include/eigen3`, or `/usr/local/include/eigen3`.

### Build and install

```bash
git clone https://github.com/yoavram-lab/VBPCApy.git
cd VBPCApy
pip install .
```

## Optional extras

```bash
# Development tools (pytest, ruff, mypy, just)
pip install .[dev]

# Plotting utilities (matplotlib)
pip install .[plot]

# Optional data utilities (pandas)
pip install .[data]

# Analysis dependencies (matplotlib, pandas, scikit-learn)
pip install .[analysis]

# Benchmark + plotting stack (joblib, pandas, scikit-learn, seaborn)
pip install .[benchmark]

# Documentation (mkdocs-material, mkdocstrings)
pip install .[docs]

# Install everything
pip install .[dev,plot,data,benchmark,docs]
```

## Using uv

[uv](https://docs.astral.sh/uv/) is the recommended environment manager for development:

```bash
# Core developer environment
uv sync --extra dev --extra data

# Include benchmark + plotting
uv sync --extra dev --extra data --extra benchmark

# Include Octave Python bridge (optional)
uv sync --extra dev --extra data --extra benchmark --extra octave
```
