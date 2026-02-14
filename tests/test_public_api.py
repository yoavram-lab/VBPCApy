"""Public API surface tests for package-level exports."""

from __future__ import annotations

from importlib.metadata import version

import vbpca_py


def test_public_symbols_are_exported() -> None:
    expected = {
        "VBPCA",
        "AutoEncoder",
        "MissingAwareMinMaxScaler",
        "MissingAwareOneHotEncoder",
        "MissingAwareStandardScaler",
        "SelectionConfig",
        "select_n_components",
    }

    exported = set(vbpca_py.__all__)
    assert expected.issubset(exported)

    for name in expected:
        assert hasattr(vbpca_py, name)


def test_package_version_matches_distribution_metadata() -> None:
    assert isinstance(vbpca_py.__version__, str)
    assert vbpca_py.__version__ == version("vbpca_py")
