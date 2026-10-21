"""Run bounded numerical references, requiring every selected check to execute."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    "tests/test_update_equations.py",
    "tests/test_update_trajectory.py",
    "tests/test_posterior_variance.py",
    "tests/test_sparse_views.py",
    "tests/test_recommended_stopping.py",
    "tests/test_mean.py::test_subtract_mu_dense_with_masking",
    "tests/test_cost.py::test_cost_basic_dense_no_priors",
    "tests/test_rotate.py::test_rotate_python_data_space_invariance_update_bias_true",
    "tests/test_rotate.py::test_rotate_python_data_space_invariance_update_bias_false",
    "tests/test_pca_full.py::test_pca_full_strict_legacy_regression_rms_value",
    "tests/test_full_update.py::test_recompute_rms_matches_manual",
    "tests/test_full_update.py::test_update_noise_variance_fully_observed_matches_manual_formula",
    "tests/test_full_update.py::test_update_noise_variance_partial_observed_matches_manual_formula",
)


class RequiredReferences:
    """A missing, skipped or expected-failing reference cannot pass the gate."""

    def __init__(self) -> None:
        self.unavailable: list[str] = []

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.unavailable.append(report.nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped or hasattr(report, "wasxfail"):
            self.unavailable.append(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.unavailable or session.testscollected == 0:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter) -> None:
        if self.unavailable:
            terminalreporter.write_sep("=", "Required numerical references unavailable")
            for name in self.unavailable:
                terminalreporter.write_line(name)


def main() -> int:
    """Execute the same required references locally and in CI.

    Returns:
        The pytest exit status, failing when a reference is unavailable.
    """
    return int(
        pytest.main(
            ["-q", "-o", "addopts=", *(str(ROOT / name) for name in CHECKS)],
            plugins=[RequiredReferences()],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
