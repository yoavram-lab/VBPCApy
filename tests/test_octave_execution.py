"""Exercise interpreter failure handling without an installed Octave."""

import subprocess
from unittest.mock import Mock

import pytest
from test_octave_parity_smoke import _run_octave_eval


@pytest.mark.parametrize("status", [1, 2, -9])
def test_failed_octave_evaluation_fails_instead_of_skipping(monkeypatch, status):
    result = subprocess.CompletedProcess(
        args=["octave"],
        returncode=status,
        stdout="partial reference output",
        stderr="reference evaluation error",
    )
    monkeypatch.setattr(subprocess, "run", Mock(return_value=result))

    with pytest.raises(pytest.fail.Exception) as failure:
        _run_octave_eval("error('deliberate failure')")

    assert f"status {status}" in str(failure.value)
    assert "partial reference output" in str(failure.value)
    assert "reference evaluation error" in str(failure.value)


def test_successful_octave_evaluation_preserves_command_and_scalar(monkeypatch):
    command = "disp(1.25);"
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=["octave"], returncode=0, stdout="  1.25\n", stderr=""
        )
    )
    monkeypatch.setattr(subprocess, "run", run)

    assert _run_octave_eval(command) == pytest.approx(1.25, rel=0, abs=0)
    run.assert_called_once_with(
        ["octave", "-qf", "--eval", command],
        check=False,
        capture_output=True,
        text=True,
    )
