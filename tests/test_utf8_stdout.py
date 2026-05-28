"""Verify CLI entry points reconfigure stdout to UTF-8 so non-ASCII content
in prompt strings or topic cards never crashes the process on a Windows
cp1252-default terminal."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run_module_help(module: str, env_overrides: dict) -> subprocess.CompletedProcess:
    """Run `python -m <module> --help` from the repo root with a narrow stdout encoding."""
    env = {
        "PYTHONIOENCODING": "cp1252",
        "PATH": "",
        "PYTHONPATH": str(REPO),
        "PYTHONUNBUFFERED": "1",
    }
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO,
        capture_output=True,
        env=env,
        text=False,
    )


def test_experiment_help_does_not_crash_on_cp1252():
    """`python -m src.experiment --help` should exit 0 even when stdout is forced to cp1252."""
    result = _run_module_help("src.experiment", {})
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_trial_help_does_not_crash_on_cp1252():
    """`python -m src.trial --help` should exit 0 even when stdout is forced to cp1252."""
    result = _run_module_help("src.trial", {})
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
