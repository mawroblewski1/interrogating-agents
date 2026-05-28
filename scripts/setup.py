"""Cross-OS bootstrap. Run: python scripts/setup.py"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.ollama_runtime import find_ollama_binary, install_hint  # noqa: E402

REQS = REPO / "requirements.txt"
MODEL = "llama3.1:8b"


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def check_python() -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"Python 3.11+ required, found {sys.version}")
    print(f"[ok] python {sys.version.split()[0]}")


def install_requirements() -> None:
    if not REQS.exists():
        sys.exit(f"missing {REQS}")
    if _run([sys.executable, "-m", "pip", "install", "-r", str(REQS)]) != 0:
        sys.exit("pip install failed")
    print("[ok] requirements installed")


def check_ollama() -> Path:
    binary = find_ollama_binary()
    if binary is None:
        print("[!] ollama not found.")
        print("    " + install_hint())
        sys.exit(1)
    print(f"[ok] ollama at {binary}")
    return binary


def pull_model(binary: Path) -> None:
    if _run([str(binary), "pull", MODEL]) != 0:
        sys.exit(f"failed to pull {MODEL}")
    print(f"[ok] model {MODEL} pulled")


def run_fast_tests() -> None:
    rc = _run([sys.executable, "-m", "pytest", "-m", "not slow", "-v"])
    if rc != 0:
        sys.exit("fast tests failed")
    print("[ok] fast tests pass")


def main() -> None:
    check_python()
    install_requirements()
    binary = check_ollama()
    pull_model(binary)
    run_fast_tests()
    print("\nSetup complete. Try:")
    print(f"  {sys.executable} -m src.trial --topic housing_prop_123 --direction -2 --condition control --n_turns 3")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
