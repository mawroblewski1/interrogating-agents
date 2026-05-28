"""OS-aware helpers for locating and invoking the Ollama binary."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _windows_fallback_paths() -> list[Path]:
    candidates: list[Path] = []
    localappdata = os.environ.get("LOCALAPPDATA", r"C:\Users\Default\AppData\Local")
    candidates.append(Path(localappdata) / "Programs" / "Ollama" / "ollama.exe")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    return candidates


def _unix_fallback_paths() -> list[Path]:
    return [
        Path("/usr/local/bin/ollama"),
        Path("/opt/homebrew/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path("/snap/bin/ollama"),
        Path.home() / ".local" / "bin" / "ollama",
    ]


def _candidate_fallbacks() -> list[Path]:
    if sys.platform.startswith("win"):
        return _windows_fallback_paths()
    return _unix_fallback_paths()


def find_ollama_binary() -> Path | None:
    """Return absolute path to ollama, or None if not found."""
    which_hit = shutil.which("ollama")
    if which_hit:
        return Path(which_hit)
    for candidate in _candidate_fallbacks():
        if candidate.exists():
            return candidate
    return None


def install_hint() -> str:
    """One-line install hint for the current OS."""
    if sys.platform == "darwin":
        return "Install: brew install ollama  (or download from https://ollama.com/download)"
    if sys.platform.startswith("linux"):
        return "Install: curl -fsSL https://ollama.com/install.sh | sh"
    if sys.platform.startswith("win"):
        return "Install: download the Windows installer from https://ollama.com/download"
    return "Install: see https://ollama.com/download"
