from pathlib import Path
from unittest.mock import patch

from scripts.ollama_runtime import find_ollama_binary, _windows_fallback_paths, _unix_fallback_paths


def test_find_ollama_returns_path_when_which_succeeds(tmp_path):
    fake = tmp_path / "ollama"
    fake.write_text("#!/bin/sh\necho hi")
    fake.chmod(0o755)
    with patch("scripts.ollama_runtime.shutil.which", return_value=str(fake)):
        assert find_ollama_binary() == Path(str(fake))


def test_find_ollama_returns_none_when_nothing_found():
    with patch("scripts.ollama_runtime.shutil.which", return_value=None), \
         patch("scripts.ollama_runtime._candidate_fallbacks", return_value=[]):
        assert find_ollama_binary() is None


def test_find_ollama_uses_fallback_when_which_fails(tmp_path):
    fake = tmp_path / "ollama"
    fake.write_text("x")
    fake.chmod(0o755)
    with patch("scripts.ollama_runtime.shutil.which", return_value=None), \
         patch("scripts.ollama_runtime._candidate_fallbacks", return_value=[fake]):
        assert find_ollama_binary() == fake


def test_windows_fallback_paths_use_localappdata_when_present(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"D:\custom\appdata")
    paths = _windows_fallback_paths()
    joined = " ".join(str(p) for p in paths)
    assert r"D:\custom\appdata" in joined or "D:/custom/appdata" in joined


def test_windows_fallback_paths_use_program_files_when_present(monkeypatch):
    monkeypatch.setenv("ProgramFiles", r"D:\custom\programfiles")
    paths = _windows_fallback_paths()
    joined = " ".join(str(p) for p in paths)
    assert r"D:\custom\programfiles" in joined or "D:/custom/programfiles" in joined


def test_unix_fallback_paths_include_homebrew_and_usr_local():
    # Use as_posix() so the assertion is portable: on Windows, str(Path("/usr/local/bin/ollama"))
    # renders as "\\usr\\local\\bin\\ollama", which would fail the substring check.
    paths = [p.as_posix() for p in _unix_fallback_paths()]
    assert "/usr/local/bin/ollama" in paths
    assert "/opt/homebrew/bin/ollama" in paths
    assert "/snap/bin/ollama" in paths


def test_candidate_fallbacks_picks_windows_branch_on_win():
    from scripts.ollama_runtime import _candidate_fallbacks
    with patch("scripts.ollama_runtime.sys") as fake_sys:
        fake_sys.platform = "win32"
        result = _candidate_fallbacks()
    joined = " ".join(str(p) for p in result)
    assert "Ollama" in joined
    assert "/usr/local/bin/ollama" not in joined


def test_candidate_fallbacks_picks_unix_branch_on_darwin():
    from scripts.ollama_runtime import _candidate_fallbacks
    with patch("scripts.ollama_runtime.sys") as fake_sys:
        fake_sys.platform = "darwin"
        result = _candidate_fallbacks()
    # as_posix() keeps the assertion portable across Windows/POSIX path separators.
    joined = " ".join(p.as_posix() for p in result)
    assert "/usr/local/bin/ollama" in joined
