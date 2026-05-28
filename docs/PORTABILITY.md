# Portability Audit

This document records what made the M2 code Windows-only and what changed.

## Audit method

Read-only greps from the repo root on branch `sanjay-portability` (HEAD off
`upstream/william-dev` @ `25275f3`):

```
grep -rn 'LOCALAPPDATA\|ProgramFiles\|cp1252\|powershell\|\.exe' src/ tests/ config/ data/ README.md
grep -rn 'Δ\|→\|—' src/ tests/
grep -rn 'os\.sep\|os\.path\.join' src/ tests/
```

Findings at HEAD:

- **No** hits for `LOCALAPPDATA`, `ProgramFiles`, `cp1252`, `powershell`, or
  `.exe` anywhere under `src/`, `tests/`, `config/`, `data/`, or `README.md`.
  Windows paths are not hardcoded in code — they live only in setup
  instructions that are not in the repo.
- **No** `Δ` glyph in `src/experiment.py`. The file already uses the ASCII
  string `delta=` at `src/experiment.py:158`. The Unicode arrows (`→`) and
  em-dashes (`—`) that do appear (e.g. `src/roles/judge.py:18-27`,
  `tests/test_quad_smoke.py:39`) live inside prompt string literals and
  comments — they are not printed to the console at runtime in a way that
  triggered the original cp1252 crash.
- **No** `os.sep` or `os.path.join` usage. File I/O in `src/experiment.py`
  uses `pathlib` and `open(..., encoding="utf-8")` (lines 40, 101), so file
  paths and file encoding are already cross-platform. The remaining
  cross-OS gap is **console encoding**, not file encoding.
- `README.md` is effectively empty: 22 bytes, contains only the heading
  `# interrogating-agents`. There are no install or run instructions in the
  repo today — Windows-only setup steps live in external session notes.

## Windows-only assumptions (state at branch point)

1. **Ollama binary discovery** — no code-side discovery. The project
   assumes `ollama` is already on `PATH` or that the user knows the
   Windows-specific install location (`%LOCALAPPDATA%\Programs\Ollama\
   ollama.exe`) from out-of-repo notes.
2. **Console encoding** — the `delta=` workaround at
   `src/experiment.py:158` is already ASCII-safe at HEAD, but the root
   cause (Python defaulting to cp1252 stdout on Windows) is untreated.
   Any future `print()` of a non-ASCII glyph from CLI entry points would
   regress. Prompt strings in `src/roles/` already contain `→` / `—`.
3. **Setup instructions** — `README.md` is empty. There is no documented
   path for macOS or Linux users to install Ollama, pull a model, or run
   the experiment.
4. **Stdlib reliance** — already cross-platform: `urllib.request`,
   `pathlib.Path`, and `subprocess` are used consistently. No Windows-only
   stdlib is relied on. File opens already pin `encoding="utf-8"`.

## Changes planned in this branch

The "After" column describes the end state of the whole `sanjay-portability`
branch and is a forward-looking promise fulfilled by later tasks.

| Concern | Before (at HEAD) | After (end of branch) |
|---|---|---|
| Ollama discovery | not in code; out-of-repo notes only | `scripts/ollama_runtime.find_ollama_binary()` via `shutil.which` + per-OS fallbacks |
| Console encoding | `delta=` ASCII workaround already in place, but cp1252 root cause untreated | force `sys.stdout.reconfigure(encoding="utf-8")` on CLI entry; ASCII workaround kept as belt-and-suspenders |
| Install docs | `README.md` is effectively empty (22 bytes) | three README sections: macOS (brew), Linux (curl install), Windows (installer + PowerShell) |
| Setup | manual pip + ollama pull (undocumented) | `python scripts/setup.py` does both, OS-aware |
| CI | none | GitHub Actions matrix on ubuntu/macos/windows running Ollama-free unit tests |

## Deliberately unchanged

- File paths in code (already `pathlib`-based).
- File encoding (`open(..., encoding="utf-8")` is already explicit).
- `urllib.request` HTTP client (stdlib, cross-OS).
- Test markers and structure (William's `slow` marker convention preserved).
- Non-ASCII glyphs inside prompt string literals in `src/roles/` — they are
  data, not console output, and `utf-8` source encoding handles them
  natively. They become safe to print once Task 4 forces UTF-8 stdout.
