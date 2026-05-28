"""
Pytest markers to separate fast unit tests from slow LLM integration tests.

Usage:
    pytest tests/ -m "not slow"          # fast tests only (no LLM calls)
    pytest tests/ -m slow                # LLM tests only
    pytest tests/test_judge_rubric.py    # run one file
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that make LLM calls (deselect with -m 'not slow')")


from scripts.ollama_runtime import find_ollama_binary


def pytest_collection_modifyitems(config, items):
    if find_ollama_binary() is not None:
        return
    skip_slow = pytest.mark.skip(reason="ollama binary not found; skipping slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
