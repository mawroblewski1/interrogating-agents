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
