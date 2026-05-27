"""2-turn quad smoke test: all 4 legs run, judge scores in range."""
import pytest
import yaml
from pathlib import Path
from src.quad import run_quad

BASE = Path(__file__).parent.parent


def _card():
    with open(BASE / "data" / "topics" / "housing_prop_123.yaml") as f:
        return yaml.safe_load(f)


def _models():
    with open(BASE / "config" / "models.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def quad_result():
    """Run one 2-turn quad once and share across all tests in this module."""
    return run_quad(_card(), seed=42, models=_models(), n_turns=2,
                    override_condition="control")


def test_quad_has_four_legs(quad_result):
    assert len(quad_result.legs) == 4


def test_quad_scores_in_range(quad_result):
    for leg in quad_result.legs:
        assert all(-2 <= s <= 2 for s in leg.scores), (
            f"Score out of range in leg dir={leg.config.direction}: {leg.scores}"
        )


def test_quad_score_count(quad_result):
    """n_turns=2 → 3 scores per leg (turn 0 + 2 reply turns)."""
    for leg in quad_result.legs:
        assert len(leg.scores) == 3, (
            f"Expected 3 scores for leg dir={leg.config.direction}, got {len(leg.scores)}"
        )


def test_quad_id_format(quad_result):
    assert quad_result.quad_id.startswith("housing_prop_123_42_")


def test_quad_directions(quad_result):
    """Legs alternate ±2 suspect directions as specified."""
    directions = [leg.config.direction for leg in quad_result.legs]
    assert directions == [-2, 2, -2, 2]
