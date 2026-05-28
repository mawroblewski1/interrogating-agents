"""2-turn treatment trial smoke test — verifies the 2-stage RAG pipeline runs end-to-end."""
import pytest
import yaml
from pathlib import Path
from src.trial import TrialConfig, run_trial
from src.roles.interrogator import SuspectModel, PersonaState

pytestmark = pytest.mark.slow

BASE = Path(__file__).parent.parent


def _card():
    with open(BASE / "data" / "topics" / "housing_prop_123.yaml") as f:
        return yaml.safe_load(f)


def _cfg(direction: int) -> TrialConfig:
    return TrialConfig(
        topic_card=_card(),
        direction=direction,
        condition="treatment",
        n_turns=2,
        seed=42,
        suspect_model="llama3.1:8b",
        interrogator_selector_model="llama3.1:8b",
        interrogator_generator_model="llama3.1:8b",
        judge_model="llama3.1:8b",
    )


def test_treatment_score_count():
    """n_turns=2 → 3 scores (turn 0 + 2 reply turns)."""
    result = run_trial(_cfg(direction=-2))
    assert len(result.scores) == 3


def test_treatment_scores_in_range():
    result = run_trial(_cfg(direction=-2))
    assert all(-2 <= s <= 2 for s in result.scores)


def test_treatment_turn_records():
    result = run_trial(_cfg(direction=-2))
    assert len(result.turns) == 3
    assert result.turns[0].interrogator_says is None
    assert all(r.interrogator_says is not None for r in result.turns[1:])


def test_treatment_pro_direction():
    """Pro-side suspect (direction=+2) treatment trial also runs cleanly."""
    result = run_trial(_cfg(direction=2))
    assert len(result.scores) == 3
    assert all(-2 <= s <= 2 for s in result.scores)
