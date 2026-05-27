"""2-turn control trial end-to-end smoke test."""
import yaml
from pathlib import Path
from src.trial import TrialConfig, run_trial

BASE = Path(__file__).parent.parent


def _card():
    with open(BASE / "data" / "topics" / "housing_prop_123.yaml") as f:
        return yaml.safe_load(f)


def _cfg(direction: int) -> TrialConfig:
    return TrialConfig(
        topic_card=_card(),
        direction=direction,
        condition="control",
        n_turns=2,
        seed=42,
        suspect_model="llama3.1:8b",
        interrogator_selector_model="llama3.1:8b",
        interrogator_generator_model="llama3.1:8b",
        judge_model="llama3.1:8b",
    )


def test_score_count():
    """n_turns=2 → 3 scores (turn 0 + 2 reply turns)."""
    result = run_trial(_cfg(direction=-2))
    assert len(result.scores) == 3, f"Expected 3 scores, got {len(result.scores)}"


def test_scores_in_range():
    result = run_trial(_cfg(direction=-2))
    assert all(-2 <= s <= 2 for s in result.scores), (
        f"Score out of range: {result.scores}"
    )


def test_turn_records_match():
    result = run_trial(_cfg(direction=-2))
    assert len(result.turns) == 3
    assert result.turns[0].interrogator_says is None  # suspect opens, no reply
    assert all(r.interrogator_says is not None for r in result.turns[1:])


def test_pro_direction():
    """Pro-side suspect (direction=+2) also runs without error."""
    result = run_trial(_cfg(direction=2))
    assert len(result.scores) == 3
    assert all(-2 <= s <= 2 for s in result.scores)
