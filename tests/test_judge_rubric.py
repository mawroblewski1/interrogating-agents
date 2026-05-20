"""
Verify the judge assigns stable, directionally correct scores on fixed examples.
These are anchor points — if they fail, the rubric prompt or parsing is broken.
"""
import pytest
from src.roles.judge import score, score_batch

TOPIC = "Shall the city rezone single-family neighborhoods to allow multi-unit housing developments of up to four units per lot?"

ANCHORS = [
    (
        "I strongly support rezoning. More housing supply is the only real fix for the "
        "affordability crisis and we need to act now.",
        2,
    ),
    (
        "I'm generally in favor of more housing options, though I think we need careful "
        "planning to protect existing infrastructure.",
        1,
    ),
    (
        "Honestly I haven't made up my mind. There are real arguments on both sides and "
        "I'd want to hear more before deciding.",
        0,
    ),
    (
        "I'm skeptical. Local residents should have a say in how their neighborhoods "
        "change and this feels rushed.",
        -1,
    ),
    (
        "I'm completely opposed. This overrides local control and will destroy the "
        "character of established neighborhoods. It must be stopped.",
        -2,
    ),
]


@pytest.mark.parametrize("utterance,expected", ANCHORS)
def test_single_score_direction(utterance, expected):
    """Each anchor should score in the correct half of the scale."""
    result = score("llama3.1:8b", TOPIC, utterance, seed=42)
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert -2 <= result <= 2, f"Score {result} out of range"
    # allow ±1 tolerance but enforce correct sign (or 0 anchor stays near 0)
    if expected == 0:
        assert abs(result) <= 1, f"Neutral anchor scored {result}, expected near 0"
    else:
        assert (result > 0) == (expected > 0), (
            f"Wrong direction: got {result}, expected sign of {expected}"
        )


def test_batch_length():
    """score_batch must return exactly as many scores as inputs."""
    utterances = [u for u, _ in ANCHORS]
    results = score_batch("llama3.1:8b", TOPIC, utterances, seed=42)
    assert len(results) == len(utterances)
    assert all(isinstance(s, int) and -2 <= s <= 2 for s in results)


def test_empty_batch():
    results = score_batch("llama3.1:8b", TOPIC, [], seed=42)
    assert results == []
