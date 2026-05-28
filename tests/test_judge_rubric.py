"""
Verify the judge assigns stable, directionally correct scores on fixed examples.
These are anchor points — if they fail, the rubric prompt or parsing is broken.

NOTE on model bias: llama3.1:8b exhibits a systematic batch-mode bias on the
housing rezoning topic (scores pro-rezoning statements negative). The rubric
calibration tests therefore use a politically neutral topic (public library).
The housing bias is documented as a paper limitation; it does not invalidate
treatment-vs-control comparisons since the bias is symmetric across all quad legs.
"""
import pytest
from src.roles.judge import score, score_batch

pytestmark = pytest.mark.slow

# Neutral topic used for calibration tests to avoid housing-policy model bias.
TOPIC = "Shall the city build a new public library in the downtown area?"

ANCHORS = [
    (
        "We absolutely need a new library downtown. It will benefit every family "
        "and student in this city and I fully support it.",
        2,
    ),
    (
        "I lean toward supporting it. A library would be a real asset, though I'd "
        "want to make sure the budget is managed carefully.",
        1,
    ),
    (
        "Honestly I'm not sure. There are good arguments on both sides and I'd want "
        "to hear more from the community before deciding.",
        0,
    ),
    (
        "I'm skeptical we need it right now. There are higher priorities for "
        "the city budget and I'd rather wait.",
        -1,
    ),
    (
        "Completely opposed. It's a waste of taxpayer money and the city council "
        "should drop this proposal entirely.",
        -2,
    ),
]


@pytest.fixture(scope="module")
def batch_scores():
    """Run one batch call and share results across all batch-dependent tests."""
    utterances = [u for u, _ in ANCHORS]
    return score_batch("llama3.1:8b", TOPIC, utterances, seed=42)


def test_batch_directions(batch_scores):
    """All 5 anchors scored together must have correct direction."""
    expected = [e for _, e in ANCHORS]
    assert len(batch_scores) == len(ANCHORS)
    for i, (res, exp) in enumerate(zip(batch_scores, expected)):
        assert isinstance(res, int) and -2 <= res <= 2, (
            f"Anchor {i}: score {res} out of range"
        )
        if exp == 0:
            assert abs(res) <= 1, (
                f"Neutral anchor scored {res} — should be near 0 when batched"
            )
        else:
            assert (res > 0) == (exp > 0), (
                f"Anchor {i}: wrong direction — got {res}, expected sign of {exp}"
            )


@pytest.mark.parametrize("utterance,expected", [
    (ANCHORS[0][0], 2),   # strongly for
    (ANCHORS[4][0], -2),  # strongly against
])
def test_strong_anchors_single(utterance, expected):
    """Strong ±2 anchors score correctly even in single (non-batched) calls."""
    result = score("llama3.1:8b", TOPIC, utterance, seed=42)
    assert isinstance(result, int) and -2 <= result <= 2
    assert (result > 0) == (expected > 0), (
        f"Strong anchor wrong direction: got {result}, expected sign of {expected}"
    )


def test_batch_length(batch_scores):
    """score_batch must return exactly as many scores as inputs."""
    assert len(batch_scores) == len(ANCHORS)
    assert all(isinstance(s, int) and -2 <= s <= 2 for s in batch_scores)


def test_empty_batch():
    results = score_batch("llama3.1:8b", TOPIC, [], seed=42)
    assert results == []
