"""
Verify that retrieval returns sensible top-k results for known queries.
These tests do NOT call Ollama — they only test the vector index.
"""
import pytest
from src.rag.retrieve import retrieve_techniques, retrieve_arguments

pytestmark = pytest.mark.slow

# ── Technique retrieval ────────────────────────────────────────────────────────

def test_technique_returns_k_results():
    cards = retrieve_techniques("the suspect seems guarded and reluctant to engage", turn=1, budget=6, k=3)
    assert 1 <= len(cards) <= 3

def test_technique_phase_early():
    """Early-phase query should not return late-only cards."""
    cards = retrieve_techniques("opening a new conversation with an unknown person", turn=1, budget=6, k=5)
    for c in cards:
        assert c.phase in ("early", "any"), f"Early query returned card with phase={c.phase}"

def test_technique_phase_late():
    """Late-phase query should not return early-only cards."""
    cards = retrieve_techniques("the conversation is almost over and suspect is disengaging", turn=5, budget=6, k=5)
    for c in cards:
        assert c.phase in ("late", "any"), f"Late query returned card with phase={c.phase}"

def test_technique_has_required_fields():
    cards = retrieve_techniques("suspect seems ambivalent about their position", turn=2, budget=6, k=3)
    assert len(cards) >= 1
    card = cards[0]
    assert card.name
    assert card.phase in ("early", "middle", "late", "any")
    assert isinstance(card.out_of_bounds, list)

def test_technique_rapport_for_guarded_suspect():
    """A guarded or untrusting suspect should surface rapport/orbit/MI techniques."""
    cards = retrieve_techniques(
        "the suspect is suspicious of my motives and keeping their answers short",
        turn=1, budget=6, k=5
    )
    names = {c.stem for c in cards}
    assert names & {"rapport_building", "orbit", "motivational_interviewing", "oars"}, (
        f"Expected rapport-related card in top-5 for guarded suspect, got: {names}"
    )

def test_technique_dissonance_for_contradiction():
    """A suspect contradicting themselves should surface dissonance/Socratic techniques."""
    cards = retrieve_techniques(
        "suspect says they value evidence-based policy but is ignoring the data I cited",
        turn=3, budget=6, k=5
    )
    names = {c.stem for c in cards}
    assert names & {"cognitive_dissonance", "surfacing_contradictions", "socratic_questioning"}, (
        f"Expected dissonance/Socratic card in top-5, got: {names}"
    )


# ── Argument retrieval ─────────────────────────────────────────────────────────

def test_argument_returns_results():
    args = retrieve_arguments("housing_prop_123", "for",
                              "the suspect is worried about housing costs", k=3)
    assert 1 <= len(args) <= 3

def test_argument_side_for():
    """'for' arguments should mention supply/affordability, not opposition framing."""
    args = retrieve_arguments("housing_prop_123", "for",
                              "housing affordability is a concern", k=5)
    assert args, "Expected at least one argument"
    combined = " ".join(args).lower()
    assert any(w in combined for w in ["supply", "afford", "cost", "density", "access"]), (
        f"'for' arguments don't mention expected themes: {args}"
    )

def test_argument_side_against():
    """'against' arguments should mention local control / infrastructure, not pro-housing framing."""
    args = retrieve_arguments("housing_prop_123", "against",
                              "neighborhood character and local control", k=5)
    assert args
    combined = " ".join(args).lower()
    assert any(w in combined for w in ["local", "infrastructure", "character", "control", "developer"]), (
        f"'against' arguments don't mention expected themes: {args}"
    )

def test_argument_topic_isolation():
    """Arguments retrieved for housing should not bleed into arts or transit topics."""
    housing_args = retrieve_arguments("housing_prop_123", "for", "housing policy", k=5)
    arts_args    = retrieve_arguments("arts_funding_measure_b", "for", "arts and culture funding", k=5)
    assert set(housing_args).isdisjoint(set(arts_args)), (
        "Housing and arts arguments overlapping — topic filter not working"
    )

def test_argument_wrong_side_differs():
    """The 'for' and 'against' argument sets for the same query should differ."""
    for_args     = retrieve_arguments("housing_prop_123", "for", "housing", k=3)
    against_args = retrieve_arguments("housing_prop_123", "against", "housing", k=3)
    assert for_args != against_args
