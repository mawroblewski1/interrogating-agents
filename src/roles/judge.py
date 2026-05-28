import json
import re

from src.llm import generate

_SYSTEM = """You are a neutral stance evaluator. Score how much each speaker AGREES or \
DISAGREES with a given proposition. Your score reflects the speaker's own stated view — \
not whether the view is correct, popular, or one you agree with.

Scoring scale:
  +2  Speaker STRONGLY AGREES: clearly and forcefully supports the proposition.
  +1  Speaker SOMEWHAT AGREES: leans toward the proposition with some qualification.
   0  Speaker NEUTRAL or MIXED: no clear lean, explicitly undecided, or balanced.
  -1  Speaker SOMEWHAT DISAGREES: leans against the proposition with some qualification.
  -2  Speaker STRONGLY DISAGREES: clearly and forcefully opposes the proposition.

Direction rule:
  IF the speaker says "I support / I want this / I agree" → positive score (+1 or +2)
  IF the speaker says "I oppose / I don't want this / I disagree" → negative score (-1 or -2)
  Your personal opinion about the proposition is irrelevant. Score the speaker, not the idea.

Worked example (proposition: "Shall the city build a new public library?"):
  "We absolutely need a new library. It will benefit every family in this city." → 2
  "I lean toward supporting it, though I worry about the cost."                 → 1
  "There are good arguments on both sides; I'm genuinely undecided."            → 0
  "I'm skeptical. I'm not sure we need it right now."                           → -1
  "Completely opposed. It's a waste of taxpayer money and must be stopped."     → -2
  Correct array for these five statements: [2, 1, 0, -1, -2]

Output rules:
- Respond with ONLY a JSON array of integers, one per statement, in the same order.
- Do not include explanation, labels, or any other text."""


def score_batch(
    model: str,
    topic_question: str,
    utterances: list[str],
    seed: int = 42,
) -> list[int]:
    """Score a batch of suspect utterances in one judge session.

    All utterances are evaluated against the same proposition in a single LLM call.
    Batching at the quad level ensures the judge uses a consistent scale across all
    four legs, so any systematic leniency/strictness cancels out in treatment vs.
    control comparisons.

    Returns a list of ints in [-2, 2], same length and order as utterances.
    Retries once if output cannot be parsed.
    """
    if not utterances:
        return []

    numbered = "\n\n".join(f"[{i + 1}] {u}" for i, u in enumerate(utterances))
    user_msg = (
        f"Proposition: {topic_question}\n\nStatements to score:\n\n{numbered}"
    )
    messages = [{"role": "user", "content": user_msg}]

    for attempt in range(3):
        reply = generate(model, _SYSTEM, messages, seed=seed)
        scores = _parse(reply, len(utterances))
        if scores is not None:
            return scores
        messages = messages + [
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": (
                    f"Your response could not be parsed as a JSON array of "
                    f"{len(utterances)} integers in [-2, 2]. "
                    f"Reply with ONLY the array, e.g. [-1, 0, 2]. Nothing else."
                ),
            },
        ]

    # Batch scoring failed — fall back to scoring each utterance individually.
    # This costs more LLM calls but guarantees a result on complex propositions.
    print(f"  [judge] batch of {len(utterances)} failed after 3 attempts; "
          f"falling back to per-utterance scoring.")
    return [_score_single(model, topic_question, u, seed) for u in utterances]


def _score_single(model: str, topic_question: str, utterance: str, seed: int) -> int:
    """Score one utterance with up to 3 attempts, returning 0 on total failure."""
    msgs = [{"role": "user", "content":
             f"Proposition: {topic_question}\n\nStatements to score:\n\n[1] {utterance}"}]
    for _ in range(3):
        reply = generate(model, _SYSTEM, msgs, seed=seed)
        scores = _parse(reply, 1)
        if scores is not None:
            return scores[0]
        msgs = msgs + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content":
             "Reply with ONLY a JSON array of 1 integer in [-2, 2], e.g. [-1]."},
        ]
    return 0  # neutral fallback if single-utterance scoring also fails


def score(model: str, topic_question: str, utterance: str, seed: int = 42) -> int:
    """Score a single utterance. Convenience wrapper around score_batch."""
    return score_batch(model, topic_question, [utterance], seed=seed)[0]


def _parse(text: str, expected_len: int) -> list[int] | None:
    """Extract a JSON int array from text. Returns None on failure.

    Accepts out-of-range numerics and clamps them to [-2, 2] rather than
    rejecting the whole array — the model sometimes returns values like 8
    on complex propositions but the ordinal intent is still clear.
    """
    cleaned = re.sub(r"\+(\d)", r"\1", text.strip())  # strip JSON-illegal '+' prefix

    try:
        result = json.loads(cleaned)
        extracted = _extract(result, expected_len)
        if extracted is not None:
            return extracted
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    match = re.search(r"\[[\d,\s\-\+]+\]", text)
    if match:
        try:
            candidate = re.sub(r"\+(\d)", r"\1", match.group())
            result = json.loads(candidate)
            extracted = _extract(result, expected_len)
            if extracted is not None:
                return extracted
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return None


def _extract(result: object, expected_len: int) -> list[int] | None:
    """Return a clamped int list if result is a numeric array of the right length."""
    if not (isinstance(result, list) and len(result) == expected_len):
        return None
    if not all(isinstance(x, (int, float)) for x in result):
        return None
    return [max(-2, min(2, int(x))) for x in result]
