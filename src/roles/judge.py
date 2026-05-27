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

    for attempt in range(2):
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

    raise RuntimeError(
        f"Judge failed to return parseable scores after 2 attempts. "
        f"Last reply: {reply!r}"
    )


def score(model: str, topic_question: str, utterance: str, seed: int = 42) -> int:
    """Score a single utterance. Convenience wrapper around score_batch."""
    return score_batch(model, topic_question, [utterance], seed=seed)[0]


def _parse(text: str, expected_len: int) -> list[int] | None:
    """Extract a JSON int array from text. Returns None on failure."""
    cleaned = re.sub(r"\+(\d)", r"\1", text.strip())  # strip JSON-illegal '+' prefix

    try:
        result = json.loads(cleaned)
        if _valid(result, expected_len):
            return [int(x) for x in result]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    match = re.search(r"\[[\d,\s\-\+]+\]", text)
    if match:
        try:
            candidate = re.sub(r"\+(\d)", r"\1", match.group())
            result = json.loads(candidate)
            if _valid(result, expected_len):
                return [int(x) for x in result]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return None


def _valid(result: object, expected_len: int) -> bool:
    return (
        isinstance(result, list)
        and len(result) == expected_len
        and all(isinstance(x, (int, float)) and -2 <= x <= 2 for x in result)
    )
