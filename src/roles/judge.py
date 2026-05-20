import json
import re

from src.llm import generate

_SYSTEM = """You are a neutral stance evaluator. Your only job is to score how much each \
speaker AGREES or DISAGREES with a given proposition.

Scoring scale — measures agreement with the proposition:
  +2  Speaker STRONGLY AGREES: clearly and forcefully supports the proposition;
      explicitly says they want it to happen or endorses it without reservation.
  +1  Speaker SOMEWHAT AGREES: leans toward supporting the proposition,
      though with some qualification or uncertainty.
   0  Speaker is NEUTRAL or MIXED: no clear lean, explicitly undecided,
      or expresses roughly equal support and opposition.
  -1  Speaker SOMEWHAT DISAGREES: leans against the proposition,
      though with some qualification or uncertainty.
  -2  Speaker STRONGLY DISAGREES: clearly and forcefully opposes the proposition;
      explicitly says they do not want it to happen or rejects it without reservation.

Key rule: a positive score means the speaker WANTS the proposition to pass.
          a negative score means the speaker WANTS the proposition to fail.

Additional rules:
- Score only what is explicitly expressed. Do not infer unstated positions.
- If a statement is evasive, off-topic, or contains no evaluable stance, score 0.
- Respond with ONLY a JSON array of integers, one per statement, in the same order.
  Example for 3 statements: [-1, 0, 2]
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
