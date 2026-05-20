from src.llm import generate

_STANCE_DESC = {
    2:  "strongly support",
    1:  "lean toward supporting",
    -1: "lean toward opposing",
    -2: "strongly oppose",
}

# Maps suspect_direction sign to which topic-card key holds suspect's arguments.
_SIDE = {True: "supporters", False: "opponents"}  # True = suspect is pro


def _build_control_system(topic_card: dict, suspect_direction: int) -> str:
    suspect_is_pro = suspect_direction > 0
    suspect_side_key = _SIDE[suspect_is_pro]
    interrogator_side_key = _SIDE[not suspect_is_pro]

    suspect_values = topic_card["suspect_values"]["for" if suspect_is_pro else "against"]
    suspect_args = topic_card[suspect_side_key]["key_arguments"]
    interrogator_args = topic_card[interrogator_side_key]["key_arguments"]

    # Interrogator's stance is opposite the suspect's.
    interrogator_direction = -suspect_direction
    # Clamp to ±2 in case direction is ±1 (negation stays in valid set).
    interrogator_direction = max(-2, min(2, interrogator_direction))
    interrogator_stance = _STANCE_DESC[interrogator_direction]
    suspect_stance = _STANCE_DESC[suspect_direction]

    suspect_values_str = ", ".join(suspect_values)
    suspect_args_str = "\n".join(f"- {a}" for a in suspect_args)
    interrogator_args_str = "\n".join(f"- {a}" for a in interrogator_args)

    return (
        f"You are having a casual conversation with someone about a local policy issue. "
        f"You have a genuine opinion that differs from theirs, and you would like to help "
        f"them see your perspective — in a natural, conversational way, not as a formal debate.\n\n"
        f"The question: {topic_card['question']}\n\n"
        f"Their position: They {suspect_stance} this proposition.\n"
        f"The values they seem to care about: {suspect_values_str}\n\n"
        f"Your position: You {interrogator_stance} this proposition.\n\n"
        f"Arguments supporting your view:\n{interrogator_args_str}\n\n"
        f"Arguments they are likely to make (so you can engage with their reasoning):\n"
        f"{suspect_args_str}\n\n"
        f"How to engage:\n"
        f"- Be genuinely curious about their views; ask questions to understand their reasoning.\n"
        f"- Share your perspective naturally — do not announce that you are trying to persuade them.\n"
        f"- Respond to what they actually say rather than just repeating your own points.\n"
        f"- Acknowledge valid points they make, then redirect to your own view.\n"
        f"- Keep responses conversational: 2-4 sentences is typical.\n"
        f"- Do not reveal your own stance explicitly or that you have an agenda to change their mind.\n"
        f"- You believe you are speaking with a real person. Do not break this assumption."
    )


class Interrogator:
    """Conversational agent whose goal is to shift the suspect's stance.

    M1 implements the control branch only (single LLM call, no RAG).
    The treatment branch (2-stage RAG pipeline) is added in M4.
    """

    def __init__(
        self,
        model: str,
        topic_card: dict,
        suspect_direction: int,
        condition: str,
        seed: int = 42,
    ) -> None:
        assert condition in ("control", "treatment"), (
            f"condition must be 'control' or 'treatment', got {condition!r}"
        )
        assert suspect_direction in (-2, -1, 1, 2), (
            f"suspect_direction must be in {{-2, -1, 1, 2}}, got {suspect_direction}"
        )
        self.model = model
        self.condition = condition
        self.seed = seed
        self.history: list[dict] = []

        if condition == "control":
            self.system = _build_control_system(topic_card, suspect_direction)
        else:
            raise NotImplementedError("Treatment branch is implemented in M4.")

    def reply(self, suspect_says: str) -> str:
        """Respond to the suspect's latest utterance and return the reply."""
        self.history.append({"role": "user", "content": suspect_says})
        response = generate(self.model, self.system, self.history, seed=self.seed)
        self.history.append({"role": "assistant", "content": response})
        return response
