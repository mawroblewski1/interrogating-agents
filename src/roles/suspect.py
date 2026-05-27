from src.llm import generate

_STANCE_DESC = {
    2:  "strongly support",
    1:  "lean toward supporting",
    -1: "lean toward opposing",
    -2: "strongly oppose",
}

_OPENING_PROMPT = "What are your thoughts on this issue? I'd love to hear your perspective."


def _build_system(topic_card: dict, direction: int) -> str:
    side = "for" if direction > 0 else "against"
    values = topic_card["suspect_values"][side]
    arguments = (
        topic_card["supporters"]["key_arguments"]
        if side == "for"
        else topic_card["opponents"]["key_arguments"]
    )

    values_str = ", ".join(values)
    arguments_str = "\n".join(f"- {a}" for a in arguments)
    stance_desc = _STANCE_DESC[direction]

    return (
        f"You are a person with a genuine opinion on a local policy question. "
        f"You are having a casual conversation with someone you just met. "
        f"You believe they are broadly sympathetic to your perspective.\n\n"
        f"The question: {topic_card['question']}\n\n"
        f"Your position: You {stance_desc} this proposition.\n\n"
        f"The values that matter to you: {values_str}\n\n"
        f"Arguments you are familiar with that support your position:\n{arguments_str}\n\n"
        f"How to behave:\n"
        f"- Speak naturally, as a real person would — not as a debater or policy expert.\n"
        f"- Draw on your values and the arguments you know, but don't recite them mechanically.\n"
        f"- Engage thoughtfully with what the other person says. If they raise a point "
        f"you haven't considered, acknowledge it genuinely.\n"
        f"- Maintain your position unless you encounter reasoning that genuinely "
        f"challenges your values or presents evidence you cannot dismiss.\n"
        f"- Do not reverse your position without a clear reason — real people don't "
        f"flip their views in a single conversation unless something truly compelling is said.\n"
        f"- Keep responses conversational: 2-4 sentences is typical unless making a "
        f"detailed argument.\n"
        f"- You believe you are speaking with a real person. Do not break this "
        f"assumption or acknowledge being an AI."
    )


class Suspect:
    """Stateful conversational agent holding one side of a debate topic.

    Maintains its own history so trial.py only needs to pass interrogator
    messages and receive replies.
    """

    def __init__(
        self,
        model: str,
        topic_card: dict,
        direction: int,
        seed: int = 42,
    ) -> None:
        assert direction in (-2, -1, 1, 2), (
            f"direction must be in {{-2, -1, 1, 2}}, got {direction}"
        )
        self.model = model
        self.system = _build_system(topic_card, direction)
        self.seed = seed
        self.history: list[dict] = []

    def opening_statement(self) -> str:
        """Generate the suspect's unprompted turn-0 statement."""
        self.history.append({"role": "user", "content": _OPENING_PROMPT})
        reply = generate(self.model, self.system, self.history, seed=self.seed)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reply(self, interrogator_says: str) -> str:
        """Respond to one interrogator turn and return the suspect's reply."""
        self.history.append({"role": "user", "content": interrogator_says})
        response = generate(self.model, self.system, self.history, seed=self.seed)
        self.history.append({"role": "assistant", "content": response})
        return response
