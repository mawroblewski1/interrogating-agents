"""
run_trial: executes one full trial leg and returns scores + transcript.

CLI usage:
    python -m src.trial --topic housing_prop_123 --direction -2 --condition control
"""
from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field
from pathlib import Path

from src.roles.suspect import Suspect
from src.roles.interrogator import Interrogator
from src.roles.judge import score_batch


@dataclass
class TrialConfig:
    topic_card: dict
    direction: int        # suspect's initial direction: -2 | -1 | 1 | 2
    condition: str        # "control" | "treatment"
    n_turns: int
    seed: int
    suspect_model: str
    interrogator_model: str
    judge_model: str


@dataclass
class TurnRecord:
    turn: int
    suspect_says: str
    interrogator_says: str | None  # None for turn 0 (suspect opens, no reply yet)
    stance_score: int


@dataclass
class DialogueResult:
    """Dialogue loop output before judge scoring. Used by quad.py for batch scoring."""
    config: TrialConfig
    utterances: list[str]      # suspect utterances only, one per turn
    turns: list[TurnRecord]    # stance_score=0 until filled by caller


@dataclass
class TrialResult:
    config: TrialConfig
    scores: list[int]           # one per suspect turn, length = n_turns + 1
    turns: list[TurnRecord]


def run_dialogue(cfg: TrialConfig) -> DialogueResult:
    """Run the conversation loop and return utterances without judge scoring.

    Separated from run_trial so quad.py can batch all four legs into one
    judge call for calibration consistency.
    """
    suspect = Suspect(cfg.suspect_model, cfg.topic_card, cfg.direction, seed=cfg.seed)
    interrogator = Interrogator(
        cfg.interrogator_model,
        cfg.topic_card,
        suspect_direction=cfg.direction,
        condition=cfg.condition,
        seed=cfg.seed,
    )

    utterances: list[str] = []
    turns: list[TurnRecord] = []

    opening = suspect.opening_statement()
    utterances.append(opening)
    turns.append(TurnRecord(turn=0, suspect_says=opening, interrogator_says=None, stance_score=0))

    for t in range(1, cfg.n_turns + 1):
        interrogator_says = interrogator.reply(utterances[-1])
        suspect_says = suspect.reply(interrogator_says)
        utterances.append(suspect_says)
        turns.append(TurnRecord(turn=t, suspect_says=suspect_says,
                                interrogator_says=interrogator_says, stance_score=0))

    return DialogueResult(config=cfg, utterances=utterances, turns=turns)


def run_trial(cfg: TrialConfig) -> TrialResult:
    """Run one full trial leg including judge scoring."""
    dialogue = run_dialogue(cfg)
    scores = score_batch(
        cfg.judge_model,
        cfg.topic_card["question"],
        dialogue.utterances,
        seed=cfg.seed,
    )
    for i, record in enumerate(dialogue.turns):
        record.stance_score = scores[i]
    return TrialResult(config=cfg, scores=scores, turns=dialogue.turns)


def _load_config() -> TrialConfig:
    base = Path(__file__).parent.parent
    with open(base / "config" / "models.yaml") as f:
        models = yaml.safe_load(f)
    with open(base / "config" / "trial_defaults.yaml") as f:
        defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--direction", type=int, required=True, choices=[-2, -1, 1, 2])
    parser.add_argument("--condition", required=True, choices=["control", "treatment"])
    parser.add_argument("--n_turns", type=int, default=defaults["n_turns"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    args = parser.parse_args()

    topic_path = base / "data" / "topics" / f"{args.topic}.yaml"
    with open(topic_path) as f:
        topic_card = yaml.safe_load(f)

    return TrialConfig(
        topic_card=topic_card,
        direction=args.direction,
        condition=args.condition,
        n_turns=args.n_turns,
        seed=args.seed,
        suspect_model=models["suspect"],
        interrogator_model=models["interrogator_selector"],
        judge_model=models["judge"],
    )


def _print_trial(result: TrialResult) -> None:
    q = result.config.topic_card["question"]
    print(f"\nTopic: {q}")
    print(f"Direction: {result.config.direction}  Condition: {result.config.condition}")
    print("=" * 70)
    for rec in result.turns:
        print(f"\n[Turn {rec.turn}]  Score: {rec.stance_score:+d}")
        print(f"  SUSPECT:      {rec.suspect_says[:200]}")
        if rec.interrogator_says:
            print(f"  INTERROGATOR: {rec.interrogator_says[:200]}")
    print("\nStance trajectory:", result.scores)


if __name__ == "__main__":
    cfg = _load_config()
    result = run_trial(cfg)
    _print_trial(result)
