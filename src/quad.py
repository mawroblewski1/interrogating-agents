"""
run_quad: runs all four legs of one experiment unit and returns a QuadResult.

Quad legs (fixed):
  Leg 0: condition=control,   suspect_direction=-2  (suspect strongly against)
  Leg 1: condition=control,   suspect_direction=+2  (suspect strongly for)
  Leg 2: condition=treatment, suspect_direction=-2
  Leg 3: condition=treatment, suspect_direction=+2

Judge scoring is batched across all four legs in a single session so any
systematic calibration bias cancels out in treatment vs. control comparisons.

For M2 (treatment not yet implemented), pass override_condition="control" to
run all legs as control. Removed in M4 when the treatment branch is ready.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.trial import TrialConfig, TrialResult, DialogueResult, run_dialogue
from src.roles.judge import score_batch


@dataclass
class LegDef:
    condition: str
    suspect_direction: int


_QUAD_LEGS = [
    LegDef("control",   -2),
    LegDef("control",    2),
    LegDef("treatment", -2),
    LegDef("treatment",  2),
]


@dataclass
class QuadResult:
    quad_id: str
    topic_id: str
    seed: int
    legs: list[TrialResult]   # always 4, in _QUAD_LEGS order


def run_quad(
    topic_card: dict,
    seed: int,
    models: dict,
    n_turns: int,
    override_condition: str | None = None,
) -> QuadResult:
    """Run all four legs and return a QuadResult with batch-scored turns.

    Args:
        topic_card:          Parsed YAML topic card.
        seed:                Reproducibility seed (same for all legs).
        models:              Dict from config/models.yaml.
        n_turns:             Turns per leg.
        override_condition:  If set, all legs use this condition instead of
                             the per-leg default. Pass "control" for M2.
    """
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    quad_id = f"{topic_card['id']}_{seed}_{ts}"

    configs = [
        TrialConfig(
            topic_card=topic_card,
            direction=leg.suspect_direction,
            condition=override_condition if override_condition else leg.condition,
            n_turns=n_turns,
            seed=seed,
            suspect_model=models["suspect"],
            interrogator_model=models["interrogator_selector"],
            judge_model=models["judge"],
        )
        for leg in _QUAD_LEGS
    ]

    # Run all four dialogues before touching the judge.
    dialogues: list[DialogueResult] = [run_dialogue(cfg) for cfg in configs]

    # Concatenate all suspect utterances for one batched judge call.
    all_utterances: list[str] = []
    boundaries: list[int] = [0]
    for d in dialogues:
        all_utterances.extend(d.utterances)
        boundaries.append(len(all_utterances))

    all_scores = score_batch(
        models["judge"],
        topic_card["question"],
        all_utterances,
        seed=seed,
    )

    # Distribute scores back to each leg and assemble TrialResults.
    trial_results: list[TrialResult] = []
    for i, (d, cfg) in enumerate(zip(dialogues, configs)):
        scores = all_scores[boundaries[i]: boundaries[i + 1]]
        for j, record in enumerate(d.turns):
            record.stance_score = scores[j]
        trial_results.append(TrialResult(config=cfg, scores=scores, turns=d.turns))

    return QuadResult(
        quad_id=quad_id,
        topic_id=topic_card["id"],
        seed=seed,
        legs=trial_results,
    )
