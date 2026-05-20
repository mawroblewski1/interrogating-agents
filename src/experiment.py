"""
Sweep quads across topics and write three output files per run.

CLI usage (M2 — control only):
    python -m src.experiment --topic housing_prop_123 --n_quads 2 --condition control

CLI usage (M4+ — control + treatment):
    python -m src.experiment --n_quads 5
"""
from __future__ import annotations

import argparse
import csv
import json
import yaml
from datetime import datetime
from pathlib import Path

from src.quad import run_quad, QuadResult
from src.metrics import magnitude, direction, consistency, directional_accuracy

BASE = Path(__file__).parent.parent


def _interrogator_stance(suspect_direction: int) -> str:
    return "for" if suspect_direction < 0 else "against"


def _write_outputs(quads: list[QuadResult], ts: str) -> None:
    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)

    transcript_path = results_dir / f"transcripts_{ts}.jsonl"
    trials_path     = results_dir / f"trials_{ts}.csv"
    quads_path      = results_dir / f"quads_{ts}.csv"

    trials_rows: list[dict] = []
    quads_rows:  list[dict] = []

    with open(transcript_path, "w", encoding="utf-8") as jf:
        for quad in quads:
            for leg in quad.legs:
                cfg = leg.config
                int_stance = _interrogator_stance(cfg.direction)
                for rec in leg.turns:
                    jf.write(json.dumps({
                        "quad_id":               quad.quad_id,
                        "topic":                 quad.topic_id,
                        "seed":                  quad.seed,
                        "condition":             cfg.condition,
                        "interrogator_stance":   int_stance,
                        "suspect_init_direction": cfg.direction,
                        "turn":                  rec.turn,
                        "suspect_says":          rec.suspect_says,
                        "interrogator_says":     rec.interrogator_says,
                        "technique_selected":    None,
                        "suspect_model_snapshot": None,
                    }, ensure_ascii=False) + "\n")

                trials_rows.append({
                    "quad_id":               quad.quad_id,
                    "topic":                 quad.topic_id,
                    "seed":                  quad.seed,
                    "condition":             cfg.condition,
                    "interrogator_stance":   int_stance,
                    "suspect_init_direction": cfg.direction,
                    "turn":                  rec.turn,
                    "stance_score":          rec.stance_score,
                })

                int_direction = 1 if int_stance == "for" else -1
                quads_rows.append({
                    "quad_id":               quad.quad_id,
                    "topic":                 quad.topic_id,
                    "seed":                  quad.seed,
                    "condition":             cfg.condition,
                    "interrogator_stance":   int_stance,
                    "suspect_init_direction": cfg.direction,
                    "magnitude":             magnitude(leg.scores),
                    "direction":             direction(leg.scores),
                    "consistency":           round(consistency(leg.scores), 4),
                    "directional_accuracy":  directional_accuracy(leg.scores, int_direction),
                    "techniques_used":       "",
                })

    _write_csv(trials_path,
               ["quad_id","topic","seed","condition","interrogator_stance",
                "suspect_init_direction","turn","stance_score"],
               trials_rows)

    _write_csv(quads_path,
               ["quad_id","topic","seed","condition","interrogator_stance",
                "suspect_init_direction","magnitude","direction","consistency",
                "directional_accuracy","techniques_used"],
               quads_rows)

    print(f"Wrote:\n  {transcript_path}\n  {trials_path}\n  {quads_path}")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_topic_card(topic_id: str) -> dict:
    with open(BASE / "data" / "topics" / f"{topic_id}.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    with open(BASE / "config" / "models.yaml") as f:
        models = yaml.safe_load(f)
    with open(BASE / "config" / "trial_defaults.yaml") as f:
        defaults = yaml.safe_load(f)
    with open(BASE / "config" / "topics.yaml") as f:
        all_topic_ids: list[str] = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic",     default=None, help="Single topic ID; omit to run all topics")
    parser.add_argument("--n_quads",   type=int, default=None)
    parser.add_argument("--n_turns",   type=int, default=defaults["n_turns"])
    parser.add_argument("--seed",      type=int, default=defaults["seed"])
    parser.add_argument("--condition", default=None, choices=["control"],
                        help="Override condition for all legs (use 'control' for M2 baseline)")
    args = parser.parse_args()

    topic_ids = [args.topic] if args.topic else all_topic_ids
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    all_quads: list[QuadResult] = []
    for topic_id in topic_ids:
        card = _load_topic_card(topic_id)
        n_quads = args.n_quads if args.n_quads is not None else card.get("n_quads", 5)
        for q_idx in range(n_quads):
            seed = args.seed + q_idx
            print(f"Running quad {q_idx + 1}/{n_quads} for topic '{topic_id}' (seed={seed})...")
            quad = run_quad(
                topic_card=card,
                seed=seed,
                models=models,
                n_turns=args.n_turns,
                override_condition=args.condition,
            )
            all_quads.append(quad)
            _print_quad_summary(quad)

    _write_outputs(all_quads, ts)


def _print_quad_summary(quad: QuadResult) -> None:
    print(f"  Quad {quad.quad_id}")
    for leg in quad.legs:
        cfg = leg.config
        print(f"    [{cfg.condition:9s} | dir={cfg.direction:+d}] "
              f"scores: {leg.scores}  "
              f"delta={leg.scores[-1] - leg.scores[0]:+d}")


if __name__ == "__main__":
    main()
