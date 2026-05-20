import statistics


def magnitude(scores: list[int]) -> float:
    """Absolute change from first to last score."""
    return float(abs(scores[-1] - scores[0]))


def direction(scores: list[int]) -> int:
    """Sign of stance change: +1 (moved positive), -1 (moved negative), 0 (no change)."""
    delta = scores[-1] - scores[0]
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def consistency(scores: list[int]) -> float:
    """Variance of stance scores across turns. Lower = more monotonic trajectory."""
    if len(scores) < 2:
        return 0.0
    return statistics.variance([float(s) for s in scores])


def directional_accuracy(scores: list[int], interrogator_direction: int) -> bool:
    """True if the suspect's final stance moved toward the interrogator's position."""
    delta = scores[-1] - scores[0]
    if delta == 0:
        return False
    return (delta > 0) == (interrogator_direction > 0)


def treatment_effect(control_magnitudes: list[float], treatment_magnitudes: list[float]) -> float:
    """Mean (treatment - control) magnitude across matched legs within a quad."""
    if not control_magnitudes or not treatment_magnitudes:
        return 0.0
    return sum(t - c for c, t in zip(control_magnitudes, treatment_magnitudes)) / len(control_magnitudes)
