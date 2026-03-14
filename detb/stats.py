"""Statistics helpers for DETB studies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median

from detb.models import AggregateMetric, EpisodeMetric


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def confidence_interval(values: list[float], z_score: float) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    center = mean(values)
    if len(values) == 1:
        return center, center
    stderr = _stddev(values) / math.sqrt(len(values))
    margin = z_score * stderr
    return center - margin, center + margin


def aggregate_episode_metrics(
    episodes: list[EpisodeMetric],
    z_score: float,
    scope: str = "run",
) -> list[AggregateMetric]:
    metrics = {
        "task_success_rate": [float(item.success) for item in episodes],
        "distance_m": [item.distance_m for item in episodes],
        "elapsed_time_s": [item.elapsed_time_s for item in episodes],
        "energy_proxy": [item.energy_proxy for item in episodes],
    }
    seed_count = len({item.seed for item in episodes})
    aggregates: list[AggregateMetric] = []
    for name, values in metrics.items():
        ci_low, ci_high = confidence_interval(values, z_score)
        aggregates.append(
            AggregateMetric(
                metric_name=name,
                aggregation_scope=scope,
                mean=mean(values),
                median=median(values),
                stddev=_stddev(values),
                ci_low=ci_low,
                ci_high=ci_high,
                n=len(values),
                seed_count=seed_count,
            )
        )
    return aggregates


def terrain_generalization_score(episodes: list[EpisodeMetric]) -> float:
    by_level: dict[int, list[float]] = defaultdict(list)
    for item in episodes:
        by_level[item.terrain_level].append(float(item.success))
    if not by_level:
        return 0.0
    per_level = [mean(values) for _, values in sorted(by_level.items())]
    return sum(per_level) / len(per_level)


def failure_counts(episodes: list[EpisodeMetric]) -> dict[str, int]:
    return dict(Counter(item.failure_label for item in episodes if item.failure_label != "none"))
