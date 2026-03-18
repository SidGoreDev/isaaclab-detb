from detb.models import EpisodeMetric
from detb.stats import aggregate_episode_metrics, terrain_generalization_score


def test_aggregate_episode_metrics():
    episodes = [
        EpisodeMetric("a", 0, "L0_flat", 0.0, "nominal", 1, 9.0, 20.0, 1.2, "none", 1, "proprio"),
        EpisodeMetric("b", 0, "L0_flat", 0.0, "nominal", 0, 7.0, 24.0, 1.4, "slip", 2, "proprio"),
        EpisodeMetric("c", 0, "L0_flat", 0.0, "nominal", 1, 8.5, 21.5, 1.3, "none", 3, "proprio"),
    ]
    aggregates = aggregate_episode_metrics(episodes, 1.96)
    metrics = {item.metric_name: item for item in aggregates}
    assert "task_success_rate" in metrics
    assert metrics["task_success_rate"].seed_count == 3
    assert metrics["distance_m"].mean > 8.0


def test_terrain_generalization_score():
    episodes = [
        EpisodeMetric("a", 0, "L0_flat", 0.0, "nominal", 1, 8.0, 21.0, 1.1, "none", 1, "proprio"),
        EpisodeMetric("b", 1, "L1_slopes", 0.0, "nominal", 1, 7.6, 22.0, 1.2, "none", 1, "proprio"),
        EpisodeMetric("c", 2, "L2_rough", 0.0, "nominal", 0, 6.1, 25.0, 1.5, "slip", 1, "proprio"),
    ]
    score = terrain_generalization_score(episodes)
    assert 0.0 <= score <= 1.0
