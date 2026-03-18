"""Evidence-tier helpers for DETB study outputs."""

from __future__ import annotations


def minimum_study_seeds(cfg) -> int:
    return int(getattr(cfg.analysis, "minimum_study_seeds", 3))


def minimum_study_eval_episodes(cfg) -> int:
    return int(getattr(cfg.analysis, "minimum_study_eval_episodes", 10))


def configured_seed_count(cfg) -> int:
    return len(list(cfg.execution.seeds))


def configured_eval_episodes(cfg) -> int:
    return int(cfg.execution.eval_episodes)


def run_tier(cfg) -> str:
    return str(getattr(cfg.execution, "run_tier", "smoke"))


def assert_study_tier_ready(cfg) -> None:
    if run_tier(cfg) != "study":
        return
    seed_count = configured_seed_count(cfg)
    eval_episodes = configured_eval_episodes(cfg)
    minimum_seeds = minimum_study_seeds(cfg)
    minimum_eval_episodes = minimum_study_eval_episodes(cfg)
    if seed_count < minimum_seeds or eval_episodes < minimum_eval_episodes:
        raise ValueError(
            "Study-tier runs require at least "
            f"{minimum_seeds} seeds and {minimum_eval_episodes} evaluation episodes per seed. "
            f"Received seeds={seed_count}, eval_episodes={eval_episodes}."
        )


def manifest_supports_requirements(manifest, cfg) -> tuple[bool, str]:
    minimum_seeds = minimum_study_seeds(cfg)
    minimum_eval_episodes = minimum_study_eval_episodes(cfg)
    manifest_tier = str(getattr(manifest, "run_tier", "smoke"))
    manifest_seed_count = len(getattr(manifest, "seeds", []))
    manifest_eval_episodes = int(getattr(manifest, "configured_eval_episodes", 0))

    if manifest_tier != "study":
        return False, f"run_tier={manifest_tier}"
    if manifest_seed_count < minimum_seeds:
        return False, f"seed_count={manifest_seed_count} < {minimum_seeds}"
    if manifest_eval_episodes < minimum_eval_episodes:
        return False, f"eval_episodes={manifest_eval_episodes} < {minimum_eval_episodes}"
    return True, "ready"
