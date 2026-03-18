"""Deterministic mock backend for DETB smoke tests."""

from __future__ import annotations

import random

from detb.models import EpisodeMetric


class MockBackend:
    """Synthetic backend that models study behavior from config plus seed."""

    name = "mock"

    def train(self, cfg) -> dict:
        sensor_cost = float(cfg.sensor.compute_cost)
        terrain_level = int(cfg.terrain.level)
        final_reward = max(0.2, 8.5 - terrain_level * 1.1 - (sensor_cost - 1.0) * 1.4)
        curve = []
        for index in range(1, 21):
            progress = index / 20.0
            reward = round(final_reward * progress + 0.25 * progress, 4)
            curve.append((index * int(cfg.execution.train_steps / 20), reward))
        return {
            "steps_per_second": round(2250 / max(sensor_cost, 1.0), 2),
            "final_reward": curve[-1][1],
            "convergence_step": curve[-1][0],
            "reward_curve": curve,
            "approx_vram_gb": round(float(cfg.sensor.vram_gb) + 1.4 + terrain_level * 0.2, 2),
        }

    def evaluate(self, cfg) -> list[EpisodeMetric]:
        episodes: list[EpisodeMetric] = []
        terrain_level = int(cfg.terrain.level)
        success_target = self._success_probability(cfg)
        base_energy = self._energy_proxy(cfg)
        for seed in cfg.execution.seeds:
            rng = random.Random(f"{seed}:{cfg.sensor.name}:{cfg.terrain.name}:{cfg.fault.name}")
            for episode_index in range(int(cfg.execution.eval_episodes)):
                trial = success_target + rng.uniform(-0.08, 0.08)
                success = 1 if trial >= 0.5 else 0
                distance = max(0.5, round((7.0 + rng.uniform(0.0, 2.5)) * (0.45 + success_target), 3))
                elapsed = round(20.0 + terrain_level * 4.0 + (1.0 - success_target) * 8.0 + rng.uniform(0.0, 2.5), 3)
                energy = round(base_energy + rng.uniform(-0.15, 0.15), 3)
                failure = "none" if success else self._failure_label(cfg, terrain_level, rng)
                episodes.append(
                    EpisodeMetric(
                        episode_id=f"{seed}-{episode_index}",
                        terrain_level=terrain_level,
                        terrain_name=str(cfg.terrain.name),
                        fault_level=float(cfg.fault.severity),
                        fault_name=str(cfg.fault.name),
                        success=success,
                        distance_m=distance,
                        elapsed_time_s=elapsed,
                        energy_proxy=energy,
                        failure_label=failure,
                        seed=int(seed),
                        sensor_profile=str(cfg.sensor.name),
                    )
                )
        return episodes

    def _success_probability(self, cfg) -> float:
        terrain_level = int(cfg.terrain.level)
        severity = float(cfg.fault.severity)
        leg_scale = float(cfg.robot.leg_length_scale)
        mass_kg = float(cfg.robot.body_mass_kg)
        torque_scale = float(cfg.robot.torque_limit_scale)
        stiffness = float(cfg.robot.stiffness)
        damping = float(cfg.robot.damping)

        success = 0.94
        success -= terrain_level * 0.12
        success -= severity * 0.55
        success -= abs(leg_scale - 1.0) * 0.45
        success -= abs(mass_kg - 32.0) / 32.0 * 0.22
        success += (torque_scale - 1.0) * 0.18
        success -= abs(stiffness - 180.0) / 180.0 * 0.08
        success -= abs(damping - 8.0) / 8.0 * 0.05

        sensor_bonus = {
            "proprio": 0.0,
            "raycaster": 0.04 + (0.03 if terrain_level >= 2 else 0.0),
            "depth_forward": 0.03 + (0.06 if terrain_level >= 2 else 0.0) + (0.03 if terrain_level >= 3 else 0.0),
        }.get(str(cfg.sensor.name), 0.0)
        success += sensor_bonus
        success -= float(cfg.sensor.noise_sigma) * 0.5
        return max(0.05, min(0.98, success))

    def _energy_proxy(self, cfg) -> float:
        terrain_level = int(cfg.terrain.level)
        mass_kg = float(cfg.robot.body_mass_kg)
        severity = float(cfg.fault.severity)
        torque_scale = float(cfg.robot.torque_limit_scale)
        return round(
            1.1
            + terrain_level * 0.22
            + abs(mass_kg - 32.0) / 32.0 * 0.6
            + severity * 0.8
            + abs(torque_scale - 1.0) * 0.25
            + (float(cfg.sensor.compute_cost) - 1.0) * 0.12,
            3,
        )

    def _failure_label(self, cfg, terrain_level: int, rng: random.Random) -> str:
        if str(cfg.fault.class_name) == "latency":
            return "timeout"
        if str(cfg.fault.class_name) == "torque_reduction" and float(cfg.fault.severity) >= 0.3:
            return "fall"
        if terrain_level >= 3:
            return rng.choice(["body_strike", "slip"])
        if terrain_level >= 2:
            return rng.choice(["slip", "fall"])
        return "timeout"
