"""Execution pipeline for DETB commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from omegaconf import OmegaConf

from detb.artifacts import (
    rebuild_summary,
    write_playback_summary,
    write_markdown_summary,
    write_requirements_markdown,
    write_success_plot,
    write_training_curve,
)
from detb.backends import IsaacLabBackend, MockBackend
from detb.config import load_group_entry, merge_cfg
from detb.evidence import assert_study_tier_ready, manifest_supports_requirements
from detb.extension import task_registry_id
from detb.io import (
    create_manifest,
    ensure_directory,
    make_run_id,
    read_csv,
    read_json,
    resolve_output_root,
    write_csv,
    write_json,
    write_manifest_bundle,
)
from detb.models import ArtifactRecord, RequirementRecord, RunManifest
from detb.stats import aggregate_episode_metrics, failure_counts, terrain_generalization_score


@dataclass
class CommandResult:
    run_dir: Path
    manifest: RunManifest


def _backend_for(name: str):
    if name == "mock":
        return MockBackend()
    if name == "isaaclab":
        return IsaacLabBackend()
    raise ValueError(f"Unsupported backend: {name}")


def _prepare_run(cfg, command: str, checkpoint_name: str | None = None) -> tuple[Path, RunManifest, list[ArtifactRecord]]:
    output_root = resolve_output_root(cfg.execution.output_root)
    run_dir = ensure_directory(output_root / command / make_run_id(command.replace("-", "_")))
    checkpoint_path = run_dir / (checkpoint_name or str(cfg.execution.checkpoint_name))
    manifest = create_manifest(cfg, command, run_dir.name, run_dir, checkpoint_path)
    artifacts: list[ArtifactRecord] = []
    write_manifest_bundle(cfg, manifest, run_dir, artifacts)
    return run_dir, manifest, artifacts


def _finalize_run(cfg, run_dir: Path, manifest: RunManifest, artifacts: list[ArtifactRecord]) -> CommandResult:
    _sync_artifact_registry(run_dir, artifacts)
    write_manifest_bundle(cfg, manifest, run_dir, artifacts)
    return CommandResult(run_dir=run_dir, manifest=manifest)


def _artifact_type_for_path(path: Path) -> str:
    mapping = {
        ".csv": "csv",
        ".json": "json",
        ".log": "log",
        ".md": "markdown",
        ".pt": "checkpoint",
        ".svg": "svg",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return mapping.get(path.suffix.lower(), "file")


def _sync_artifact_registry(run_dir: Path, artifacts: list[ArtifactRecord]) -> None:
    recorded = {item.relative_path for item in artifacts}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = str(path.relative_to(run_dir)).replace("\\", "/")
        if relative_path == "artifact_registry.json" or relative_path in recorded:
            continue
        artifacts.append(ArtifactRecord(_artifact_type_for_path(path), relative_path, "Recorded DETB artifact"))
        recorded.add(relative_path)


def bundle_artifacts(run_dir: str | Path) -> Path:
    return rebuild_summary(Path(run_dir))


def run_sweep(cfg, config_dir: str | Path | None = None) -> CommandResult:
    assert_study_tier_ready(cfg)
    run_dir, manifest, artifacts = _prepare_run(cfg, "sweep")
    backend = _backend_for(str(cfg.execution.backend))
    results: list[dict] = []
    for point in cfg.study.screening_points:
        case_cfg = merge_cfg(
            cfg,
            {
                "robot": {
                    "leg_length_scale": float(point.leg_length_scale),
                    "body_mass_kg": float(point.body_mass_kg),
                    "torque_limit_scale": float(point.torque_limit_scale),
                    "stiffness": float(point.stiffness),
                    "damping": float(point.damping),
                }
            },
        )
        episodes = backend.evaluate(case_cfg)
        success = mean(float(item.success) for item in episodes)
        energy = mean(item.energy_proxy for item in episodes)
        pareto_score = round(success - 0.15 * energy, 4)
        results.append(
            {
                "design_point": str(point.id),
                "task_success_rate": round(success, 4),
                "energy_proxy": round(energy, 4),
                "pareto_score": pareto_score,
                "leg_length_scale": float(point.leg_length_scale),
                "body_mass_kg": float(point.body_mass_kg),
                "torque_limit_scale": float(point.torque_limit_scale),
            }
        )
    results.sort(key=lambda row: row["pareto_score"], reverse=True)
    best = results[0]
    write_csv(run_dir / "sweep_results.csv", results)
    write_json(run_dir / "sweep_results.json", results)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# DETB Parameter Sweep: {manifest.run_id}",
                "",
                f"Top candidate: `{best['design_point']}`",
                f"- Success rate: {best['task_success_rate']:.3f}",
                f"- Energy proxy: {best['energy_proxy']:.3f}",
                f"- Pareto score: {best['pareto_score']:.3f}",
                "",
                "The sweep uses the staged screening points from `configs/study/sweep.yaml`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.extend(
        [
            ArtifactRecord("csv", "sweep_results.csv", "Screening DOE results"),
            ArtifactRecord("json", "sweep_results.json", "Screening DOE results"),
            ArtifactRecord("markdown", "summary.md", "Sweep summary"),
        ]
    )
    return _finalize_run(cfg, run_dir, manifest, artifacts)


def run_sensor_eval(cfg, config_dir: str | Path | None = None) -> CommandResult:
    assert_study_tier_ready(cfg)
    run_dir, manifest, artifacts = _prepare_run(cfg, "sensor_eval")
    backend = _backend_for(str(cfg.execution.backend))
    rows: list[dict] = []
    for name in cfg.analysis.sensor_profiles:
        sensor_cfg = load_group_entry("sensor", str(name), config_dir=config_dir)
        case_cfg = merge_cfg(cfg, {"sensor": OmegaConf.to_container(sensor_cfg, resolve=True)})
        episodes = backend.evaluate(case_cfg)
        success = mean(float(item.success) for item in episodes)
        energy = mean(item.energy_proxy for item in episodes)
        rows.append(
            {
                "sensor_profile": str(name),
                "task_success_rate": round(success, 4),
                "energy_proxy": round(energy, 4),
                "compute_cost": float(sensor_cfg.compute_cost),
                "approx_vram_gb": float(sensor_cfg.vram_gb),
                "roi_score": round(success / float(sensor_cfg.compute_cost), 4),
            }
        )
    rows.sort(key=lambda row: row["roi_score"], reverse=True)
    write_csv(run_dir / "sensor_eval.csv", rows)
    write_json(run_dir / "sensor_eval.json", rows)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# DETB Sensor Evaluation: {manifest.run_id}",
                "",
                f"Recommended minimum viable sensor profile: `{rows[0]['sensor_profile']}`",
                "",
                "Profiles are compared under matched seeds and terrain settings.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.extend(
        [
            ArtifactRecord("csv", "sensor_eval.csv", "Sensor comparison matrix"),
            ArtifactRecord("json", "sensor_eval.json", "Sensor comparison matrix"),
            ArtifactRecord("markdown", "summary.md", "Sensor evaluation summary"),
        ]
    )
    return _finalize_run(cfg, run_dir, manifest, artifacts)


def run_terrain_eval(cfg, config_dir: str | Path | None = None) -> CommandResult:
    assert_study_tier_ready(cfg)
    run_dir, manifest, artifacts = _prepare_run(cfg, "terrain_eval")
    backend = _backend_for(str(cfg.execution.backend))
    all_episodes = []
    terrain_rows: list[dict] = []
    for terrain_name in cfg.analysis.terrain_profiles:
        terrain_cfg = load_group_entry("terrain", str(terrain_name), config_dir=config_dir)
        case_cfg = merge_cfg(cfg, {"terrain": OmegaConf.to_container(terrain_cfg, resolve=True)})
        episodes = backend.evaluate(case_cfg)
        all_episodes.extend(episodes)
        terrain_rows.append(
            {
                "terrain_name": str(terrain_cfg.name),
                "terrain_level": int(terrain_cfg.level),
                "task_success_rate": round(mean(float(item.success) for item in episodes), 4),
                "failure_count": sum(1 for item in episodes if item.failure_label != "none"),
            }
        )
    tgs = terrain_generalization_score(all_episodes)
    write_csv(run_dir / "terrain_eval.csv", terrain_rows)
    write_json(run_dir / "terrain_eval.json", {"terrain_rows": terrain_rows, "terrain_generalization_score": tgs})
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# DETB Terrain Evaluation: {manifest.run_id}",
                "",
                f"- Terrain Generalization Score: {tgs:.3f}",
                f"- Failure labels captured: {failure_counts(all_episodes)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.extend(
        [
            ArtifactRecord("csv", "terrain_eval.csv", "Per-terrain success curve data"),
            ArtifactRecord("json", "terrain_eval.json", "Terrain evaluation summary"),
            ArtifactRecord("markdown", "summary.md", "Terrain evaluation summary"),
        ]
    )
    return _finalize_run(cfg, run_dir, manifest, artifacts)


def run_failure_eval(cfg, config_dir: str | Path | None = None) -> CommandResult:
    assert_study_tier_ready(cfg)
    run_dir, manifest, artifacts = _prepare_run(cfg, "failure_eval")
    backend = _backend_for(str(cfg.execution.backend))
    failure_profile = load_group_entry("fault", str(cfg.analysis.failure_profile), config_dir=config_dir)
    rows: list[dict] = []
    threshold = None
    for level in cfg.analysis.failure_levels:
        case_cfg = merge_cfg(
            cfg,
            {
                "fault": {
                    "name": str(failure_profile.name),
                    "class_name": str(failure_profile.class_name),
                    "severity": float(level),
                    "latency_steps": int(failure_profile.latency_steps),
                }
            },
        )
        episodes = backend.evaluate(case_cfg)
        aggregates = aggregate_episode_metrics(episodes, float(cfg.analysis.confidence_z), scope="failure_level")
        success_metric = next(item for item in aggregates if item.metric_name == "task_success_rate")
        rows.append(
            {
                "fault_name": str(failure_profile.name),
                "severity": float(level),
                "task_success_rate": round(success_metric.mean, 4),
                "ci_low": round(success_metric.ci_low, 4),
                "ci_high": round(success_metric.ci_high, 4),
            }
        )
        if threshold is None and success_metric.mean < 0.5 and success_metric.ci_high < 0.5:
            threshold = float(level)
    write_csv(run_dir / "failure_eval.csv", rows)
    write_json(run_dir / "failure_eval.json", {"fault_curve": rows, "critical_threshold": threshold})
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# DETB Failure Evaluation: {manifest.run_id}",
                "",
                f"- Fault profile: `{failure_profile.name}`",
                f"- Critical threshold: `{threshold}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.extend(
        [
            ArtifactRecord("csv", "failure_eval.csv", "Fault severity sweep"),
            ArtifactRecord("json", "failure_eval.json", "Failure threshold summary"),
            ArtifactRecord("markdown", "summary.md", "Failure evaluation summary"),
        ]
    )
    return _finalize_run(cfg, run_dir, manifest, artifacts)


def generate_requirements(cfg, source_dir: str | Path) -> CommandResult:
    run_dir = Path(source_dir).resolve()
    manifest = RunManifest(**read_json(run_dir / "run_manifest.json"))
    requirements: list[RequirementRecord] = []
    evidence_ready, evidence_reason = manifest_supports_requirements(manifest, cfg)

    if not evidence_ready:
        requirements.append(
            RequirementRecord(
                req_id="DETB-REQ-0000",
                statement="No candidate requirement met the configured evidence threshold.",
                status="candidate",
                source_run_id=manifest.run_id,
                source_metric="none",
                confidence_interval="[n/a]",
                assumptions=f"Evidence gate not met: {evidence_reason}",
                reviewer="unassigned",
                artifact_links="summary.md",
            )
        )
    else:
        aggregates_path = run_dir / "aggregate_metrics.csv"
        if aggregates_path.exists():
            for row in read_csv(aggregates_path):
                if row["metric_name"] != "task_success_rate":
                    continue
                mean_value = float(row["mean"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                if (
                    mean_value >= float(cfg.analysis.success_requirement_threshold)
                    and ci_low >= 0.75
                    and int(row["seed_count"]) >= 2
                ):
                    requirements.append(
                        RequirementRecord(
                            req_id="DETB-REQ-0001",
                            statement=(
                                f"The {manifest.robot_variant} baseline shall achieve at least "
                                f"{int(cfg.analysis.success_requirement_threshold * 100)} percent success "
                                f"on {manifest.terrain_name} using the {manifest.sensor_profile} profile."
                            ),
                            status="candidate",
                            source_run_id=manifest.run_id,
                            source_metric="task_success_rate",
                            confidence_interval=f"[{ci_low:.3f}, {ci_high:.3f}]",
                            assumptions=f"fault={manifest.fault_profile}; backend={manifest.backend}",
                            reviewer="unassigned",
                            artifact_links="aggregate_metrics.csv,summary.md",
                        )
                    )

        terrain_path = run_dir / "terrain_eval.json"
        if terrain_path.exists():
            terrain_payload = read_json(terrain_path)
            tgs = float(terrain_payload["terrain_generalization_score"])
            if tgs >= float(cfg.analysis.tgs_requirement_threshold):
                requirements.append(
                    RequirementRecord(
                        req_id="DETB-REQ-0002",
                        statement=f"The baseline configuration shall maintain a terrain generalization score of at least {cfg.analysis.tgs_requirement_threshold:.2f}.",
                        status="candidate",
                        source_run_id=manifest.run_id,
                        source_metric="terrain_generalization_score",
                        confidence_interval=f"[{tgs:.3f}, {tgs:.3f}]",
                        assumptions=f"terrain_profiles={list(cfg.analysis.terrain_profiles)}",
                        reviewer="unassigned",
                        artifact_links="terrain_eval.json,summary.md",
                    )
                )

        failure_path = run_dir / "failure_eval.json"
        if failure_path.exists():
            failure_payload = read_json(failure_path)
            threshold = failure_payload.get("critical_threshold")
            if threshold is not None:
                requirements.append(
                    RequirementRecord(
                        req_id="DETB-REQ-0003",
                        statement=f"The design shall retain at least 50 percent task success up to a fault severity of {threshold:.2f}.",
                        status="candidate",
                        source_run_id=manifest.run_id,
                        source_metric="critical_threshold",
                        confidence_interval=f"[{threshold:.3f}, {threshold:.3f}]",
                        assumptions=f"fault_profile={cfg.analysis.failure_profile}",
                        reviewer="unassigned",
                        artifact_links="failure_eval.json,summary.md",
                    )
                )

    if not requirements:
        requirements.append(
            RequirementRecord(
                req_id="DETB-REQ-0000",
                statement="No candidate requirement met the configured evidence threshold.",
                status="candidate",
                source_run_id=manifest.run_id,
                source_metric="none",
                confidence_interval="[n/a]",
                assumptions="Evidence thresholds not met",
                reviewer="unassigned",
                artifact_links="summary.md",
            )
        )

    write_csv(run_dir / "requirement_ledger.csv", [item.to_dict() for item in requirements])
    write_json(run_dir / "requirement_ledger.json", [item.to_dict() for item in requirements])
    write_requirements_markdown(run_dir, requirements)
    artifacts = [
        ArtifactRecord("csv", "requirement_ledger.csv", "Candidate requirement ledger"),
        ArtifactRecord("json", "requirement_ledger.json", "Candidate requirement ledger"),
        ArtifactRecord("markdown", "candidate_requirements.md", "Candidate requirement report"),
    ]
    _sync_artifact_registry(run_dir, artifacts)
    write_manifest_bundle(cfg, manifest, run_dir, artifacts)
    return CommandResult(run_dir=run_dir, manifest=manifest)

def _critical_threshold(backend, cfg, config_dir: str | Path | None = None) -> float | None:
    failure_profile = load_group_entry("fault", str(cfg.analysis.failure_profile), config_dir=config_dir)
    threshold = None
    for level in cfg.analysis.failure_levels:
        case_cfg = merge_cfg(
            cfg,
            {
                "fault": {
                    "name": str(failure_profile.name),
                    "class_name": str(failure_profile.class_name),
                    "severity": float(level),
                    "latency_steps": int(failure_profile.latency_steps),
                }
            },
        )
        episodes = backend.evaluate(case_cfg)
        aggregates = aggregate_episode_metrics(episodes, float(cfg.analysis.confidence_z), scope="failure_level")
        success_metric = next(item for item in aggregates if item.metric_name == "task_success_rate")
        if threshold is None and success_metric.mean < 0.5 and success_metric.ci_high < 0.5:
            threshold = float(level)
    return threshold


def _normalized_robustness(threshold: float | None, cfg) -> float:
    max_level = max(float(level) for level in cfg.analysis.failure_levels)
    if threshold is None:
        return 1.0
    if max_level == 0:
        return 0.0
    return threshold / max_level


def _objective_score(
    cfg,
    success_rate: float,
    energy_proxy: float,
    elapsed_time_s: float,
    terrain_score: float,
    robustness_score: float,
) -> float:
    elapsed_ratio = elapsed_time_s / max(float(cfg.objective.target_elapsed_time_s), 1.0)
    score = (
        float(cfg.objective.success_weight) * success_rate
        + float(cfg.objective.terrain_weight) * terrain_score
        + float(cfg.objective.robustness_weight) * robustness_score
        - float(cfg.objective.energy_weight) * energy_proxy
        - float(cfg.objective.elapsed_weight) * elapsed_ratio
    )
    return round(score, 4)


def run_visualize(cfg) -> CommandResult:
    visual_cfg = merge_cfg(cfg, {"execution": {"backend": "isaaclab"}})
    run_dir, manifest, artifacts = _prepare_run(visual_cfg, "visualize")
    runtime_cfg = _with_runtime_context(visual_cfg, run_dir, manifest)
    backend = IsaacLabBackend(runtime_cfg)
    command, cwd = backend.build_visualize_command(runtime_cfg)
    launch_spec = {
        "mode": "isaaclab_play",
        "execute": bool(runtime_cfg.visualization.execute),
        "cwd": str(cwd),
        "command": command,
        "task": str(task_registry_id(runtime_cfg)),
        "device": str(runtime_cfg.execution.device),
        "num_envs": int(runtime_cfg.visualization.num_envs),
        "rollout_steps": int(runtime_cfg.visualization.rollout_steps),
        "real_time": bool(runtime_cfg.visualization.real_time),
        "video": bool(runtime_cfg.visualization.video),
        "use_pretrained_checkpoint": bool(runtime_cfg.visualization.use_pretrained_checkpoint),
        "load_run": str(runtime_cfg.visualization.load_run),
        "checkpoint": str(runtime_cfg.visualization.checkpoint),
        "headless": bool(runtime_cfg.visualization.headless),
    }
    if runtime_cfg.visualization.execute:
        playback = backend.visualize(runtime_cfg)
        launch_spec = playback.get("launch_spec", launch_spec)
        launch_spec["execute"] = True
        _record_isaac_play_artifacts(run_dir, artifacts, backend)
        _apply_runtime_stack_to_manifest(manifest, playback.get("runtime_stack"))
        summary_path = write_playback_summary(
            run_dir,
            manifest,
            playback,
            "Playback diagnostics were captured from the DETB-owned Isaac Lab runner.",
        )
        artifacts.append(ArtifactRecord("markdown", summary_path.name, "Playback diagnostics summary"))
        return_code = int(launch_spec.get("return_code", 0))
    else:
        return_code = 0
        (run_dir / "summary.md").write_text(
            "\n".join(
                [
                    f"# DETB Visualization Launch: {manifest.run_id}",
                    "",
                    "This command delegates to the pinned Isaac Lab playback runtime.",
                    f"- Execute: `{runtime_cfg.visualization.execute}`",
                    f"- Task: `{runtime_cfg.task.command}`",
                    f"- Command file: `visualize_command.json`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(ArtifactRecord("markdown", "summary.md", "Visualization launch summary"))
    launch_spec["return_code"] = return_code
    write_json(run_dir / "visualize_command.json", launch_spec)
    artifacts.append(ArtifactRecord("json", "visualize_command.json", "Isaac Lab playback launch specification"))
    return _finalize_run(runtime_cfg, run_dir, manifest, artifacts)


def run_train_gui(cfg) -> CommandResult:
    gui_cfg = merge_cfg(cfg, {"execution": {"backend": "isaaclab"}})
    run_dir, manifest, artifacts = _prepare_run(gui_cfg, "train_gui")
    backend = IsaacLabBackend(gui_cfg)
    command, cwd = backend.build_train_gui_command(gui_cfg)
    launch_spec = {
        "mode": "isaaclab_train",
        "execute": bool(gui_cfg.visualization.train_execute),
        "cwd": str(cwd),
        "command": command,
        "task": str(task_registry_id(gui_cfg)),
        "device": str(gui_cfg.execution.device),
        "num_envs": int(gui_cfg.visualization.train_num_envs),
        "max_iterations": int(gui_cfg.visualization.train_max_iterations),
        "seed": int(gui_cfg.visualization.train_seed),
        "video": bool(gui_cfg.visualization.video),
        "headless": bool(gui_cfg.visualization.headless),
    }
    if gui_cfg.visualization.train_execute:
        return_code = backend.run_command(command, cwd)
    else:
        return_code = 0
    launch_spec["return_code"] = return_code
    write_json(run_dir / "train_gui_command.json", launch_spec)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# DETB GUI Training Launch: {manifest.run_id}",
                "",
                "This command delegates to the pinned Isaac Lab training runtime.",
                f"- Execute: `{gui_cfg.visualization.train_execute}`",
                f"- Task: `{gui_cfg.task.command}`",
                f"- Command file: `train_gui_command.json`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.extend(
        [
            ArtifactRecord("json", "train_gui_command.json", "Isaac Lab training launch specification"),
            ArtifactRecord("markdown", "summary.md", "GUI training launch summary"),
        ]
    )
    return _finalize_run(gui_cfg, run_dir, manifest, artifacts)

def run_tune(cfg, config_dir: str | Path | None = None) -> CommandResult:
    assert_study_tier_ready(cfg)
    run_dir, manifest, artifacts = _prepare_run(cfg, "tune")
    backend = _backend_for(str(cfg.execution.backend))
    rows: list[dict] = []
    for point in cfg.study.screening_points:
        case_cfg = merge_cfg(
            cfg,
            {
                "robot": {
                    "leg_length_scale": float(point.leg_length_scale),
                    "body_mass_kg": float(point.body_mass_kg),
                    "torque_limit_scale": float(point.torque_limit_scale),
                    "stiffness": float(point.stiffness),
                    "damping": float(point.damping),
                }
            },
        )
        eval_episodes = backend.evaluate(case_cfg)
        eval_aggregates = aggregate_episode_metrics(eval_episodes, float(cfg.analysis.confidence_z))
        success = next(item.mean for item in eval_aggregates if item.metric_name == "task_success_rate")
        energy = next(item.mean for item in eval_aggregates if item.metric_name == "energy_proxy")
        elapsed = next(item.mean for item in eval_aggregates if item.metric_name == "elapsed_time_s")

        terrain_episodes = []
        for terrain_name in case_cfg.analysis.terrain_profiles:
            terrain_cfg = load_group_entry("terrain", str(terrain_name), config_dir=config_dir)
            terrain_case = merge_cfg(case_cfg, {"terrain": OmegaConf.to_container(terrain_cfg, resolve=True)})
            terrain_episodes.extend(backend.evaluate(terrain_case))
        tgs = terrain_generalization_score(terrain_episodes)

        threshold = _critical_threshold(backend, case_cfg, config_dir=config_dir)
        robustness_score = _normalized_robustness(threshold, case_cfg)
        composite_score = _objective_score(case_cfg, success, energy, elapsed, tgs, robustness_score)
        realized_threshold = threshold if threshold is not None else max(float(level) for level in cfg.analysis.failure_levels)
        meets_targets = (
            success >= float(cfg.objective.target_success_rate)
            and tgs >= float(cfg.objective.target_tgs)
            and realized_threshold >= float(cfg.objective.target_failure_threshold)
        )
        rows.append(
            {
                "design_point": str(point.id),
                "task_success_rate": round(success, 4),
                "terrain_generalization_score": round(tgs, 4),
                "energy_proxy": round(energy, 4),
                "elapsed_time_s": round(elapsed, 4),
                "critical_threshold": threshold,
                "robustness_score": round(robustness_score, 4),
                "composite_score": composite_score,
                "meets_targets": meets_targets,
                "leg_length_scale": float(point.leg_length_scale),
                "body_mass_kg": float(point.body_mass_kg),
                "torque_limit_scale": float(point.torque_limit_scale),
            }
        )
    rows.sort(key=lambda row: row["composite_score"], reverse=True)
    top_rows = rows[: int(cfg.objective.candidate_limit)]
    write_csv(run_dir / "tune_results.csv", rows)
    write_json(
        run_dir / "tune_results.json",
        {
            "objective": OmegaConf.to_container(cfg.objective, resolve=True),
            "top_candidates": top_rows,
            "all_candidates": rows,
        },
    )
    best = rows[0]
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# DETB Tuning Summary: {manifest.run_id}",
                "",
                f"Recommended design point: `{best['design_point']}`",
                f"- Composite score: {best['composite_score']:.3f}",
                f"- Success rate: {best['task_success_rate']:.3f}",
                f"- TGS: {best['terrain_generalization_score']:.3f}",
                f"- Critical threshold: {best['critical_threshold']}",
                f"- Meets targets: `{best['meets_targets']}`",
                "",
                "Objective weights and thresholds are stored in `tune_results.json`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.extend(
        [
            ArtifactRecord("csv", "tune_results.csv", "Tuning candidate ranking"),
            ArtifactRecord("json", "tune_results.json", "Tuning objectives and candidate ranking"),
            ArtifactRecord("markdown", "summary.md", "Tuning summary"),
        ]
    )
    return _finalize_run(cfg, run_dir, manifest, artifacts)

def _record_isaac_train_artifacts(run_dir: Path, manifest: RunManifest, artifacts: list[ArtifactRecord], backend: IsaacLabBackend, training: dict) -> None:
    launch_spec = training.get("launch_spec")
    if launch_spec is not None:
        write_json(run_dir / "isaac_train_command.json", launch_spec)
        artifacts.append(ArtifactRecord("json", "isaac_train_command.json", "Isaac Lab training launch specification"))

    source_checkpoint = training.get("source_checkpoint_path")
    if source_checkpoint:
        backend.copy_checkpoint(Path(source_checkpoint), Path(manifest.checkpoint_path))
        artifacts.append(ArtifactRecord("checkpoint", Path(manifest.checkpoint_path).name, "Trained Isaac Lab checkpoint copy"))

    if (run_dir / "isaac_train_result.json").exists():
        artifacts.append(ArtifactRecord("json", "isaac_train_result.json", "Isaac Lab training result metadata"))
    if (run_dir / "isaac_train_stdout.log").exists():
        artifacts.append(ArtifactRecord("log", "isaac_train_stdout.log", "Isaac Lab training stdout log"))
    if (run_dir / "isaac_train_stderr.log").exists():
        artifacts.append(ArtifactRecord("log", "isaac_train_stderr.log", "Isaac Lab training stderr log"))
    if (run_dir / "isaac_train_debug.log").exists():
        artifacts.append(ArtifactRecord("log", "isaac_train_debug.log", "Isaac Lab training stage debug log"))

    if backend.copy_if_exists(training.get("source_env_yaml", ""), run_dir / "isaac_env.yaml"):
        artifacts.append(ArtifactRecord("yaml", "isaac_env.yaml", "Resolved Isaac Lab environment config"))
    if backend.copy_if_exists(training.get("source_agent_yaml", ""), run_dir / "isaac_agent.yaml"):
        artifacts.append(ArtifactRecord("yaml", "isaac_agent.yaml", "Resolved Isaac Lab agent config"))



def _record_isaac_eval_artifacts(run_dir: Path, manifest: RunManifest, artifacts: list[ArtifactRecord], backend: IsaacLabBackend) -> None:
    metadata = backend.last_eval_metadata or {}
    source_checkpoint = metadata.get("source_checkpoint_path")
    if source_checkpoint:
        backend.copy_checkpoint(Path(source_checkpoint), Path(manifest.checkpoint_path))
        artifacts.append(ArtifactRecord("checkpoint", Path(manifest.checkpoint_path).name, "Evaluated Isaac Lab checkpoint copy"))

    if metadata:
        write_json(run_dir / "isaac_eval_runs.json", metadata)
        artifacts.append(ArtifactRecord("json", "isaac_eval_runs.json", "Isaac Lab evaluation run metadata"))

    for seed_run in metadata.get("seed_runs", []):
        for key in ("result_json", "stdout_log", "stderr_log"):
            name = seed_run.get(key)
            if not name:
                continue
            path = run_dir / str(name)
            if not path.exists():
                continue
            if name.endswith(".json"):
                artifacts.append(ArtifactRecord("json", path.name, f"Isaac Lab evaluation payload for seed {seed_run['seed']}"))
            else:
                artifacts.append(ArtifactRecord("log", path.name, f"Isaac Lab evaluation log for seed {seed_run['seed']}"))


def _record_isaac_play_artifacts(run_dir: Path, artifacts: list[ArtifactRecord], backend: IsaacLabBackend) -> None:
    metadata = backend.last_play_metadata or {}
    if metadata:
        write_json(run_dir / "isaac_play_runs.json", metadata)
        artifacts.append(ArtifactRecord("json", "isaac_play_runs.json", "Isaac Lab playback run metadata"))

    for name, artifact_type, description in (
        ("isaac_play_result.json", "json", "Isaac Lab playback diagnostics payload"),
        ("playback_telemetry.csv", "csv", "Playback telemetry samples"),
        ("isaac_play_stdout.log", "log", "Isaac Lab playback stdout log"),
        ("isaac_play_stderr.log", "log", "Isaac Lab playback stderr log"),
        ("isaac_play_debug.log", "log", "Isaac Lab playback stage debug log"),
    ):
        if (run_dir / name).exists():
            artifacts.append(ArtifactRecord(artifact_type, name, description))


def _apply_runtime_stack_to_manifest(manifest: RunManifest, runtime_stack: dict | None) -> None:
    if not runtime_stack:
        return
    manifest.torch_version = str(runtime_stack.get("torch_version", manifest.torch_version))
    manifest.cuda_version = str(runtime_stack.get("cuda_version", manifest.cuda_version))
    manifest.rsl_rl_version = str(runtime_stack.get("rsl_rl_version", manifest.rsl_rl_version))



def run_train(cfg) -> CommandResult:
    run_dir, manifest, artifacts = _prepare_run(cfg, "train")
    runtime_cfg = _with_runtime_context(cfg, run_dir, manifest)
    backend = _backend_for(str(cfg.execution.backend))
    training = backend.train(runtime_cfg)

    checkpoint = Path(manifest.checkpoint_path)
    if isinstance(backend, IsaacLabBackend):
        _record_isaac_train_artifacts(run_dir, manifest, artifacts, backend, training)
        _apply_runtime_stack_to_manifest(manifest, training.get("runtime_stack"))
    else:
        checkpoint.write_text("mock checkpoint\n", encoding="utf-8")
        artifacts.append(ArtifactRecord("checkpoint", checkpoint.name, "Synthetic checkpoint placeholder"))

    write_json(run_dir / "training_summary.json", training)
    write_csv(
        run_dir / "training_reward_curve.csv",
        [{"step": step, "reward": reward} for step, reward in training["reward_curve"]],
    )
    curve_path = write_training_curve(run_dir, training["reward_curve"])
    artifacts.extend(
        [
            ArtifactRecord("json", "training_summary.json", "Training summary metadata"),
            ArtifactRecord("csv", "training_reward_curve.csv", "Training reward curve samples"),
            ArtifactRecord("svg", curve_path.name, "Training curve plot"),
        ]
    )
    return _finalize_run(runtime_cfg, run_dir, manifest, artifacts)



def run_evaluate(cfg) -> CommandResult:
    assert_study_tier_ready(cfg)
    run_dir, manifest, artifacts = _prepare_run(cfg, "evaluate")
    runtime_cfg = _with_runtime_context(cfg, run_dir, manifest)
    backend = _backend_for(str(cfg.execution.backend))
    episodes = backend.evaluate(runtime_cfg)
    if isinstance(backend, IsaacLabBackend):
        _record_isaac_eval_artifacts(run_dir, manifest, artifacts, backend)
        _apply_runtime_stack_to_manifest(manifest, (backend.last_eval_metadata or {}).get("runtime_stack"))
        note = "Held-out evaluation battery complete using the Isaac Lab backend."
    else:
        note = "Held-out evaluation battery complete. Metrics are synthetic when the mock backend is selected."
    aggregates = aggregate_episode_metrics(episodes, float(cfg.analysis.confidence_z))
    write_csv(run_dir / "episode_metrics.csv", [item.to_dict() for item in episodes])
    write_csv(run_dir / "aggregate_metrics.csv", [item.to_dict() for item in aggregates])
    write_json(run_dir / "aggregate_metrics.json", [item.to_dict() for item in aggregates])
    overview_path = write_success_plot(run_dir, aggregates)
    summary_path = write_markdown_summary(run_dir, manifest, aggregates, note)
    artifacts.extend(
        [
            ArtifactRecord("csv", "episode_metrics.csv", "Episode-level evaluation metrics"),
            ArtifactRecord("csv", "aggregate_metrics.csv", "Aggregate metrics with confidence intervals"),
            ArtifactRecord("json", "aggregate_metrics.json", "Aggregate metrics in JSON"),
            ArtifactRecord("svg", overview_path.name, "Aggregate overview plot"),
            ArtifactRecord("markdown", summary_path.name, "Run summary"),
        ]
    )
    return _finalize_run(runtime_cfg, run_dir, manifest, artifacts)

def _with_runtime_context(cfg, run_dir: Path, manifest: RunManifest):
    runtime_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    runtime_cfg.execution.detb_run_dir = str(run_dir)
    runtime_cfg.execution.detb_run_id = manifest.run_id
    return runtime_cfg
