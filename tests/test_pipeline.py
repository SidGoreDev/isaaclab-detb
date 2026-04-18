from pathlib import Path

import pytest

from detb.backends import IsaacLabBackend
from detb.config import default_config_dir, load_config
from detb.extension import detb_lab_version
from detb.io import read_json, write_csv, write_json
from detb.models import EpisodeMetric, RunManifest
from detb.pipeline import (
    bundle_artifacts,
    generate_requirements,
    run_evaluate,
    run_failure_eval,
    run_sensor_eval,
    run_sweep,
    run_terrain_eval,
    run_train,
    run_train_gui,
    run_tune,
    run_visualize,
)


def _cfg(tmp_path: Path, *extra: str):
    overrides = [f"execution.output_root={tmp_path.as_posix()}"]
    overrides.extend(extra)
    return load_config("base", default_config_dir(), overrides)


def _artifact_paths(run_dir: Path) -> set[str]:
    return {item["relative_path"] for item in read_json(run_dir / "artifact_registry.json")}


def test_v1_train_evaluate_bundle_and_requirements_contract(tmp_path: Path):
    train_result = run_train(_cfg(tmp_path))
    eval_result = run_evaluate(_cfg(tmp_path))

    train_manifest = read_json(train_result.run_dir / "run_manifest.json")
    eval_manifest = read_json(eval_result.run_dir / "run_manifest.json")
    train_paths = _artifact_paths(train_result.run_dir)
    eval_paths = _artifact_paths(eval_result.run_dir)

    assert train_manifest["command"] == "train"
    assert train_manifest["run_tier"] == "smoke"
    assert Path(train_manifest["checkpoint_path"]).exists()
    assert Path(train_manifest["checkpoint_path"]).name in train_paths
    assert {
        "resolved_config.yaml",
        "run_manifest.json",
        "training_summary.json",
        "training_reward_curve.csv",
        "training_curve.svg",
    }.issubset(train_paths)
    assert (train_result.run_dir / "artifact_registry.json").exists()

    train_summary = read_json(train_result.run_dir / "training_summary.json")
    assert train_summary["reward_curve"]
    assert train_summary["final_reward"] >= 0.0
    assert train_summary["convergence_step"] > 0

    assert eval_manifest["command"] == "evaluate"
    assert eval_manifest["run_tier"] == "smoke"
    assert {
        "resolved_config.yaml",
        "run_manifest.json",
        "episode_metrics.csv",
        "aggregate_metrics.csv",
        "aggregate_metrics.json",
        "aggregate_overview.svg",
        "summary.md",
    }.issubset(eval_paths)
    assert (eval_result.run_dir / "artifact_registry.json").exists()

    summary_path = bundle_artifacts(eval_result.run_dir)
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Rebuilt from stored aggregate metrics." in summary_text
    assert "## Aggregate Metrics" in summary_text
    assert "task_success_rate" in summary_text

    requirements_result = generate_requirements(_cfg(tmp_path), eval_result.run_dir)
    requirements_paths = _artifact_paths(requirements_result.run_dir)
    requirements_payload = read_json(requirements_result.run_dir / "requirement_ledger.json")
    requirements_md = (requirements_result.run_dir / "candidate_requirements.md").read_text(encoding="utf-8")

    assert requirements_result.run_dir == eval_result.run_dir
    assert {
        "requirement_ledger.csv",
        "requirement_ledger.json",
        "candidate_requirements.md",
    }.issubset(requirements_paths)
    assert requirements_payload[0]["req_id"] == "DETB-REQ-0000"
    assert requirements_payload[0]["status"] == "candidate"
    assert requirements_payload[0]["source_run_id"] == eval_manifest["run_id"]
    assert requirements_payload[0]["source_metric"] == "none"
    assert requirements_payload[0]["artifact_links"] == "summary.md"
    assert "Evidence gate not met" in requirements_payload[0]["assumptions"]
    assert "# DETB Candidate Requirements" in requirements_md
    assert "DETB-REQ-0000" in requirements_md


def test_v1_visualize_execute_artifact_contract(tmp_path: Path, monkeypatch):
    source_checkpoint_path = tmp_path / "isaac-play-source" / "baseline_policy.pt"

    def fake_visualize(self, cfg):
        run_dir = Path(str(cfg.execution.detb_run_dir))
        source_dir = source_checkpoint_path.parent
        source_dir.mkdir(parents=True, exist_ok=True)
        source_checkpoint_path.write_text("checkpoint", encoding="utf-8")

        (run_dir / "isaac_play_result.json").write_text(
            '{\n'
            '  "task": "DETB-Velocity-Flat-Anymal-C-Play-v0",\n'
            '  "task_registry_id": "DETB-Velocity-Flat-Anymal-C-Play-v0",\n'
            f'  "checkpoint": "{source_checkpoint_path.as_posix()}",\n'
            '  "video_files": ["C:/video.mp4"],\n'
            '  "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},\n'
            '  "diagnostics": {\n'
            '    "verdict": "insufficient_motion",\n'
            '    "timed_out": false,\n'
            '    "rollout_limit_reached": true,\n'
            '    "net_displacement_m": 0.08,\n'
            '    "path_length_m": 0.12,\n'
            '    "mean_planar_speed_mps": 0.03,\n'
            '    "mean_command_planar_speed_mps": 0.41,\n'
            '    "initial_position_m": [0.0, 0.0, 0.55],\n'
            '    "final_position_m": [0.08, 0.01, 0.54],\n'
            '    "min_height_m": 0.52,\n'
            '    "steps_completed": 40,\n'
            '    "command_motion_expected": true\n'
            '  }\n'
            '}\n',
            encoding="utf-8",
        )
        (run_dir / "playback_telemetry.csv").write_text(
            "step,sim_time_s,base_pos_x_m\n0,0.0,0.0\n1,0.02,0.01\n",
            encoding="utf-8",
        )
        (run_dir / "isaac_play_stdout.log").write_text("stdout\n", encoding="utf-8")
        (run_dir / "isaac_play_stderr.log").write_text("stderr\n", encoding="utf-8")
        (run_dir / "isaac_play_debug.log").write_text("debug\n", encoding="utf-8")
        self.last_play_metadata = {
            "mode": "isaaclab_play",
            "task_registry_id": "DETB-Velocity-Flat-Anymal-C-Play-v0",
            "result_json": "isaac_play_result.json",
            "telemetry_csv": "playback_telemetry.csv",
            "stdout_log": "isaac_play_stdout.log",
            "stderr_log": "isaac_play_stderr.log",
            "return_code": 0,
            "source_checkpoint_path": str(source_checkpoint_path),
            "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},
            "video_files": ["C:/video.mp4"],
        }
        return {
            "task": "DETB-Velocity-Flat-Anymal-C-Play-v0",
            "task_registry_id": "DETB-Velocity-Flat-Anymal-C-Play-v0",
            "checkpoint": str(source_checkpoint_path),
            "video_files": ["C:/video.mp4"],
            "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},
            "diagnostics": {
                "verdict": "insufficient_motion",
                "timed_out": False,
                "rollout_limit_reached": True,
                "net_displacement_m": 0.08,
                "path_length_m": 0.12,
                "mean_planar_speed_mps": 0.03,
                "mean_command_planar_speed_mps": 0.41,
                "initial_position_m": [0.0, 0.0, 0.55],
                "final_position_m": [0.08, 0.01, 0.54],
                "min_height_m": 0.52,
                "steps_completed": 40,
                "command_motion_expected": True,
            },
            "launch_spec": {
                "mode": "isaaclab_play",
                "cwd": str(tmp_path),
                "command": ["echo", "visualize"],
                "stdout_log": "isaac_play_stdout.log",
                "stderr_log": "isaac_play_stderr.log",
                "return_code": 0,
                "task": "DETB-Velocity-Flat-Anymal-C-Play-v0",
                "experiment_name": "detb_anymal_c_flat",
                "telemetry_csv": "playback_telemetry.csv",
                "video_dir": str(run_dir / "videos" / "play"),
            },
        }

    monkeypatch.setattr(
        IsaacLabBackend,
        "build_visualize_command",
        classmethod(lambda cls, cfg, **kwargs: (["echo", "visualize"], tmp_path)),
    )
    monkeypatch.setattr(IsaacLabBackend, "visualize", fake_visualize)

    visualize_result = run_visualize(_cfg(tmp_path, "visualization.execute=true"))
    visualize_manifest = read_json(visualize_result.run_dir / "run_manifest.json")
    visualize_payload = read_json(visualize_result.run_dir / "visualize_command.json")
    playback_result = read_json(visualize_result.run_dir / "isaac_play_result.json")
    playback_payload = read_json(visualize_result.run_dir / "isaac_play_runs.json")
    summary_text = (visualize_result.run_dir / "summary.md").read_text(encoding="utf-8")
    visualize_paths = _artifact_paths(visualize_result.run_dir)

    expected_visualize_paths = {
        "resolved_config.yaml",
        "run_manifest.json",
        "visualize_command.json",
        "summary.md",
        "isaac_play_result.json",
        "isaac_play_runs.json",
        "playback_telemetry.csv",
        "isaac_play_stdout.log",
        "isaac_play_stderr.log",
        "isaac_play_debug.log",
        Path(visualize_manifest["checkpoint_path"]).name,
    }

    assert visualize_manifest["task_registry_id"] == "DETB-Velocity-Flat-Anymal-C-Play-v0"
    assert visualize_manifest["command"] == "visualize"
    assert expected_visualize_paths.issubset(visualize_paths)
    assert (visualize_result.run_dir / "artifact_registry.json").exists()
    assert visualize_payload["mode"] == "isaaclab_play"
    assert visualize_payload["execute"] is True
    assert visualize_payload["task"] == "DETB-Velocity-Flat-Anymal-C-Play-v0"
    assert visualize_payload["return_code"] == 0
    assert playback_payload["mode"] == "isaaclab_play"
    assert playback_payload["source_checkpoint_path"] == str(source_checkpoint_path)
    assert playback_result["diagnostics"]["timed_out"] is False
    assert playback_result["diagnostics"]["rollout_limit_reached"] is True
    assert "Verdict: `insufficient_motion`" in summary_text
    assert "Video files: `1`" in summary_text
    assert "Playback diagnostics were captured from the DETB-owned Isaac Lab runner." in summary_text
    assert Path(visualize_manifest["checkpoint_path"]).read_text(encoding="utf-8") == "checkpoint"


def test_train_and_evaluate_create_artifacts(tmp_path: Path):
    train_result = run_train(_cfg(tmp_path))
    eval_result = run_evaluate(_cfg(tmp_path))

    assert (train_result.run_dir / "training_summary.json").exists()
    assert (train_result.run_dir / "training_curve.svg").exists()
    assert (eval_result.run_dir / "episode_metrics.csv").exists()
    assert (eval_result.run_dir / "aggregate_metrics.csv").exists()

    summary_path = bundle_artifacts(eval_result.run_dir)
    assert summary_path.exists()

    train_manifest = read_json(train_result.run_dir / "run_manifest.json")
    assert train_manifest["task_registry_id"] == "DETB-Velocity-Flat-Anymal-C-v0"
    assert train_manifest["robot_asset_id"] == "detb.anymal_c"
    assert train_manifest["run_tier"] == "smoke"
    assert train_manifest["detb_lab_version"] == detb_lab_version()


def test_custom_task_variant_writes_task_registry_id(tmp_path: Path):
    train_result = run_train(_cfg(tmp_path, "task=flat_walk_stability"))

    train_manifest = read_json(train_result.run_dir / "run_manifest.json")

    assert train_manifest["task_registry_id"] == "DETB-Velocity-Flat-Anymal-C-Stability-v0"


def test_custom_robot_variant_writes_robot_and_task_metadata(tmp_path: Path):
    train_result = run_train(_cfg(tmp_path, "task=flat_walk_simple_actuator", "robot=anymal_c_simple_actuator"))

    train_manifest = read_json(train_result.run_dir / "run_manifest.json")

    assert train_manifest["task_registry_id"] == "DETB-Velocity-Flat-Anymal-C-SimpleActuator-v0"
    assert train_manifest["robot_asset_id"] == "detb.anymal_c_simple_actuator"


def test_analysis_commands_and_requirements(tmp_path: Path):
    sweep_cfg = _cfg(tmp_path, "study=sweep")
    sweep_result = run_sweep(sweep_cfg, config_dir=default_config_dir())
    sensor_result = run_sensor_eval(_cfg(tmp_path), config_dir=default_config_dir())
    terrain_result = run_terrain_eval(_cfg(tmp_path), config_dir=default_config_dir())
    failure_result = run_failure_eval(_cfg(tmp_path), config_dir=default_config_dir())
    tune_result = run_tune(sweep_cfg, config_dir=default_config_dir())

    assert (sweep_result.run_dir / "sweep_results.csv").exists()
    assert (sensor_result.run_dir / "sensor_eval.csv").exists()
    assert (terrain_result.run_dir / "terrain_eval.json").exists()
    assert (failure_result.run_dir / "failure_eval.json").exists()
    assert (tune_result.run_dir / "tune_results.csv").exists()

    tune_payload = read_json(tune_result.run_dir / "tune_results.json")
    assert len(tune_payload["top_candidates"]) >= 3
    assert tune_payload["top_candidates"][0]["composite_score"] >= tune_payload["top_candidates"][-1]["composite_score"]
    assert any(not row["meets_targets"] for row in tune_payload["all_candidates"])

    requirements_result = generate_requirements(_cfg(tmp_path), terrain_result.run_dir)
    assert (requirements_result.run_dir / "requirement_ledger.csv").exists()
    requirements_payload = read_json(requirements_result.run_dir / "requirement_ledger.json")
    assert requirements_payload[0]["assumptions"].startswith("Evidence gate not met")


def test_visualization_launch_specs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        IsaacLabBackend,
        "build_visualize_command",
        classmethod(lambda cls, cfg, **kwargs: (["echo", "visualize"], tmp_path)),
    )
    monkeypatch.setattr(
        IsaacLabBackend,
        "build_train_gui_command",
        classmethod(lambda cls, cfg: (["echo", "train_gui"], tmp_path)),
    )
    monkeypatch.setattr(IsaacLabBackend, "run_command", staticmethod(lambda command, cwd: 0))

    visualize_result = run_visualize(_cfg(tmp_path))
    train_gui_result = run_train_gui(_cfg(tmp_path))

    visualize_payload = read_json(visualize_result.run_dir / "visualize_command.json")
    train_payload = read_json(train_gui_result.run_dir / "train_gui_command.json")

    assert visualize_payload["mode"] == "isaaclab_play"
    assert visualize_payload["task"] == "DETB-Velocity-Flat-Anymal-C-Play-v0"
    assert visualize_payload["execute"] is False
    assert (visualize_result.run_dir / "visualize_command.json").exists()
    assert train_payload["mode"] == "isaaclab_train"
    assert train_payload["execute"] is False
    assert (train_gui_result.run_dir / "train_gui_command.json").exists()


def test_visualization_execute_records_playback_artifacts(tmp_path: Path, monkeypatch):
    def fake_visualize(self, cfg):
        run_dir = Path(str(cfg.execution.detb_run_dir))
        source_checkpoint = tmp_path / "source_checkpoint.pt"
        source_checkpoint.write_text("checkpoint", encoding="utf-8")
        (run_dir / "isaac_play_result.json").write_text(
            '{\n'
            '  "task": "DETB-Velocity-Flat-Anymal-C-Play-v0",\n'
            '  "task_registry_id": "DETB-Velocity-Flat-Anymal-C-Play-v0",\n'
            f'  "checkpoint": "{source_checkpoint.as_posix()}",\n'
            '  "video_files": ["C:/video.mp4"],\n'
            '  "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},\n'
            '  "diagnostics": {\n'
            '    "verdict": "insufficient_motion",\n'
            '    "net_displacement_m": 0.08,\n'
            '    "path_length_m": 0.12,\n'
            '    "mean_planar_speed_mps": 0.03,\n'
            '    "mean_command_planar_speed_mps": 0.41,\n'
            '    "initial_position_m": [0.0, 0.0, 0.55],\n'
            '    "final_position_m": [0.08, 0.01, 0.54],\n'
            '    "min_height_m": 0.52,\n'
            '    "steps_completed": 40,\n'
            '    "command_motion_expected": true\n'
            '  }\n'
            '}\n',
            encoding="utf-8",
        )
        (run_dir / "playback_telemetry.csv").write_text(
            "step,sim_time_s,base_pos_x_m\n0,0.0,0.0\n1,0.02,0.01\n",
            encoding="utf-8",
        )
        (run_dir / "isaac_play_stdout.log").write_text("stdout\n", encoding="utf-8")
        (run_dir / "isaac_play_stderr.log").write_text("stderr\n", encoding="utf-8")
        (run_dir / "isaac_play_debug.log").write_text("debug\n", encoding="utf-8")
        self.last_play_metadata = {
            "mode": "isaaclab_play",
            "result_json": "isaac_play_result.json",
            "telemetry_csv": "playback_telemetry.csv",
            "stdout_log": "isaac_play_stdout.log",
            "stderr_log": "isaac_play_stderr.log",
            "return_code": 0,
            "source_checkpoint_path": str(source_checkpoint),
            "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},
            "video_files": ["C:/video.mp4"],
        }
        return {
            "task": "DETB-Velocity-Flat-Anymal-C-Play-v0",
            "task_registry_id": "DETB-Velocity-Flat-Anymal-C-Play-v0",
            "checkpoint": str(source_checkpoint),
            "video_files": ["C:/video.mp4"],
            "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},
            "diagnostics": {
                "verdict": "insufficient_motion",
                "net_displacement_m": 0.08,
                "path_length_m": 0.12,
                "mean_planar_speed_mps": 0.03,
                "mean_command_planar_speed_mps": 0.41,
                "initial_position_m": [0.0, 0.0, 0.55],
                "final_position_m": [0.08, 0.01, 0.54],
                "min_height_m": 0.52,
                "steps_completed": 40,
                "command_motion_expected": True,
            },
            "launch_spec": {
                "mode": "isaaclab_play",
                "cwd": str(tmp_path),
                "command": ["echo", "visualize"],
                "stdout_log": "isaac_play_stdout.log",
                "stderr_log": "isaac_play_stderr.log",
                "return_code": 0,
                "task": "DETB-Velocity-Flat-Anymal-C-Play-v0",
                "experiment_name": "detb_anymal_c_flat",
                "telemetry_csv": "playback_telemetry.csv",
                "video_dir": str(run_dir / "videos" / "play"),
            },
        }

    monkeypatch.setattr(
        IsaacLabBackend,
        "build_visualize_command",
        classmethod(lambda cls, cfg, **kwargs: (["echo", "visualize"], tmp_path)),
    )
    monkeypatch.setattr(IsaacLabBackend, "visualize", fake_visualize)

    visualize_result = run_visualize(_cfg(tmp_path, "visualization.execute=true"))

    command_payload = read_json(visualize_result.run_dir / "visualize_command.json")
    artifact_payload = read_json(visualize_result.run_dir / "artifact_registry.json")
    manifest = read_json(visualize_result.run_dir / "run_manifest.json")
    summary_text = (visualize_result.run_dir / "summary.md").read_text(encoding="utf-8")
    artifact_paths = {item["relative_path"] for item in artifact_payload}

    assert command_payload["execute"] is True
    assert command_payload["task"] == "DETB-Velocity-Flat-Anymal-C-Play-v0"
    assert command_payload["telemetry_csv"] == "playback_telemetry.csv"
    assert "isaac_play_result.json" in artifact_paths
    assert "playback_telemetry.csv" in artifact_paths
    assert "isaac_play_debug.log" in artifact_paths
    assert "isaac_play_runs.json" in artifact_paths
    assert "baseline_policy.pt" in artifact_paths
    assert manifest["task_registry_id"] == "DETB-Velocity-Flat-Anymal-C-Play-v0"
    assert "Verdict: `insufficient_motion`" in summary_text
    assert (visualize_result.run_dir / "baseline_policy.pt").exists()


def test_study_tier_requires_minimum_eval_episodes(tmp_path: Path):
    cfg = _cfg(tmp_path, "execution.run_tier=study")

    try:
        run_evaluate(cfg)
    except ValueError as exc:
        assert "Study-tier runs require at least" in str(exc)
    else:
        raise AssertionError("Expected study-tier evaluation to reject under-threshold settings.")


def test_isaac_train_packaging_records_configs_logs_and_runtime_stack(tmp_path: Path, monkeypatch):
    import detb.pipeline as pipeline

    class FakeIsaacBackend(IsaacLabBackend):
        def train(self, cfg):
            run_dir = Path(str(cfg.execution.detb_run_dir))
            source_dir = tmp_path / "isaac-source"
            source_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = source_dir / "model_0.pt"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            env_yaml = source_dir / "env.yaml"
            agent_yaml = source_dir / "agent.yaml"
            env_yaml.write_text("env: true\n", encoding="utf-8")
            agent_yaml.write_text("agent: true\n", encoding="utf-8")
            (run_dir / "isaac_train_result.json").write_text("{}", encoding="utf-8")
            (run_dir / "isaac_train_stdout.log").write_text("stdout\n", encoding="utf-8")
            (run_dir / "isaac_train_stderr.log").write_text("stderr\n", encoding="utf-8")
            (run_dir / "isaac_train_debug.log").write_text("debug\n", encoding="utf-8")
            self.last_train_metadata = {
                "result_json": "isaac_train_result.json",
                "stdout_log": "isaac_train_stdout.log",
                "stderr_log": "isaac_train_stderr.log",
            }
            return {
                "steps_per_second": 10.0,
                "final_reward": 1.0,
                "convergence_step": 24,
                "reward_curve": [(24, 1.0)],
                "approx_vram_gb": 2.5,
                "source_checkpoint_path": str(checkpoint),
                "source_log_dir": str(source_dir),
                "source_event_file": "",
                "source_env_yaml": str(env_yaml),
                "source_agent_yaml": str(agent_yaml),
                "source_debug_log": str(run_dir / "isaac_train_debug.log"),
                "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},
                "launch_spec": {"mode": "isaaclab_train", "command": ["python", "train.py"]},
            }

    monkeypatch.setattr(pipeline, "_backend_for", lambda name: FakeIsaacBackend())

    result = run_train(_cfg(tmp_path, "execution.backend=isaaclab"))
    artifacts = read_json(result.run_dir / "artifact_registry.json")
    manifest = read_json(result.run_dir / "run_manifest.json")
    paths = {item["relative_path"] for item in artifacts}

    assert (result.run_dir / "baseline_policy.pt").exists()
    assert (result.run_dir / "isaac_env.yaml").exists()
    assert (result.run_dir / "isaac_agent.yaml").exists()
    assert (result.run_dir / "isaac_train_command.json").exists()
    assert "isaac_train_debug.log" in paths
    assert "isaac_train_command.json" in paths
    assert "baseline_policy.pt" in paths
    assert "isaac_env.yaml" in paths
    assert "isaac_agent.yaml" in paths
    assert "run_manifest.json" in paths
    assert "resolved_config.yaml" in paths
    assert manifest["torch_version"] == "2.7.0"
    assert manifest["cuda_version"] == "12.4"
    assert manifest["rsl_rl_version"] == "3.1.2"


def test_isaac_evaluate_packaging_records_seed_artifacts_and_runtime_stack(tmp_path: Path, monkeypatch):
    import detb.pipeline as pipeline

    class FakeIsaacBackend(IsaacLabBackend):
        def evaluate(self, cfg):
            run_dir = Path(str(cfg.execution.detb_run_dir))
            source_dir = tmp_path / "isaac-eval-source"
            source_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = source_dir / "model_0.pt"
            checkpoint.write_text("checkpoint", encoding="utf-8")

            seed_runs = []
            episodes = []
            for seed in cfg.execution.seeds:
                result_name = f"isaac_eval_seed_{int(seed)}.json"
                stdout_name = f"isaac_eval_seed_{int(seed)}_stdout.log"
                stderr_name = f"isaac_eval_seed_{int(seed)}_stderr.log"
                (run_dir / result_name).write_text("{}", encoding="utf-8")
                (run_dir / stdout_name).write_text("stdout\n", encoding="utf-8")
                (run_dir / stderr_name).write_text("stderr\n", encoding="utf-8")
                seed_runs.append(
                    {
                        "seed": int(seed),
                        "result_json": result_name,
                        "stdout_log": stdout_name,
                        "stderr_log": stderr_name,
                        "return_code": 0,
                        "command": ["python", "evaluate.py"],
                        "cwd": str(tmp_path),
                    }
                )
                episodes.append(
                    EpisodeMetric(
                        episode_id=f"{int(seed)}-0",
                        terrain_level=0,
                        terrain_name="flat_eval",
                        fault_level=0.0,
                        fault_name="nominal",
                        success=1,
                        distance_m=9.5,
                        elapsed_time_s=20.0,
                        energy_proxy=0.4,
                        failure_label="none",
                        seed=int(seed),
                        sensor_profile="proprio",
                    )
                )

            self.last_eval_metadata = {
                "mode": "isaaclab_eval",
                "source_checkpoint_path": str(checkpoint),
                "seed_runs": seed_runs,
                "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},
            }
            return episodes

    monkeypatch.setattr(pipeline, "_backend_for", lambda name: FakeIsaacBackend())

    result = run_evaluate(_cfg(tmp_path, "execution.backend=isaaclab"))
    artifacts = read_json(result.run_dir / "artifact_registry.json")
    manifest = read_json(result.run_dir / "run_manifest.json")
    paths = {item["relative_path"] for item in artifacts}

    assert (result.run_dir / "baseline_policy.pt").exists()
    assert (result.run_dir / "isaac_eval_runs.json").exists()
    assert "baseline_policy.pt" in paths
    assert "isaac_eval_runs.json" in paths
    assert "isaac_eval_seed_11.json" in paths
    assert "isaac_eval_seed_11_stdout.log" in paths
    assert "isaac_eval_seed_11_stderr.log" in paths
    assert manifest["torch_version"] == "2.7.0"
    assert manifest["cuda_version"] == "12.4"
    assert manifest["rsl_rl_version"] == "3.1.2"


def _seed_run_manifest(run_dir: Path, **overrides) -> RunManifest:
    defaults = dict(
        run_id="seeded-run",
        command="evaluate",
        timestamp="2026-04-18T00:00:00Z",
        git_commit="0000000",
        isaac_sim_version="5.1.0",
        isaaclab_version="2.3.0",
        driver_version="unknown",
        operating_system="test",
        gpu_model="test",
        robot_variant="anymal_c",
        task_family="flat_walk",
        terrain_name="flat_eval",
        sensor_profile="proprio",
        fault_profile="nominal",
        seeds=[11],
        backend="mock",
        checkpoint_path="",
        config_snapshot_path="",
        run_tier="smoke",
        configured_eval_episodes=0,
    )
    defaults.update(overrides)
    manifest = RunManifest(**defaults)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_manifest.json", manifest.to_dict())
    return manifest


def _seed_aggregate_csv(run_dir: Path, rows: list[dict]) -> None:
    write_csv(run_dir / "aggregate_metrics.csv", rows)


def test_bundle_artifacts_rebuilds_summary_from_csv(tmp_path: Path):
    run_dir = tmp_path / "evaluate" / "seeded"
    _seed_run_manifest(run_dir, command="evaluate")
    _seed_aggregate_csv(
        run_dir,
        [
            {
                "metric_name": "task_success_rate",
                "aggregation_scope": "overall",
                "mean": 0.82,
                "median": 0.82,
                "stddev": 0.05,
                "ci_low": 0.78,
                "ci_high": 0.86,
                "n": 8,
                "seed_count": 3,
            }
        ],
    )

    summary_path = bundle_artifacts(run_dir)

    assert summary_path == run_dir / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Rebuilt from stored aggregate metrics." in summary_text
    assert "## Aggregate Metrics" in summary_text
    assert "task_success_rate" in summary_text


def test_bundle_artifacts_missing_manifest_raises(tmp_path: Path):
    run_dir = tmp_path / "evaluate" / "empty"
    run_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        bundle_artifacts(run_dir)


def test_generate_requirements_no_evidence_emits_req_0000(tmp_path: Path):
    run_dir = tmp_path / "evaluate" / "smoke"
    _seed_run_manifest(run_dir, run_tier="smoke", seeds=[11], configured_eval_episodes=4)

    result = generate_requirements(_cfg(tmp_path), run_dir)
    ledger = read_json(result.run_dir / "requirement_ledger.json")

    assert result.run_dir == run_dir
    assert len(ledger) == 1
    assert ledger[0]["req_id"] == "DETB-REQ-0000"
    assert ledger[0]["source_metric"] == "none"
    assert "Evidence gate not met" in ledger[0]["assumptions"]
    assert (run_dir / "candidate_requirements.md").exists()
    assert (run_dir / "requirement_ledger.csv").exists()


def test_generate_requirements_emits_req_0001_when_threshold_met(tmp_path: Path):
    run_dir = tmp_path / "evaluate" / "study"
    _seed_run_manifest(
        run_dir,
        run_tier="study",
        seeds=[11, 22, 33],
        configured_eval_episodes=10,
    )
    _seed_aggregate_csv(
        run_dir,
        [
            {
                "metric_name": "task_success_rate",
                "aggregation_scope": "overall",
                "mean": 0.90,
                "median": 0.90,
                "stddev": 0.02,
                "ci_low": 0.86,
                "ci_high": 0.94,
                "n": 30,
                "seed_count": 3,
            }
        ],
    )

    result = generate_requirements(_cfg(tmp_path), run_dir)
    ledger = read_json(result.run_dir / "requirement_ledger.json")
    req_ids = {record["req_id"] for record in ledger}

    assert "DETB-REQ-0001" in req_ids
    req_0001 = next(record for record in ledger if record["req_id"] == "DETB-REQ-0001")
    assert req_0001["source_metric"] == "task_success_rate"
    assert req_0001["confidence_interval"] == "[0.860, 0.940]"


def test_generate_requirements_emits_req_0002_from_terrain_eval(tmp_path: Path):
    run_dir = tmp_path / "evaluate" / "study_terrain"
    _seed_run_manifest(
        run_dir,
        run_tier="study",
        seeds=[11, 22, 33],
        configured_eval_episodes=10,
    )
    _seed_aggregate_csv(run_dir, [])
    write_json(run_dir / "terrain_eval.json", {"terrain_generalization_score": 0.72})

    result = generate_requirements(_cfg(tmp_path), run_dir)
    ledger = read_json(result.run_dir / "requirement_ledger.json")
    req_ids = {record["req_id"] for record in ledger}

    assert "DETB-REQ-0002" in req_ids
    req_0002 = next(record for record in ledger if record["req_id"] == "DETB-REQ-0002")
    assert req_0002["source_metric"] == "terrain_generalization_score"
    assert req_0002["confidence_interval"] == "[0.720, 0.720]"


def test_generate_requirements_emits_req_0003_from_failure_eval(tmp_path: Path):
    run_dir = tmp_path / "evaluate" / "study_failure"
    _seed_run_manifest(
        run_dir,
        run_tier="study",
        seeds=[11, 22, 33],
        configured_eval_episodes=10,
    )
    _seed_aggregate_csv(run_dir, [])
    write_json(run_dir / "failure_eval.json", {"critical_threshold": 0.35})

    result = generate_requirements(_cfg(tmp_path), run_dir)
    ledger = read_json(result.run_dir / "requirement_ledger.json")
    req_ids = {record["req_id"] for record in ledger}

    assert "DETB-REQ-0003" in req_ids
    req_0003 = next(record for record in ledger if record["req_id"] == "DETB-REQ-0003")
    assert req_0003["source_metric"] == "critical_threshold"
    assert "0.350" in req_0003["statement"] or "0.35" in req_0003["statement"]
