from pathlib import Path

from omegaconf import OmegaConf

from detb.backends import IsaacLabBackend
from detb.config import default_config_dir, load_config


def _cfg(tmp_path: Path, *extra: str):
    isaaclab_root = tmp_path / "isaaclab-root"
    isaaclab_root.mkdir(exist_ok=True)
    (isaaclab_root / "isaaclab.bat").write_text("@echo off\n", encoding="utf-8")
    fake_python = tmp_path / "isaaclab51" / "python.exe"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("python", encoding="utf-8")
    overrides = [
        f"execution.isaaclab_root={isaaclab_root.as_posix()}",
        f"execution.isaaclab_log_root={(tmp_path / 'logs' / 'rsl_rl').as_posix()}",
        f"execution.isaaclab_python={fake_python.as_posix()}",
    ]
    overrides.extend(extra)
    return load_config("base", default_config_dir(), overrides)



def _runtime_cfg(tmp_path: Path, *extra: str):
    cfg = _cfg(tmp_path, *extra)
    run_dir = tmp_path / "detb-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    runtime_cfg.execution.detb_run_dir = str(run_dir)
    runtime_cfg.execution.detb_run_id = "detb_test_run"
    return runtime_cfg, run_dir



def test_build_visualize_command_uses_python_runner(tmp_path: Path):
    cfg = _cfg(tmp_path)
    command, cwd = IsaacLabBackend.build_visualize_command(cfg)

    assert cwd == (tmp_path / "isaaclab-root").resolve()
    assert command[0].endswith("python.exe")
    assert command[1].endswith("detb_isaaclab_play.py")
    assert "--task" in command
    assert "DETB-Velocity-Flat-Anymal-C-Play-v0" in command
    assert "--device" in command
    assert "cuda:0" in command
    assert "--output_json" in command
    assert "--telemetry_csv" in command
    assert "--rollout_steps" in command
    assert "--use_pretrained_checkpoint" in command
    assert "--real-time" in command



def test_build_train_gui_command_includes_video_controls(tmp_path: Path):
    cfg = _cfg(tmp_path, "visualization.video=true", "visualization.train_num_envs=48", "visualization.train_max_iterations=120")
    command, cwd = IsaacLabBackend.build_train_gui_command(cfg)

    assert cwd == (tmp_path / "isaaclab-root").resolve()
    assert command[0].endswith("python.exe")
    assert command[1].endswith("run_with_detb_lab.py")
    assert command[2].endswith("train.py")
    assert "--num_envs" in command
    assert "48" in command
    assert "--max_iterations" in command
    assert "120" in command
    assert "DETB-Velocity-Flat-Anymal-C-v0" in command
    assert "--video" in command
    assert "--video_length" in command
    assert "--video_interval" in command



def test_train_reads_result_metadata_and_curve(tmp_path: Path, monkeypatch):
    cfg, run_dir = _runtime_cfg(tmp_path)
    backend = IsaacLabBackend(cfg)
    source_log_dir = tmp_path / "logs" / "rsl_rl" / "anymal_c_flat" / "2026-01-01_00-00-00_detb_test_run"
    params_dir = source_log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = source_log_dir / "model_299.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    event_file = source_log_dir / "events.out.tfevents.test"
    event_file.write_text("events", encoding="utf-8")
    (params_dir / "env.yaml").write_text("env: true\n", encoding="utf-8")
    (params_dir / "agent.yaml").write_text("num_steps_per_env: 24\n", encoding="utf-8")

    def fake_run(command, cwd, **kwargs):
        result_path = run_dir / "isaac_train_result.json"
        result_path.write_text(
            '{\n'
            f'  "task": "Isaac-Velocity-Flat-Anymal-C-v0",\n'
            f'  "experiment_name": "anymal_c_flat",\n'
            f'  "run_name": "detb_detb_test_run",\n'
            f'  "log_dir": "{source_log_dir.as_posix()}",\n'
            f'  "checkpoint_path": "{checkpoint_path.as_posix()}",\n'
            f'  "event_file": "{event_file.as_posix()}",\n'
            f'  "env_yaml": "{(params_dir / "env.yaml").as_posix()}",\n'
            f'  "agent_yaml": "{(params_dir / "agent.yaml").as_posix()}",\n'
            '  "device": "cuda:0",\n'
            '  "num_envs": 64,\n'
            '  "seed": 11,\n'
            '  "max_iterations": 300\n'
            '}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(IsaacLabBackend, "run_command", staticmethod(fake_run))
    monkeypatch.setattr(IsaacLabBackend, "_reward_curve", staticmethod(lambda event_file: [(24, 1.5), (48, 2.5)]))
    monkeypatch.setattr(IsaacLabBackend, "_steps_per_second", staticmethod(lambda event_file, **kwargs: 321.0))

    training = backend.train(cfg)

    assert training["source_checkpoint_path"] == str(checkpoint_path)
    assert training["reward_curve"][-1] == (48, 2.5)
    assert training["steps_per_second"] == 321.0
    assert backend.last_train_metadata is not None
    assert backend.last_train_metadata["source_log_dir"] == str(source_log_dir)



def test_evaluate_reads_seed_payloads(tmp_path: Path, monkeypatch):
    checkpoint_path = tmp_path / "baseline_policy.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    cfg, run_dir = _runtime_cfg(tmp_path, f"execution.checkpoint={checkpoint_path.as_posix()}")
    backend = IsaacLabBackend(cfg)

    def fake_run(command, cwd, **kwargs):
        output_path = Path(command[command.index("--output_json") + 1])
        seed = int(command[command.index("--seed") + 1])
        output_path.write_text(
            '{\n'
            f'  "task": "Isaac-Velocity-Flat-Anymal-C-Play-v0",\n'
            f'  "checkpoint": "{checkpoint_path.as_posix()}",\n'
            '  "episodes": [\n'
            '    {\n'
            f'      "episode_id": "{seed}-0",\n'
            '      "terrain_level": 0,\n'
            '      "terrain_name": "L0_flat",\n'
            '      "fault_level": 0.0,\n'
            '      "fault_name": "nominal",\n'
            '      "success": 1,\n'
            '      "distance_m": 9.25,\n'
            '      "elapsed_time_s": 20.0,\n'
            '      "energy_proxy": 0.45,\n'
            '      "failure_label": "none",\n'
            f'      "seed": {seed},\n'
            '      "sensor_profile": "proprio"\n'
            '    }\n'
            '  ],\n'
            f'  "device": "cuda:0",\n'
            f'  "seed": {seed}\n'
            '}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(IsaacLabBackend, "run_command", staticmethod(fake_run))

    episodes = backend.evaluate(cfg)

    assert len(episodes) == 3
    assert all(item.success == 1 for item in episodes)
    assert backend.last_eval_metadata is not None
    assert backend.last_eval_metadata["source_checkpoint_path"] == str(checkpoint_path)
    assert len(backend.last_eval_metadata["seed_runs"]) == 3


def test_visualize_reads_result_metadata_and_logs(tmp_path: Path, monkeypatch):
    cfg, run_dir = _runtime_cfg(tmp_path)
    backend = IsaacLabBackend(cfg)

    def fake_run(command, cwd, **kwargs):
        (run_dir / "isaac_play_stdout.log").write_text("stdout\n", encoding="utf-8")
        (run_dir / "isaac_play_stderr.log").write_text("stderr\n", encoding="utf-8")
        (run_dir / "isaac_play_debug.log").write_text("debug\n", encoding="utf-8")
        (run_dir / "playback_telemetry.csv").write_text(
            "step,sim_time_s,base_pos_x_m\n0,0.0,0.0\n1,0.02,0.15\n",
            encoding="utf-8",
        )
        (run_dir / "videos" / "play").mkdir(parents=True, exist_ok=True)
        (run_dir / "videos" / "play" / "rl-video-step-0.mp4").write_text("video", encoding="utf-8")
        result_path = run_dir / "isaac_play_result.json"
        result_path.write_text(
            '{\n'
            '  "task": "DETB-Velocity-Flat-Anymal-C-Play-v0",\n'
            '  "task_registry_id": "DETB-Velocity-Flat-Anymal-C-Play-v0",\n'
            f'  "checkpoint": "{(tmp_path / "baseline_policy.pt").as_posix()}",\n'
            '  "video_files": [\n'
            f'    "{(run_dir / "videos" / "play" / "rl-video-step-0.mp4").as_posix()}"\n'
            '  ],\n'
            '  "runtime_stack": {"torch_version": "2.7.0", "cuda_version": "12.4", "rsl_rl_version": "3.1.2"},\n'
            '  "diagnostics": {\n'
            '    "verdict": "locomoting",\n'
            '    "net_displacement_m": 1.25,\n'
            '    "path_length_m": 1.42,\n'
            '    "mean_planar_speed_mps": 0.61,\n'
            '    "mean_command_planar_speed_mps": 0.74,\n'
            '    "initial_position_m": [0.0, 0.0, 0.55],\n'
            '    "final_position_m": [1.25, 0.05, 0.54],\n'
            '    "min_height_m": 0.51,\n'
            '    "steps_completed": 50,\n'
            '    "command_motion_expected": true\n'
            '  }\n'
            '}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(IsaacLabBackend, "run_command", staticmethod(fake_run))

    payload = backend.visualize(cfg)

    assert payload["diagnostics"]["verdict"] == "locomoting"
    assert payload["launch_spec"]["stdout_log"] == "isaac_play_stdout.log"
    assert backend.last_play_metadata is not None
    assert backend.last_play_metadata["telemetry_csv"] == "playback_telemetry.csv"
    assert len(backend.last_play_metadata["video_files"]) == 1

def test_train_command_passes_repo_local_log_root(tmp_path: Path):
    cfg, _ = _runtime_cfg(tmp_path)
    command, cwd = IsaacLabBackend._train_command(cfg, tmp_path / 'result.json')

    assert cwd == (tmp_path / 'isaaclab-root').resolve()
    assert "DETB-Velocity-Flat-Anymal-C-v0" in command
    assert '--log_root' in command
    assert command[command.index('--log_root') + 1] == str((tmp_path / 'logs' / 'rsl_rl').resolve())
    assert '--robot_asset_id' in command
    assert command[command.index('--robot_asset_id') + 1] == 'detb.anymal_c'


def test_task_command_falls_back_when_registry_id_missing(tmp_path: Path):
    cfg = _cfg(tmp_path, "task.registry_id=")
    command, _ = IsaacLabBackend.build_visualize_command(cfg)

    assert "Isaac-Velocity-Flat-Anymal-C-Play-v0" in command


def test_visualize_command_resolves_explicit_checkpoint_path(tmp_path: Path):
    checkpoint = tmp_path / "outputs" / "baseline_policy.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    cfg = _cfg(tmp_path, f"visualization.checkpoint={checkpoint.as_posix()}", "visualization.use_pretrained_checkpoint=false")
    command, _ = IsaacLabBackend.build_visualize_command(cfg)

    assert command[command.index("--checkpoint") + 1] == str(checkpoint.resolve())


def test_stability_task_uses_detb_registry_ids(tmp_path: Path):
    cfg = _cfg(tmp_path, "task=flat_walk_stability")
    runtime_cfg, _ = _runtime_cfg(tmp_path, "task=flat_walk_stability")
    visualize_command, _ = IsaacLabBackend.build_visualize_command(cfg)
    train_command, _ = IsaacLabBackend._train_command(runtime_cfg, tmp_path / "result.json")

    assert "DETB-Velocity-Flat-Anymal-C-Stability-Play-v0" in visualize_command
    assert "DETB-Velocity-Flat-Anymal-C-Stability-v0" in train_command


def test_simple_actuator_task_uses_matching_robot_and_registry_ids(tmp_path: Path):
    cfg = _cfg(tmp_path, "task=flat_walk_simple_actuator", "robot=anymal_c_simple_actuator")
    runtime_cfg, _ = _runtime_cfg(tmp_path, "task=flat_walk_simple_actuator", "robot=anymal_c_simple_actuator")
    visualize_command, _ = IsaacLabBackend.build_visualize_command(cfg)
    train_command, _ = IsaacLabBackend._train_command(runtime_cfg, tmp_path / "result.json")

    assert "DETB-Velocity-Flat-Anymal-C-SimpleActuator-Play-v0" in visualize_command
    assert "DETB-Velocity-Flat-Anymal-C-SimpleActuator-v0" in train_command
    assert train_command[train_command.index("--robot_asset_id") + 1] == "detb.anymal_c_simple_actuator"
    assert train_command[train_command.index("--robot_actuator_profile") + 1] == "dc_motor"


def test_mismatched_task_and_robot_are_rejected(tmp_path: Path):
    cfg = _cfg(tmp_path, "task=flat_walk_simple_actuator")

    try:
        IsaacLabBackend.build_visualize_command(cfg)
    except RuntimeError as exc:
        assert "not compatible" in str(exc)
    else:
        raise AssertionError("Expected mismatched task and robot selection to be rejected.")
