"""Thin Isaac Lab backend seam for DETB."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing import event_accumulator

from detb.extension import (
    detb_lab_root,
    expected_robot_asset_id_for_task,
    experiment_name,
    resolve_play_task_id,
    resolve_train_task_id,
    robot_actuator_profile,
    robot_asset_id,
    robot_spec_for_id,
)
from detb.io import read_json, write_json
from detb.models import EpisodeMetric


class IsaacLabBackend:
    """Backend helper for delegating real runs to the pinned Isaac Lab runtime."""

    name = "isaaclab"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.last_train_metadata: dict[str, Any] | None = None
        self.last_eval_metadata: dict[str, Any] | None = None
        self.last_play_metadata: dict[str, Any] | None = None

    @staticmethod
    def repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_path(cls, value: str, *, base: Path | None = None) -> Path:
        raw = Path(str(value))
        if raw.is_absolute():
            return raw
        anchor = base if base is not None else cls.repo_root()
        return (anchor / raw).resolve()

    @classmethod
    def root_path(cls, cfg) -> Path:
        root = cls._resolve_path(str(cfg.execution.isaaclab_root))
        if not root.exists():
            raise RuntimeError(f"Isaac Lab root not found: {root}")
        return root

    @classmethod
    def log_root_path(cls, cfg) -> Path:
        return cls._resolve_path(str(cfg.execution.isaaclab_log_root))

    @classmethod
    def _bootstrap_script_path(cls) -> Path:
        return cls.repo_root() / "scripts" / "run_with_detb_lab.py"

    @classmethod
    def _python_executable(cls, cfg) -> Path:
        requested = str(getattr(cfg.execution, "isaaclab_python", "")).strip()
        candidates: list[Path] = []
        if requested:
            candidates.append(cls._resolve_path(requested))
        candidates.append((Path.home() / "miniconda3" / "envs" / "isaaclab51" / "python.exe").resolve())
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if requested:
            raise RuntimeError(f"Configured Isaac Lab Python executable does not exist: {candidates[0]}")
        raise RuntimeError(
            "Unable to locate the Isaac Lab Python executable. "
            "Set execution.isaaclab_python to the isaaclab51 python.exe path."
        )

    @staticmethod
    def _runtime_run_dir(cfg) -> Path:
        run_dir = getattr(cfg.execution, "detb_run_dir", None)
        if not run_dir:
            raise RuntimeError("Missing DETB runtime context: execution.detb_run_dir")
        return Path(str(run_dir)).resolve()

    @staticmethod
    def _runtime_run_id(cfg) -> str:
        run_id = getattr(cfg.execution, "detb_run_id", None)
        if not run_id:
            raise RuntimeError("Missing DETB runtime context: execution.detb_run_id")
        return str(run_id)

    @classmethod
    def _train_script_path(cls, cfg) -> Path:
        return cls._resolve_path(str(cfg.execution.isaaclab_train_script))

    @classmethod
    def _eval_script_path(cls, cfg) -> Path:
        return cls._resolve_path(str(cfg.execution.isaaclab_eval_script))

    @classmethod
    def _play_script_path(cls, cfg) -> Path:
        return cls._resolve_path(str(cfg.execution.isaaclab_play_script))

    @staticmethod
    def _sanitize_name(value: str) -> str:
        allowed = []
        for char in value:
            if char.isalnum() or char in {"-", "_"}:
                allowed.append(char)
            else:
                allowed.append("_")
        return "".join(allowed)

    @classmethod
    def _run_name(cls, cfg) -> str:
        return cls._sanitize_name(f"detb_{cls._runtime_run_id(cfg)}")

    @staticmethod
    def _default_gui_kit_args(*, headless: bool) -> str:
        if os.name == "nt" and not headless:
            return "--/app/vulkan=false"
        return ""

    @classmethod
    def _effective_kit_args(cls, cfg, *, headless: bool) -> str:
        explicit = str(getattr(cfg.execution, "isaaclab_kit_args", "")).strip()
        fallback = cls._default_gui_kit_args(headless=headless)
        parts: list[str] = []
        if fallback and "/app/vulkan" not in explicit:
            parts.append(fallback)
        if explicit:
            parts.append(explicit)
        return " ".join(parts)

    @classmethod
    def _append_kit_args(cls, command: list[str], cfg, *, headless: bool) -> None:
        kit_args = cls._effective_kit_args(cfg, headless=headless)
        if kit_args:
            command.append(f"--kit_args={kit_args}")

    @staticmethod
    def _sensor_supported(cfg) -> None:
        if str(cfg.sensor.name) != "proprio":
            raise RuntimeError(
                "The real Isaac backend currently supports only the proprio sensor profile. "
                f"Received: {cfg.sensor.name}."
            )

    @staticmethod
    def _robot_supported(cfg) -> None:
        spec = robot_spec_for_id(robot_asset_id(cfg))
        if spec is None:
            raise RuntimeError(
                "The real Isaac backend does not recognize this DETB robot asset. "
                f"Received: {robot_asset_id(cfg)}."
            )
        if abs(float(cfg.robot.leg_length_scale) - 1.0) > 1e-6:
            raise RuntimeError(
                "The real Isaac backend does not yet support leg-length overrides. "
                f"Received: {cfg.robot.leg_length_scale}."
            )
        if abs(float(cfg.robot.stiffness) - float(spec.baseline_stiffness)) > 1e-6:
            raise RuntimeError(
                "The real Isaac backend does not yet support stiffness overrides for the "
                f"'{spec.actuator_profile}' profile. Expected {spec.baseline_stiffness}, "
                f"received {cfg.robot.stiffness}."
            )
        if abs(float(cfg.robot.damping) - float(spec.baseline_damping)) > 1e-6:
            raise RuntimeError(
                "The real Isaac backend does not yet support damping overrides for the "
                f"'{spec.actuator_profile}' profile. Expected {spec.baseline_damping}, "
                f"received {cfg.robot.damping}."
            )
        if robot_actuator_profile(cfg) != spec.actuator_profile:
            raise RuntimeError(
                "The real Isaac backend currently expects the configured DETB actuator profile. "
                f"Received: {robot_actuator_profile(cfg)}."
            )

    @classmethod
    def _assert_supported_real_cfg(cls, cfg) -> None:
        cls._sensor_supported(cfg)
        cls._robot_supported(cfg)

    @staticmethod
    def _assert_task_matches_robot(task_name: str, cfg) -> None:
        expected_asset_id = expected_robot_asset_id_for_task(task_name)
        if expected_asset_id and robot_asset_id(cfg) != expected_asset_id:
            raise RuntimeError(
                "The selected DETB task family is not compatible with the selected robot asset. "
                f"Task '{task_name}' expects '{expected_asset_id}', received '{robot_asset_id(cfg)}'."
            )

    @classmethod
    def build_visualize_command(
        cls,
        cfg,
        *,
        output_json: Path | None = None,
        telemetry_csv: Path | None = None,
        video_dir: Path | None = None,
    ) -> tuple[list[str], Path]:
        root = cls.root_path(cfg)
        python_exe = cls._python_executable(cfg)
        script = cls._play_script_path(cfg)
        task_name = resolve_play_task_id(cfg)
        train_task_name = resolve_train_task_id(cfg)
        cls._assert_supported_real_cfg(cfg)
        cls._assert_task_matches_robot(task_name, cfg)
        runtime_run_dir = getattr(cfg.execution, "detb_run_dir", "")
        preview_root = Path(str(runtime_run_dir)).resolve() if runtime_run_dir else cls.repo_root() / ".detb_preview" / "visualize"
        resolved_output_json = output_json or (preview_root / "isaac_play_result.json")
        resolved_telemetry_csv = telemetry_csv or (preview_root / "playback_telemetry.csv")
        resolved_video_dir = video_dir or (preview_root / "videos" / "play")
        command = [
            str(python_exe),
            str(script),
            "--task",
            task_name,
            "--num_envs",
            str(cfg.visualization.num_envs),
            "--seed",
            str(int(cfg.execution.seeds[0])),
            "--device",
            str(cfg.execution.device),
            "--log_root",
            str(cls.log_root_path(cfg)),
            "--experiment_name",
            experiment_name(cfg, train_task_name),
            "--output_json",
            str(resolved_output_json),
            "--telemetry_csv",
            str(resolved_telemetry_csv),
            "--video_dir",
            str(resolved_video_dir),
            "--rollout_steps",
            str(cfg.visualization.rollout_steps),
            "--robot_asset_id",
            robot_asset_id(cfg),
            "--robot_actuator_profile",
            robot_actuator_profile(cfg),
            "--sensor_profile",
            str(cfg.sensor.name),
            "--terrain_name",
            str(cfg.terrain.name),
            "--terrain_level",
            str(cfg.terrain.level),
            "--fault_class",
            str(cfg.fault.class_name),
            "--fault_severity",
            str(cfg.fault.severity),
            "--latency_steps",
            str(cfg.fault.latency_steps),
            "--success_distance_m",
            str(cfg.task.success_distance_m),
            "--body_mass_kg",
            str(cfg.robot.body_mass_kg),
            "--torque_limit_scale",
            str(cfg.robot.torque_limit_scale),
            "--leg_length_scale",
            str(cfg.robot.leg_length_scale),
            "--stiffness",
            str(cfg.robot.stiffness),
            "--damping",
            str(cfg.robot.damping),
            "--diagnostic_min_displacement_m",
            str(cfg.visualization.diagnostic_min_displacement_m),
            "--diagnostic_min_path_length_m",
            str(cfg.visualization.diagnostic_min_path_length_m),
            "--diagnostic_fall_height_m",
            str(cfg.visualization.diagnostic_fall_height_m),
            "--diagnostic_min_command_speed_mps",
            str(cfg.visualization.diagnostic_min_command_speed_mps),
        ]
        checkpoint = str(cfg.visualization.checkpoint).strip()
        load_run = str(cfg.visualization.load_run).strip()
        if checkpoint:
            command.extend(["--checkpoint", str(cls._resolve_path(checkpoint))])
        elif load_run:
            command.extend(["--load_run", load_run])
        elif bool(cfg.visualization.use_pretrained_checkpoint):
            command.append("--use_pretrained_checkpoint")
        if bool(cfg.visualization.real_time):
            command.append("--real_time")
        if bool(cfg.visualization.video):
            command.extend(["--video", "--video_length", str(cfg.visualization.video_length)])
        if bool(cfg.visualization.headless):
            command.append("--headless")
        cls._append_kit_args(command, cfg, headless=bool(cfg.visualization.headless))
        return command, root

    def visualize(self, cfg) -> dict[str, Any]:
        run_dir = self._runtime_run_dir(cfg)
        result_json = run_dir / "isaac_play_result.json"
        telemetry_csv = run_dir / "playback_telemetry.csv"
        stdout_path = run_dir / "isaac_play_stdout.log"
        stderr_path = run_dir / "isaac_play_stderr.log"
        video_dir = run_dir / "videos" / "play"
        command, cwd = self.build_visualize_command(
            cfg,
            output_json=result_json,
            telemetry_csv=telemetry_csv,
            video_dir=video_dir,
        )
        timeout_s = int(cfg.execution.isaaclab_timeout_s)
        try:
            return_code = self.run_command(
                command,
                cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=self._subprocess_env(),
                timeout_s=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Isaac Lab playback timed out after "
                f"{timeout_s} seconds. See {stdout_path.name} and {stderr_path.name} in {run_dir}."
            ) from exc
        if return_code != 0:
            raise RuntimeError(
                "Isaac Lab playback failed. "
                f"See {stdout_path.name} and {stderr_path.name} in {run_dir}."
            )
        if not result_json.exists():
            raise RuntimeError(
                "Isaac Lab playback exited without writing isaac_play_result.json. "
                f"See {stdout_path.name} and {stderr_path.name} in {run_dir}."
            )

        payload = read_json(result_json)
        recorded_videos = [str(path.resolve()) for path in sorted(video_dir.rglob("*.mp4"))]
        if recorded_videos:
            payload["video_files"] = recorded_videos
            write_json(result_json, payload)
        launch_spec = {
            "mode": "isaaclab_play",
            "cwd": str(cwd),
            "command": command,
            "stdout_log": stdout_path.name,
            "stderr_log": stderr_path.name,
            "return_code": return_code,
            "task": payload.get("task", resolve_play_task_id(cfg)),
            "experiment_name": experiment_name(cfg, resolve_train_task_id(cfg)),
            "telemetry_csv": telemetry_csv.name,
            "video_dir": str(video_dir),
        }
        payload["launch_spec"] = launch_spec
        self.last_play_metadata = launch_spec | {
            "result_json": result_json.name,
            "telemetry_csv": telemetry_csv.name,
            "source_checkpoint_path": str(payload.get("checkpoint", "")),
            "runtime_stack": payload.get("runtime_stack", {}),
            "video_files": payload.get("video_files", []),
        }
        return payload

    @classmethod
    def build_train_gui_command(cls, cfg) -> tuple[list[str], Path]:
        root = cls.root_path(cfg)
        python_exe = cls._python_executable(cfg)
        bootstrap_script = cls._bootstrap_script_path()
        script = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
        task_name = resolve_train_task_id(cfg)
        cls._assert_supported_real_cfg(cfg)
        cls._assert_task_matches_robot(task_name, cfg)
        command = [
            str(python_exe),
            str(bootstrap_script),
            str(script),
            "--task",
            task_name,
            "--num_envs",
            str(cfg.visualization.train_num_envs),
            "--max_iterations",
            str(cfg.visualization.train_max_iterations),
            "--seed",
            str(cfg.visualization.train_seed),
            "--device",
            str(cfg.execution.device),
        ]
        if bool(cfg.visualization.video):
            command.extend(
                [
                    "--video",
                    "--video_length",
                    str(cfg.visualization.video_length),
                    "--video_interval",
                    str(cfg.visualization.video_interval),
                ]
            )
        if bool(cfg.visualization.headless):
            command.append("--headless")
        cls._append_kit_args(command, cfg, headless=bool(cfg.visualization.headless))
        return command, root

    @classmethod
    def build_train_command(cls, cfg, output_json: Path) -> tuple[list[str], Path]:
        cls._assert_supported_real_cfg(cfg)
        root = cls.root_path(cfg)
        python_exe = cls._python_executable(cfg)
        script = cls._train_script_path(cfg)
        task_name = resolve_train_task_id(cfg)
        cls._assert_task_matches_robot(task_name, cfg)
        resolved_experiment_name = experiment_name(cfg, task_name)
        seed = int(cfg.execution.seeds[0])
        log_root = cls.log_root_path(cfg)
        command = [
            str(python_exe),
            str(script),
            "--task",
            task_name,
            "--num_envs",
            str(cfg.execution.num_envs),
            "--seed",
            str(seed),
            "--max_iterations",
            str(cfg.execution.train_max_iterations),
            "--device",
            str(cfg.execution.device),
            "--experiment_name",
            resolved_experiment_name,
            "--run_name",
            cls._run_name(cfg),
            "--output_json",
            str(output_json),
            "--log_root",
            str(log_root),
            "--robot_asset_id",
            robot_asset_id(cfg),
            "--robot_actuator_profile",
            robot_actuator_profile(cfg),
            "--sensor_profile",
            str(cfg.sensor.name),
            "--terrain_name",
            str(cfg.terrain.name),
            "--terrain_level",
            str(cfg.terrain.level),
            "--body_mass_kg",
            str(cfg.robot.body_mass_kg),
            "--torque_limit_scale",
            str(cfg.robot.torque_limit_scale),
            "--leg_length_scale",
            str(cfg.robot.leg_length_scale),
            "--stiffness",
            str(cfg.robot.stiffness),
            "--damping",
            str(cfg.robot.damping),
        ]
        if bool(cfg.execution.headless):
            command.append("--headless")
        cls._append_kit_args(command, cfg, headless=bool(cfg.execution.headless))
        return command, root

    @classmethod
    def build_evaluate_command(cls, cfg, output_json: Path, checkpoint_path: Path, seed: int) -> tuple[list[str], Path]:
        cls._assert_supported_real_cfg(cfg)
        root = cls.root_path(cfg)
        python_exe = cls._python_executable(cfg)
        script = cls._eval_script_path(cfg)
        task_name = resolve_play_task_id(cfg)
        cls._assert_task_matches_robot(task_name, cfg)
        command = [
            str(python_exe),
            str(script),
            "--task",
            task_name,
            "--num_envs",
            str(cfg.execution.eval_num_envs),
            "--seed",
            str(seed),
            "--device",
            str(cfg.execution.device),
            "--checkpoint",
            str(checkpoint_path),
            "--eval_episodes",
            str(cfg.execution.eval_episodes),
            "--output_json",
            str(output_json),
            "--robot_asset_id",
            robot_asset_id(cfg),
            "--robot_actuator_profile",
            robot_actuator_profile(cfg),
            "--sensor_profile",
            str(cfg.sensor.name),
            "--terrain_name",
            str(cfg.terrain.name),
            "--terrain_level",
            str(cfg.terrain.level),
            "--fault_name",
            str(cfg.fault.name),
            "--fault_class",
            str(cfg.fault.class_name),
            "--fault_severity",
            str(cfg.fault.severity),
            "--latency_steps",
            str(cfg.fault.latency_steps),
            "--success_distance_m",
            str(cfg.task.success_distance_m),
            "--body_mass_kg",
            str(cfg.robot.body_mass_kg),
            "--torque_limit_scale",
            str(cfg.robot.torque_limit_scale),
            "--leg_length_scale",
            str(cfg.robot.leg_length_scale),
            "--stiffness",
            str(cfg.robot.stiffness),
            "--damping",
            str(cfg.robot.damping),
        ]
        if bool(cfg.execution.headless):
            command.append("--headless")
        cls._append_kit_args(command, cfg, headless=bool(cfg.execution.headless))
        return command, root

    @classmethod
    def _subprocess_env(cls) -> dict[str, str]:
        env = os.environ.copy()
        repo_root = str(cls.repo_root())
        extension_root = str(detb_lab_root().resolve())
        current = env.get("PYTHONPATH", "")
        prefixes = [repo_root]
        if Path(extension_root).exists():
            prefixes.append(extension_root)
        env["PYTHONPATH"] = os.pathsep.join(prefixes + ([current] if current else []))
        return env

    @staticmethod
    def run_command(
        command: list[str],
        cwd: Path,
        *,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> int:
        stdout_handle = stdout_path.open("w", encoding="utf-8") if stdout_path is not None else None
        stderr_handle = stderr_path.open("w", encoding="utf-8") if stderr_path is not None else None
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=timeout_s,
            )
            return completed.returncode
        finally:
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()

    @staticmethod
    def _latest_checkpoint_in_dir(log_dir: Path) -> Path:
        checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
        if not checkpoints:
            raise FileNotFoundError(f"No Isaac Lab checkpoints found in: {log_dir}")
        return checkpoints[-1]

    @classmethod
    def _resolve_checkpoint(cls, cfg) -> Path:
        explicit = str(getattr(cfg.execution, "checkpoint", "")).strip()
        if explicit:
            checkpoint_path = cls._resolve_path(explicit)
            if checkpoint_path.exists():
                return checkpoint_path
            raise FileNotFoundError(f"Configured Isaac checkpoint does not exist: {checkpoint_path}")

        task_name = resolve_train_task_id(cfg)
        resolved_experiment_name = experiment_name(cfg, task_name)
        experiment_root = cls.log_root_path(cfg) / resolved_experiment_name
        if not experiment_root.exists():
            raise FileNotFoundError(f"Isaac experiment log directory does not exist: {experiment_root}")

        requested_run = str(getattr(cfg.execution, "load_run", "")).strip()
        if requested_run:
            run_dir = experiment_root / requested_run
            if not run_dir.exists():
                raise FileNotFoundError(f"Configured Isaac run directory does not exist: {run_dir}")
            return cls._latest_checkpoint_in_dir(run_dir)

        runs = sorted(path for path in experiment_root.iterdir() if path.is_dir())
        if not runs:
            raise FileNotFoundError(f"No Isaac run directories found in: {experiment_root}")
        return cls._latest_checkpoint_in_dir(runs[-1])

    @staticmethod
    def _reward_curve(event_file: Path) -> list[tuple[int, float]]:
        if not event_file.exists():
            return []
        accumulator = event_accumulator.EventAccumulator(str(event_file))
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        if "Train/mean_reward" in scalar_tags:
            tag = "Train/mean_reward"
        else:
            reward_tags = [name for name in scalar_tags if "reward" in name.lower()]
            if not reward_tags:
                return []
            tag = reward_tags[0]
        return [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]

    @staticmethod
    def _steps_per_second(event_file: Path, *, num_envs: int, num_steps_per_env: int) -> float:
        if not event_file.exists():
            return 0.0
        accumulator = event_accumulator.EventAccumulator(str(event_file))
        accumulator.Reload()
        if "Train/mean_reward" not in accumulator.Tags().get("scalars", []):
            return 0.0
        events = accumulator.Scalars("Train/mean_reward")
        if len(events) < 2:
            return 0.0
        wall_time = events[-1].wall_time - events[0].wall_time
        if wall_time <= 0:
            return 0.0
        total_steps = max(events[-1].step - events[0].step, 1) * int(num_envs) * int(num_steps_per_env)
        return round(total_steps / wall_time, 2)

    @staticmethod
    def _num_steps_per_env(agent_yaml: Path) -> int:
        if not agent_yaml.exists():
            return 24
        payload = yaml.safe_load(agent_yaml.read_text(encoding="utf-8")) or {}
        return int(payload.get("num_steps_per_env", 24))

    def train(self, cfg) -> dict:
        run_dir = self._runtime_run_dir(cfg)
        result_json = run_dir / "isaac_train_result.json"
        stdout_path = run_dir / "isaac_train_stdout.log"
        stderr_path = run_dir / "isaac_train_stderr.log"
        command, cwd = self.build_train_command(cfg, result_json)
        timeout_s = int(cfg.execution.isaaclab_timeout_s)
        try:
            return_code = self.run_command(
                command,
                cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=self._subprocess_env(),
                timeout_s=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Isaac Lab training timed out after "
                f"{timeout_s} seconds. See {stdout_path.name} and {stderr_path.name} in {run_dir}."
            ) from exc
        if return_code != 0:
            raise RuntimeError(
                "Isaac Lab training failed. "
                f"See {stdout_path.name} and {stderr_path.name} in {run_dir}."
            )
        if not result_json.exists():
            raise RuntimeError(
                "Isaac Lab training exited without writing isaac_train_result.json. "
                f"See {stdout_path.name} and {stderr_path.name} in {run_dir}."
            )

        payload = read_json(result_json)
        log_dir = Path(str(payload["log_dir"]))
        checkpoint_path = Path(str(payload["checkpoint_path"]))
        if payload.get("event_file"):
            event_file = Path(str(payload["event_file"]))
        else:
            event_file = log_dir / "missing.tfevents"
        reward_curve = self._reward_curve(event_file)
        steps_per_second = self._steps_per_second(
            event_file,
            num_envs=int(payload["num_envs"]),
            num_steps_per_env=self._num_steps_per_env(Path(str(payload["agent_yaml"]))),
        )
        training = {
            "steps_per_second": steps_per_second,
            "final_reward": reward_curve[-1][1] if reward_curve else 0.0,
            "convergence_step": reward_curve[-1][0] if reward_curve else 0,
            "reward_curve": reward_curve,
            "approx_vram_gb": round(float(cfg.sensor.vram_gb), 2),
            "source_checkpoint_path": str(checkpoint_path),
            "source_log_dir": str(log_dir),
            "source_event_file": str(event_file) if event_file.exists() else "",
            "source_env_yaml": str(payload.get("env_yaml", "")),
            "source_agent_yaml": str(payload.get("agent_yaml", "")),
            "source_debug_log": str(run_dir / "isaac_train_debug.log") if (run_dir / "isaac_train_debug.log").exists() else "",
            "runtime_stack": payload.get("runtime_stack", {}),
            "launch_spec": {
                "mode": "isaaclab_train",
                "cwd": str(cwd),
                "command": command,
                "stdout_log": stdout_path.name,
                "stderr_log": stderr_path.name,
                "return_code": return_code,
                "task": payload["task"],
                "experiment_name": payload["experiment_name"],
                "run_name": payload["run_name"],
            },
        }
        self.last_train_metadata = training["launch_spec"] | {
            "result_json": result_json.name,
            "source_log_dir": str(log_dir),
            "source_checkpoint_path": str(checkpoint_path),
            "source_env_yaml": training["source_env_yaml"],
            "source_agent_yaml": training["source_agent_yaml"],
            "source_event_file": training["source_event_file"],
            "source_debug_log": training["source_debug_log"],
            "runtime_stack": training["runtime_stack"],
        }
        return training

    def evaluate(self, cfg) -> list[EpisodeMetric]:
        run_dir = self._runtime_run_dir(cfg)
        checkpoint_path = self._resolve_checkpoint(cfg)
        episodes: list[EpisodeMetric] = []
        seed_runs: list[dict[str, Any]] = []
        timeout_s = int(cfg.execution.isaaclab_timeout_s)
        for seed in cfg.execution.seeds:
            result_json = run_dir / f"isaac_eval_seed_{int(seed)}.json"
            stdout_path = run_dir / f"isaac_eval_seed_{int(seed)}_stdout.log"
            stderr_path = run_dir / f"isaac_eval_seed_{int(seed)}_stderr.log"
            command, cwd = self.build_evaluate_command(cfg, result_json, checkpoint_path, int(seed))
            try:
                return_code = self.run_command(
                    command,
                    cwd,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    env=self._subprocess_env(),
                    timeout_s=timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "Isaac Lab evaluation timed out after "
                    f"{timeout_s} seconds. See {stdout_path.name} and {stderr_path.name} in {run_dir}."
                ) from exc
            if return_code != 0:
                raise RuntimeError(
                    "Isaac Lab evaluation failed. "
                    f"See {stdout_path.name} and {stderr_path.name} in {run_dir}."
                )
            if not result_json.exists():
                raise RuntimeError(
                    "Isaac Lab evaluation exited without writing its result payload. "
                    f"See {stdout_path.name} and {stderr_path.name} in {run_dir}."
                )
            payload = read_json(result_json)
            for row in payload["episodes"]:
                episodes.append(EpisodeMetric(**row))
            seed_runs.append(
                {
                    "seed": int(seed),
                    "result_json": result_json.name,
                    "stdout_log": stdout_path.name,
                    "stderr_log": stderr_path.name,
                    "return_code": return_code,
                    "command": command,
                    "cwd": str(cwd),
                }
            )
        self.last_eval_metadata = {
            "mode": "isaaclab_eval",
            "source_checkpoint_path": str(checkpoint_path),
            "seed_runs": seed_runs,
            "runtime_stack": payload.get("runtime_stack", {}) if seed_runs else {},
        }
        return episodes

    @staticmethod
    def copy_checkpoint(source_path: Path, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    @staticmethod
    def copy_if_exists(source_path: str, target_path: Path) -> bool:
        if not source_path:
            return False
        source = Path(source_path)
        if not source.exists():
            return False
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        return True
