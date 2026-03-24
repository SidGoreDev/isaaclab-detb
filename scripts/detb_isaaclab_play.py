from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run DETB-managed Isaac Lab playback with diagnostics.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--log_root", type=str, required=True)
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--load_run", type=str, default="")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", default=False)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--telemetry_csv", type=str, required=True)
parser.add_argument("--video_dir", type=str, default="")
parser.add_argument("--rollout_steps", type=int, required=True)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--real_time", action="store_true", default=False)
parser.add_argument("--robot_asset_id", type=str, required=True)
parser.add_argument("--robot_actuator_profile", type=str, required=True)
parser.add_argument("--sensor_profile", type=str, required=True)
parser.add_argument("--terrain_name", type=str, required=True)
parser.add_argument("--terrain_level", type=int, required=True)
parser.add_argument("--fault_class", type=str, required=True)
parser.add_argument("--fault_severity", type=float, required=True)
parser.add_argument("--latency_steps", type=int, required=True)
parser.add_argument("--success_distance_m", type=float, required=True)
parser.add_argument("--body_mass_kg", type=float, required=True)
parser.add_argument("--torque_limit_scale", type=float, required=True)
parser.add_argument("--leg_length_scale", type=float, required=True)
parser.add_argument("--stiffness", type=float, required=True)
parser.add_argument("--damping", type=float, required=True)
parser.add_argument("--diagnostic_min_displacement_m", type=float, required=True)
parser.add_argument("--diagnostic_min_path_length_m", type=float, required=True)
parser.add_argument("--diagnostic_fall_height_m", type=float, required=True)
parser.add_argument("--diagnostic_min_command_speed_mps", type=float, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401

from detb_isaaclab_common import (
    apply_fault_to_actions,
    prepare_cfgs,
    resolve_pretrained_checkpoint_task_name,
    validate_supported_configuration,
)


def _debug_log_path(output_json: str) -> Path:
    return Path(output_json).resolve().with_name("isaac_play_debug.log")


def _make_stage_logger(output_json: str):
    debug_path = _debug_log_path(output_json)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    handle = debug_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    return handle, log


def _runtime_stack() -> dict[str, str]:
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(getattr(torch.version, "cuda", "unknown") or "unknown"),
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
    }


def _latest_checkpoint(log_dir: Path) -> Path:
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints were written to {log_dir}")
    return checkpoints[-1]


def _resolve_checkpoint() -> Path:
    task_name = args.task.split(":")[-1]

    if args.checkpoint:
        raw_path = Path(args.checkpoint).expanduser()
        if raw_path.exists():
            return raw_path.resolve()
        repo_relative = Path(__file__).resolve().parents[1] / raw_path
        if repo_relative.exists():
            return repo_relative.resolve()
        return Path(str(retrieve_file_path(args.checkpoint))).resolve()

    if args.use_pretrained_checkpoint:
        pretrained_task_name = resolve_pretrained_checkpoint_task_name(task_name)
        published = get_published_pretrained_checkpoint("rsl_rl", pretrained_task_name)
        if published:
            return Path(str(published)).resolve()
        raise FileNotFoundError(
            "A published pretrained checkpoint is not available for "
            f"'{task_name}' (resolved pretrained source '{pretrained_task_name}')."
        )

    experiment_root = Path(args.log_root).expanduser().resolve() / args.experiment_name
    if not experiment_root.exists():
        raise FileNotFoundError(f"Isaac Lab experiment directory does not exist: {experiment_root}")

    if args.load_run:
        run_dir = experiment_root / args.load_run
        if not run_dir.exists():
            raise FileNotFoundError(f"Requested Isaac Lab run does not exist: {run_dir}")
        return _latest_checkpoint(run_dir)

    run_dirs = sorted(path for path in experiment_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No Isaac Lab run directories were found in: {experiment_root}")
    return _latest_checkpoint(run_dirs[-1])


def _extract_policy_components(runner):
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None
    return policy_nn, normalizer


def _flatten_recurrent_policy_memory(policy_nn) -> None:
    for memory_name in ("memory_a", "memory_c"):
        memory = getattr(policy_nn, memory_name, None)
        rnn = getattr(memory, "rnn", None)
        if rnn is not None and hasattr(rnn, "flatten_parameters"):
            rnn.flatten_parameters()


def _flatten_actuator_lstm_modules(robot) -> None:
    actuators = getattr(robot, "actuators", {})
    for actuator in getattr(actuators, "values", lambda: [])():
        network = getattr(actuator, "network", None)
        lstm = getattr(network, "lstm", None)
        if lstm is not None and hasattr(lstm, "flatten_parameters"):
            lstm.flatten_parameters()


def _make_runner(env, agent_cfg, checkpoint_path: Path):
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(str(checkpoint_path))
    return wrapper, runner


def _command_tensor(env) -> torch.Tensor:
    try:
        return env.command_manager.get_command("base_velocity")[0].detach().cpu()
    except Exception:
        return torch.zeros(3)


def _planar_speed_from_robot(robot) -> float:
    velocity = getattr(robot.data, "root_lin_vel_w", None)
    if velocity is None:
        velocity = getattr(robot.data, "root_lin_vel_b", None)
    if velocity is None:
        return 0.0
    return float(torch.linalg.vector_norm(velocity[0, :2]).item())


def _telemetry_row(step: int, sim_time_s: float, robot, command: torch.Tensor, initial_xy, path_length_m: float) -> dict:
    root_pos = robot.data.root_pos_w[0].detach().cpu()
    planar_displacement = float(torch.linalg.vector_norm(root_pos[:2] - initial_xy).item())
    command_planar_speed = float(torch.linalg.vector_norm(command[:2]).item())
    planar_speed = _planar_speed_from_robot(robot)
    return {
        "step": int(step),
        "sim_time_s": round(sim_time_s, 4),
        "base_pos_x_m": round(float(root_pos[0].item()), 6),
        "base_pos_y_m": round(float(root_pos[1].item()), 6),
        "base_pos_z_m": round(float(root_pos[2].item()), 6),
        "planar_displacement_m": round(planar_displacement, 6),
        "path_length_m": round(path_length_m, 6),
        "planar_speed_mps": round(planar_speed, 6),
        "command_lin_vel_x_mps": round(float(command[0].item()), 6),
        "command_lin_vel_y_mps": round(float(command[1].item()), 6),
        "command_ang_vel_z_rps": round(float(command[2].item()), 6),
        "command_planar_speed_mps": round(command_planar_speed, 6),
    }


def _verdict(
    *,
    fell: bool,
    moved_enough: bool,
    command_motion_expected: bool,
    max_distance_m: float,
    path_length_m: float,
) -> str:
    if fell:
        return "fell"
    if moved_enough:
        return "locomoting"
    if not command_motion_expected:
        return "standing_command"
    if max_distance_m <= 0.05 and path_length_m <= 0.10:
        return "stuck"
    return "insufficient_motion"


def main() -> None:
    debug_handle, log_stage = _make_stage_logger(args.output_json)
    env = None
    faulthandler.enable(file=debug_handle, all_threads=True)
    faulthandler.dump_traceback_later(300, repeat=True, file=debug_handle)
    try:
        log_stage("Validating Isaac playback configuration.")
        validate_supported_configuration(
            args.sensor_profile,
            args.task,
            args.leg_length_scale,
            args.stiffness,
            args.damping,
            fault_class=args.fault_class,
            robot_asset_id=args.robot_asset_id,
            actuator_profile=args.robot_actuator_profile,
        )

        checkpoint_path = _resolve_checkpoint()
        log_stage(f"Resolved checkpoint: {checkpoint_path}")

        log_stage("Preparing Isaac playback configs.")
        env_cfg, agent_cfg = prepare_cfgs(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            seed=args.seed,
            experiment_name=args.experiment_name,
            run_name="play",
            sensor_name=args.sensor_profile,
            terrain_name=args.terrain_name,
            terrain_level=args.terrain_level,
            body_mass_kg=args.body_mass_kg,
            torque_limit_scale=args.torque_limit_scale,
            leg_length_scale=args.leg_length_scale,
            stiffness=args.stiffness,
            damping=args.damping,
            max_iterations=None,
            robot_asset_id=args.robot_asset_id,
            actuator_profile=args.robot_actuator_profile,
        )

        render_mode = "rgb_array" if args.video else None
        log_stage(f"Creating playback environment with render_mode={render_mode}.")
        env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
            log_stage("Converted multi-agent environment to single-agent wrapper.")

        if args.video:
            video_dir = Path(args.video_dir).expanduser().resolve()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_kwargs = {
                "video_folder": str(video_dir),
                "step_trigger": lambda step: step == 0,
                "video_length": int(args.video_length),
                "disable_logger": True,
            }
            print("[INFO] Recording videos during DETB playback.", flush=True)
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            log_stage(f"Video wrapper attached: {video_dir}")

        env, runner = _make_runner(env, agent_cfg, checkpoint_path)
        log_stage(f"Runner initialized: {agent_cfg.class_name}")

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn, _ = _extract_policy_components(runner)
        _flatten_recurrent_policy_memory(policy_nn)
        robot = env.unwrapped.scene["robot"]
        _flatten_actuator_lstm_modules(robot)

        obs = env.get_observations()
        root_pos = robot.data.root_pos_w[0].detach().cpu()
        initial_xy = root_pos[:2].clone()
        previous_xy = root_pos[:2].clone()
        telemetry_rows: list[dict] = []
        telemetry_rows.append(_telemetry_row(0, 0.0, robot, _command_tensor(env.unwrapped), initial_xy, 0.0))

        path_length_m = 0.0
        min_height_m = float(root_pos[2].item())
        max_distance_m = 0.0
        command_speed_samples: list[float] = []
        planar_speed_samples: list[float] = []
        stationary_steps = 0
        fell = False
        terminated = False
        timed_out = False
        rollout_limit_reached = False

        fault_history = deque()

        log_stage(f"Starting playback rollout for {args.rollout_steps} step(s).")
        for step in range(1, int(args.rollout_steps) + 1):
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                actions, fault_history = apply_fault_to_actions(
                    actions,
                    args.fault_class,
                    args.fault_severity,
                    args.latency_steps,
                    fault_history,
                )
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            root_pos = robot.data.root_pos_w[0].detach().cpu()
            current_xy = root_pos[:2].clone()
            path_length_m += float(torch.linalg.vector_norm(current_xy - previous_xy).item())
            previous_xy = current_xy

            command = _command_tensor(env.unwrapped)
            row = _telemetry_row(step, step * float(env.unwrapped.step_dt), robot, command, initial_xy, path_length_m)
            telemetry_rows.append(row)
            command_speed_samples.append(float(row["command_planar_speed_mps"]))
            planar_speed_samples.append(float(row["planar_speed_mps"]))

            if float(row["planar_speed_mps"]) < 0.05:
                stationary_steps += 1
            min_height_m = min(min_height_m, float(row["base_pos_z_m"]))
            max_distance_m = max(max_distance_m, float(row["planar_displacement_m"]))
            if float(row["base_pos_z_m"]) < float(args.diagnostic_fall_height_m):
                fell = True

            if bool(dones[0].item()):
                terminated = bool(env.unwrapped.termination_manager.terminated[0].item())
                timed_out = bool(env.unwrapped.termination_manager.time_outs[0].item())
                rollout_limit_reached = step >= int(args.rollout_steps)
                if timed_out and rollout_limit_reached and not terminated:
                    timed_out = False
                    log_stage(
                        "Playback rollout reached the configured rollout limit "
                        f"at step {step} (terminated={terminated}, rollout_limit_reached={rollout_limit_reached})."
                    )
                else:
                    log_stage(
                        "Playback rollout ended early "
                        f"at step {step} (terminated={terminated}, timed_out={timed_out})."
                    )
                break

            sleep_time = float(env.unwrapped.step_dt) - (time.time() - start_time)
            if args.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        final_row = telemetry_rows[-1]
        if int(final_row["step"]) >= int(args.rollout_steps):
            rollout_limit_reached = True
        moved_enough = (
            float(final_row["planar_displacement_m"]) >= float(args.diagnostic_min_displacement_m)
            or path_length_m >= float(args.diagnostic_min_path_length_m)
        )
        mean_command_speed = sum(command_speed_samples) / max(len(command_speed_samples), 1)
        peak_command_speed = max(command_speed_samples, default=0.0)
        command_motion_expected = (
            mean_command_speed >= float(args.diagnostic_min_command_speed_mps)
            or peak_command_speed >= float(args.diagnostic_min_command_speed_mps)
        )
        verdict = _verdict(
            fell=fell,
            moved_enough=moved_enough,
            command_motion_expected=command_motion_expected,
            max_distance_m=float(final_row["planar_displacement_m"]),
            path_length_m=path_length_m,
        )

        video_files = []
        if args.video and args.video_dir:
            video_root = Path(args.video_dir).expanduser().resolve()
            video_files = [str(path.resolve()) for path in sorted(video_root.rglob("*.mp4"))]

        payload = {
            "task": args.task,
            "task_registry_id": args.task,
            "checkpoint": str(checkpoint_path),
            "robot_asset_id": args.robot_asset_id,
            "robot_actuator_profile": args.robot_actuator_profile,
            "device": args.device,
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "rollout_steps": int(args.rollout_steps),
            "video_enabled": bool(args.video),
            "video_files": video_files,
            "runtime_stack": _runtime_stack(),
            "diagnostics": {
                "verdict": verdict,
                "moved_enough": moved_enough,
                "command_motion_expected": command_motion_expected,
                "fell": fell,
                "terminated": terminated,
                "timed_out": timed_out,
                "rollout_limit_reached": rollout_limit_reached,
                "initial_position_m": [
                    round(float(telemetry_rows[0]["base_pos_x_m"]), 6),
                    round(float(telemetry_rows[0]["base_pos_y_m"]), 6),
                    round(float(telemetry_rows[0]["base_pos_z_m"]), 6),
                ],
                "final_position_m": [
                    round(float(final_row["base_pos_x_m"]), 6),
                    round(float(final_row["base_pos_y_m"]), 6),
                    round(float(final_row["base_pos_z_m"]), 6),
                ],
                "net_displacement_m": round(float(final_row["planar_displacement_m"]), 6),
                "path_length_m": round(path_length_m, 6),
                "max_distance_from_origin_m": round(max_distance_m, 6),
                "mean_planar_speed_mps": round(sum(planar_speed_samples) / max(len(planar_speed_samples), 1), 6),
                "peak_planar_speed_mps": round(max(planar_speed_samples, default=0.0), 6),
                "mean_command_planar_speed_mps": round(mean_command_speed, 6),
                "peak_command_planar_speed_mps": round(peak_command_speed, 6),
                "stationary_fraction": round(stationary_steps / max(len(planar_speed_samples), 1), 6),
                "min_height_m": round(min_height_m, 6),
                "sim_time_s": round(float(final_row["sim_time_s"]), 6),
                "steps_completed": int(final_row["step"]),
                "success_distance_m": round(float(args.success_distance_m), 6),
            },
        }

        telemetry_path = Path(args.telemetry_csv)
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with telemetry_path.open("w", encoding="utf-8", newline="") as handle:
            header = list(telemetry_rows[0].keys())
            handle.write(",".join(header) + "\n")
            for row in telemetry_rows:
                handle.write(",".join(str(row[key]) for key in header) + "\n")

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log_stage(f"Wrote playback telemetry CSV: {telemetry_path}")
        log_stage(f"Wrote playback diagnostics JSON: {output_path}")
    finally:
        if env is not None:
            log_stage("Closing playback environment.")
            env.close()
            log_stage("Playback environment closed.")
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        debug_handle.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
