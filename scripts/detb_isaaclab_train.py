from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run DETB-managed Isaac Lab training.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max_iterations", type=int, required=True)
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--log_root", type=str, required=True)
parser.add_argument("--robot_asset_id", type=str, required=True)
parser.add_argument("--robot_actuator_profile", type=str, required=True)
parser.add_argument("--sensor_profile", type=str, required=True)
parser.add_argument("--terrain_name", type=str, required=True)
parser.add_argument("--terrain_level", type=int, required=True)
parser.add_argument("--body_mass_kg", type=float, required=True)
parser.add_argument("--torque_limit_scale", type=float, required=True)
parser.add_argument("--leg_length_scale", type=float, required=True)
parser.add_argument("--stiffness", type=float, required=True)
parser.add_argument("--damping", type=float, required=True)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401

from detb_isaaclab_common import prepare_cfgs


def _latest_checkpoint(log_dir: Path) -> Path:
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints were written to {log_dir}")
    return checkpoints[-1]


def _debug_log_path(output_json: str) -> Path:
    return Path(output_json).resolve().with_name("isaac_train_debug.log")


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
    import torch

    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(getattr(torch.version, "cuda", "unknown") or "unknown"),
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
    }


def _flatten_actuator_lstm_modules(robot) -> None:
    actuators = getattr(robot, "actuators", {})
    for actuator in getattr(actuators, "values", lambda: [])():
        network = getattr(actuator, "network", None)
        lstm = getattr(network, "lstm", None)
        if lstm is not None and hasattr(lstm, "flatten_parameters"):
            lstm.flatten_parameters()


def main() -> None:
    debug_handle, log_stage = _make_stage_logger(args.output_json)
    env = None
    faulthandler.enable(file=debug_handle, all_threads=True)
    faulthandler.dump_traceback_later(300, repeat=True, file=debug_handle)
    try:
        log_stage("Preparing Isaac Lab configs.")
        env_cfg, agent_cfg = prepare_cfgs(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            seed=args.seed,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            sensor_name=args.sensor_profile,
            terrain_name=args.terrain_name,
            terrain_level=args.terrain_level,
            body_mass_kg=args.body_mass_kg,
            torque_limit_scale=args.torque_limit_scale,
            leg_length_scale=args.leg_length_scale,
            stiffness=args.stiffness,
            damping=args.damping,
            max_iterations=args.max_iterations,
            robot_asset_id=args.robot_asset_id,
            actuator_profile=args.robot_actuator_profile,
        )

        log_root_path = Path(args.log_root).expanduser().resolve() / args.experiment_name
        log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.run_name:
            log_dir_name += f"_{args.run_name}"
        log_dir = log_root_path / log_dir_name
        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        env_cfg.log_dir = str(log_dir)
        log_stage(f"Resolved log directory: {log_dir}")

        log_stage("Writing config snapshots before environment construction.")
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        log_stage("Config snapshots written.")

        log_stage("Creating gym environment.")
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        log_stage("Gym environment created.")
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
            log_stage("Converted multi-agent environment to single-agent wrapper.")
        _flatten_actuator_lstm_modules(env.unwrapped.scene["robot"])

        if args.video:
            video_kwargs = {
                "video_folder": str(log_dir / "videos" / "train"),
                "step_trigger": lambda step: step % args.video_interval == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during DETB training.", flush=True)
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            log_stage("Video wrapper attached.")

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        log_stage("RSL-RL environment wrapper attached.")

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        log_stage(f"Runner initialized: {agent_cfg.class_name}")

        runner.add_git_repo_to_log(__file__)
        log_stage("Registered DETB repo path for RSL-RL git logging.")

        log_stage(f"Starting runner.learn for {agent_cfg.max_iterations} iteration(s).")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        log_stage("runner.learn completed.")

        checkpoint_path = _latest_checkpoint(log_dir)
        event_files = sorted(log_dir.glob("*.tfevents.*"))
        log_stage(f"Latest checkpoint: {checkpoint_path}")
        payload = {
            "task": args.task,
            "task_registry_id": args.task,
            "experiment_name": args.experiment_name,
            "run_name": args.run_name,
            "log_dir": str(log_dir),
            "checkpoint_path": str(checkpoint_path),
            "event_file": str(event_files[-1]) if event_files else "",
            "env_yaml": str(params_dir / "env.yaml"),
            "agent_yaml": str(params_dir / "agent.yaml"),
            "robot_asset_id": args.robot_asset_id,
            "robot_actuator_profile": args.robot_actuator_profile,
            "device": args.device,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "max_iterations": args.max_iterations,
            "runtime_stack": _runtime_stack(),
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log_stage(f"Wrote training result payload: {output_path}")
    finally:
        if env is not None:
            log_stage("Closing environment.")
            env.close()
            log_stage("Environment closed.")
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        debug_handle.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
