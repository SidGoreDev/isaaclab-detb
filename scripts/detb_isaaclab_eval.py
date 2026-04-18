from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run DETB-managed Isaac Lab evaluation.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--eval_episodes", type=int, required=True)
parser.add_argument("--output_json", type=str, required=True)
parser.add_argument("--robot_asset_id", type=str, required=True)
parser.add_argument("--robot_actuator_profile", type=str, required=True)
parser.add_argument("--sensor_profile", type=str, required=True)
parser.add_argument("--terrain_name", type=str, required=True)
parser.add_argument("--terrain_level", type=int, required=True)
parser.add_argument("--fault_name", type=str, required=True)
parser.add_argument("--fault_class", type=str, required=True)
parser.add_argument("--fault_severity", type=float, required=True)
parser.add_argument("--latency_steps", type=int, required=True)
parser.add_argument("--success_distance_m", type=float, required=True)
parser.add_argument("--body_mass_kg", type=float, required=True)
parser.add_argument("--torque_limit_scale", type=float, required=True)
parser.add_argument("--leg_length_scale", type=float, required=True)
parser.add_argument("--stiffness", type=float, required=True)
parser.add_argument("--damping", type=float, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401

from detb_isaaclab_common import (
    apply_fault_to_actions,
    prepare_cfgs,
    resolve_policy_module,
    validate_supported_configuration,
)


FALL_HEIGHT_THRESHOLD = 0.25


def _debug_log_path(output_json: str) -> Path:
    return Path(output_json).resolve().with_name("isaac_eval_debug.log")


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


def _make_runner(env, agent_cfg):
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(args.checkpoint)
    return runner



def _step_without_auto_reset(env, action):
    env.action_manager.process_action(action.to(env.device))
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    env.episode_length_buf += 1
    env.common_step_counter += 1
    reset_buf = env.termination_manager.compute()
    terminated = env.termination_manager.terminated.clone()
    time_outs = env.termination_manager.time_outs.clone()
    env.reward_manager.compute(dt=env.step_dt)
    env.command_manager.compute(dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[env.num_envs])
    return obs, reset_buf, terminated, time_outs



def _episode_failure_label(root_height: float, terminated: bool, timed_out: bool) -> str:
    if terminated or root_height < FALL_HEIGHT_THRESHOLD:
        return "fall"
    if timed_out:
        return "timeout"
    return "timeout"


def _runtime_stack() -> dict[str, str]:
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(getattr(torch.version, "cuda", "unknown") or "unknown"),
        "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
    }


def _flatten_recurrent_policy_memory(policy_nn) -> None:
    for memory_name in ("memory_a", "memory_c"):
        memory = getattr(policy_nn, memory_name, None)
        rnn = getattr(memory, "rnn", None)
        if rnn is not None and hasattr(rnn, "flatten_parameters"):
            rnn.flatten_parameters()


def _policy_module(runner):
    return resolve_policy_module(runner)


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
        log_stage("Validating Isaac evaluation configuration.")
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

        log_stage("Preparing Isaac evaluation configs.")
        env_cfg, agent_cfg = prepare_cfgs(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            seed=args.seed,
            experiment_name="detb_eval",
            run_name="eval",
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

        log_stage("Creating evaluation environment.")
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
            log_stage("Converted multi-agent environment to single-agent wrapper.")

        log_stage("Loading checkpoint into runner.")
        runner = _make_runner(env, agent_cfg)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        _flatten_recurrent_policy_memory(_policy_module(runner))
        robot = env.unwrapped.scene["robot"]
        _flatten_actuator_lstm_modules(robot)
        episodes: list[dict] = []

        log_stage(f"Starting evaluation loop for {args.eval_episodes} episode(s).")
        for episode_index in range(args.eval_episodes):
            obs_dict, _ = env.reset()
            obs = TensorDict(obs_dict, batch_size=[env.unwrapped.num_envs])
            fault_history = deque()
            torque_accumulator = 0.0
            steps = 0

            while True:
                with torch.inference_mode():
                    actions = policy(obs)
                    actions, fault_history = apply_fault_to_actions(
                        actions,
                        args.fault_class,
                        args.fault_severity,
                        args.latency_steps,
                        fault_history,
                    )

                obs, reset_buf, terminated, time_outs = _step_without_auto_reset(env.unwrapped, actions)
                torque_accumulator += float(torch.mean(torch.square(robot.data.applied_torque[0])).item())
                steps += 1

                if bool(reset_buf[0].item()):
                    distance = float(
                        torch.norm(robot.data.root_pos_w[0, :2] - env.unwrapped.scene.env_origins[0, :2]).item()
                    )
                    elapsed_time_s = float(steps * env.unwrapped.step_dt)
                    energy_proxy = torque_accumulator / max(steps, 1)
                    root_height = float(robot.data.root_pos_w[0, 2].item())
                    timed_out = bool(time_outs[0].item())
                    terminated_flag = bool(terminated[0].item())
                    success = int(
                        timed_out and distance >= args.success_distance_m and root_height >= FALL_HEIGHT_THRESHOLD
                    )
                    failure_label = "none" if success else _episode_failure_label(root_height, terminated_flag, timed_out)
                    episodes.append(
                        {
                            "episode_id": f"{args.seed}-{episode_index}",
                            "terrain_level": int(args.terrain_level),
                            "terrain_name": args.terrain_name,
                            "fault_level": float(args.fault_severity),
                            "fault_name": args.fault_name,
                            "success": success,
                            "distance_m": round(distance, 4),
                            "elapsed_time_s": round(elapsed_time_s, 4),
                            "energy_proxy": round(energy_proxy, 6),
                            "failure_label": failure_label,
                            "seed": int(args.seed),
                            "sensor_profile": args.sensor_profile,
                        }
                    )
                    break

        payload = {
            "task": args.task,
            "task_registry_id": args.task,
            "checkpoint": args.checkpoint,
            "robot_asset_id": args.robot_asset_id,
            "robot_actuator_profile": args.robot_actuator_profile,
            "episodes": episodes,
            "device": args.device,
            "seed": args.seed,
            "runtime_stack": _runtime_stack(),
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log_stage(f"Wrote evaluation result payload: {output_path}")
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
