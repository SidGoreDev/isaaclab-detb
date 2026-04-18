from types import SimpleNamespace

from scripts.detb_isaaclab_common import apply_common_overrides, resolve_policy_module


def test_apply_common_overrides_sets_obs_groups_and_disables_command_debug_vis():
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=0),
        seed=0,
        sim=SimpleNamespace(device="cpu"),
        commands=SimpleNamespace(base_velocity=SimpleNamespace(debug_vis=True)),
    )
    agent_cfg = SimpleNamespace(
        seed=0,
        device="cpu",
        experiment_name="",
        run_name="",
        max_iterations=0,
        obs_groups={},
    )

    apply_common_overrides(
        env_cfg,
        agent_cfg,
        device="cuda:0",
        num_envs=16,
        seed=11,
        experiment_name="detb_anymal_c_flat",
        run_name="play",
        max_iterations=300,
    )

    assert env_cfg.scene.num_envs == 16
    assert env_cfg.seed == 11
    assert env_cfg.sim.device == "cuda:0"
    assert env_cfg.commands.base_velocity.debug_vis is False
    assert agent_cfg.seed == 11
    assert agent_cfg.device == "cuda:0"
    assert agent_cfg.experiment_name == "detb_anymal_c_flat"
    assert agent_cfg.run_name == "play"
    assert agent_cfg.max_iterations == 300
    assert agent_cfg.obs_groups == {"policy": ["policy"], "critic": ["policy"]}


def test_resolve_policy_module_prefers_policy_and_falls_back_to_actor_critic():
    policy_runner = SimpleNamespace(alg=SimpleNamespace(policy="policy_module"))
    actor_runner = SimpleNamespace(alg=SimpleNamespace(actor_critic="actor_module"))

    assert resolve_policy_module(policy_runner) == "policy_module"
    assert resolve_policy_module(actor_runner) == "actor_module"
