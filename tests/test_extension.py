from detb.extension import ensure_detb_lab_on_path


def test_detb_lab_registry_metadata_is_available():
    ensure_detb_lab_on_path()

    from detb_lab.registry import published_pretrained_task_id_for_task, robot_spec_for_id, spec_for_task_id

    spec = spec_for_task_id("DETB-Velocity-Flat-Anymal-C-v0")
    robot = robot_spec_for_id("detb.anymal_c")

    assert spec is not None
    assert spec.rough_train_registry_id == "DETB-Velocity-Rough-Anymal-C-v0"
    assert spec.flat_env_cfg_entry_point.startswith(
        "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg:"
    )
    assert robot is not None
    assert robot.actuator_profile == "actuator_net"
    assert robot.cfg_entry_point == "detb_lab.assets.robots.anymal_c:DETB_ANYMAL_C_CFG"
    assert published_pretrained_task_id_for_task("DETB-Velocity-Flat-Anymal-C-Play-v0") == "Isaac-Velocity-Flat-Anymal-C-v0"


def test_stability_task_registry_metadata_is_available():
    ensure_detb_lab_on_path()

    from detb_lab.registry import spec_for_task_id

    spec = spec_for_task_id("DETB-Velocity-Flat-Anymal-C-Stability-v0")

    assert spec is not None
    assert spec.rough_train_registry_id == "DETB-Velocity-Rough-Anymal-C-Stability-v0"
    assert spec.flat_experiment_name == "detb_anymal_c_stability_flat"


def test_simple_actuator_robot_and_task_metadata_are_available():
    ensure_detb_lab_on_path()

    from detb_lab.registry import published_pretrained_task_id_for_task, robot_spec_for_id, spec_for_task_id

    spec = spec_for_task_id("DETB-Velocity-Flat-Anymal-C-SimpleActuator-v0")
    robot = robot_spec_for_id("detb.anymal_c_simple_actuator")

    assert spec is not None
    assert spec.robot_asset_id == "detb.anymal_c_simple_actuator"
    assert spec.flat_experiment_name == "detb_anymal_c_simple_actuator_flat"
    assert robot is not None
    assert robot.actuator_profile == "dc_motor"
    assert robot.baseline_stiffness == 40.0
    assert robot.baseline_damping == 5.0
    assert (
        published_pretrained_task_id_for_task("DETB-Velocity-Flat-Anymal-C-SimpleActuator-Play-v0")
        == "DETB-Velocity-Flat-Anymal-C-SimpleActuator-v0"
    )
