"""Command line entry point for DETB."""

from __future__ import annotations

import argparse
from pathlib import Path

from detb.config import default_config_dir, load_config
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DETB command line interface")
    parser.add_argument(
        "command",
        choices=[
            "train",
            "evaluate",
            "bundle-artifacts",
            "sweep",
            "sensor-eval",
            "terrain-eval",
            "failure-eval",
            "generate-requirements",
            "visualize",
            "train-gui",
            "tune",
        ],
    )
    parser.add_argument("--config-name", default="base")
    parser.add_argument("--config-dir", default=str(default_config_dir()))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--source-dir", default=None, help="Existing run directory for bundling or requirements")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config_name, args.config_dir, args.overrides)
    command = args.command
    config_dir = Path(args.config_dir)

    if command == "train":
        result = run_train(cfg)
        print(result.run_dir)
        return 0
    if command == "evaluate":
        result = run_evaluate(cfg)
        print(result.run_dir)
        return 0
    if command == "bundle-artifacts":
        if not args.source_dir:
            raise SystemExit("--source-dir is required for bundle-artifacts")
        summary = bundle_artifacts(args.source_dir)
        print(summary)
        return 0
    if command == "sweep":
        result = run_sweep(cfg, config_dir=config_dir)
        print(result.run_dir)
        return 0
    if command == "sensor-eval":
        result = run_sensor_eval(cfg, config_dir=config_dir)
        print(result.run_dir)
        return 0
    if command == "terrain-eval":
        result = run_terrain_eval(cfg, config_dir=config_dir)
        print(result.run_dir)
        return 0
    if command == "failure-eval":
        result = run_failure_eval(cfg, config_dir=config_dir)
        print(result.run_dir)
        return 0
    if command == "tune":
        result = run_tune(cfg, config_dir=config_dir)
        print(result.run_dir)
        return 0
    if command == "visualize":
        result = run_visualize(cfg)
        print(result.run_dir)
        return 0
    if command == "train-gui":
        result = run_train_gui(cfg)
        print(result.run_dir)
        return 0
    if command == "generate-requirements":
        if not args.source_dir:
            raise SystemExit("--source-dir is required for generate-requirements")
        result = generate_requirements(cfg, args.source_dir)
        print(result.run_dir)
        return 0
    raise SystemExit(f"Unhandled command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
