"""Drift-guard tests for the DETB CLI command registry."""

from __future__ import annotations

from pathlib import Path

from detb import cli


def test_argparse_choices_match_command_tuples():
    parser = cli._parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    assert command_action.choices is not None
    assert set(command_action.choices) == set(cli.V1_SUPPORTED_COMMANDS) | set(cli.EXPERIMENTAL_COMMANDS)


def test_v1_and_experimental_are_disjoint():
    assert set(cli.V1_SUPPORTED_COMMANDS).isdisjoint(cli.EXPERIMENTAL_COMMANDS)


def test_cli_command_order_matches_taxonomy():
    assert set(cli._CLI_COMMAND_ORDER) == set(cli.V1_SUPPORTED_COMMANDS) | set(cli.EXPERIMENTAL_COMMANDS)
    assert len(cli._CLI_COMMAND_ORDER) == len(set(cli._CLI_COMMAND_ORDER))


def test_every_command_has_a_dispatch_branch():
    source = Path(cli.__file__).read_text(encoding="utf-8")
    for name in cli.V1_SUPPORTED_COMMANDS + cli.EXPERIMENTAL_COMMANDS:
        assert f'command == "{name}"' in source, (
            f"CLI command {name!r} appears in V1_SUPPORTED_COMMANDS or "
            "EXPERIMENTAL_COMMANDS but has no dispatch branch in detb/cli.py."
        )
