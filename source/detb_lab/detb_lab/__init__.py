"""DETB-owned Isaac Lab extension package."""

from __future__ import annotations

__version__ = "0.1.3"


def register_all() -> None:
    from detb_lab.registry import register_all_tasks

    register_all_tasks()
