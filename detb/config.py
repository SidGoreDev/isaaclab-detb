"""Hydra configuration helpers for DETB."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def default_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def load_config(
    config_name: str = "base",
    config_dir: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    resolved_dir = Path(config_dir) if config_dir else default_config_dir()
    with initialize_config_dir(version_base=None, config_dir=str(resolved_dir)):
        return compose(config_name=config_name, overrides=list(overrides or []))


def load_group_entry(
    group: str,
    name: str,
    config_dir: str | Path | None = None,
) -> DictConfig:
    resolved_dir = Path(config_dir) if config_dir else default_config_dir()
    path = resolved_dir / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config group entry: {path}")
    return OmegaConf.load(path)


def merge_cfg(base_cfg: DictConfig, patch: dict) -> DictConfig:
    return OmegaConf.merge(base_cfg, OmegaConf.create(patch))


def config_to_builtin(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
