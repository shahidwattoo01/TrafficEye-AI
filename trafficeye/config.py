"""Configuration loading with deep-merge of user overrides onto the defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access (cfg.model.conf) for readability."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return Config(cfg)


def resolve_path(p: str | Path) -> Path:
    """Resolve paths relative to the project root when they are not absolute."""
    p = Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p
