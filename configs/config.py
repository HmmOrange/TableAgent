from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = _resolve_existing_config_path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    if "include" in payload:
        raise ValueError("Config includes are no longer supported; use one complete config.yaml")
    return _expand_env(payload)


def _resolve_existing_config_path(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"Config file not found: {path}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        env_name, default = match.group(1), match.group(2)
        resolved = os.environ.get(env_name)
        if resolved is None:
            resolved = default if default is not None else ""
        return resolved

    expanded = _ENV_PATTERN.sub(replace, value)
    return expanded or None if _ENV_PATTERN.fullmatch(value) else expanded

