"""The service configuration: one YAML file with the models, the QA settings and the service settings.

`${NAME}` or `${NAME:-default}` in a value is replaced by the environment variable NAME.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return _expand_env(payload)


def model_profile(config: dict[str, Any], kind: str, name: str | None = None) -> tuple[str, dict[str, Any]]:
    """The profile `name`, or the default of `kind` ("llm" or "vlm") named at the top of the config."""
    profile = name or config.get(kind)
    if not profile:
        raise ValueError(f"No {kind} model: set '{kind}' at the top of the config to a profile under 'models'")
    models = config.get("models") or {}
    if str(profile) not in models or not isinstance(models[str(profile)], dict):
        raise ValueError(f"Model profile '{profile}' is not defined under 'models'")
    return str(profile), dict(models[str(profile)])


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        resolved = os.environ.get(match.group(1))
        if resolved is None:
            resolved = match.group(2) if match.group(2) is not None else ""
        return resolved

    expanded = _ENV_PATTERN.sub(replace, value)
    return expanded or None if _ENV_PATTERN.fullmatch(value) else expanded
