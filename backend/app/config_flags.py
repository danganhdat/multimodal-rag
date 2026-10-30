"""Feature flags: load from YAML, merge overrides, compute config_id."""

from pathlib import Path

import yaml

from .observe import config_id

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PATH = _PROJECT_ROOT / "configs" / "default.yaml"

_defaults: dict | None = None


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_defaults() -> dict:
    global _defaults
    if _defaults is None:
        if _DEFAULT_PATH.exists():
            _defaults = _load_yaml(_DEFAULT_PATH)
        else:
            _defaults = {}
    return _defaults


def load_flags(overrides: dict | None = None) -> dict:
    """Merge defaults with overrides (one level deep). Returns a new dict."""
    base = {k: dict(v) if isinstance(v, dict) else v for k, v in get_defaults().items()}
    if not overrides:
        return base
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def flags_id(flags: dict) -> str:
    return config_id(flags)
