"""Read project configuration without depending on the working directory."""

from pathlib import Path
import os
import copy
import threading
import uuid

import yaml

_checkout = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("SAT_CLAS_HOME", str(
    _checkout if (_checkout / "configs/default.yaml").is_file() else Path.cwd()
))).resolve()
_settings_lock = threading.RLock()


def merge_config(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def settings_path():
    return Path(os.environ.get("SAT_CLAS_SETTINGS", PROJECT_ROOT / "configs/local.yaml"))


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML mapping; use configs/default.yaml when no path is supplied."""
    config_path = Path(path or os.environ.get("SAT_CLAS_CONFIG", PROJECT_ROOT / "configs/default.yaml"))
    if path is None and not config_path.is_file() and "SAT_CLAS_CONFIG" not in os.environ:
        config_path = Path(__file__).resolve().parents[1] / "default_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    if path is None:
        from src.utils.settings import default_sections
        config = merge_config(default_sections(), config)
        with _settings_lock:
            local = settings_path()
            if local.is_file():
                overrides = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
                if not isinstance(overrides, dict):
                    raise ValueError("Local settings must be a YAML mapping.")
                config = merge_config(config, overrides)
    return config


def save_settings(values):
    """Validate the complete editable settings and atomically persist overrides."""
    from src.utils.settings import AppSettings
    with _settings_lock:
        current = load_config()
        merged = merge_config(current, values)
        validated = AppSettings.model_validate({k: merged[k] for k in AppSettings.model_fields}).model_dump()
        target = settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(yaml.safe_dump(validated, sort_keys=False), encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return validated
