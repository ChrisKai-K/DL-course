from __future__ import annotations

from pathlib import Path
from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    config: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line in {config_path}: {raw_line.rstrip()}")
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_scalar(value.strip())
    return config


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value
