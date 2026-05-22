from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


METRIC_KEYS = [
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevance",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Config root must be a mapping.")
    return data


def require_field(data: Dict[str, Any], key: str):
    if key not in data:
        raise KeyError(f"Missing required config field: {key}")
    return data[key]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
