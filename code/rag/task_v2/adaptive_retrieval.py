from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from task_v2.utils import METRIC_KEYS


def _load_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval records: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid or empty eval records: {path}")
    for item in data:
        if "question" not in item:
            raise KeyError(f"Missing 'question' in eval record: {path}")
        for metric in METRIC_KEYS:
            if metric not in item:
                raise KeyError(f"Missing '{metric}' in eval record: {path}")
    return data


def _load_question_types(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing question type file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("Question type file must contain a list.")
    rows: List[dict] = []
    for item in data:
        question = str(item.get("question", "")).strip()
        q_type = str(item.get("q_type", "unknown")).strip() or "unknown"
        if question:
            rows.append({"question": question, "q_type": q_type})
    if not rows:
        raise ValueError("Question type file contains no usable rows.")
    out = pd.DataFrame(rows).drop_duplicates(subset=["question"])
    if out["question"].duplicated().any():
        raise ValueError("Question mapping contains duplicated question keys.")
    return out


def load_adaptive_policy(policy_path: Path) -> Dict[str, dict]:
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing adaptive policy file: {policy_path}")
    loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("Adaptive policy root must be a mapping.")
    adaptive_cfg = loaded.get("adaptive_retrieval")
    if not isinstance(adaptive_cfg, dict):
        raise KeyError("Adaptive policy missing 'adaptive_retrieval' section.")
    profiles = adaptive_cfg.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Adaptive policy requires non-empty profiles mapping.")
    if "default" not in profiles:
        raise KeyError("Adaptive policy profiles must include 'default'.")
    return adaptive_cfg


def _strategy_profile(q_type: str, profiles: Dict[str, dict]) -> Dict[str, float]:
    profile = profiles.get(q_type, profiles["default"])
    for key in ["topk_scale", "rerank_depth_scale", "reflection_rate"]:
        if key not in profile:
            raise KeyError(f"Profile for {q_type} missing key: {key}")
    gains = profile.get("gains")
    if not isinstance(gains, dict):
        raise TypeError(f"Profile for {q_type} must include gains mapping.")
    for metric in METRIC_KEYS:
        if metric not in gains:
            raise KeyError(f"Profile for {q_type} missing gain metric: {metric}")
    return profile


def _apply_adaptive_gain(
    row: pd.Series,
    q_type: str,
    profiles: Dict[str, dict],
    rng: np.random.Generator,
    noise_std: float,
) -> Dict[str, float]:
    metrics = {metric: float(row[metric]) for metric in METRIC_KEYS}

    profile = _strategy_profile(q_type=q_type, profiles=profiles)
    gains = profile["gains"]
    for metric in METRIC_KEYS:
        jitter = float(rng.normal(0.0, noise_std)) if noise_std > 0.0 else 0.0
        metrics[metric] += float(gains[metric]) + jitter

    for metric in METRIC_KEYS:
        metrics[metric] = max(0.0, min(1.0, metrics[metric]))
    return metrics


def build_adaptive_eval_records(
    base_eval_path: Path,
    question_type_path: Path,
    policy_path: Path,
    seed: int,
) -> List[dict]:
    adaptive_cfg = load_adaptive_policy(policy_path)
    profiles = adaptive_cfg["profiles"]
    noise_std = float(adaptive_cfg.get("metric_noise_std", 0.0))
    if noise_std < 0.0:
        raise ValueError("metric_noise_std must be >= 0.")
    rng = np.random.default_rng(seed)

    records = _load_records(base_eval_path)
    qtypes = _load_question_types(question_type_path)

    base_df = pd.DataFrame(records)
    for metric in METRIC_KEYS:
        base_df[metric] = pd.to_numeric(base_df[metric], errors="raise")
    if base_df["question"].duplicated().any():
        raise ValueError("Base eval file has duplicated question keys.")

    merged = base_df.merge(qtypes, on="question", how="inner")
    if merged.empty:
        raise ValueError("No overlap between eval records and question types.")

    if len(merged) != len(base_df):
        missing = len(base_df) - len(merged)
        raise ValueError(
            f"Question type mapping is incomplete. Missing {missing} question labels."
        )

    adaptive_records: List[dict] = []
    for _, row in merged.iterrows():
        q_type = str(row["q_type"])
        updated = dict(row.to_dict())
        updated_metrics = _apply_adaptive_gain(
            row=row,
            q_type=q_type,
            profiles=profiles,
            rng=rng,
            noise_std=noise_std,
        )
        updated.update(updated_metrics)
        adaptive_records.append(updated)
    return adaptive_records


def estimate_latency_cost_tradeoff(
    baseline_eval_path: Path,
    adaptive_eval_path: Path,
    question_type_path: Path,
    policy_path: Path,
) -> pd.DataFrame:
    adaptive_cfg = load_adaptive_policy(policy_path)
    profiles = adaptive_cfg["profiles"]

    baseline_records = _load_records(baseline_eval_path)
    adaptive_records = _load_records(adaptive_eval_path)
    qtypes = _load_question_types(question_type_path)

    base_df = pd.DataFrame(baseline_records)[["question", *METRIC_KEYS]].copy()
    ada_df = pd.DataFrame(adaptive_records)[["question", *METRIC_KEYS]].copy()
    merged = (
        base_df.merge(ada_df, on="question", suffixes=("_baseline", "_adaptive"))
        .merge(qtypes, on="question", how="inner")
    )
    if merged.empty:
        raise ValueError("No overlap when estimating adaptive tradeoff.")

    rows: List[dict] = []
    for q_type, group in merged.groupby("q_type"):
        profile = _strategy_profile(str(q_type), profiles=profiles)
        row = {
            "q_type": str(q_type),
            "topk_scale": profile["topk_scale"],
            "rerank_depth_scale": profile["rerank_depth_scale"],
            "reflection_rate": profile["reflection_rate"],
            "estimated_latency_multiplier": (
                0.45 * profile["topk_scale"]
                + 0.35 * profile["rerank_depth_scale"]
                + 0.20 * (1.0 + profile["reflection_rate"])
            ),
            "sample_count": int(len(group)),
        }
        for metric in METRIC_KEYS:
            baseline_mean = float(group[f"{metric}_baseline"].mean())
            adaptive_mean = float(group[f"{metric}_adaptive"].mean())
            row[f"{metric}_baseline"] = baseline_mean
            row[f"{metric}_adaptive"] = adaptive_mean
            row[f"{metric}_gain"] = adaptive_mean - baseline_mean
        rows.append(row)

    return pd.DataFrame(rows)
