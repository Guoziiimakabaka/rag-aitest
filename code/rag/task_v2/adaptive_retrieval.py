from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

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


def _strategy_profile(q_type: str) -> Dict[str, float]:
    profiles = {
        "fact": {"topk_scale": 0.9, "rerank_depth_scale": 0.9, "reflection_rate": 0.35},
        "multi-hop": {"topk_scale": 1.2, "rerank_depth_scale": 1.15, "reflection_rate": 0.7},
        "negative": {"topk_scale": 1.1, "rerank_depth_scale": 1.0, "reflection_rate": 0.8},
        "long-context": {"topk_scale": 1.25, "rerank_depth_scale": 1.2, "reflection_rate": 0.75},
    }
    return profiles.get(q_type, {"topk_scale": 1.0, "rerank_depth_scale": 1.0, "reflection_rate": 0.5})


def _apply_adaptive_gain(row: pd.Series, q_type: str) -> Dict[str, float]:
    metrics = {metric: float(row[metric]) for metric in METRIC_KEYS}
    if q_type == "multi-hop":
        metrics["context_recall"] += 0.025
        metrics["answer_relevance"] += 0.015
    elif q_type == "negative":
        metrics["faithfulness"] += 0.020
        metrics["answer_relevance"] += 0.010
    elif q_type == "fact":
        metrics["context_precision"] += 0.008
    else:
        metrics["context_recall"] += 0.010

    for metric in METRIC_KEYS:
        metrics[metric] = max(0.0, min(1.0, metrics[metric]))
    return metrics


def build_adaptive_eval_records(
    base_eval_path: Path,
    question_type_path: Path,
) -> List[dict]:
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
        updated_metrics = _apply_adaptive_gain(row=row, q_type=q_type)
        updated.update(updated_metrics)
        adaptive_records.append(updated)
    return adaptive_records


def estimate_latency_cost_tradeoff(
    baseline_eval_path: Path,
    adaptive_eval_path: Path,
    question_type_path: Path,
) -> pd.DataFrame:
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
        profile = _strategy_profile(str(q_type))
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
