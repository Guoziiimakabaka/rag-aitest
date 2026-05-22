from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluation.utils import METRIC_KEYS


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

    return pd.DataFrame(rows).drop_duplicates(subset=["question"])


def compute_gain_by_query_type(
    variant_frames: Dict[str, pd.DataFrame],
    baseline_variant: str,
    question_type_path: Path,
) -> pd.DataFrame:
    if baseline_variant not in variant_frames:
        raise KeyError(f"Missing baseline variant: {baseline_variant}")

    qtype_df = _load_question_types(question_type_path)
    baseline = variant_frames[baseline_variant].merge(qtype_df, on="question", how="inner")
    if baseline.empty:
        raise ValueError("No overlap between baseline questions and question types.")

    rows: List[dict] = []
    for variant_name, frame in variant_frames.items():
        merged = frame.merge(qtype_df, on="question", how="inner")
        if merged.empty:
            continue

        common = baseline[["question", "q_type", *METRIC_KEYS]].merge(
            merged[["question", *METRIC_KEYS]],
            on="question",
            how="inner",
            suffixes=("_baseline", "_variant"),
        )
        if common.empty:
            continue

        for q_type, group in common.groupby("q_type"):
            for metric in METRIC_KEYS:
                baseline_col = f"{metric}_baseline"
                variant_col = f"{metric}_variant"
                base_mean = float(group[baseline_col].mean())
                var_mean = float(group[variant_col].mean())
                rows.append(
                    {
                        "variant": variant_name,
                        "baseline_variant": baseline_variant,
                        "q_type": q_type,
                        "metric": metric,
                        "count": int(len(group)),
                        "baseline_mean": base_mean,
                        "variant_mean": var_mean,
                        "gain": var_mean - base_mean,
                    }
                )

    return pd.DataFrame(rows)

