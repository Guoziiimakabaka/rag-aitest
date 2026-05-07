from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _label_error_case(row: pd.Series) -> str:
    recall = float(row["context_recall"])
    precision = float(row["context_precision"])
    faithfulness = float(row["faithfulness"])

    if recall < 0.5:
        return "no_recall"
    if recall >= 0.6 and precision < 0.6:
        return "bad_rank"
    if recall >= 0.6 and precision >= 0.6 and faithfulness < 0.6:
        return "good_retrieval_bad_generation"
    if precision < 0.5:
        return "over_retrieval_noise"
    return "ok"


def build_error_dashboard(
    variant_frames: Dict[str, pd.DataFrame],
    top_k: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    dashboard_rows: List[dict] = []
    case_payload: Dict[str, List[dict]] = {}

    for variant_name, frame in variant_frames.items():
        df = frame.copy()
        df["error_label"] = df.apply(_label_error_case, axis=1)

        grouped = df.groupby("error_label", as_index=False).agg(
            count=("question", "count"),
            avg_context_recall=("context_recall", "mean"),
            avg_context_precision=("context_precision", "mean"),
            avg_faithfulness=("faithfulness", "mean"),
            avg_answer_relevance=("answer_relevance", "mean"),
        )
        grouped["variant"] = variant_name
        dashboard_rows.extend(grouped.to_dict(orient="records"))

        df["risk_score"] = 1.0 - (df["faithfulness"] + df["answer_relevance"]) / 2.0
        top = df.sort_values(["risk_score", "faithfulness"], ascending=False).head(top_k)
        case_payload[variant_name] = top[
            [
                "question",
                "error_label",
                "context_recall",
                "context_precision",
                "faithfulness",
                "answer_relevance",
                "risk_score",
            ]
        ].to_dict(orient="records")

    return pd.DataFrame(dashboard_rows), case_payload
