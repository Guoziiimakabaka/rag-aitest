from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from evaluation.utils import METRIC_KEYS, require_field


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _load_eval_records(path: Path, required_keys: List[str] | None = None) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval json for calibration: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Invalid or empty eval records for calibration: {path}")
    extra_keys = required_keys or []
    for rec in records:
        if "question" not in rec:
            raise KeyError(f"Missing key 'question' in eval record from {path}")
        for key in METRIC_KEYS:
            if key not in rec:
                raise KeyError(f"Missing key '{key}' in eval record from {path}")
        for key in extra_keys:
            if key not in rec:
                raise KeyError(f"Missing key '{key}' in eval record from {path}")
    return records


def _to_float(value, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - keep fast-fail traceback context
        raise TypeError(f"Invalid numeric value for key '{key}': {value}") from exc


def _extract_confidence(record: dict, weights: Dict[str, float]) -> float:
    if "confidence" in record:
        return _clip01(_to_float(record["confidence"], "confidence"))

    score = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        metric_value = _to_float(record[key], key)
        score += float(weight) * metric_value
        weight_sum += float(weight)
    if weight_sum <= 0.0:
        raise ValueError("Calibration confidence weights sum must be positive.")
    return _clip01(score / weight_sum)


def _is_correct_proxy(record: dict, faithfulness_min: float, relevance_min: float) -> int:
    faithfulness = _to_float(record["faithfulness"], "faithfulness")
    relevance = _to_float(record["answer_relevance"], "answer_relevance")
    return int(faithfulness >= faithfulness_min and relevance >= relevance_min)


def _normalize_text(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[\W_]+", "", lowered)


def _is_correct_by_ground_truth(record: dict, mode: str) -> int:
    answer = _normalize_text(str(record["answer"]))
    ground_truth = _normalize_text(str(record["ground_truth"]))
    if not answer or not ground_truth:
        return 0
    if mode == "ground_truth_contains":
        return int(ground_truth in answer)
    if mode == "normalized_exact":
        return int(answer == ground_truth)
    raise ValueError(f"Unsupported correctness mode: {mode}")


def _compute_ece(confidence: np.ndarray, labels: np.ndarray, bins: int) -> float:
    if bins < 2:
        raise ValueError("calibration_bins must be >= 2")
    if confidence.size == 0:
        raise ValueError("confidence array must not be empty")

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = confidence.size

    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        count = int(np.sum(mask))
        if count == 0:
            continue

        conf_mean = float(np.mean(confidence[mask]))
        acc_mean = float(np.mean(labels[mask]))
        ece += (count / total) * abs(acc_mean - conf_mean)
    return float(ece)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def run_answer_calibration(config: dict, root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    calibration_cfg = require_field(config, "calibration")
    variants_cfg = require_field(config, "variants")
    if not isinstance(variants_cfg, list) or not variants_cfg:
        raise ValueError("Config must contain non-empty variants list.")

    calibration_bins = int(calibration_cfg.get("bins", 10))
    refusal_threshold = float(calibration_cfg.get("refusal_threshold", 0.65))
    faithfulness_min = float(calibration_cfg.get("correct_proxy_faithfulness_min", 0.8))
    relevance_min = float(calibration_cfg.get("correct_proxy_relevance_min", 0.8))
    correctness_mode = str(calibration_cfg.get("correctness_mode", "proxy"))
    supported_modes = {"proxy", "ground_truth_contains", "normalized_exact"}
    if correctness_mode not in supported_modes:
        raise ValueError(
            f"Unsupported calibration.correctness_mode: {correctness_mode}. "
            f"Supported: {sorted(supported_modes)}"
        )

    weights_raw = calibration_cfg.get(
        "confidence_weights",
        {
            "context_recall": 0.1,
            "context_precision": 0.2,
            "faithfulness": 0.4,
            "answer_relevance": 0.3,
        },
    )
    if not isinstance(weights_raw, dict):
        raise TypeError("calibration.confidence_weights must be a mapping.")
    missing_weight_keys = [key for key in METRIC_KEYS if key not in weights_raw]
    if missing_weight_keys:
        raise KeyError(f"Missing confidence weights for metrics: {missing_weight_keys}")
    weights = {key: float(weights_raw[key]) for key in METRIC_KEYS}

    summary_rows: List[dict] = []
    detail_rows: List[dict] = []

    for variant_item in variants_cfg:
        variant_name = str(require_field(variant_item, "name"))
        eval_json_rel = str(require_field(variant_item, "eval_json"))
        eval_json_path = (root / eval_json_rel).resolve()
        extra_required = []
        if correctness_mode in {"ground_truth_contains", "normalized_exact"}:
            extra_required = ["answer", "ground_truth"]
        records = _load_eval_records(eval_json_path, required_keys=extra_required)

        confidence_values: List[float] = []
        correct_values: List[int] = []

        for rec in records:
            confidence = _extract_confidence(rec, weights=weights)
            if correctness_mode == "proxy":
                correct_proxy = _is_correct_proxy(
                    rec,
                    faithfulness_min=faithfulness_min,
                    relevance_min=relevance_min,
                )
            else:
                correct_proxy = _is_correct_by_ground_truth(
                    rec,
                    mode=correctness_mode,
                )
            refuse = int(confidence < refusal_threshold)

            confidence_values.append(confidence)
            correct_values.append(correct_proxy)

            detail_rows.append(
                {
                    "variant": variant_name,
                    "question": str(rec["question"]),
                    "confidence_proxy": confidence,
                    "correct_proxy": correct_proxy,
                    "refuse": refuse,
                    "faithfulness": _to_float(rec["faithfulness"], "faithfulness"),
                    "answer_relevance": _to_float(rec["answer_relevance"], "answer_relevance"),
                    "context_recall": _to_float(rec["context_recall"], "context_recall"),
                    "context_precision": _to_float(rec["context_precision"], "context_precision"),
                }
            )

        confidence_arr = np.asarray(confidence_values, dtype=float)
        correct_arr = np.asarray(correct_values, dtype=float)
        refused_mask = confidence_arr < refusal_threshold
        accepted_mask = ~refused_mask
        error_arr = 1.0 - correct_arr

        total = int(confidence_arr.size)
        accepted_count = int(np.sum(accepted_mask))
        refused_count = int(np.sum(refused_mask))
        total_errors = float(np.sum(error_arr))
        accepted_errors = float(np.sum(error_arr[accepted_mask]))
        refused_errors = float(np.sum(error_arr[refused_mask]))

        ece = _compute_ece(confidence=confidence_arr, labels=correct_arr, bins=calibration_bins)
        brier = float(np.mean((confidence_arr - correct_arr) ** 2))
        accepted_accuracy = (
            float(np.mean(correct_arr[accepted_mask])) if accepted_count > 0 else 0.0
        )

        summary_rows.append(
            {
                "variant": variant_name,
                "eval_count": total,
                "ece": ece,
                "brier_score": brier,
                "refusal_threshold": refusal_threshold,
                "correctness_mode": correctness_mode,
                "refusal_rate": _safe_div(refused_count, total),
                "accepted_count": accepted_count,
                "accepted_accuracy_proxy": accepted_accuracy,
                "error_capture_rate_by_refusal": _safe_div(refused_errors, total_errors),
                "error_leakage_rate_after_accept": _safe_div(accepted_errors, max(accepted_count, 1)),
                "correct_proxy_rate": float(np.mean(correct_arr)),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def run_calibration_threshold_sweep(
    calibration_detail_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if calibration_detail_df.empty:
        raise ValueError("calibration_detail_df must not be empty.")

    calibration_cfg = require_field(config, "calibration")
    sweep_cfg = calibration_cfg.get("threshold_sweep", {})
    if not isinstance(sweep_cfg, dict):
        raise TypeError("calibration.threshold_sweep must be a mapping.")

    start = float(sweep_cfg.get("start", 0.4))
    end = float(sweep_cfg.get("end", 0.9))
    step = float(sweep_cfg.get("step", 0.05))
    lambda_refusal = float(sweep_cfg.get("lambda_refusal", 0.2))
    lambda_leak = float(sweep_cfg.get("lambda_leak", 0.3))

    if end < start:
        raise ValueError("threshold_sweep.end must be >= start")
    if step <= 0.0:
        raise ValueError("threshold_sweep.step must be > 0")

    thresholds = np.arange(start, end + 1e-12, step, dtype=float)

    rows: List[dict] = []
    variants = calibration_detail_df["variant"].dropna().astype(str).unique().tolist()
    for variant in variants:
        view = calibration_detail_df[calibration_detail_df["variant"] == variant].copy()
        if view.empty:
            continue

        conf = view["confidence_proxy"].astype(float).to_numpy()
        correct = view["correct_proxy"].astype(float).to_numpy()
        error = 1.0 - correct
        total = int(conf.size)
        total_errors = float(np.sum(error))

        for threshold in thresholds:
            refused = conf < threshold
            accepted = ~refused
            refused_count = int(np.sum(refused))
            accepted_count = int(np.sum(accepted))
            refusal_rate = _safe_div(refused_count, total)

            accepted_accuracy = (
                float(np.mean(correct[accepted])) if accepted_count > 0 else 0.0
            )
            accepted_errors = float(np.sum(error[accepted]))
            refused_errors = float(np.sum(error[refused]))
            error_capture = _safe_div(refused_errors, total_errors)
            error_leakage = _safe_div(accepted_errors, max(accepted_count, 1))

            utility = accepted_accuracy - lambda_refusal * refusal_rate - lambda_leak * error_leakage

            rows.append(
                {
                    "variant": variant,
                    "threshold": float(threshold),
                    "accepted_count": accepted_count,
                    "refused_count": refused_count,
                    "refusal_rate": refusal_rate,
                    "accepted_accuracy_proxy": accepted_accuracy,
                    "error_capture_rate_by_refusal": error_capture,
                    "error_leakage_rate_after_accept": error_leakage,
                    "utility_score": float(utility),
                }
            )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("Threshold sweep produced empty result.")
    return result_df

