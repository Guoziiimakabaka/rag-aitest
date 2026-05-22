from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "code" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

# ruff: noqa: E402
from task_v2.calibration import run_answer_calibration, run_calibration_threshold_sweep
from task_v2.decision_gate import run_decision_gate


def _write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_eval_payload() -> list[dict]:
    return [
        {
            "question": "Q1",
            "ground_truth": "A1",
            "answer": "A1",
            "context_recall": 0.9,
            "context_precision": 0.9,
            "faithfulness": 0.9,
            "answer_relevance": 0.9,
        },
        {
            "question": "Q2",
            "ground_truth": "A2",
            "answer": "WRONG",
            "context_recall": 0.8,
            "context_precision": 0.8,
            "faithfulness": 0.7,
            "answer_relevance": 0.7,
        },
        {
            "question": "Q3",
            "ground_truth": "A3",
            "answer": "A3",
            "context_recall": 0.6,
            "context_precision": 0.6,
            "faithfulness": 0.6,
            "answer_relevance": 0.6,
        },
    ]


def test_calibration_modes() -> None:
    eval_rel = "tests/tmp/unit_eval.json"
    eval_path = ROOT / eval_rel
    _write_json(eval_path, _build_eval_payload())

    config = {
        "variants": [{"name": "v1", "eval_json": eval_rel}],
        "calibration": {
            "bins": 5,
            "refusal_threshold": 0.75,
            "correctness_mode": "ground_truth_contains",
            "confidence_weights": {
                "context_recall": 0.1,
                "context_precision": 0.2,
                "faithfulness": 0.4,
                "answer_relevance": 0.3,
            },
            "threshold_sweep": {
                "start": 0.5,
                "end": 0.8,
                "step": 0.1,
                "lambda_refusal": 0.2,
                "lambda_leak": 0.3,
            },
        },
    }

    summary_df, detail_df = run_answer_calibration(config=config, root=ROOT)
    assert not summary_df.empty
    assert not detail_df.empty
    assert summary_df.iloc[0]["correctness_mode"] == "ground_truth_contains"

    sweep_df = run_calibration_threshold_sweep(detail_df, config=config)
    assert not sweep_df.empty
    assert set(["threshold", "utility_score"]).issubset(set(sweep_df.columns))


def test_decision_gate_basic() -> None:
    ablation_df = pd.DataFrame(
        [
            {
                "variant": "full",
                "context_recall": 0.9,
                "context_precision": 0.9,
                "faithfulness": 0.9,
                "answer_relevance": 0.9,
            },
            {
                "variant": "v2",
                "context_recall": 0.8,
                "context_precision": 0.8,
                "faithfulness": 0.8,
                "answer_relevance": 0.8,
            },
        ]
    )
    calibration_df = pd.DataFrame(
        [
            {
                "variant": "full",
                "ece": 0.1,
                "accepted_accuracy_proxy": 0.9,
                "error_leakage_rate_after_accept": 0.1,
            },
            {
                "variant": "v2",
                "ece": 0.3,
                "accepted_accuracy_proxy": 0.7,
                "error_leakage_rate_after_accept": 0.4,
            },
        ]
    )
    stats_df = pd.DataFrame(
        [
            {
                "baseline_variant": "full",
                "variant": "v2",
                "metric": "faithfulness",
                "p_value": 0.5,
                "mean_diff": -0.1,
            }
        ]
    )
    config = {
        "decision_gate": {
            "metric_minimums": {
                "context_recall": 0.5,
                "context_precision": 0.5,
                "faithfulness": 0.85,
                "answer_relevance": 0.85,
            },
            "require_significant_improvement_metrics": ["faithfulness"],
            "max_ece": 0.25,
            "min_accepted_accuracy_proxy": 0.85,
            "max_error_leakage_rate_after_accept": 0.2,
        }
    }

    summary_df, detail_df = run_decision_gate(
        ablation_df=ablation_df,
        stats_df=stats_df,
        calibration_df=calibration_df,
        baseline_variant="full",
        config=config,
    )

    assert not summary_df.empty
    assert not detail_df.empty
    row_full = summary_df[summary_df["variant"] == "full"].iloc[0]
    row_v2 = summary_df[summary_df["variant"] == "v2"].iloc[0]
    assert bool(row_full["gate_passed"]) is True
    assert bool(row_v2["gate_passed"]) is False


if __name__ == "__main__":
    test_calibration_modes()
    test_decision_gate_basic()
    print("task_v2_unit_quality_tests_passed=true")
