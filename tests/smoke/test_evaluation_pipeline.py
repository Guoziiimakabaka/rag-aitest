from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "code" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from phase4_tools import compute_metrics
from evaluation_pipeline import main as evaluation_main


def main() -> None:
    root = ROOT
    config = root / "configs" / "evaluation.yaml"
    out_dir = root / "code" / "rag" / "reports" / "evaluation" / "smoke"

    sys.argv = [
        "evaluation_pipeline.py",
        "--config",
        str(config),
        "--output-dir",
        str(out_dir),
        "--seed",
        "42",
    ]
    evaluation_main()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    variants = summary["variants"]
    if "full" not in variants:
        raise KeyError("Missing full variant in summary.")

    expected = compute_metrics(
        json.loads(
            (root / "code" / "rag" / "outputs" / "large" / "rag_eval_results.json").read_text(
                encoding="utf-8"
            )
        )
    )

    got = variants["full"]["metrics"]
    for key, value in expected.items():
        if abs(float(got[key]) - float(value)) > 1e-9:
            raise AssertionError(f"Metric mismatch for {key}: {got[key]} vs {value}")

    if not (out_dir / "stats_significance.csv").exists():
        raise FileNotFoundError("stats_significance.csv not found")
    if not (out_dir / "calibration_metrics.csv").exists():
        raise FileNotFoundError("calibration_metrics.csv not found")
    if not (out_dir / "calibration_detail.csv").exists():
        raise FileNotFoundError("calibration_detail.csv not found")
    if not (out_dir / "calibration_threshold_sweep.csv").exists():
        raise FileNotFoundError("calibration_threshold_sweep.csv not found")
    if not (out_dir / "decision_gate_summary.csv").exists():
        raise FileNotFoundError("decision_gate_summary.csv not found")
    if not (out_dir / "decision_gate_metric_detail.csv").exists():
        raise FileNotFoundError("decision_gate_metric_detail.csv not found")
    if not (out_dir / "competition_scorecard_summary.csv").exists():
        raise FileNotFoundError("competition_scorecard_summary.csv not found")
    if not (out_dir / "competition_scorecard_detail.csv").exists():
        raise FileNotFoundError("competition_scorecard_detail.csv not found")

    print("evaluation_smoke_test_passed=true")


if __name__ == "__main__":
    main()

