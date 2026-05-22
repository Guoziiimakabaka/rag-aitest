from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "code" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from phase4_tools import compute_metrics
from task_v2_pipeline import main as task_v2_main
from task_v2_adaptive_runner import build_adaptive_config


def main() -> None:
    root = ROOT
    config = root / "configs" / "task_v2.yaml"
    out_dir = root / "code" / "rag" / "reports" / "task_v2" / "smoke"

    sys.argv = [
        "task_v2_pipeline.py",
        "--config",
        str(config),
        "--output-dir",
        str(out_dir),
        "--seed",
        "42",
    ]
    task_v2_main()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    variants = summary["variants"]
    if "full" not in variants:
        raise KeyError("Missing full variant in summary.")
    if int(variants["full"].get("run_count", 0)) < 1:
        raise AssertionError("run_count should be >= 1 for variant full.")

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
    if not (out_dir / "stability_summary.csv").exists():
        raise FileNotFoundError("stability_summary.csv not found")

    stability_rows = (out_dir / "stability_summary.csv").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    if len(stability_rows) <= 1:
        raise AssertionError("stability_summary.csv should contain data rows.")

    stats_df = pd.read_csv(out_dir / "stats_significance.csv")
    for field in ["p_value_adjusted", "significant_adjusted", "p_adjust_method"]:
        if field not in stats_df.columns:
            raise AssertionError(f"Missing required stats field: {field}")

    adaptive_dir = root / "code" / "rag" / "outputs" / "adaptive" / "smoke"
    adaptive_cfg = root / "configs" / "task_v2.adaptive.smoke.yaml"
    adaptive_result = build_adaptive_config(
        config_path=config,
        output_dir=adaptive_dir,
        output_config_path=adaptive_cfg,
        question_type_path=root / "code" / "rag" / "outputs" / "generated_testset.json",
        policy_path=root / "configs" / "adaptive_retrieval.yaml",
        repeat_runs=3,
        seed=42,
        source_variant_name="full",
        adaptive_variant_name="adaptive_variant",
    )
    if not adaptive_result["tradeoff_csv"].exists():
        raise FileNotFoundError("Adaptive tradeoff csv not generated.")
    if not adaptive_result["cost_summary_csv"].exists():
        raise FileNotFoundError("Adaptive cost summary csv not generated.")

    print("task_v2_smoke_test_passed=true")


if __name__ == "__main__":
    main()
