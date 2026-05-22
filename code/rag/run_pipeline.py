from __future__ import annotations

import argparse
from pathlib import Path

from export_report import export_report
from pull_models import main as pull_models_main
from evaluation_pipeline import main as evaluation_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command pipeline: optional model pull + report export."
    )
    parser.add_argument(
        "--pull-models",
        action="store_true",
        help="Pull embedding/reranker models from configured HF mirror.",
    )
    parser.add_argument(
        "--eval-json",
        default="code/rag/outputs/large/rag_eval_results.json",
        help="Path to evaluation JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="code/rag/reports/latest",
        help="Output directory for exported report files.",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help=(
            "Run evaluation pipeline "
            "(real ablation + significance + gain + calibration + decision gate)."
        ),
    )
    parser.add_argument(
        "--evaluation-config",
        default="configs/evaluation.yaml",
        help="Config path for evaluation pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pull_models:
        pull_models_main()

    if args.evaluation:
        import sys

        sys.argv = [
            "evaluation_pipeline.py",
            "--config",
            args.evaluation_config,
            "--output-dir",
            args.output_dir,
        ]
        evaluation_main()
        return

    eval_json_path = Path(args.eval_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    paths = export_report(eval_json_path=eval_json_path, output_dir=output_dir)

    print("pipeline_done=true")
    print(f"summary_json={paths.summary_json}")
    print(f"ablation_csv={paths.ablation_csv}")
    print(f"error_csv={paths.error_csv}")
    print(f"markdown={paths.markdown}")


if __name__ == "__main__":
    main()

