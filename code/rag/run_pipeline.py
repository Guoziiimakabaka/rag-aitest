from __future__ import annotations

import argparse
from pathlib import Path

from export_report import export_report


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
        "--task-v2",
        action="store_true",
        help="Run task-v2 pipeline (real ablation + significance + error analysis).",
    )
    parser.add_argument(
        "--task-v2-config",
        default="configs/task_v2.yaml",
        help="Config path for task-v2 pipeline.",
    )
    parser.add_argument(
        "--task-v2-repeat-setup",
        action="store_true",
        help="Generate repeated-run config and files for task-v2 stability tests.",
    )
    parser.add_argument(
        "--task-v2-repeat-count",
        type=int,
        default=3,
        help="Repeat count used by --task-v2-repeat-setup.",
    )
    parser.add_argument(
        "--task-v2-repeat-output-dir",
        default="code/rag/outputs/repeat_runs/latest",
        help="Output directory for repeated eval_json files.",
    )
    parser.add_argument(
        "--task-v2-repeat-output-config",
        default="configs/task_v2.repeated.generated.yaml",
        help="Generated task-v2 config path for repeated runs.",
    )
    parser.add_argument(
        "--task-v2-repeat-overwrite",
        action="store_true",
        help="Overwrite existing repeated run files.",
    )
    parser.add_argument(
        "--task-v2-adaptive-setup",
        action="store_true",
        help="Generate adaptive retrieval variant artifacts and config.",
    )
    parser.add_argument(
        "--task-v2-adaptive-output-dir",
        default="code/rag/outputs/adaptive/latest",
        help="Output directory for adaptive variant artifacts.",
    )
    parser.add_argument(
        "--task-v2-adaptive-output-config",
        default="configs/task_v2.adaptive.generated.yaml",
        help="Generated config path with adaptive variant.",
    )
    parser.add_argument(
        "--task-v2-adaptive-source-variant",
        default="full",
        help="Source variant name used to synthesize adaptive variant.",
    )
    parser.add_argument(
        "--task-v2-adaptive-variant-name",
        default="adaptive_variant",
        help="Name of generated adaptive variant.",
    )
    parser.add_argument(
        "--task-v2-adaptive-policy-path",
        default="configs/adaptive_retrieval.yaml",
        help="Adaptive retrieval policy yaml path.",
    )
    parser.add_argument(
        "--task-v2-adaptive-repeat-runs",
        type=int,
        default=3,
        help="Repeated adaptive run files to generate.",
    )
    parser.add_argument(
        "--task-v2-adaptive-seed",
        type=int,
        default=42,
        help="Base seed used by adaptive setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pull_models:
        from pull_models import main as pull_models_main

        pull_models_main()

    if args.task_v2_repeat_setup:
        import sys
        from task_v2_repeat_runner import main as task_v2_repeat_main

        sys.argv = [
            "task_v2_repeat_runner.py",
            "--config",
            args.task_v2_config,
            "--run-count",
            str(args.task_v2_repeat_count),
            "--output-dir",
            args.task_v2_repeat_output_dir,
            "--output-config",
            args.task_v2_repeat_output_config,
        ]
        if args.task_v2_repeat_overwrite:
            sys.argv.append("--overwrite")
        task_v2_repeat_main()
        return

    if args.task_v2_adaptive_setup:
        import sys
        from task_v2_adaptive_runner import main as task_v2_adaptive_main

        sys.argv = [
            "task_v2_adaptive_runner.py",
            "--config",
            args.task_v2_config,
            "--output-dir",
            args.task_v2_adaptive_output_dir,
            "--output-config",
            args.task_v2_adaptive_output_config,
            "--source-variant",
            args.task_v2_adaptive_source_variant,
            "--adaptive-variant-name",
            args.task_v2_adaptive_variant_name,
            "--policy-path",
            args.task_v2_adaptive_policy_path,
            "--repeat-runs",
            str(args.task_v2_adaptive_repeat_runs),
            "--seed",
            str(args.task_v2_adaptive_seed),
        ]
        task_v2_adaptive_main()
        return

    if args.task_v2:
        import sys
        from task_v2_pipeline import main as task_v2_main

        sys.argv = [
            "task_v2_pipeline.py",
            "--config",
            args.task_v2_config,
            "--output-dir",
            args.output_dir,
        ]
        task_v2_main()
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
