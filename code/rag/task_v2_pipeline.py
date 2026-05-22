from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from task_v2 import (
    build_error_dashboard,
    compute_gain_by_query_type,
    run_real_ablation,
    run_significance_tests,
)
from task_v2.statistics import TestConfig
from task_v2.utils import ensure_output_dir, load_yaml_config, require_field, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task-v2 pipeline: real ablation + significance + query gain + error dashboard"
    )
    parser.add_argument(
        "--config",
        default="configs/task_v2.yaml",
        help="Path to task-v2 yaml config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output dir from config.",
    )
    return parser.parse_args()


def _render_report(
    generated_at: str,
    baseline_variant: str,
    ablation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    error_df: pd.DataFrame,
) -> str:
    lines = [
        "# Task-v2 Experiment Report",
        "",
        f"- Generated at: {generated_at}",
        f"- Baseline variant: {baseline_variant}",
        "",
        "## Real Ablation Summary",
        "",
    ]

    lines.append(
        "| variant | recall | precision | faithfulness | relevance |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in ablation_df.iterrows():
        lines.append(
            f"| {row['variant']} | {row['context_recall']:.4f} | {row['context_precision']:.4f} | {row['faithfulness']:.4f} | {row['answer_relevance']:.4f} |"
        )

    lines.extend([
        "",
        "## Significance Tests (vs baseline)",
        "",
        "| variant | metric | mean_diff | cohen_d | p_value | p_adj | ci95 | significant | significant_adj |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    if stats_df.empty:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for _, row in stats_df.iterrows():
            ci = f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
            lines.append(
                f"| {row['variant']} | {row['metric']} | {row['mean_diff']:.4f} | {row['effect_size_cohen_d']:.4f} | {row['p_value']:.6f} | {row['p_value_adjusted']:.6f} | {ci} | {bool(row['significant_p_lt_0_05'])} | {bool(row['significant_adjusted'])} |"
            )

    lines.extend([
        "",
        "## Stability Summary",
        "",
        "| variant | metric | runs | mean | std | cv |",
        "|---|---|---:|---:|---:|---:|",
    ])
    if stability_df.empty:
        lines.append("| NA | NA | NA | NA | NA | NA |")
    else:
        for _, row in stability_df.iterrows():
            lines.append(
                f"| {row['variant']} | {row['metric']} | {int(row['runs'])} | {row['mean']:.4f} | {row['std']:.4f} | {row['cv']:.4f} |"
            )

    lines.extend([
        "",
        "## Gain by Query Type",
        "",
        "| variant | q_type | metric | gain | count |",
        "|---|---|---|---:|---:|",
    ])
    if gain_df.empty:
        lines.append("| NA | NA | NA | NA | NA |")
    else:
        for _, row in gain_df.iterrows():
            lines.append(
                f"| {row['variant']} | {row['q_type']} | {row['metric']} | {row['gain']:.4f} | {int(row['count'])} |"
            )

    lines.extend([
        "",
        "## Error Dashboard",
        "",
        "| variant | error_label | count | avg_faithfulness | avg_relevance |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in error_df.iterrows():
        lines.append(
            f"| {row['variant']} | {row['error_label']} | {int(row['count'])} | {row['avg_faithfulness']:.4f} | {row['avg_answer_relevance']:.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = (root / args.config).resolve()

    config = load_yaml_config(config_path)
    experiment = require_field(config, "experiment")
    stats_cfg = config.get("statistics", {})

    seed = int(args.seed if args.seed is not None else experiment.get("seed", 42))
    set_global_seed(seed)

    output_dir_raw = args.output_dir if args.output_dir else experiment.get("output_dir")
    if not output_dir_raw:
        raise ValueError("Missing experiment.output_dir in config and --output-dir not provided.")
    output_dir = ensure_output_dir((root / str(output_dir_raw)).resolve())

    baseline_variant = str(experiment.get("baseline_variant", "full"))
    bootstrap_iters = int(stats_cfg.get("bootstrap_iters", 1500))
    p_adjust_method = str(stats_cfg.get("p_adjust_method", "holm"))
    p_alpha = float(stats_cfg.get("p_alpha", 0.05))

    ablation_df, variant_frames, summary_payload, stability_df = run_real_ablation(
        config=config,
        root=root,
    )
    stats_df = run_significance_tests(
        variant_frames=variant_frames,
        baseline_variant=baseline_variant,
        seed=seed,
        test_config=TestConfig(
            bootstrap_iters=bootstrap_iters,
            alpha=p_alpha,
            correction_method=p_adjust_method,
        ),
    )

    benchmark_cfg = require_field(config, "benchmark")
    qtype_path = (root / str(require_field(benchmark_cfg, "question_type_path"))).resolve()
    gain_df = compute_gain_by_query_type(
        variant_frames=variant_frames,
        baseline_variant=baseline_variant,
        question_type_path=qtype_path,
    )

    error_df, error_cases = build_error_dashboard(variant_frames=variant_frames, top_k=20)

    generated_at = datetime.now(timezone.utc).isoformat()

    summary_json = output_dir / "summary.json"
    ablation_csv = output_dir / "ablation_real.csv"
    stats_csv = output_dir / "stats_significance.csv"
    stability_csv = output_dir / "stability_summary.csv"
    gain_csv = output_dir / "gain_by_query_type.csv"
    error_csv = output_dir / "error_dashboard.csv"
    cases_json = output_dir / "error_cases_topk.json"
    report_md = output_dir / "report.md"

    summary_json.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "seed": seed,
                "config_path": str(config_path),
                "baseline_variant": baseline_variant,
                "statistics": {
                    "bootstrap_iters": bootstrap_iters,
                    "p_adjust_method": p_adjust_method,
                    "p_alpha": p_alpha,
                },
                "variants": summary_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ablation_df.to_csv(ablation_csv, index=False, encoding="utf-8-sig")
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    stability_df.to_csv(stability_csv, index=False, encoding="utf-8-sig")
    gain_df.to_csv(gain_csv, index=False, encoding="utf-8-sig")
    error_df.to_csv(error_csv, index=False, encoding="utf-8-sig")
    cases_json.write_text(
        json.dumps(error_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_md.write_text(
        _render_report(
            generated_at=generated_at,
            baseline_variant=baseline_variant,
            ablation_df=ablation_df,
            stats_df=stats_df,
            stability_df=stability_df,
            gain_df=gain_df,
            error_df=error_df,
        ),
        encoding="utf-8",
    )

    print("task_v2_pipeline_done=true")
    print(f"summary_json={summary_json}")
    print(f"ablation_csv={ablation_csv}")
    print(f"stats_csv={stats_csv}")
    print(f"stability_csv={stability_csv}")
    print(f"gain_csv={gain_csv}")
    print(f"error_csv={error_csv}")
    print(f"cases_json={cases_json}")
    print(f"report_md={report_md}")


if __name__ == "__main__":
    main()
