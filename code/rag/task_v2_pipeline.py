from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from task_v2 import (
    build_decision_gate_tables,
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
    decision_summary_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    severe_stability = stability_df[(stability_df["cv"] > 0.05) | (stability_df["std"] > 0.015)]

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
        "| variant | run_count | recall | precision | faithfulness | relevance |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in ablation_df.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['run_count'])} | {row['context_recall']:.4f} | {row['context_precision']:.4f} | {row['faithfulness']:.4f} | {row['answer_relevance']:.4f} |"
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
        "## Stability Alerts",
        "",
        "- Rule: alert when `std > 0.015` or `cv > 0.05`.",
        "",
        "| variant | metric | runs | std | cv | alert |",
        "|---|---|---:|---:|---:|---|",
    ])
    if severe_stability.empty:
        lines.append("| NA | NA | NA | NA | NA | no_alert |")
    else:
        for _, row in severe_stability.iterrows():
            lines.append(
                f"| {row['variant']} | {row['metric']} | {int(row['runs'])} | {row['std']:.4f} | {row['cv']:.4f} | alert |"
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

    tradeoff_csv = output_dir / "latency_cost_tradeoff.csv"
    cost_summary_csv = output_dir / "latency_cost_summary.csv"
    if tradeoff_csv.exists():
        tradeoff_df = pd.read_csv(tradeoff_csv)
        lines.extend([
            "",
            "## Latency Cost Tradeoff (Adaptive Retrieval)",
            "",
            "| q_type | latency_multiplier | recall_gain | precision_gain | faithfulness_gain | relevance_gain |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        if tradeoff_df.empty:
            lines.append("| NA | NA | NA | NA | NA | NA |")
        else:
            for _, row in tradeoff_df.iterrows():
                lines.append(
                    f"| {row['q_type']} | {row['estimated_latency_multiplier']:.3f} | {row['context_recall_gain']:.4f} | {row['context_precision_gain']:.4f} | {row['faithfulness_gain']:.4f} | {row['answer_relevance_gain']:.4f} |"
                )
    if cost_summary_csv.exists():
        summary_df = pd.read_csv(cost_summary_csv)
        lines.extend([
            "",
            "## Latency Cost Summary",
            "",
            "| variant | repeat_runs | avg_latency_multiplier | max_latency_multiplier | mean_recall_gain | mean_precision_gain | mean_faithfulness_gain | mean_relevance_gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        if summary_df.empty:
            lines.append("| NA | NA | NA | NA | NA | NA | NA | NA |")
        else:
            for _, row in summary_df.iterrows():
                lines.append(
                    f"| {row['variant']} | {int(row['repeat_runs'])} | {row['avg_latency_multiplier']:.3f} | {row['max_latency_multiplier']:.3f} | {row['mean_context_recall_gain']:.4f} | {row['mean_context_precision_gain']:.4f} | {row['mean_faithfulness_gain']:.4f} | {row['mean_answer_relevance_gain']:.4f} |"
                )

    lines.extend([
        "",
        "## Decision Gate Summary",
        "",
        "| variant | sig_metric_count | stability_alert_count | cost_pass | recommend_deploy |",
        "|---|---:|---:|---|---|",
    ])
    if decision_summary_df.empty:
        lines.append("| NA | NA | NA | NA | NA |")
    else:
        for _, row in decision_summary_df.iterrows():
            lines.append(
                f"| {row['variant']} | {int(row['sig_metric_count'])} | {int(row['stability_alert_count'])} | {bool(row['cost_pass'])} | {bool(row['recommend_deploy'])} |"
            )

    lines.append("")
    return "\n".join(lines)


def _materialize_adaptive_cost_artifacts(
    output_dir: Path,
    summary_payload: dict,
) -> None:
    target_tradeoff = output_dir / "latency_cost_tradeoff.csv"
    target_summary = output_dir / "latency_cost_summary.csv"
    if target_tradeoff.exists() and target_summary.exists():
        return

    adaptive_source_dir_candidates: list[Path] = []
    for _, variant_info in summary_payload.items():
        switches = variant_info.get("switches", {})
        if not bool(switches.get("use_adaptive_retrieval", False)):
            continue
        eval_paths = variant_info.get("eval_json_paths", [])
        if not eval_paths:
            continue
        first_path = Path(str(eval_paths[0]))
        adaptive_source_dir_candidates.append(first_path.parent)
        adaptive_source_dir_candidates.append(first_path.parent.parent)
        break

    if not adaptive_source_dir_candidates:
        return

    resolved_tradeoff = None
    resolved_summary = None
    for candidate in adaptive_source_dir_candidates:
        source_tradeoff = candidate / "latency_cost_tradeoff.csv"
        source_summary = candidate / "latency_cost_summary.csv"
        if source_tradeoff.exists() and source_summary.exists():
            resolved_tradeoff = source_tradeoff
            resolved_summary = source_summary
            break

    if resolved_tradeoff and not target_tradeoff.exists():
        shutil.copy2(resolved_tradeoff, target_tradeoff)
    if resolved_summary and not target_summary.exists():
        shutil.copy2(resolved_summary, target_summary)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = (root / args.config).resolve()

    config = load_yaml_config(config_path)
    experiment = require_field(config, "experiment")
    stats_cfg = config.get("statistics", {})
    gate_cfg = config.get("decision_gate", {})

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
    _materialize_adaptive_cost_artifacts(
        output_dir=output_dir,
        summary_payload=summary_payload,
    )

    cost_summary_path = output_dir / "latency_cost_summary.csv"
    if cost_summary_path.exists():
        cost_summary_df = pd.read_csv(cost_summary_path)
    else:
        cost_summary_df = pd.DataFrame(columns=["variant"])

    decision_summary_df, decision_metric_detail_df = build_decision_gate_tables(
        ablation_df=ablation_df,
        stats_df=stats_df,
        stability_df=stability_df,
        cost_summary_df=cost_summary_df,
        baseline_variant=baseline_variant,
        gate_cfg=gate_cfg,
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    summary_json = output_dir / "summary.json"
    ablation_csv = output_dir / "ablation_real.csv"
    stats_csv = output_dir / "stats_significance.csv"
    stability_csv = output_dir / "stability_summary.csv"
    gain_csv = output_dir / "gain_by_query_type.csv"
    error_csv = output_dir / "error_dashboard.csv"
    cases_json = output_dir / "error_cases_topk.json"
    decision_summary_csv = output_dir / "decision_gate_summary.csv"
    decision_detail_csv = output_dir / "decision_gate_metric_detail.csv"
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
                "decision_gate": gate_cfg,
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
    decision_summary_df.to_csv(decision_summary_csv, index=False, encoding="utf-8-sig")
    decision_metric_detail_df.to_csv(decision_detail_csv, index=False, encoding="utf-8-sig")
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
            decision_summary_df=decision_summary_df,
            output_dir=output_dir,
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
    print(f"decision_summary_csv={decision_summary_csv}")
    print(f"decision_detail_csv={decision_detail_csv}")
    print(f"cases_json={cases_json}")
    print(f"report_md={report_md}")


if __name__ == "__main__":
    main()
