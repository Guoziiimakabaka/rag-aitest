from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from evaluation import (
    build_error_dashboard,
    compute_gain_by_query_type,
    run_competition_scorecard,
    run_answer_calibration,
    run_calibration_threshold_sweep,
    run_decision_gate,
    run_real_ablation,
    run_significance_tests,
)
from evaluation.statistics import TestConfig
from evaluation.utils import ensure_output_dir, load_yaml_config, require_field, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline: real ablation + significance + query gain + error dashboard"
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Path to evaluation yaml config.",
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
    gain_df: pd.DataFrame,
    error_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    calibration_sweep_df: pd.DataFrame,
    gate_summary_df: pd.DataFrame,
    scorecard_summary_df: pd.DataFrame,
) -> str:
    lines = [
        "# Evaluation Experiment Report",
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
        "| variant | metric | mean_diff | cohen_d | p_value | ci95 | significant |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    if stats_df.empty:
        lines.append("| NA | NA | NA | NA | NA | NA | NA |")
    else:
        for _, row in stats_df.iterrows():
            ci = f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
            lines.append(
                f"| {row['variant']} | {row['metric']} | {row['mean_diff']:.4f} | {row['effect_size_cohen_d']:.4f} | {row['p_value']:.6f} | {ci} | {bool(row['significant_p_lt_0_05'])} |"
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

    lines.extend([
        "",
        "## Answer Calibration & Refusal",
        "",
        "| variant | correctness_mode | ece | brier | refusal_rate | accepted_accuracy_proxy | error_capture_rate_by_refusal |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    if calibration_df.empty:
        lines.append("| NA | NA | NA | NA | NA | NA | NA |")
    else:
        for _, row in calibration_df.iterrows():
            lines.append(
                f"| {row['variant']} | {row['correctness_mode']} | {row['ece']:.4f} | {row['brier_score']:.4f} | {row['refusal_rate']:.4f} | {row['accepted_accuracy_proxy']:.4f} | {row['error_capture_rate_by_refusal']:.4f} |"
            )

    lines.extend([
        "",
        "## Refusal Threshold Sweep (Top Utility)",
        "",
        "| variant | threshold | utility_score | refusal_rate | accepted_accuracy_proxy | error_leakage_rate_after_accept |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    if calibration_sweep_df.empty:
        lines.append("| NA | NA | NA | NA | NA | NA |")
    else:
        for variant in calibration_sweep_df["variant"].dropna().astype(str).unique().tolist():
            view = calibration_sweep_df[calibration_sweep_df["variant"] == variant]
            if view.empty:
                continue
            best = view.sort_values("utility_score", ascending=False).iloc[0]
            lines.append(
                f"| {variant} | {best['threshold']:.2f} | {best['utility_score']:.4f} | {best['refusal_rate']:.4f} | {best['accepted_accuracy_proxy']:.4f} | {best['error_leakage_rate_after_accept']:.4f} |"
            )

    lines.extend([
        "",
        "## Decision Gate",
        "",
        "| variant | gate_passed | rules_passed | rules_total | failed_rules |",
        "|---|---|---:|---:|---|",
    ])
    if gate_summary_df.empty:
        lines.append("| NA | NA | NA | NA | NA |")
    else:
        for _, row in gate_summary_df.iterrows():
            lines.append(
                f"| {row['variant']} | {bool(row['gate_passed'])} | {int(row['rules_passed'])} | {int(row['rules_total'])} | {row['failed_rules']} |"
            )

    lines.extend([
        "",
        "## Competition Scorecard",
        "",
        "| variant | weighted_pass_rate | competition_readiness |",
        "|---|---:|---|",
    ])
    if scorecard_summary_df.empty:
        lines.append("| NA | NA | NA |")
    else:
        for _, row in scorecard_summary_df.iterrows():
            lines.append(
                f"| {row['variant']} | {row['weighted_pass_rate']:.4f} | {row['competition_readiness']} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = (root / args.config).resolve()

    config = load_yaml_config(config_path)
    experiment = require_field(config, "experiment")

    seed = int(args.seed if args.seed is not None else experiment.get("seed", 42))
    set_global_seed(seed)

    output_dir_raw = args.output_dir if args.output_dir else experiment.get("output_dir")
    if not output_dir_raw:
        raise ValueError("Missing experiment.output_dir in config and --output-dir not provided.")
    output_dir = ensure_output_dir((root / str(output_dir_raw)).resolve())

    baseline_variant = str(experiment.get("baseline_variant", "full"))

    ablation_df, variant_frames, summary_payload = run_real_ablation(config=config, root=root)
    stats_df = run_significance_tests(
        variant_frames=variant_frames,
        baseline_variant=baseline_variant,
        seed=seed,
        test_config=TestConfig(bootstrap_iters=1500),
    )

    benchmark_cfg = require_field(config, "benchmark")
    qtype_path = (root / str(require_field(benchmark_cfg, "question_type_path"))).resolve()
    gain_df = compute_gain_by_query_type(
        variant_frames=variant_frames,
        baseline_variant=baseline_variant,
        question_type_path=qtype_path,
    )

    error_df, error_cases = build_error_dashboard(variant_frames=variant_frames, top_k=20)
    calibration_df, calibration_detail_df = run_answer_calibration(config=config, root=root)
    calibration_sweep_df = run_calibration_threshold_sweep(
        calibration_detail_df=calibration_detail_df,
        config=config,
    )
    gate_summary_df, gate_detail_df = run_decision_gate(
        ablation_df=ablation_df,
        stats_df=stats_df,
        calibration_df=calibration_df,
        baseline_variant=baseline_variant,
        config=config,
    )
    scorecard_summary_df, scorecard_detail_df = run_competition_scorecard(
        gate_detail_df=gate_detail_df,
        config=config,
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    summary_json = output_dir / "summary.json"
    ablation_csv = output_dir / "ablation_real.csv"
    stats_csv = output_dir / "stats_significance.csv"
    gain_csv = output_dir / "gain_by_query_type.csv"
    error_csv = output_dir / "error_dashboard.csv"
    cases_json = output_dir / "error_cases_topk.json"
    calibration_csv = output_dir / "calibration_metrics.csv"
    calibration_detail_csv = output_dir / "calibration_detail.csv"
    calibration_sweep_csv = output_dir / "calibration_threshold_sweep.csv"
    gate_summary_csv = output_dir / "decision_gate_summary.csv"
    gate_detail_csv = output_dir / "decision_gate_metric_detail.csv"
    scorecard_summary_csv = output_dir / "competition_scorecard_summary.csv"
    scorecard_detail_csv = output_dir / "competition_scorecard_detail.csv"
    report_md = output_dir / "report.md"

    summary_json.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "seed": seed,
                "config_path": str(config_path),
                "baseline_variant": baseline_variant,
                "variants": summary_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ablation_df.to_csv(ablation_csv, index=False, encoding="utf-8-sig")
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    gain_df.to_csv(gain_csv, index=False, encoding="utf-8-sig")
    error_df.to_csv(error_csv, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(calibration_csv, index=False, encoding="utf-8-sig")
    calibration_detail_df.to_csv(calibration_detail_csv, index=False, encoding="utf-8-sig")
    calibration_sweep_df.to_csv(calibration_sweep_csv, index=False, encoding="utf-8-sig")
    gate_summary_df.to_csv(gate_summary_csv, index=False, encoding="utf-8-sig")
    gate_detail_df.to_csv(gate_detail_csv, index=False, encoding="utf-8-sig")
    scorecard_summary_df.to_csv(scorecard_summary_csv, index=False, encoding="utf-8-sig")
    scorecard_detail_df.to_csv(scorecard_detail_csv, index=False, encoding="utf-8-sig")
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
            gain_df=gain_df,
            error_df=error_df,
            calibration_df=calibration_df,
            calibration_sweep_df=calibration_sweep_df,
            gate_summary_df=gate_summary_df,
            scorecard_summary_df=scorecard_summary_df,
        ),
        encoding="utf-8",
    )

    print("evaluation_pipeline_done=true")
    print(f"summary_json={summary_json}")
    print(f"ablation_csv={ablation_csv}")
    print(f"stats_csv={stats_csv}")
    print(f"gain_csv={gain_csv}")
    print(f"error_csv={error_csv}")
    print(f"calibration_csv={calibration_csv}")
    print(f"calibration_detail_csv={calibration_detail_csv}")
    print(f"calibration_sweep_csv={calibration_sweep_csv}")
    print(f"gate_summary_csv={gate_summary_csv}")
    print(f"gate_detail_csv={gate_detail_csv}")
    print(f"scorecard_summary_csv={scorecard_summary_csv}")
    print(f"scorecard_detail_csv={scorecard_detail_csv}")
    print(f"cases_json={cases_json}")
    print(f"report_md={report_md}")


if __name__ == "__main__":
    main()

