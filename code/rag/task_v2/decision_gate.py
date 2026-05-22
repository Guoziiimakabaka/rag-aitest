from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from task_v2.utils import require_field


def _to_float(value, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Invalid numeric value for '{field_name}': {value}") from exc


def _find_stat_row(
    stats_df: pd.DataFrame,
    baseline_variant: str,
    variant: str,
    metric: str,
) -> pd.Series | None:
    if stats_df.empty:
        return None
    mask = (
        (stats_df["baseline_variant"] == baseline_variant)
        & (stats_df["variant"] == variant)
        & (stats_df["metric"] == metric)
    )
    view = stats_df[mask]
    if view.empty:
        return None
    return view.iloc[0]


def run_decision_gate(
    ablation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    baseline_variant: str,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gate_cfg = require_field(config, "decision_gate")

    metric_mins = gate_cfg.get("metric_minimums", {})
    if not isinstance(metric_mins, dict):
        raise TypeError("decision_gate.metric_minimums must be a mapping.")

    required_sig_metrics = gate_cfg.get("require_significant_improvement_metrics", [])
    if not isinstance(required_sig_metrics, list):
        raise TypeError(
            "decision_gate.require_significant_improvement_metrics must be a list."
        )
    required_sig_metrics = [str(x) for x in required_sig_metrics]

    max_ece = _to_float(gate_cfg.get("max_ece", 0.2), "decision_gate.max_ece")
    min_accepted_acc = _to_float(
        gate_cfg.get("min_accepted_accuracy_proxy", 0.85),
        "decision_gate.min_accepted_accuracy_proxy",
    )
    max_error_leak = _to_float(
        gate_cfg.get("max_error_leakage_rate_after_accept", 0.2),
        "decision_gate.max_error_leakage_rate_after_accept",
    )

    compare_variants = gate_cfg.get("compare_variants", [])
    if compare_variants:
        target_variants = [str(x) for x in compare_variants]
    else:
        target_variants = [str(x) for x in ablation_df["variant"].dropna().tolist()]

    summary_rows: List[dict] = []
    detail_rows: List[dict] = []

    for variant in target_variants:
        row_view = ablation_df[ablation_df["variant"] == variant]
        if row_view.empty:
            raise ValueError(f"Variant '{variant}' not found in ablation data.")
        ablation_row = row_view.iloc[0]

        cal_view = calibration_df[calibration_df["variant"] == variant]
        if cal_view.empty:
            raise ValueError(f"Variant '{variant}' not found in calibration data.")
        cal_row = cal_view.iloc[0]

        failed_rules: List[str] = []
        passed_count = 0
        total_rules = 0

        for metric, min_value in metric_mins.items():
            total_rules += 1
            got = _to_float(ablation_row[metric], f"ablation.{metric}")
            threshold = _to_float(min_value, f"decision_gate.metric_minimums.{metric}")
            passed = got >= threshold
            if passed:
                passed_count += 1
            else:
                failed_rules.append(f"metric_min:{metric}")
            detail_rows.append(
                {
                    "variant": variant,
                    "rule_type": "metric_minimum",
                    "rule_name": metric,
                    "operator": ">=",
                    "threshold": threshold,
                    "actual_value": got,
                    "passed": passed,
                }
            )

        total_rules += 1
        ece = _to_float(cal_row["ece"], "calibration.ece")
        ece_pass = ece <= max_ece
        if ece_pass:
            passed_count += 1
        else:
            failed_rules.append("calibration:max_ece")
        detail_rows.append(
            {
                "variant": variant,
                "rule_type": "calibration",
                "rule_name": "ece",
                "operator": "<=",
                "threshold": max_ece,
                "actual_value": ece,
                "passed": ece_pass,
            }
        )

        total_rules += 1
        accepted_acc = _to_float(
            cal_row["accepted_accuracy_proxy"],
            "calibration.accepted_accuracy_proxy",
        )
        accepted_acc_pass = accepted_acc >= min_accepted_acc
        if accepted_acc_pass:
            passed_count += 1
        else:
            failed_rules.append("calibration:min_accepted_accuracy_proxy")
        detail_rows.append(
            {
                "variant": variant,
                "rule_type": "calibration",
                "rule_name": "accepted_accuracy_proxy",
                "operator": ">=",
                "threshold": min_accepted_acc,
                "actual_value": accepted_acc,
                "passed": accepted_acc_pass,
            }
        )

        total_rules += 1
        error_leak = _to_float(
            cal_row["error_leakage_rate_after_accept"],
            "calibration.error_leakage_rate_after_accept",
        )
        error_leak_pass = error_leak <= max_error_leak
        if error_leak_pass:
            passed_count += 1
        else:
            failed_rules.append("calibration:max_error_leakage_rate_after_accept")
        detail_rows.append(
            {
                "variant": variant,
                "rule_type": "calibration",
                "rule_name": "error_leakage_rate_after_accept",
                "operator": "<=",
                "threshold": max_error_leak,
                "actual_value": error_leak,
                "passed": error_leak_pass,
            }
        )

        if variant != baseline_variant:
            for metric in required_sig_metrics:
                total_rules += 1
                stat_row = _find_stat_row(
                    stats_df=stats_df,
                    baseline_variant=baseline_variant,
                    variant=variant,
                    metric=metric,
                )
                if stat_row is None:
                    raise ValueError(
                        f"Missing significance row for variant={variant}, metric={metric}"
                    )
                p_value = _to_float(stat_row["p_value"], "stats.p_value")
                mean_diff = _to_float(stat_row["mean_diff"], "stats.mean_diff")
                sig_pass = bool(p_value < 0.05 and mean_diff > 0.0)
                if sig_pass:
                    passed_count += 1
                else:
                    failed_rules.append(f"significant_improve:{metric}")
                detail_rows.append(
                    {
                        "variant": variant,
                        "rule_type": "significance",
                        "rule_name": metric,
                        "operator": "p<0.05 and mean_diff>0",
                        "threshold": 0.05,
                        "actual_value": p_value,
                        "passed": sig_pass,
                    }
                )

        gate_passed = len(failed_rules) == 0
        summary_rows.append(
            {
                "variant": variant,
                "baseline_variant": baseline_variant,
                "rules_total": total_rules,
                "rules_passed": passed_count,
                "rules_failed": total_rules - passed_count,
                "pass_rate": passed_count / total_rules if total_rules > 0 else 0.0,
                "gate_passed": gate_passed,
                "failed_rules": ";".join(failed_rules),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)
