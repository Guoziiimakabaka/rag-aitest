from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def _cfg_float(cfg: Dict[str, object], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def _cfg_int(cfg: Dict[str, object], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def build_decision_gate_tables(
    ablation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    cost_summary_df: pd.DataFrame,
    baseline_variant: str,
    gate_cfg: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = gate_cfg or {}
    use_adjusted = bool(cfg.get("use_adjusted_significance", True))
    require_positive = bool(cfg.get("require_positive_mean_diff", True))
    min_sig_metrics = _cfg_int(cfg, "min_significant_metrics", 1)
    max_std = _cfg_float(cfg, "max_stability_std", 0.015)
    max_cv = _cfg_float(cfg, "max_stability_cv", 0.05)
    max_avg_latency = _cfg_float(cfg, "max_avg_latency_multiplier", 1.25)
    max_peak_latency = _cfg_float(cfg, "max_latency_multiplier", 1.35)
    min_recall_gain = _cfg_float(cfg, "min_mean_context_recall_gain", 0.0)
    min_relevance_gain = _cfg_float(cfg, "min_mean_answer_relevance_gain", 0.0)

    variants = [str(v) for v in ablation_df["variant"].dropna().tolist() if str(v) != baseline_variant]
    variants = sorted(set(variants))

    metric_rows = []
    summary_rows = []

    for variant in variants:
        variant_stats = stats_df[stats_df["variant"] == variant].copy()
        variant_stab = stability_df[stability_df["variant"] == variant].copy()
        if variant_stats.empty or variant_stab.empty:
            continue

        merged = variant_stats.merge(
            variant_stab[["metric", "runs", "std", "cv"]],
            on="metric",
            how="inner",
        )
        if merged.empty:
            continue

        merged["sig_flag"] = (
            merged["significant_adjusted"]
            if use_adjusted and "significant_adjusted" in merged.columns
            else merged["significant_p_lt_0_05"]
        )
        if require_positive:
            merged["sig_flag"] = merged["sig_flag"] & (merged["mean_diff"] > 0.0)

        merged["stability_pass"] = (merged["std"] <= max_std) & (merged["cv"] <= max_cv)
        merged["metric_pass"] = merged["sig_flag"] & merged["stability_pass"]
        merged["variant"] = variant
        metric_rows.extend(merged.to_dict(orient="records"))

        sig_metric_count = int(merged["metric_pass"].sum())
        stability_alert_count = int((~merged["stability_pass"]).sum())
        passed_metrics = merged[merged["metric_pass"]]["metric"].tolist()

        cost_slice = cost_summary_df[cost_summary_df["variant"] == variant].copy()
        if cost_slice.empty:
            cost_pass = True
            cost_reason = "cost_summary_missing_treated_as_pass"
            avg_latency = None
            peak_latency = None
            recall_gain = None
            relevance_gain = None
        else:
            row = cost_slice.iloc[0]
            avg_latency = float(row["avg_latency_multiplier"])
            peak_latency = float(row["max_latency_multiplier"])
            recall_gain = float(row["mean_context_recall_gain"])
            relevance_gain = float(row["mean_answer_relevance_gain"])
            cost_pass = (
                avg_latency <= max_avg_latency
                and peak_latency <= max_peak_latency
                and recall_gain >= min_recall_gain
                and relevance_gain >= min_relevance_gain
            )
            cost_reason = "pass" if cost_pass else "cost_or_gain_threshold_failed"

        overall_pass = (
            sig_metric_count >= min_sig_metrics
            and stability_alert_count == 0
            and bool(cost_pass)
        )
        summary_rows.append(
            {
                "variant": variant,
                "baseline_variant": baseline_variant,
                "sig_metric_count": sig_metric_count,
                "min_sig_metrics_required": min_sig_metrics,
                "stability_alert_count": stability_alert_count,
                "cost_pass": bool(cost_pass),
                "cost_reason": cost_reason,
                "avg_latency_multiplier": avg_latency,
                "max_latency_multiplier": peak_latency,
                "mean_context_recall_gain": recall_gain,
                "mean_answer_relevance_gain": relevance_gain,
                "passed_metrics": ",".join(str(x) for x in passed_metrics),
                "recommend_deploy": bool(overall_pass),
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, metric_df
