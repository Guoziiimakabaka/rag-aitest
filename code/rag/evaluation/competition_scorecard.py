from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _build_rule_key(rule_row: pd.Series) -> str:
    rule_type = str(rule_row["rule_type"])
    rule_name = str(rule_row["rule_name"])
    if rule_type == "metric_minimum":
        return f"metric_min:{rule_name}"
    if rule_type == "significance":
        return f"significant_improve:{rule_name}"
    if rule_type == "calibration":
        if rule_name == "ece":
            return "calibration:max_ece"
        if rule_name == "accepted_accuracy_proxy":
            return "calibration:min_accepted_accuracy_proxy"
        if rule_name == "error_leakage_rate_after_accept":
            return "calibration:max_error_leakage_rate_after_accept"
    return f"{rule_type}:{rule_name}"


def _default_rule_mapping() -> Dict[str, dict]:
    return {
        "metric_min:context_recall": {
            "score_item": "检索覆盖能力",
            "score_weight": 10.0,
            "judge_tip": "确保检索覆盖达标，减少漏召回。",
        },
        "metric_min:context_precision": {
            "score_item": "检索精准能力",
            "score_weight": 10.0,
            "judge_tip": "控制噪声召回，减少证据污染。",
        },
        "metric_min:faithfulness": {
            "score_item": "回答可信与安全",
            "score_weight": 20.0,
            "judge_tip": "重点关注幻觉风险与事实一致性。",
        },
        "metric_min:answer_relevance": {
            "score_item": "业务问答相关性",
            "score_weight": 10.0,
            "judge_tip": "确保回答切题，不答非所问。",
        },
        "calibration:max_ece": {
            "score_item": "置信校准质量",
            "score_weight": 15.0,
            "judge_tip": "置信度应与真实正确率一致。",
        },
        "calibration:min_accepted_accuracy_proxy": {
            "score_item": "可放行回答准确性",
            "score_weight": 20.0,
            "judge_tip": "放行答案应保持高准确率。",
        },
        "calibration:max_error_leakage_rate_after_accept": {
            "score_item": "风险外放控制",
            "score_weight": 10.0,
            "judge_tip": "降低错误答案对外输出概率。",
        },
        "significant_improve:faithfulness": {
            "score_item": "关键指标显著提升",
            "score_weight": 5.0,
            "judge_tip": "核心指标提升需具统计显著性。",
        },
    }


def run_competition_scorecard(
    gate_detail_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if gate_detail_df.empty:
        raise ValueError("gate_detail_df must not be empty for scorecard mapping.")

    mapping_cfg = config.get("competition_scorecard", {})
    if mapping_cfg and not isinstance(mapping_cfg, dict):
        raise TypeError("competition_scorecard must be a mapping if provided.")

    rule_mapping = _default_rule_mapping()
    user_rule_mapping = mapping_cfg.get("rule_mapping", {})
    if user_rule_mapping:
        if not isinstance(user_rule_mapping, dict):
            raise TypeError("competition_scorecard.rule_mapping must be a mapping.")
        for key, value in user_rule_mapping.items():
            if not isinstance(value, dict):
                raise TypeError(f"rule_mapping[{key}] must be a mapping.")
            merged = dict(rule_mapping.get(str(key), {}))
            merged.update(value)
            rule_mapping[str(key)] = merged

    default_weight = float(mapping_cfg.get("default_score_weight", 5.0))
    readiness_threshold = float(mapping_cfg.get("readiness_threshold", 0.85))

    detail_rows: List[dict] = []
    for _, row in gate_detail_df.iterrows():
        rule_key = _build_rule_key(row)
        mapping = rule_mapping.get(rule_key, {})
        score_item = str(mapping.get("score_item", "未映射评分项"))
        score_weight = float(mapping.get("score_weight", default_weight))
        judge_tip = str(mapping.get("judge_tip", "建议补充对应评分说明。"))
        passed = bool(row["passed"])

        detail_rows.append(
            {
                "variant": str(row["variant"]),
                "rule_key": rule_key,
                "score_item": score_item,
                "score_weight": score_weight,
                "passed": passed,
                "status": "PASS" if passed else "FAIL",
                "actual_value": float(row["actual_value"]),
                "threshold": float(row["threshold"]),
                "operator": str(row["operator"]),
                "judge_tip": judge_tip,
            }
        )

    detail_df = pd.DataFrame(detail_rows)

    summary_rows: List[dict] = []
    for variant, group in detail_df.groupby("variant"):
        total_weight = float(group["score_weight"].sum())
        passed_weight = float(group[group["passed"]]["score_weight"].sum())
        weighted_pass_rate = 0.0 if total_weight == 0.0 else passed_weight / total_weight
        readiness = "READY" if weighted_pass_rate >= readiness_threshold else "NOT_READY"
        summary_rows.append(
            {
                "variant": str(variant),
                "score_total_weight": total_weight,
                "score_passed_weight": passed_weight,
                "weighted_pass_rate": weighted_pass_rate,
                "readiness_threshold": readiness_threshold,
                "competition_readiness": readiness,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("weighted_pass_rate", ascending=False)
    return summary_df, detail_df
