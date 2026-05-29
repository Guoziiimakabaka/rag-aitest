from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


METRIC_LABELS = {
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
}


def _load_json_compat(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))

ERROR_GUIDANCE = {
    "no_recall": "优化方向：增强召回覆盖率，完善 Query Rewrite，并校准 Hybrid 检索策略。",
    "bad_rank": "优化方向：增强排序质量，调整 reranker 深度与阈值策略。",
    "good_retrieval_bad_generation": "优化方向：强化生成约束与反思机制，降低证据外扩写风险。",
    "over_retrieval_noise": "优化方向：降低噪声召回比例，优化 top-k 与过滤策略。",
    "ok": "当前样本表现稳定，可作为参考基线。",
}

MODULE_NOTES = [
    {
        "name": "Query Rewrite / HyDE",
        "input": "用户原始问题",
        "output": "增强检索查询",
        "impact": "提升召回，尤其对 multi-hop 问题更敏感",
        "switch": "use_query_rewrite",
    },
    {
        "name": "Hybrid Retrieval",
        "input": "增强后查询",
        "output": "Dense + BM25 融合候选",
        "impact": "提升证据覆盖率，降低漏召回",
        "switch": "use_hybrid",
    },
    {
        "name": "Cross-Encoder Reranker",
        "input": "候选证据块",
        "output": "重排后的高相关证据",
        "impact": "提升 precision，减少噪声上下文",
        "switch": "use_reranker",
    },
    {
        "name": "Reflection / Judge",
        "input": "初次回答 + 证据",
        "output": "置信度与重试决策",
        "impact": "降低幻觉风险，提高 faithfulness",
        "switch": "use_reflection",
    },
]

PAGE_GUIDE = {
    "Home": {
        "purpose": "总览当前系统能力、交付结果与页面导航。",
        "meaning": "提供统一入口，快速建立项目全局认知。",
    },
    "Overview": {
        "purpose": "查看核心指标与关键结论摘要。",
        "meaning": "以统一指标体系评估整体交付质量。",
    },
    "Rigor": {
        "purpose": "查看真实消融与统计显著性结果。",
        "meaning": "验证方案优化具备统计可靠性与可解释性。",
    },
    "Gain Decomposition": {
        "purpose": "查看不同问题类型下的收益差异。",
        "meaning": "识别业务场景收益结构，支持策略配置决策。",
    },
    "Error X-Ray": {
        "purpose": "查看错误分型与高风险案例。",
        "meaning": "将异常样本转化为可执行的优化路径。",
    },
    "Method Graph": {
        "purpose": "查看方法链路和模块开关。",
        "meaning": "明确模块职责、依赖关系与成本收益权衡。",
    },
    "Repro & CI": {
        "purpose": "查看复现命令、工件和 CI 信息。",
        "meaning": "体现可复现性、可审计性与持续交付能力。",
    },
    "Calibration": {
        "purpose": "查看置信度校准质量与拒答策略效果。",
        "meaning": "量化模型自知能力，降低错误答案外放风险。",
    },
    "Decision Gate": {
        "purpose": "查看是否达到发布门槛与失败规则。",
        "meaning": "把多维指标转成明确的通过/不通过结论。",
    },
    "Competition Scorecard": {
        "purpose": "查看规则到评分项映射与交付就绪度。",
        "meaning": "把工程指标翻译成可操作的项目评分语言。",
    },
}


def _default_report_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "code" / "rag" / "reports" / "evaluation"


@st.cache_data(show_spinner=False)
def _load_run_data(run_dir: str) -> Dict[str, object]:
    path = Path(run_dir)
    required = {
        "summary_json": path / "summary.json",
        "ablation_csv": path / "ablation_real.csv",
        "stats_csv": path / "stats_significance.csv",
        "gain_csv": path / "gain_by_query_type.csv",
        "error_csv": path / "error_dashboard.csv",
        "cases_json": path / "error_cases_topk.json",
    }
    optional = {
        "calibration_csv": path / "calibration_metrics.csv",
        "calibration_detail_csv": path / "calibration_detail.csv",
        "calibration_sweep_csv": path / "calibration_threshold_sweep.csv",
        "gate_summary_csv": path / "decision_gate_summary.csv",
        "gate_detail_csv": path / "decision_gate_metric_detail.csv",
        "scorecard_summary_csv": path / "competition_scorecard_summary.csv",
        "scorecard_detail_csv": path / "competition_scorecard_detail.csv",
    }

    missing = [name for name, file_path in required.items() if not file_path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Run 目录缺少必要文件: {missing}; 当前目录: {path}"
        )

    summary = _load_json_compat(required["summary_json"])
    ablation_df = pd.read_csv(required["ablation_csv"])
    stats_df = pd.read_csv(required["stats_csv"])
    gain_df = pd.read_csv(required["gain_csv"])
    error_df = pd.read_csv(required["error_csv"])
    cases = _load_json_compat(required["cases_json"])
    calibration_df = (
        pd.read_csv(optional["calibration_csv"])
        if optional["calibration_csv"].exists()
        else pd.DataFrame()
    )
    calibration_detail_df = (
        pd.read_csv(optional["calibration_detail_csv"])
        if optional["calibration_detail_csv"].exists()
        else pd.DataFrame()
    )
    calibration_sweep_df = (
        pd.read_csv(optional["calibration_sweep_csv"])
        if optional["calibration_sweep_csv"].exists()
        else pd.DataFrame()
    )
    gate_summary_df = (
        pd.read_csv(optional["gate_summary_csv"])
        if optional["gate_summary_csv"].exists()
        else pd.DataFrame()
    )
    gate_detail_df = (
        pd.read_csv(optional["gate_detail_csv"])
        if optional["gate_detail_csv"].exists()
        else pd.DataFrame()
    )
    scorecard_summary_df = (
        pd.read_csv(optional["scorecard_summary_csv"])
        if optional["scorecard_summary_csv"].exists()
        else pd.DataFrame()
    )
    scorecard_detail_df = (
        pd.read_csv(optional["scorecard_detail_csv"])
        if optional["scorecard_detail_csv"].exists()
        else pd.DataFrame()
    )

    return {
        "summary": summary,
        "ablation_df": ablation_df,
        "stats_df": stats_df,
        "gain_df": gain_df,
        "error_df": error_df,
        "calibration_df": calibration_df,
        "calibration_detail_df": calibration_detail_df,
        "calibration_sweep_df": calibration_sweep_df,
        "gate_summary_df": gate_summary_df,
        "gate_detail_df": gate_detail_df,
        "scorecard_summary_df": scorecard_summary_df,
        "scorecard_detail_df": scorecard_detail_df,
        "cases": cases,
        "required": required,
        "optional": optional,
    }


def _safe_float(value: float | int) -> float:
    return float(value) if value is not None else 0.0


def _metric_delta_text(current: float, baseline: float) -> str:
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    return f"{current:.4f} ({sign}{delta:.4f} vs baseline)"


def _extract_top_findings(
    ablation_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    baseline_variant: str,
    compare_variant: str,
) -> List[str]:
    findings: List[str] = []

    baseline_row = ablation_df[ablation_df["variant"] == baseline_variant]
    compare_row = ablation_df[ablation_df["variant"] == compare_variant]
    if not baseline_row.empty and not compare_row.empty:
        deltas = {}
        for metric in METRIC_LABELS:
            deltas[metric] = _safe_float(compare_row.iloc[0][metric]) - _safe_float(
                baseline_row.iloc[0][metric]
            )
        top_metric = max(deltas, key=lambda x: abs(deltas[x]))
        findings.append(
            f"{compare_variant} 相比 {baseline_variant} 变化最大的指标是 {METRIC_LABELS[top_metric]}，差值 {deltas[top_metric]:+.4f}。"
        )

    gain_slice = gain_df[
        (gain_df["variant"] == compare_variant) & (gain_df["metric"] == "context_recall")
    ]
    if not gain_slice.empty:
        top_q_type_row = gain_slice.sort_values("gain", ascending=False).iloc[0]
        findings.append(
            f"在 query type 维度上，{compare_variant} 对 {top_q_type_row['q_type']} 的 Recall 增益最高（{top_q_type_row['gain']:+.4f}）。"
        )

    penalty_row = gain_df[
        (gain_df["variant"] == compare_variant)
    ]
    if not penalty_row.empty:
        worst = penalty_row.sort_values("gain", ascending=True).iloc[0]
        findings.append(
            f"当前最大短板出现在 {worst['q_type']} / {METRIC_LABELS.get(worst['metric'], worst['metric'])}，增益为 {worst['gain']:+.4f}。"
        )

    return findings[:5]


def _render_page_guide(page_name: str) -> None:
    guide = PAGE_GUIDE.get(page_name)
    if not guide:
        return
    st.caption(f"功能说明：{guide['purpose']}  |  工程价值：{guide['meaning']}")


def _render_home(
    summary: dict,
    ablation_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    baseline_variant: str,
    compare_variant: str,
) -> None:
    st.title("RAG-Eye 平台主页")
    _render_page_guide("Home")

    generated_at = summary.get("generated_at", "N/A")
    seed = summary.get("seed", "N/A")
    st.info(f"run_time={generated_at} | seed={seed}")

    baseline_row = ablation_df[ablation_df["variant"] == baseline_variant]
    compare_row = ablation_df[ablation_df["variant"] == compare_variant]
    if baseline_row.empty or compare_row.empty:
        st.error("baseline 或 compare variant 不存在于 ablation 数据中。")
        return

    baseline_row = baseline_row.iloc[0]
    compare_row = compare_row.iloc[0]

    st.subheader("平台价值定位")
    st.write(
        "面向行业知识服务场景，提供可验证、可解释、可复现的 RAG 评估与优化能力。"
    )

    st.subheader("核心指标总览")
    cols = st.columns(4)
    for idx, metric in enumerate(METRIC_LABELS.keys()):
        with cols[idx]:
            st.metric(
                METRIC_LABELS[metric],
                f"{_safe_float(compare_row[metric]):.4f}",
                f"{_safe_float(compare_row[metric]) - _safe_float(baseline_row[metric]):+.4f}",
            )

    st.subheader("功能导航")
    st.write("1. 可靠性验证：进入 `Rigor`。")
    st.write("2. 场景收益分析：进入 `Gain Decomposition`。")
    st.write("3. 异常定位与改进：进入 `Error X-Ray`。")
    st.write("4. 技术架构与交付能力：进入 `Method Graph` 与 `Repro & CI`。")

    findings = _extract_top_findings(
        ablation_df=ablation_df,
        gain_df=gain_df,
        baseline_variant=baseline_variant,
        compare_variant=compare_variant,
    )
    if findings:
        st.subheader("关键结论摘要")
        for item in findings:
            st.write(f"- {item}")


def _render_overview(
    summary: dict,
    ablation_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    baseline_variant: str,
    compare_variant: str,
) -> None:
    st.title("RAG-Eye 综合看板")
    _render_page_guide("Overview")
    st.caption("面向实际工程交付的可证据化 RAG 能力展示")

    generated_at = summary.get("generated_at", "N/A")
    seed = summary.get("seed", "N/A")
    st.info(f"run_time={generated_at} | seed={seed} | baseline={baseline_variant} | compare={compare_variant}")

    baseline_row = ablation_df[ablation_df["variant"] == baseline_variant]
    compare_row = ablation_df[ablation_df["variant"] == compare_variant]
    if baseline_row.empty or compare_row.empty:
        st.error("baseline 或 compare variant 不存在于 ablation 数据中。")
        return

    baseline_row = baseline_row.iloc[0]
    compare_row = compare_row.iloc[0]

    cols = st.columns(4)
    metrics = list(METRIC_LABELS.keys())
    for idx, metric in enumerate(metrics):
        with cols[idx]:
            st.metric(
                METRIC_LABELS[metric],
                f"{_safe_float(compare_row[metric]):.4f}",
                f"{_safe_float(compare_row[metric]) - _safe_float(baseline_row[metric]):+.4f}",
            )

    st.subheader("核心发现")
    findings = _extract_top_findings(
        ablation_df=ablation_df,
        gain_df=gain_df,
        baseline_variant=baseline_variant,
        compare_variant=compare_variant,
    )
    if not findings:
        st.warning("暂未生成关键结论，请检查输入数据完整性。")
    else:
        for item in findings:
            st.write(f"- {item}")

    st.subheader("关联分析入口")
    st.write("- 结果可信性验证：查看「实验严谨性（Rigor）」页。")
    st.write("- 收益来源分析：查看「算法增益拆解（Gain Decomposition）」页。")
    st.write("- 异常样本分析：查看「错误分析与风险案例（Error X-Ray）」页。")


def _render_rigor(
    ablation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    baseline_variant: str,
    compare_variant: str,
    selected_metric: str,
) -> None:
    st.header("实验严谨性（Rigor）")
    _render_page_guide("Rigor")

    st.subheader("真实消融矩阵")
    display_cols = [
        "variant",
        "use_hybrid",
        "use_reranker",
        "use_reflection",
        "use_query_rewrite",
        *list(METRIC_LABELS.keys()),
    ]
    st.dataframe(ablation_df[display_cols], use_container_width=True)

    st.subheader("显著性检验")
    if stats_df.empty:
        st.warning("当前无显著性检验数据。")
        return

    focus = stats_df[
        (stats_df["baseline_variant"] == baseline_variant)
        & (stats_df["variant"] == compare_variant)
        & (stats_df["metric"] == selected_metric)
    ]
    if focus.empty:
        st.info("当前筛选条件没有对应显著性条目。")
    else:
        row = focus.iloc[0]
        sig_text = "显著" if bool(row["significant_p_lt_0_05"]) else "不显著"
        st.success(
            f"{METRIC_LABELS[selected_metric]}: p={row['p_value']:.6f}, d={row['effect_size_cohen_d']:.4f}, 95%CI=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] -> {sig_text}"
        )

    st.dataframe(stats_df.sort_values(["metric", "p_value"]), use_container_width=True)

    ci_df = stats_df[
        (stats_df["baseline_variant"] == baseline_variant)
        & (stats_df["variant"] == compare_variant)
    ].copy()
    if not ci_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[METRIC_LABELS.get(x, x) for x in ci_df["metric"]],
                y=ci_df["mean_diff"],
                mode="markers+lines",
                name="mean_diff",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=ci_df["ci95_high"] - ci_df["mean_diff"],
                    arrayminus=ci_df["mean_diff"] - ci_df["ci95_low"],
                ),
            )
        )
        fig.update_layout(
            title=f"{compare_variant} vs {baseline_variant} 的均值差与95%CI",
            yaxis_title="mean_diff",
            xaxis_title="metric",
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_gain(gain_df: pd.DataFrame, compare_variant: str, selected_metric: str) -> None:
    st.header("算法增益拆解（Gain Decomposition）")
    _render_page_guide("Gain Decomposition")

    data = gain_df[gain_df["variant"] == compare_variant].copy()
    if data.empty:
        st.warning("当前 compare variant 没有增益数据。")
        return

    st.subheader("增益热力图")
    pivot = data.pivot_table(
        index="q_type",
        columns="metric",
        values="gain",
        aggfunc="mean",
    ).fillna(0.0)

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdBu",
        origin="lower",
        text_auto=True,
        aspect="auto",
        title=f"{compare_variant} 的 query type x metric 增益",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("按问题类型的指标增益")
    metric_slice = data[data["metric"] == selected_metric].copy()
    fig_bar = px.bar(
        metric_slice,
        x="q_type",
        y="gain",
        color="q_type",
        title=f"{METRIC_LABELS[selected_metric]} 在不同 q_type 的增益",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    top = data.sort_values("gain", ascending=False).head(3)
    bottom = data.sort_values("gain", ascending=True).head(3)

    st.write("**增益最佳 Top3：**")
    for _, row in top.iterrows():
        st.write(
            f"- {row['q_type']} / {METRIC_LABELS.get(row['metric'], row['metric'])}: {row['gain']:+.4f}"
        )

    st.write("**待优化 Bottom3：**")
    for _, row in bottom.iterrows():
        st.write(
            f"- {row['q_type']} / {METRIC_LABELS.get(row['metric'], row['metric'])}: {row['gain']:+.4f}"
        )


def _render_error_xray(
    error_df: pd.DataFrame,
    cases: dict,
    compare_variant: str,
    q_type_filter: str,
    risk_threshold: float,
) -> None:
    st.header("错误分析与风险案例（Error X-Ray）")
    _render_page_guide("Error X-Ray")

    variant_error = error_df[error_df["variant"] == compare_variant].copy()
    if variant_error.empty:
        st.warning("当前 variant 没有错误分型统计。")
        return

    st.subheader("错误分型分布")
    fig = px.bar(
        variant_error,
        x="error_label",
        y="count",
        color="error_label",
        title=f"{compare_variant} 的错误分型分布",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(variant_error.sort_values("count", ascending=False), use_container_width=True)

    st.subheader("高风险样例")
    case_rows = pd.DataFrame(cases.get(compare_variant, []))
    if case_rows.empty:
        st.info("当前 variant 无高风险样例。")
        return

    filtered = case_rows[case_rows["risk_score"] >= risk_threshold].copy()
    if q_type_filter != "all" and "q_type" in filtered.columns:
        filtered = filtered[filtered["q_type"] == q_type_filter]

    st.caption(f"筛选后样例数: {len(filtered)}")
    if filtered.empty:
        st.warning("筛选条件下无样例。")
        return

    for idx, row in filtered.head(20).iterrows():
        title = (
            f"Case {idx+1} | label={row['error_label']} | "
            f"risk={row['risk_score']:.3f} | faith={row['faithfulness']:.3f}"
        )
        with st.expander(title):
            st.write(f"question: {row['question']}")
            st.write(
                f"metrics: recall={row['context_recall']:.3f}, precision={row['context_precision']:.3f}, "
                f"faithfulness={row['faithfulness']:.3f}, relevance={row['answer_relevance']:.3f}"
            )
            guidance = ERROR_GUIDANCE.get(row["error_label"], "暂无策略说明")
            st.info(f"优化策略: {guidance}")


def _render_calibration(
    calibration_df: pd.DataFrame,
    calibration_detail_df: pd.DataFrame,
    calibration_sweep_df: pd.DataFrame,
    compare_variant: str,
) -> None:
    st.header("答案校准与拒答（Calibration）")
    _render_page_guide("Calibration")

    if calibration_df.empty:
        st.warning("当前 run 未产出校准结果文件。")
        return

    st.subheader("校准指标总览")
    st.dataframe(calibration_df, use_container_width=True)

    focus = calibration_df[calibration_df["variant"] == compare_variant]
    if focus.empty:
        st.info("当前 compare variant 无校准指标，显示全部 variant。")
    else:
        focus_row = focus.iloc[0]
        cols = st.columns(4)
        cols[0].metric("ECE", f"{float(focus_row['ece']):.4f}")
        cols[1].metric("Brier", f"{float(focus_row['brier_score']):.4f}")
        cols[2].metric("Refusal Rate", f"{float(focus_row['refusal_rate']):.4f}")
        cols[3].metric(
            "Accepted Acc (proxy)",
            f"{float(focus_row['accepted_accuracy_proxy']):.4f}",
        )

    if calibration_detail_df.empty:
        st.info("当前 run 未产出样本级校准明细。")
        return

    detail = calibration_detail_df[
        calibration_detail_df["variant"] == compare_variant
    ].copy()
    if detail.empty:
        st.info("当前 compare variant 无样本级校准明细。")
        return

    st.subheader("置信度分布与正确率代理")
    fig_hist = px.histogram(
        detail,
        x="confidence_proxy",
        nbins=20,
        color="correct_proxy",
        barmode="overlay",
        title=f"{compare_variant} confidence_proxy 分布",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    bin_edges = np.linspace(0.0, 1.0, 11)
    detail["bin"] = pd.cut(
        detail["confidence_proxy"],
        bins=bin_edges,
        include_lowest=True,
        right=True,
    )
    by_bin = detail.groupby("bin", as_index=False).agg(
        confidence_mean=("confidence_proxy", "mean"),
        accuracy_proxy=("correct_proxy", "mean"),
        count=("question", "count"),
    )
    by_bin["bin_label"] = by_bin["bin"].astype(str)

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=by_bin["bin_label"],
            y=by_bin["confidence_mean"],
            mode="lines+markers",
            name="mean_confidence",
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=by_bin["bin_label"],
            y=by_bin["accuracy_proxy"],
            mode="lines+markers",
            name="accuracy_proxy",
        )
    )
    fig_line.update_layout(
        title=f"{compare_variant} 分箱校准曲线（代理）",
        xaxis_title="confidence bin",
        yaxis_title="score",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    if calibration_sweep_df.empty:
        st.info("当前 run 未产出阈值扫描结果。")
        return

    st.subheader("拒答阈值扫描（风险-覆盖权衡）")
    sweep = calibration_sweep_df[
        calibration_sweep_df["variant"] == compare_variant
    ].copy()
    if sweep.empty:
        st.info("当前 compare variant 无阈值扫描结果。")
        return

    fig_util = px.line(
        sweep,
        x="threshold",
        y="utility_score",
        title=f"{compare_variant} utility_score vs threshold",
        markers=True,
    )
    st.plotly_chart(fig_util, use_container_width=True)

    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["accepted_accuracy_proxy"],
            mode="lines+markers",
            name="accepted_accuracy_proxy",
        )
    )
    fig_tradeoff.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["error_leakage_rate_after_accept"],
            mode="lines+markers",
            name="error_leakage_rate_after_accept",
        )
    )
    fig_tradeoff.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["refusal_rate"],
            mode="lines+markers",
            name="refusal_rate",
        )
    )
    fig_tradeoff.update_layout(
        title=f"{compare_variant} 阈值权衡曲线",
        xaxis_title="threshold",
        yaxis_title="score",
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)


def _render_method_graph(
    ablation_df: pd.DataFrame,
    compare_variant: str,
) -> None:
    st.header("方法链路与开关（Method Graph）")
    _render_page_guide("Method Graph")

    st.subheader("方法流程")
    st.markdown(
        """
`Query` -> `Rewrite/HyDE` -> `Hybrid Retrieval` -> `Reranker` -> `Generation` -> `Reflection/Judge`
        """
    )

    row = ablation_df[ablation_df["variant"] == compare_variant]
    if row.empty:
        st.warning("当前 variant 未找到开关信息。")
        return
    row = row.iloc[0]

    cols = st.columns(2)
    for idx, module in enumerate(MODULE_NOTES):
        target_col = cols[idx % 2]
        with target_col:
            enabled = bool(row.get(module["switch"], False))
            st.markdown(f"### {module['name']}")
            st.write(f"- 输入: {module['input']}")
            st.write(f"- 输出: {module['output']}")
            st.write(f"- 主要影响: {module['impact']}")
            st.write(f"- 当前开关: {'ON' if enabled else 'OFF'}")

    st.subheader("模块 trade-off（当前静态估计）")
    tradeoff = pd.DataFrame(
        [
            {"module": "rewrite/hyde", "latency_cost": "中", "quality_gain": "中高"},
            {"module": "hybrid", "latency_cost": "中", "quality_gain": "高"},
            {"module": "reranker", "latency_cost": "中高", "quality_gain": "中高"},
            {"module": "reflection", "latency_cost": "高", "quality_gain": "高（可信度）"},
        ]
    )
    st.dataframe(tradeoff, use_container_width=True)


def _render_decision_gate(
    gate_summary_df: pd.DataFrame,
    gate_detail_df: pd.DataFrame,
    compare_variant: str,
) -> None:
    st.header("发布门控判定（Decision Gate）")
    _render_page_guide("Decision Gate")

    if gate_summary_df.empty:
        st.warning("当前 run 未产出 decision gate 结果。")
        return

    st.subheader("门控总览")
    st.dataframe(gate_summary_df, use_container_width=True)

    focus = gate_summary_df[gate_summary_df["variant"] == compare_variant]
    if not focus.empty:
        row = focus.iloc[0]
        status_text = "PASS" if bool(row["gate_passed"]) else "FAIL"
        if bool(row["gate_passed"]):
            st.success(
                f"{compare_variant}: {status_text} | "
                f"rules={int(row['rules_passed'])}/{int(row['rules_total'])}"
            )
        else:
            st.error(
                f"{compare_variant}: {status_text} | "
                f"rules={int(row['rules_passed'])}/{int(row['rules_total'])} | "
                f"failed={row['failed_rules']}"
            )

    if gate_detail_df.empty:
        st.info("当前 run 未产出 decision gate 规则明细。")
        return

    st.subheader("规则明细")
    detail = gate_detail_df[gate_detail_df["variant"] == compare_variant].copy()
    if detail.empty:
        st.info("当前 compare variant 无规则明细。")
        return
    st.dataframe(detail, use_container_width=True)


def _render_competition_scorecard(
    scorecard_summary_df: pd.DataFrame,
    scorecard_detail_df: pd.DataFrame,
    compare_variant: str,
) -> None:
    st.header("工程评分映射（Competition Scorecard）")
    _render_page_guide("Competition Scorecard")

    if scorecard_summary_df.empty:
        st.warning("当前 run 未产出 competition scorecard 结果。")
        return

    st.subheader("就绪度总览")
    st.dataframe(scorecard_summary_df, use_container_width=True)

    focus = scorecard_summary_df[scorecard_summary_df["variant"] == compare_variant]
    if not focus.empty:
        row = focus.iloc[0]
        readiness = str(row["competition_readiness"])
        if readiness == "READY":
            st.success(
                f"{compare_variant}: READY | weighted_pass_rate={float(row['weighted_pass_rate']):.4f}"
            )
        else:
            st.error(
                f"{compare_variant}: NOT_READY | weighted_pass_rate={float(row['weighted_pass_rate']):.4f}"
            )

    if scorecard_detail_df.empty:
        st.info("当前 run 未产出 scorecard 规则明细。")
        return
    detail = scorecard_detail_df[scorecard_detail_df["variant"] == compare_variant].copy()
    if detail.empty:
        st.info("当前 compare variant 无 scorecard 规则明细。")
        return

    st.subheader("评分项明细")
    st.dataframe(detail, use_container_width=True)


def _render_repro_ci(summary: dict, required_files: Dict[str, Path]) -> None:
    st.header("复现与工程可信度（Repro & CI）")
    _render_page_guide("Repro & CI")

    st.subheader("运行元信息")
    st.json(
        {
            "generated_at": summary.get("generated_at"),
            "seed": summary.get("seed"),
            "baseline_variant": summary.get("baseline_variant"),
            "config_path": summary.get("config_path"),
        }
    )

    st.subheader("一键命令")
    st.code(
        "python code/rag/run_pipeline.py --evaluation --evaluation-config configs/evaluation.yaml --output-dir code/rag/reports/evaluation/manual_run",
        language="bash",
    )
    st.code("python tests/smoke/test_evaluation_pipeline.py", language="bash")

    st.subheader("工件清单")
    rows = [
        {"artifact": key, "path": str(path), "exists": path.exists()}
        for key, path in required_files.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("CI 信息")
    ci_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    if ci_path.exists():
        st.success("已检测到 CI 配置文件: .github/workflows/ci.yml")
        st.code(ci_path.read_text(encoding="utf-8"), language="yaml")
    else:
        st.warning("未检测到 CI workflow 文件。")


def _get_variants(ablation_df: pd.DataFrame) -> List[str]:
    variants = [str(x) for x in ablation_df["variant"].dropna().tolist()]
    unique = []
    for item in variants:
        if item not in unique:
            unique.append(item)
    return unique


def main() -> None:
    st.set_page_config(
        page_title="RAG-Eye v3 Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    report_root = _default_report_root()
    if not report_root.exists():
        st.error(f"未找到 evaluation 报告目录: {report_root}")
        return

    run_dirs = sorted([x for x in report_root.iterdir() if x.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        st.error("evaluation 报告目录下没有可用 run。")
        return

    with st.sidebar:
        st.title("RAG-Eye v3")
        st.caption("工程能力展示版")

        run_name = st.selectbox("选择 run_id", [x.name for x in run_dirs], index=len(run_dirs) - 1)
        selected_run = next(x for x in run_dirs if x.name == run_name)

    try:
        data = _load_run_data(str(selected_run))
    except Exception as exc:
        st.exception(exc)
        return

    ablation_df = data["ablation_df"]
    stats_df = data["stats_df"]
    gain_df = data["gain_df"]
    error_df = data["error_df"]
    calibration_df = data["calibration_df"]
    calibration_detail_df = data["calibration_detail_df"]
    calibration_sweep_df = data["calibration_sweep_df"]
    gate_summary_df = data["gate_summary_df"]
    gate_detail_df = data["gate_detail_df"]
    scorecard_summary_df = data["scorecard_summary_df"]
    scorecard_detail_df = data["scorecard_detail_df"]
    cases = data["cases"]
    summary = data["summary"]

    variants = _get_variants(ablation_df)
    if not variants:
        st.error("ablation 数据中没有 variant 字段。")
        return

    baseline_default = summary.get("baseline_variant", variants[0])
    if baseline_default not in variants:
        baseline_default = variants[0]

    compare_default = variants[0] if len(variants) == 1 else variants[min(1, len(variants) - 1)]

    with st.sidebar:
        baseline_variant = st.selectbox(
            "baseline_variant",
            variants,
            index=variants.index(baseline_default),
        )
        compare_variant = st.selectbox(
            "compare_variant",
            variants,
            index=variants.index(compare_default),
        )
        selected_metric = st.selectbox("metric", list(METRIC_LABELS.keys()), format_func=lambda x: METRIC_LABELS[x])
        q_type_options = ["all"] + sorted(list(set(gain_df["q_type"].dropna().astype(str).tolist())))
        q_type_filter = st.selectbox("q_type", q_type_options)
        risk_threshold = st.slider("confidence/risk threshold", 0.0, 1.0, 0.4, 0.05)

        page = st.radio(
            "页面导航",
            [
                "Home",
                "Overview",
                "Rigor",
                "Gain Decomposition",
                "Error X-Ray",
                "Calibration",
                "Decision Gate",
                "Competition Scorecard",
                "Method Graph",
                "Repro & CI",
            ],
        )

    if page == "Home":
        _render_home(summary, ablation_df, gain_df, baseline_variant, compare_variant)
    elif page == "Overview":
        _render_overview(summary, ablation_df, gain_df, baseline_variant, compare_variant)
    elif page == "Rigor":
        _render_rigor(ablation_df, stats_df, baseline_variant, compare_variant, selected_metric)
    elif page == "Gain Decomposition":
        _render_gain(gain_df, compare_variant, selected_metric)
    elif page == "Error X-Ray":
        _render_error_xray(error_df, cases, compare_variant, q_type_filter, risk_threshold)
    elif page == "Calibration":
        _render_calibration(
            calibration_df,
            calibration_detail_df,
            calibration_sweep_df,
            compare_variant,
        )
    elif page == "Decision Gate":
        _render_decision_gate(gate_summary_df, gate_detail_df, compare_variant)
    elif page == "Competition Scorecard":
        _render_competition_scorecard(
            scorecard_summary_df,
            scorecard_detail_df,
            compare_variant,
        )
    elif page == "Method Graph":
        _render_method_graph(ablation_df, compare_variant)
    elif page == "Repro & CI":
        _render_repro_ci(summary, data["required"])


if __name__ == "__main__":
    main()

