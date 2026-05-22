from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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
}


def _default_report_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "code" / "rag" / "reports" / "task_v2"


@st.cache_data(show_spinner=False)
def _load_run_data(run_dir: str) -> Dict[str, object]:
    path = Path(run_dir)
    required = {
        "summary_json": path / "summary.json",
        "ablation_csv": path / "ablation_real.csv",
        "stats_csv": path / "stats_significance.csv",
        "stability_csv": path / "stability_summary.csv",
        "gain_csv": path / "gain_by_query_type.csv",
        "error_csv": path / "error_dashboard.csv",
        "cases_json": path / "error_cases_topk.json",
    }
    optional = {
        "tradeoff_csv": path / "latency_cost_tradeoff.csv",
        "cost_summary_csv": path / "latency_cost_summary.csv",
    }

    missing = [name for name, file_path in required.items() if not file_path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Run 目录缺少必要文件: {missing}; 当前目录: {path}"
        )

    summary = json.loads(required["summary_json"].read_text(encoding="utf-8"))
    ablation_df = pd.read_csv(required["ablation_csv"])
    stats_df = pd.read_csv(required["stats_csv"])
    stability_df = pd.read_csv(required["stability_csv"])
    gain_df = pd.read_csv(required["gain_csv"])
    error_df = pd.read_csv(required["error_csv"])
    cases = json.loads(required["cases_json"].read_text(encoding="utf-8"))
    tradeoff_df = (
        pd.read_csv(optional["tradeoff_csv"])
        if optional["tradeoff_csv"].exists()
        else pd.DataFrame()
    )
    cost_summary_df = (
        pd.read_csv(optional["cost_summary_csv"])
        if optional["cost_summary_csv"].exists()
        else pd.DataFrame()
    )

    return {
        "summary": summary,
        "ablation_df": ablation_df,
        "stats_df": stats_df,
        "stability_df": stability_df,
        "gain_df": gain_df,
        "error_df": error_df,
        "cases": cases,
        "tradeoff_df": tradeoff_df,
        "cost_summary_df": cost_summary_df,
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
    tradeoff_df: pd.DataFrame,
    cost_summary_df: pd.DataFrame,
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
    if not tradeoff_df.empty:
        st.write("- Adaptive 模块代价收益：查看下方成本收益卡片。")

        st.subheader("Adaptive 成本收益摘要")
        mean_multiplier = float(tradeoff_df["estimated_latency_multiplier"].mean())
        max_gain = float(tradeoff_df["context_recall_gain"].max())
        col1, col2 = st.columns(2)
        col1.metric("平均延迟倍率", f"{mean_multiplier:.3f}x")
        col2.metric("最大 Recall 增益", f"{max_gain:+.4f}")
    if not cost_summary_df.empty:
        st.subheader("统一成本汇总")
        st.dataframe(cost_summary_df, use_container_width=True)


def _render_rigor(
    ablation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    stability_df: pd.DataFrame,
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
        sig_raw = bool(row["significant_p_lt_0_05"])
        sig_adj = bool(row.get("significant_adjusted", False))
        sig_text = "校正后显著" if sig_adj else ("原始显著" if sig_raw else "不显著")
        st.success(
            f"{METRIC_LABELS[selected_metric]}: p={row['p_value']:.6f}, p_adj={row.get('p_value_adjusted', float('nan')):.6f}, d={row['effect_size_cohen_d']:.4f}, 95%CI=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] -> {sig_text}"
        )

    st.dataframe(
        stats_df.sort_values(["metric", "p_value_adjusted", "p_value"]),
        use_container_width=True,
    )

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

    st.subheader("稳定性告警")
    if stability_df.empty:
        st.warning("当前无稳定性数据。")
        return

    stable_slice = stability_df[stability_df["variant"] == compare_variant].copy()
    if stable_slice.empty:
        st.info("当前 compare variant 无稳定性条目。")
        return

    stable_slice["is_alert"] = (stable_slice["std"] > 0.015) | (stable_slice["cv"] > 0.05)
    alert_rows = stable_slice[stable_slice["is_alert"]]
    if alert_rows.empty:
        st.success("当前 variant 未触发稳定性告警（std <= 0.015 且 cv <= 0.05）。")
    else:
        st.error(f"发现 {len(alert_rows)} 条稳定性告警，请优先复查这些指标。")
        st.dataframe(
            alert_rows[["variant", "metric", "runs", "mean", "std", "cv", "is_alert"]],
            use_container_width=True,
        )

    fig_stability = px.bar(
        stable_slice,
        x="metric",
        y="std",
        color="is_alert",
        title=f"{compare_variant} 指标稳定性（std）",
        color_discrete_map={True: "#d62728", False: "#2ca02c"},
    )
    st.plotly_chart(fig_stability, use_container_width=True)


def _render_adaptive_tradeoff(tradeoff_df: pd.DataFrame) -> None:
    st.header("Adaptive Tradeoff")
    if tradeoff_df.empty:
        st.info("当前 run 无 latency_cost_tradeoff.csv，可先运行 adaptive setup 再评估。")
        return

    st.subheader("分题型延迟成本与收益")
    st.dataframe(tradeoff_df, use_container_width=True)

    fig_latency = px.bar(
        tradeoff_df,
        x="q_type",
        y="estimated_latency_multiplier",
        color="q_type",
        title="Adaptive Retrieval 估计延迟倍率",
    )
    st.plotly_chart(fig_latency, use_container_width=True)

    gain_cols = [
        "context_recall_gain",
        "context_precision_gain",
        "faithfulness_gain",
        "answer_relevance_gain",
    ]
    melt_df = tradeoff_df.melt(
        id_vars=["q_type"],
        value_vars=gain_cols,
        var_name="metric",
        value_name="gain",
    )
    fig_gain = px.bar(
        melt_df,
        x="q_type",
        y="gain",
        color="metric",
        barmode="group",
        title="Adaptive Retrieval 分题型指标增益",
    )
    st.plotly_chart(fig_gain, use_container_width=True)


def _render_cost_summary(cost_summary_df: pd.DataFrame) -> None:
    st.header("Cost Summary")
    if cost_summary_df.empty:
        st.info("当前 run 无 latency_cost_summary.csv。")
        return
    st.dataframe(cost_summary_df, use_container_width=True)

    row = cost_summary_df.iloc[0]
    cols = st.columns(3)
    cols[0].metric("Avg Latency", f"{float(row['avg_latency_multiplier']):.3f}x")
    cols[1].metric("Max Latency", f"{float(row['max_latency_multiplier']):.3f}x")
    cols[2].metric("Repeat Runs", str(int(row["repeat_runs"])))


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
        "python code/rag/run_pipeline.py --task-v2 --task-v2-config configs/task_v2.yaml --output-dir code/rag/reports/task_v2/manual_run",
        language="bash",
    )
    st.code(
        "python code/rag/run_pipeline.py --task-v2-adaptive-setup --task-v2-config configs/task_v2.yaml --task-v2-adaptive-output-config configs/task_v2.adaptive.generated.yaml",
        language="bash",
    )
    st.code("python tests/smoke/test_task_v2_pipeline.py", language="bash")

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
        st.error(f"未找到 task_v2 报告目录: {report_root}")
        return

    run_dirs = sorted([x for x in report_root.iterdir() if x.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        st.error("task_v2 报告目录下没有可用 run。")
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
    stability_df = data["stability_df"]
    gain_df = data["gain_df"]
    error_df = data["error_df"]
    cases = data["cases"]
    tradeoff_df = data["tradeoff_df"]
    cost_summary_df = data["cost_summary_df"]
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
                "Adaptive Tradeoff",
                "Cost Summary",
                "Error X-Ray",
                "Method Graph",
                "Repro & CI",
            ],
        )

    if page == "Home":
        _render_home(summary, ablation_df, gain_df, baseline_variant, compare_variant)
    elif page == "Overview":
        _render_overview(
            summary,
            ablation_df,
            gain_df,
            tradeoff_df,
            cost_summary_df,
            baseline_variant,
            compare_variant,
        )
    elif page == "Rigor":
        _render_rigor(
            ablation_df,
            stats_df,
            stability_df,
            baseline_variant,
            compare_variant,
            selected_metric,
        )
    elif page == "Gain Decomposition":
        _render_gain(gain_df, compare_variant, selected_metric)
    elif page == "Adaptive Tradeoff":
        _render_adaptive_tradeoff(tradeoff_df)
    elif page == "Cost Summary":
        _render_cost_summary(cost_summary_df)
    elif page == "Error X-Ray":
        _render_error_xray(error_df, cases, compare_variant, q_type_filter, risk_threshold)
    elif page == "Method Graph":
        _render_method_graph(ablation_df, compare_variant)
    elif page == "Repro & CI":
        _render_repro_ci(summary, data["required"])


if __name__ == "__main__":
    main()
