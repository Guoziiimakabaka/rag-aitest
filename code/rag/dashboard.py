import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px

# 引入之前的模块
# 注意：确保 rag.py 和 eval_engine.py 在同一目录下，且没有直接在 import 时运行耗时代码
import rag
import eval_engine

st.set_page_config(
    page_title="RAG-Eye 评测系统",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #0f52ba;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .stProgress .st-bo {
        background-color: #0f52ba;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def render_metric_card(col, label, value, delta=None):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

def radar_chart(df_small, df_large):
    categories = ['Context Recall', 'Context Precision', 'Faithfulness', 'Answer Relevance']
    
    # 计算平均分
    def get_means(df):
        return [
            df['context_recall'].mean(),
            df['context_precision'].mean(),
            df['faithfulness'].mean(),
            df['answer_relevance'].mean()
        ]

    fig = go.Figure()

    if df_small is not None:
        fig.add_trace(go.Scatterpolar(
            r=get_means(df_small),
            theta=categories,
            fill='toself',
            name='Baseline (Small Chunk)',
            line_color='#FF6B6B'
        ))

    if df_large is not None:
        fig.add_trace(go.Scatterpolar(
            r=get_means(df_large),
            theta=categories,
            fill='toself',
            name='Optimized (Large Chunk)',
            line_color='#4ECDC4'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="RAG 能力维度对比 (雷达图)"
    )
    return fig


st.sidebar.title("👁️ RAG-Eye")
st.sidebar.caption("垂类大模型知识检索全链路评测")

page = st.sidebar.radio("导航", ["📊 评测看板 (Dashboard)", "🐛 幻觉/Badcase 诊断", "⚙️ 运行新评测"])

PATH_SMALL_RESULTS = "outputs/small/rag_eval_results.json"
PATH_LARGE_RESULTS = "outputs/large/rag_eval_results.json"
PATH_TESTSET = "outputs/generated_testset.json"


if page == "📊 评测看板 (Dashboard)":
    st.title("📊 RAG 系统综合评测看板")

    df_small = load_results(PATH_SMALL_RESULTS)
    df_large = load_results(PATH_LARGE_RESULTS)

    if df_small is None and df_large is None:
        st.warning("⚠️ 尚未找到评测结果数据。请先到“⚙️ 运行新评测”页面执行测试。")
        st.stop()


    st.subheader("核心指标对比")
    c1, c2, c3, c4 = st.columns(4)

    current_df = df_large if df_large is not None else df_small
    label_prefix = "Optimized (Large)" if df_large is not None else "Baseline (Small)"

    if current_df is not None:
        render_metric_card(c1, "Context Recall (召回)", current_df['context_recall'].mean())
        render_metric_card(c2, "Context Precision (精准)", current_df['context_precision'].mean())
        render_metric_card(c3, "Faithfulness (信实度)", current_df['faithfulness'].mean())
        render_metric_card(c4, "Answer Relevance (相关性)", current_df['answer_relevance'].mean())

    st.markdown("---")

    c_chart1, c_chart2 = st.columns([1, 1])

    with c_chart1:
        st.write("#### 🎯 能力维度对比 (Radar)")
        fig_radar = radar_chart(df_small, df_large)
        st.plotly_chart(fig_radar, use_container_width=True)

    with c_chart2:
        st.write("#### 📈 分数分布 (Box Plot)")
        plot_data = []
        if df_small is not None:
            temp = df_small[['faithfulness', 'answer_relevance']].copy()
            temp['Model'] = 'Small Chunk'
            plot_data.append(temp)
        if df_large is not None:
            temp = df_large[['faithfulness', 'answer_relevance']].copy()
            temp['Model'] = 'Large Chunk'
            plot_data.append(temp)
        
        if plot_data:
            df_plot = pd.concat(plot_data)
            df_melt = df_plot.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
            fig_box = px.box(df_melt, x="Metric", y="Score", color="Model", points="all",
                             title="Faithfulness & Relevance 分布")
            st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("📋 详细评测记录")
    
    view_option = st.radio("查看模型结果：", ["Optimized (Large)", "Baseline (Small)"], horizontal=True)
    target_df = df_large if view_option == "Optimized (Large)" else df_small

    if target_df is not None:
        st.dataframe(
            target_df[['question', 'ground_truth', 'answer', 'faithfulness', 'answer_relevance']],
            use_container_width=True,
            column_config={
                "faithfulness": st.column_config.ProgressColumn("Faithfulness", format="%.2f", min_value=0, max_value=1),
                "answer_relevance": st.column_config.ProgressColumn("Relevance", format="%.2f", min_value=0, max_value=1),
            }
        )
    else:
        st.info("该模型暂无数据。")


elif page == "🐛 幻觉/Badcase 诊断":
    st.title("🐛 Bad Case 深度诊断")
    st.markdown("此页面专门筛选**低分样本**（Faithfulness < 0.6 或 Relevance < 0.6），并展示 LLM 评委的判定理由。")

    source = st.selectbox("选择要分析的模型结果", ["Optimized (Large Chunk)", "Baseline (Small Chunk)"])
    
    csv_path = PATH_LARGE_RESULTS if "Large" in source else PATH_SMALL_RESULTS
    
    if not os.path.exists(csv_path):
        st.error(f"找不到 {csv_path}。请先运行评测。")
        st.stop()
        
    df = load_results(csv_path)

    c1, c2 = st.columns(2)
    with c1:
        threshold_faith = st.slider("Faithfulness 阈值 (低于此值视为幻觉)", 0.0, 1.0, 0.6)
    with c2:
        threshold_rel = st.slider("Relevance 阈值 (低于此值视为答非所问)", 0.0, 1.0, 0.6)

    bad_cases = df[
        (df['faithfulness'] < threshold_faith) | 
        (df['answer_relevance'] < threshold_rel)
    ]

    st.write(f"🔍 共发现 **{len(bad_cases)}** 个 Bad Cases (总样本数: {len(df)})")

    for i, row in bad_cases.iterrows():
        with st.expander(f"Case #{i+1}: {row['question'][:50]}... (Faith: {row['faithfulness']:.2f})", expanded=False):
            c_left, c_right = st.columns([1, 1])
            
            with c_left:
                st.markdown("**User Question:**")
                st.info(row['question'])
                
                st.markdown("**RAG Answer:**")
                st.warning(row['answer'])
                
                st.markdown("**Ground Truth:**")
                st.success(row['ground_truth'])

            with c_right:
                st.markdown("### 🕵️ 裁判解释 (Judge Reasoning)")
                
                if row['faithfulness'] < threshold_faith:
                    st.markdown("**❌ Faithfulness (幻觉检测):**")
                    st.markdown(f"> {row['faithfulness_explanation']}")
                    st.progress(row['faithfulness'])
                
                if row['answer_relevance'] < threshold_rel:
                    st.markdown("**❌ Relevance (相关性):**")
                    st.markdown(f"> {row['answer_relevance_explanation']}")
                    st.progress(row['answer_relevance'])
                
                st.divider()
                st.markdown("**Retrieval Metrics:**")
                st.write(f"Recall: {row['context_recall']:.2f}")
                st.write(f"Precision: {row['context_precision']:.2f}")


elif page == "⚙️ 运行新评测":
    st.title("⚙️ 执行全链路评测")
    st.markdown("这里将调用 `rag.py` 生成答案，并使用 `eval_engine.py` 进行打分。")

    if os.path.exists(PATH_TESTSET):
        with open(PATH_TESTSET, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        st.success(f"✅ 检测到测试集 ({len(test_data)} 题): `{PATH_TESTSET}`")
    else:
        st.error("❌ 未检测到测试集，请先运行 generate_testset.py 或上传文件。")
        st.stop()

    st.markdown("### 1. RAG 参数配置")
    sample_limit = st.number_input("本次评测样本数量 (用于快速演示)", min_value=1, max_value=100, value=5)
    
    st.markdown("### 2. 执行")
    
    if st.button("🚀 开始评测 (Start Evaluation)", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("正在加载知识库和模型... (可能需要1-2分钟)")
            
            from rag import build_or_load_vectorstore, build_qa_chain, build_rag_eval_input
            from eval_engine import load_eval_samples_from_json, evaluate_samples, save_results


            pdf_docs = rag.load_pdf(rag.PDF_PATH)
            
            progress_bar.progress(10)
            status_text.text("正在运行 RAG (Small Chunk) 生成答案...")
            
            db_small = build_or_load_vectorstore(pdf_docs, rag.DB_DIR_SMALL, 200, 50)
            qa_small = build_qa_chain(db_small)
            
            build_rag_eval_input(
                qa_small, 
                PATH_TESTSET, 
                "data/rag_eval_input_small.json", 
                max_samples=sample_limit
            )
            
            progress_bar.progress(40)
            status_text.text("正在运行 RAG (Large Chunk) 生成答案...")
            
            db_large = build_or_load_vectorstore(pdf_docs, rag.DB_DIR_LARGE, 500, 80)
            qa_large = build_qa_chain(db_large)
            
            build_rag_eval_input(
                qa_large, 
                PATH_TESTSET, 
                "data/rag_eval_input_large.json", 
                max_samples=sample_limit
            )

            progress_bar.progress(60)
            status_text.text("AI 裁判正在为 Small Chunk 评分...")
            
            os.makedirs("outputs/small", exist_ok=True)
            samples_small = load_eval_samples_from_json("data/rag_eval_input_small.json")
            res_small = evaluate_samples(samples_small)
            save_results(res_small, PATH_SMALL_RESULTS, PATH_SMALL_RESULTS.replace('.json', '.csv'))

            progress_bar.progress(80)
            status_text.text("AI 裁判正在为 Large Chunk 评分...")
            
            os.makedirs("outputs/large", exist_ok=True)
            samples_large = load_eval_samples_from_json("data/rag_eval_input_large.json")
            res_large = evaluate_samples(samples_large)
            save_results(res_large, PATH_LARGE_RESULTS, PATH_LARGE_RESULTS.replace('.json', '.csv'))

            progress_bar.progress(100)
            status_text.text("评测完成！请前往 Dashboard 查看结果。")
            st.success("✅ 全链路评测已完成！点击左侧导航栏查看报表。")
            
            st.cache_data.clear()

        except Exception as e:
            st.error(f"运行过程中出错: {e}")
            import traceback
            st.code(traceback.format_exc())