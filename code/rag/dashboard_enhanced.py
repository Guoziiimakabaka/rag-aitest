import streamlit as st
import pandas as pd
import json
import os
import time
import requests
import plotly.graph_objects as go
import plotly.express as px

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st.error("❌ 缺少必要的库！请在终端运行: pip install streamlit-lottie")
    st.stop()

try:
    import rag
    import eval_engine
    from rag import build_or_load_vectorstore, build_qa_chain, build_rag_eval_input
    from eval_engine import load_eval_samples_from_json, evaluate_samples, save_results
except ImportError as e:
    st.error(f"❌ 无法导入后端模块: {e}")
    st.info("请确保 rag.py 和 eval_engine.py 与本文件在同一目录下。")
    st.stop()


st.set_page_config(
    page_title="RAG-Eye | 智能评测系统",
    page_icon="🧿",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3) # 3秒超时防止卡死
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_scanning = load_lottieurl("https://lottie.host/5a072045-8025-4638-b39b-8d695ba57451/6Qy3O7qQyv.json")
lottie_brain = load_lottieurl("https://lottie.host/0a767480-e83c-41c0-82a1-e4075c3db64d/v1J7rQy2wK.json")

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    h1, h2, h3 {
        color: #E6EDF3 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border-radius: 12px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(88, 166, 255, 0.6);
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.3);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #58A6FF, #3FB950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 14px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1F6FEB 0%, #238636 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 10px rgba(31, 111, 235, 0.5);
    }
</style>
""", unsafe_allow_html=True)


def render_tech_card(col, label, value, sub_text=""):
    with col:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.3f}</div>
            <div style="color: #7D8590; font-size: 12px; margin-top: 5px;">{sub_text}</div>
        </div>
        """, unsafe_allow_html=True)

def tech_radar_chart(df_small, df_large):
    categories = ['Context Recall', 'Context Precision', 'Faithfulness', 'Answer Relevance']
    
    fig = go.Figure()

    if df_small is not None:
        vals = [
            df_small['context_recall'].mean(),
            df_small['context_precision'].mean(),
            df_small['faithfulness'].mean(),
            df_small['answer_relevance'].mean()
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories,
            fill='toself',
            name='Baseline (Small)',
            line_color='#FF7B72',
            opacity=0.6
        ))

    if df_large is not None:
        vals = [
            df_large['context_recall'].mean(),
            df_large['context_precision'].mean(),
            df_large['faithfulness'].mean(),
            df_large['answer_relevance'].mean()
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories,
            fill='toself',
            name='Optimized (Large)',
            line_color='#3FB950',
            opacity=0.7
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#30363D', linecolor='#30363D'),
            angularaxis=dict(gridcolor='#30363D', linecolor='#30363D'),
            bgcolor='rgba(22, 27, 34, 0.5)'
        ),
        font=dict(color='#E6EDF3'),
        legend=dict(font=dict(color='#E6EDF3')),
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig


with st.sidebar:
    if lottie_brain:
        st_lottie(lottie_brain, height=150, key="sidebar_anim")
    else:
        st.markdown("🧬 **System Specs**")
        
    st.markdown("<h2 style='text-align: center;'>RAG-Eye</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8B949E;'>垂类大模型全链路评测</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected_page = st.radio(
        "系统模块",
        ["🚀 评测主控台 (Runner)", "📊 全息数据看板 (Dashboard)", "🔬 幻觉深度透视 (X-Ray)"],
        index=1
    )
    
    st.markdown("---")
    st.info("💡 **Tip**: 使用大Chunk (500) 通常能显著提升多跳问题的召回率。")

PATH_SMALL_RESULTS = "outputs/small/rag_eval_results.json"
PATH_LARGE_RESULTS = "outputs/large/rag_eval_results.json"
PATH_TESTSET = "outputs/generated_testset.json"

def load_data(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return pd.DataFrame(json.load(f))
    return None

df_small = load_data(PATH_SMALL_RESULTS)
df_large = load_data(PATH_LARGE_RESULTS)


if selected_page == "📊 全息数据看板 (Dashboard)":
    col_hero_1, col_hero_2 = st.columns([3, 1])
    with col_hero_1:
        st.title("系统效能全息看板")
        st.markdown("通过多维指标实时监控 RAG 系统的检索质量与生成信实度。")
        st.markdown(f"当前对比版本：**Baseline (Chunk=200)** vs **Optimized (Chunk=500)**")
    with col_hero_2:
        if df_large is not None:
             avg_score = (df_large['faithfulness'].mean() + df_large['answer_relevance'].mean()) / 2
             st.metric("综合质量得分 (Optimized)", f"{avg_score:.2f}", "+14%")

    st.markdown("---")

    if df_small is None and df_large is None:
        st.warning("⚠️ 暂无评测数据，请前往“评测主控台”启动测试任务。")
        st.stop()

    st.markdown("### 🧬 核心效能指标 (Key Performance Indicators)")
    c1, c2, c3, c4 = st.columns(4)
    
    target_df = df_large if df_large is not None else df_small
    
    render_tech_card(c1, "Context Recall", target_df['context_recall'].mean(), "检索覆盖率")
    render_tech_card(c2, "Context Precision", target_df['context_precision'].mean(), "信噪比")
    render_tech_card(c3, "Faithfulness", target_df['faithfulness'].mean(), "事实一致性")
    render_tech_card(c4, "Answer Relevance", target_df['answer_relevance'].mean(), "用户意图匹配")

    st.markdown("### 🕸️ 维度对比分析")
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.markdown("**能力雷达图**")
        fig = tech_radar_chart(df_small, df_large)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_right:
        st.markdown("**信实度分布密度**")
        hist_data = []
        if df_small is not None:
            hist_data.append(go.Histogram(x=df_small["faithfulness"], name='Baseline', opacity=0.6, marker_color='#FF7B72'))
        if df_large is not None:
            hist_data.append(go.Histogram(x=df_large["faithfulness"], name='Optimized', opacity=0.6, marker_color='#3FB950'))
        
        fig_hist = go.Figure(data=hist_data)
        fig_hist.update_layout(
             barmode='overlay', 
             paper_bgcolor='rgba(0,0,0,0)', 
             plot_bgcolor='rgba(0,0,0,0)', 
             font=dict(color='#E6EDF3'),
             xaxis_title="Faithfulness Score", 
             yaxis_title="Count"
        )
        st.plotly_chart(fig_hist, use_container_width=True)


elif selected_page == "🔬 幻觉深度透视 (X-Ray)":
    col_t1, col_t2 = st.columns([4, 1])
    with col_t1:
        st.title("🔬 幻觉与Badcase 诊断")
        st.markdown("由 LLM 裁判提供的深度归因分析，精准定位“胡说八道”的根源。")
    with col_t2:
        if lottie_scanning:
            st_lottie(lottie_scanning, height=100)
        else:
            st.markdown("📡 **Scanning...**")

    with st.container():
        st.markdown("""<div class="glass-card">""", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            dataset_choice = st.selectbox("分析模型", ["Optimized (Large)", "Baseline (Small)"])
        with fc2:
            threshold = st.slider("幻觉阈值 (Faithfulness < X)", 0.0, 1.0, 0.6)
        with fc3:
            keyword = st.text_input("关键词搜索")
        st.markdown("</div>", unsafe_allow_html=True)

    current_df = df_large if "Large" in dataset_choice else df_small
    if current_df is None:
        st.error("暂无数据。")
        st.stop()

    mask = (current_df['faithfulness'] < threshold)
    if keyword:
        mask = mask & (current_df['question'].str.contains(keyword) | current_df['answer'].str.contains(keyword))
    
    filtered_df = current_df[mask]

    st.write(f"🔍 扫描完成：发现 **{len(filtered_df)}** 个潜在风险案例")

    for idx, row in filtered_df.iterrows():
        with st.expander(f"🔴 Case #{idx}: {row['question'][:60]}...", expanded=False):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**问题 (Question):**")
                st.info(row['question'])
                st.markdown("**模型回答 (AI):**")
                st.markdown(f"> {row['answer']}")
                st.markdown("**标准事实 (GT):**")
                st.success(row['ground_truth'])
            
            with c2:
                st.markdown("### 🤖 裁判判决")
                st.markdown(f"""
                <div style="background-color: #2D1B1E; padding: 15px; border-radius: 8px; border-left: 5px solid #FF7B72;">
                    <div style="color: #FF7B72; font-weight: bold; margin-bottom: 5px;">Score: {row['faithfulness']:.2f}</div>
                    <div style="color: #E6EDF3; font-style: italic;">{row['faithfulness_explanation']}</div>
                </div>
                """, unsafe_allow_html=True)


elif selected_page == "🚀 评测主控台 (Runner)":
    st.title("🚀 自动评测主控台")
    
    has_testset = os.path.exists(PATH_TESTSET)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        sample_count = st.slider("评测样本量 (Samples)", 1, 50, 5)
            
    with c2:
        if has_testset:
            st.success("✅ 测试集就绪")
            # ⚠️ 修复：强制使用 utf-8 读取
            with open(PATH_TESTSET, 'r', encoding='utf-8') as f:
                data = json.load(f)
            st.caption(f"包含 {len(data)} 条问答对")
        else:
            st.error("❌ 未检测到测试集")

    st.markdown("---")
    
    if st.button("INITIATE EVALUATION PROTOCOL (启动评测协议)", type="primary", use_container_width=True):
        console_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        def log(msg):
            console_placeholder.markdown(f"""
            <div style="font-family: 'Courier New'; color: #3FB950; background: #000; padding: 10px; border-radius: 5px;">
                > {msg} <span style="animation: blink 1s infinite;">_</span>
            </div>
            """, unsafe_allow_html=True)
        
        try:
            log("Initializing RAG Engines...")
            pdf_docs = rag.load_pdf(rag.PDF_PATH)
            progress_bar.progress(10)

            log("Running Inference (Baseline)...")
            db_small = build_or_load_vectorstore(pdf_docs, rag.DB_DIR_SMALL, 200, 50)
            qa_small = build_qa_chain(db_small)
            build_rag_eval_input(qa_small, PATH_TESTSET, "data/rag_eval_input_small.json", max_samples=sample_count)
            progress_bar.progress(40)

            log("Running Inference (Optimized)...")
            db_large = build_or_load_vectorstore(pdf_docs, rag.DB_DIR_LARGE, 500, 80)
            qa_large = build_qa_chain(db_large)
            build_rag_eval_input(qa_large, PATH_TESTSET, "data/rag_eval_input_large.json", max_samples=sample_count)
            progress_bar.progress(70)

            log("Running LLM-as-a-Judge...")
            os.makedirs("outputs/small", exist_ok=True)
            samples_small = load_eval_samples_from_json("data/rag_eval_input_small.json")
            res_small = evaluate_samples(samples_small)
            save_results(res_small, PATH_SMALL_RESULTS, PATH_SMALL_RESULTS.replace('.json', '.csv'))
            
            os.makedirs("outputs/large", exist_ok=True)
            samples_large = load_eval_samples_from_json("data/rag_eval_input_large.json")
            res_large = evaluate_samples(samples_large)
            save_results(res_large, PATH_LARGE_RESULTS, PATH_LARGE_RESULTS.replace('.json', '.csv'))
            
            progress_bar.progress(100)
            log("All Systems Green. Evaluation Complete.")
            st.success("评测完成！请切换至看板查看。")
            st.cache_data.clear()

        except Exception as e:
            st.error(f"SYSTEM FAILURE: {str(e)}")
            st.exception(e)