# RAG-Eye

<p align="center">
  <strong>面向垂直领域知识库的 RAG 质量评测、优化与门禁决策平台</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/langchain-0.2+-green.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#系统架构">系统架构</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#使用方法">使用方法</a> •
  <a href="#api-接口">API</a> •
  <a href="#项目结构">项目结构</a> •
  <a href="#技术栈">技术栈</a>
</p>

---

## 简介

RAG-Eye 是一个端到端的 RAG（检索增强生成）质量评测、优化与门禁决策平台。系统以国家标准《新能源汽车维修维护技术要求》(GB/T 44510-2024) 为知识源，面向 RAG 系统质量保障的全生命周期，提供了从智能测试生成、增强推理、多维度自动评测、统计显著性检验、置信度校准到版本准入决策的完整闭环。

**核心理念**：让垂直领域 RAG 系统不仅"能回答"，还要"答得有证据、错得可定位、风险可拦截、结果可复现、版本可准入"。

## 功能特性

### 智能测试集生成
- 利用大语言模型自动解析行业标准文档
- 分层策略生成覆盖**事实召回**、**跨段推理**和**否定陷阱**三种场景的测试用例
- 已自动产出 60 条高质量测试样本，支持可追溯的测试资产

### 增强型 RAG 推理引擎
- **查询改写**：多视角检索扩展，缓解口语化表达与专业术语不一致问题
- **假设性文档嵌入 (HyDE)**：先生成假设答案，再检索相关证据
- **混合检索**：稠密检索 + BM25 稀疏检索加权融合
- **交叉编码器精排**：细粒度判断问题与候选文档的相关性

### 自检反思与重试机制
- 独立裁判模型二次评估答案置信度
- 低置信度自动触发重试（最多 3 轮）
- 主动拒答策略，降低错误答案外放风险

### 自动化评测流水线
- **真实消融分析**：量化每个模块对整体质量的贡献
- **统计显著性检验**：Wilcoxon 配对检验、Bootstrap 置信区间、Cohen's d 效应量
- **分场景收益分解**：按问题类型分析优化效果
- **错误分型诊断**：检索未召回、排序不佳、生成失真、过度召回噪声
- **置信度校准**：ECE、Brier 分数、拒答阈值扫描
- **门禁准入判定**：多维指标自动判定版本是否可上线

### 交互式可视化看板
- 基于 Streamlit 的 9 维度分析仪表盘
- 支持交互式筛选、阈值调节和多维度下钻分析

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG-Eye 系统架构                             │
├─────────────────────────────────────────────────────────────────────┤
│   报告与展示层   │  Streamlit Dashboard (9页面) │ FastAPI RESTful    │
├─────────────────────────────────────────────────────────────────────┤
│   评测与分析层   │  消融 │ 统计检验 │ 错误诊断 │ 校准 │ 门禁 │ 评分 │
├─────────────────────────────────────────────────────────────────────┤
│   生成层         │  LLM 推理 │ 自检反思 │ 多角色评审 (4角色)        │
├─────────────────────────────────────────────────────────────────────┤
│   检索层         │  Dense + BM25 融合 │ Query Rewrite │ HyDE        │
│                  │  Cross-Encoder Rerank │ 去重                      │
├─────────────────────────────────────────────────────────────────────┤
│   数据层         │  ChromaDB 向量库 │ 测试集 (60条) │ 评测产物       │
└─────────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.11+
- 依赖包见 `requirements.txt`

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/rag-eye.git
cd rag-eye

# 安装依赖
pip install -U -r requirements.txt
```

### 配置

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1  # 可选，兼容 OpenAI 的 API
OPENAI_MODEL=deepseek-chat                     # 可选，默认模型
HF_ENDPOINT=https://hf-mirror.com              # 可选，HuggingFace 镜像加速
HF_HOME=.cache/huggingface                     # 可选，模型缓存路径
RERANKER_MODEL=BAAI/bge-reranker-base          # 可选，重排序模型
```

### 模型预拉取

首次运行前，建议预拉取嵌入和重排序模型到本地缓存：

```bash
python code/rag/pull_models.py
```

### 运行评测

```bash
# 使用代理标签口径（日常快速迭代）
python code/rag/run_pipeline.py --evaluation --evaluation-config configs/evaluation.yaml --output-dir code/rag/reports/evaluation/latest

# 使用标准答案口径（终验评估）
python code/rag/run_pipeline.py --evaluation --evaluation-config configs/evaluation_groundtruth.yaml --output-dir code/rag/reports/evaluation/latest
```

## 使用方法

### 启动在线服务

```bash
uvicorn code.rag.api:app --host 0.0.0.0 --port 8000
```

服务启动后可访问：
- API 文档：`http://localhost:8000/docs`
- ReDoc 文档：`http://localhost:8000/redoc`

### 启动可视化看板

```bash
streamlit run code/rag/dashboard_app.py
```

看板包含以下分析页面：
- **Home** - 首页概览
- **Overview** - 核心指标总览
- **Rigor** - 实验严谨性分析
- **Gain Decomposition** - 场景收益分解
- **Error X-Ray** - 错误诊断透视
- **Calibration** - 置信度校准曲线
- **Decision Gate** - 门禁准入判定
- **Competition Scorecard** - 评分映射
- **Method Graph** - 方法链路图

### 运行测试

```bash
# 冒烟测试
python tests/smoke/test_evaluation_pipeline.py

# 单元测试
python tests/unit/test_evaluation_quality.py

# 代码质量检查
ruff check code/rag/evaluation/
mypy code/rag/evaluation/
```

## API 接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/retrieve` | POST | 检索相关文档，返回扩展查询和重排序结果 |
| `/ask` | POST | 问答接口，返回答案和证据文档 |
| `/ask_with_reflection` | POST | 带自检反思的问答，支持置信度评估和自动重试 |
| `/phase3/evaluate_sample` | POST | 样本评测，含多角色评审和硬负例挖掘 |
| `/phase4/ablation` | POST | 消融分析，量化各模块贡献 |
| `/phase4/error_analysis` | POST | 错误分析，输出错误类型统计 |
| `/phase4/long_context_probe` | POST | 长上下文探针生成 |

### 接口示例

**问答请求**

```bash
curl -X POST "http://localhost:8000/ask_with_reflection" \
  -H "Content-Type: application/json" \
  -d '{"question": "高压系统检修前需要哪些防护措施？"}'
```

**响应示例**

```json
{
  "answer": "根据标准要求，高压系统检修前需要...",
  "confidence": 0.92,
  "needs_retry": false,
  "retries_used": 0,
  "judge_reason": "回答准确且有充分证据支持"
}
```

## 项目结构

```
czbAiTest/
├── code/
│   └── rag/
│       ├── api.py                        # FastAPI 接口服务
│       ├── rag.py                        # 基础 RAG 流程
│       ├── eval_engine.py                # 基础评测器
│       ├── evaluation_pipeline.py        # Evaluation 总流水线
│       ├── generate_testset.py           # 测试集生成
│       ├── run_pipeline.py               # 一键入口脚本
│       ├── pull_models.py                # 模型预拉取
│       ├── env_utils.py                  # 环境变量管理
│       ├── dashboard_app.py              # Streamlit 展示看板
│       ├── dashboard_enhanced.py         # 增强看板（含 Runner）
│       ├── phase3_eval.py                # Phase3 多角色评审
│       ├── phase4_tools.py               # Phase4 工具集
│       ├── export_report.py              # 报告导出
│       ├── evaluation/                   # 评测子模块
│       │   ├── real_ablation.py          # 真实消融分析
│       │   ├── statistics.py             # 统计检验
│       │   ├── query_gain.py             # 分场景收益
│       │   ├── error_analysis.py         # 错误分型诊断
│       │   ├── calibration.py            # 置信度校准
│       │   ├── decision_gate.py          # 决策门禁
│       │   └── competition_scorecard.py  # 评分映射
│       ├── data/                         # 数据文件
│       │   └── GBT+44510-2024.pdf        # 知识源文档
│       ├── outputs/                      # 评测产物
│       ├── reports/                      # 分析报告
│       └── vectorstores/                 # 向量数据库
├── configs/
│   ├── evaluation.yaml                   # 评测配置（代理标签口径）
│   └── evaluation_groundtruth.yaml       # 评测配置（标准答案口径）
├── benchmark/
│   ├── README.md                         # Benchmark 协议说明
│   ├── benchmark_schema.json             # 测试集 Schema
│   └── splits/
│       └── manifest.json                 # 数据集划分元数据
├── tests/
│   ├── smoke/
│   │   └── test_evaluation_pipeline.py   # 冒烟测试
│   └── unit/
│       └── test_evaluation_quality.py    # 单元测试
├── .github/
│   └── workflows/
│       └── ci.yml                        # CI 流水线配置
└── requirements.txt                      # 依赖清单
```

## 核心指标

系统采用四维评测指标体系，分别对应 RAG 链路的不同环节：

| 指标 | 含义 | 评估维度 |
|------|------|----------|
| **上下文召回率** (Context Recall) | 检索是否覆盖回答所需的关键证据 | 检索质量 |
| **上下文精准度** (Context Precision) | 召回内容中有多少比例是有用信息 | 检索质量 |
| **回答忠实度** (Faithfulness) | 回答是否忠于检索到的证据，而非模型自行编造 | 生成质量 |
| **答案相关度** (Answer Relevance) | 回答是否真正解决了问题核心 | 生成质量 |

## 评测结果示例

| 版本 | 上下文召回率 | 上下文精准度 | 回答忠实度 | 答案相关性 | 门禁状态 |
|------|:---:|:---:|:---:|:---:|:---:|
| 增强版本 | 0.4600 | 0.5100 | 0.9700 | 0.8800 | 通过 |
| 基线版本 | 0.5800 | 0.6100 | 0.8500 | 0.9700 | 未通过 |

> **说明**：增强版本在回答忠实度上明显更优，并通过了校准、错误泄漏和准入规则的综合判定。对于高风险行业知识库，"宁可少答也不能错答"的取向更符合真实上线要求。

## 输出产物

评测流水线运行后，会在指定目录生成以下产物：

| 类型 | 文件 | 说明 |
|------|------|------|
| 摘要 | `summary.json` | 评测核心指标汇总 |
| 消融 | `ablation_real.csv` | 各模块贡献度分析 |
| 统计 | `stats_significance.csv` | 显著性检验结果 |
| 收益 | `gain_by_query_type.csv` | 分场景收益分解 |
| 错误 | `error_dashboard.csv` | 错误诊断看板 |
| 错误 | `error_cases_topk.json` | 高风险样例详情 |
| 校准 | `calibration_metrics.csv` | 校准指标 |
| 校准 | `calibration_threshold_sweep.csv` | 阈值扫描结果 |
| 门禁 | `decision_gate_summary.csv` | 门禁判定结果 |
| 评分 | `competition_scorecard_summary.csv` | 评分映射 |
| 报告 | `report.md` | 综合分析报告 |

## 技术栈

| 类别 | 技术 |
|------|------|
| **RAG 框架** | LangChain |
| **向量数据库** | ChromaDB |
| **嵌入模型** | BGE-M3 (BAAI) |
| **重排序模型** | BGE-Reranker-Base (BAAI) |
| **LLM** | OpenAI-compatible API (DeepSeek) |
| **Web 框架** | FastAPI |
| **可视化** | Streamlit + Plotly |
| **统计分析** | SciPy |
| **数据处理** | Pandas |
| **代码质量** | Ruff + Mypy |
| **CI/CD** | GitHub Actions |

## 错误分型诊断

系统将评测中发现的问题自动归因为四种类型，并提供针对性优化建议：

| 错误类型 | 含义 | 优化方向 |
|----------|------|----------|
| `no_recall` | 关键证据未被检索到 | 优化检索策略 |
| `bad_rank` | 关键证据被召回但排名靠后 | 优化重排序模型 |
| `good_retrieval_bad_generation` | 证据正确但生成阶段产生失真 | 优化生成提示词 |
| `over_retrieval_noise` | 检索到过多无关片段 | 优化召回过滤 |

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- **知识源**：国家标准《新能源汽车维修维护技术要求》(GB/T 44510-2024)
- **比赛**：传智杯 AI 测试赛道

---

<p align="center">
  Made with ❤️ by RAG-Eye Team
</p>
