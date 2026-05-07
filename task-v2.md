# task-v2：RAG-Eye 高质量算法面试项目优化计划

## 0. 背景与目标
当前项目已经具备完整 RAG 链路（Hybrid Retrieval / Rewrite / HyDE / Reranker / Reflection / 分层评测 / 报告导出），但要达到“高质量算法岗面试项目”标准，还需要在以下三个维度补齐：

1. 实验严谨性：从“可跑”升级为“可验证、可复现、可统计推断”。
2. 算法深度：从“工程整合”升级为“有研究问题、有方法创新、有机制解释”。
3. 工程可信度：从“单次演示”升级为“标准化 benchmark + 自动化流水线 + CI 质量门禁”。

本计划聚焦：在不盲目造轮子的前提下，基于成熟库快速迭代出可被面试官信服的研究型工程成果。

---

## 1. 现状评估与主要缺口

### 1.1 已有优势
- 已有可运行 API 和一键报告产线。
- 已有多阶段能力：检索增强、反思重试、分层评测、消融与误差分析。
- 已形成可展示指标基线（Context Recall/Precision/Faithfulness/Relevance）。

### 1.2 关键缺口
- 消融仍是模拟降分，缺少“真实开关 + 真实重跑”实验。
- 缺少统计显著性检验，难证明提升不是随机波动。
- 缺少统一配置与随机种子控制，复现实验成本高。
- 缺少 query 类型分桶收益归因，无法解释“为什么提升”。
- 缺少标准化 benchmark 与 CI 自动评测，工程可信度仍偏演示型。

---

## 2. v2 量化目标（验收导向）

### 2.1 实验严谨性目标
- 所有核心模块支持真实开关（hybrid/reranker/rewrite/reflection/router/calibration）。
- 同一配置重复 3 次结果方差可控（关键指标标准差 <= 0.015）。
- 每项主要改动输出显著性结论（p-value + effect size + CI）。

### 2.2 算法深度目标
- 完成 query 分型收益分析：`fact / multi-hop / negative / long-context`。
- 引入至少 2 个“研究味”模块并完成对照实验：
  - Query Router（按问题类型分流检索策略）
  - Adaptive Retrieval（动态 top-k / 动态重排深度）
  - Answer Calibration（置信度标定 + 拒答阈值）
- 输出可解释错误归因：retrieval miss / rerank miss / generation hallucination / uncertainty misuse。

### 2.3 工程可信度目标
- 建立标准 benchmark 切分与评测协议（固定输入、固定评测脚本、固定输出格式）。
- 建立自动化评测流水线（本地一键 + CI nightly/smoke）。
- README 与报告中明确“模块收益 + 代价（时延/成本）”trade-off。

---

## 3. 具体优化工作包（Workstreams）

## WS-A：实验严谨性（P0）

### A1. 真实可开关 ablation（替换模拟消融）
- 将现有 `phase4_tools.run_ablation_simulation` 升级为“真实运行模式”。
- 通过统一配置驱动模块开关，逐一真实重跑并落盘。
- 输出每个 variant 的完整结果文件与聚合指标。

交付物：
- `ablation_config.yaml`（或同等配置）
- `run_ablation.py`（真实重跑）
- `ablation_results/*.json|csv|md`

### A2. 统计显著性检验
- 使用成熟统计库（scipy/statsmodels/pingouin）做 paired test：
  - 配对 t-test / Wilcoxon（二选一或并行）
  - Bootstrap 95% CI
  - effect size（Cohen's d 或 Cliff's delta）
- 在报告中输出“显著/不显著 + 效应大小 + 实际意义”。

交付物：
- `stats_report.json`
- `report.md` 新增 Statistical Significance 章节

### A3. 复现性基础设施
- 统一 seed 控制：Python/Numpy/检索采样/LLM评测调用层。
- 统一配置管理：Hydra/Pydantic Settings/Typer CLI（选成熟方案）。
- 所有实验命令支持 `--config` 与 `--seed`。

交付物：
- `configs/` 目录
- `reproduce.sh` / `reproduce.ps1`
- 复现实验说明文档

---

## WS-B：算法深度（P1）

### B1. Retrieval Error Case 细粒度分析
- 新增错误标签体系：
  - `no_recall`（关键证据未召回）
  - `bad_rank`（召回但排序过低）
  - `good_retrieval_bad_generation`
  - `over_retrieval_noise`
- 自动统计各标签占比 + 样例输出。

交付物：
- `error_taxonomy.md`
- `error_cases_topk.json`
- `error_dashboard.csv`

### B2. Query 类型收益分解
- 按 query type 分桶对比各模块收益。
- 输出 per-type 提升矩阵：`module x query_type x metric`。

交付物：
- `gain_by_query_type.csv`
- 报告新增“模块收益分解”图表

### B3. 研究味模块（优先 2 项）
- Query Router：基于轻量分类器/规则 + LLM judge 路由到检索策略。
- Adaptive Retrieval：按 query 难度调整 top-k、rerank depth、是否二次检索。
- Answer Calibration：输出 calibrated confidence，低置信触发拒答或反思重试。

交付物：
- 新模块代码 + ablation 对照结果
- 延迟/成本/效果三维对比表

---

## WS-C：工程可信度（P1）

### C1. 标准 benchmark 协议化
- 固定数据切分：dev/test/challenge。
- 固定输入格式、固定评测指标、固定输出目录结构。
- 版本化 benchmark 元数据（schema + version）。

交付物：
- `benchmark/README.md`
- `benchmark_schema.json`

### C2. 自动化评测流水线
- 一键命令串联：数据准备 -> 运行 -> 评测 -> 统计 -> 报告。
- 使用现有 `run_pipeline.py` 作为入口继续扩展。
- 引入任务编排（Makefile/tox/nox/Invoke 选一，优先轻量）。

交付物：
- `make evaluate`（或等价命令）
- `reports/<timestamp>/` 完整工件

### C3. CI 与质量门禁
- CI 最小集：lint + basic tests + smoke eval（小样本）。
- 失败即中断（快速失败），不吞异常。
- 输出 artifact（关键指标 JSON + markdown 摘要）。

交付物：
- `.github/workflows/*.yml`
- `tests/smoke/` 测试脚本

---

## 4. 里程碑与节奏（建议 4 周）

### Week 1（P0）
- 完成真实 ablation 框架
- 完成配置管理与 seed 统一
- 产出 v2 baseline

### Week 2（P0）
- 完成统计显著性检验链路
- 完成首版“真实消融 + 显著性”报告

### Week 3（P1）
- 完成 query 类型收益分解与错误归因
- 上线 Query Router 或 Adaptive Retrieval（至少 1 项）

### Week 4（P1）
- 上线第二项研究模块（Router/Adaptive/Calibration 中再选 1 项）
- 完成 CI 与 benchmark 协议化
- 输出面试版最终报告与项目陈述

---

## 5. DoD（Definition of Done）
满足以下条件，task-v2 判定完成：

1. 可复现：
- 任意机器按文档配置后可一键复现实验并产出同结构报告。

2. 可证明：
- 每项主要改动都有对照实验与显著性结论，而非单次跑分。

3. 可解释：
- 报告可清楚回答“哪类问题受益、为什么受益、代价是什么”。

4. 可工程化：
- CI 可自动执行最小评测闭环；失败快速暴露。

---

## 6. 风险与应对
- 网络/模型下载不稳定：继续使用 `HF_ENDPOINT` 镜像 + 本地 cache + 预拉取。
- 评测成本过高：引入 `smoke set`（快速）和 `full set`（离线）双模式。
- 统计结果不显著：扩大样本、改进分桶、报告 effect size 与置信区间避免误导。
- 模块变复杂：坚持配置化开关和统一接口，避免分叉脚本。

---

## 7. 面试表达映射（最终产出建议）
- 问题定义：行业标准问答中“可追溯性 + 抗幻觉”矛盾。
- 方法创新：Hybrid + Rewrite + Reflection + Router/Adaptive/Calibration。
- 量化收益：按 query type 的提升矩阵 + 显著性检验。
- 工程能力：可复现实验平台、自动化评测与 CI 门禁。
- trade-off：效果提升与时延/成本增加的平衡策略。

---

## 8. 执行原则
- 优先复用成熟第三方库，不重复造轮子。
- 不吞异常，严格快速失败。
- 代码与脚本遵循 PEP8 与清晰可维护原则。
