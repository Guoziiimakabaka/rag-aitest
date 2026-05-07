# task-v3：RAG-Eye 演示网站升级改造方案（完整详细版）

## 0. 文档定位
本方案用于将当前“简易演示页面”升级为“算法面试可讲、可证、可复现”的展示系统。

约束：
- 本阶段仅输出改造方案，不改代码。
- 方案优先复用现有 Task-v2 产物，不重复造轮子。

---

## 1. 改造背景与问题定义

### 1.1 当前页面主要问题
1. 展示以“静态指标”居多，缺少“问题-方法-证据-结论”的叙事闭环。  
2. Task-v2 核心亮点（真实消融、显著性检验、分类型收益、错误归因）未被结构化呈现。  
3. 面试场景下无法快速回答“为什么有效、对谁有效、代价是什么”。  
4. 工程可信度展示不足（配置、seed、benchmark 协议、CI 状态可见性弱）。

### 1.2 目标用户
1. 算法面试官：关注方法有效性、显著性、泛化与可解释性。  
2. 工程面试官：关注复现性、流程自动化、质量门禁。  
3. 项目本人：用于 5-15 分钟结构化演示。

---

## 2. 产品目标与非目标

### 2.1 产品目标（必须达成）
1. 3 分钟内讲清项目核心贡献。  
2. 每个方法模块都能看到收益、显著性和 trade-off。  
3. 支持 baseline 与 variant 的实时对比。  
4. 支持从总体指标 drill down 到 badcase 证据。  
5. 明确可复现信息：配置、seed、命令、产物路径、CI。

### 2.2 非目标（本阶段不做）
1. 不做复杂权限系统。  
2. 不做在线训练或在线索引构建。  
3. 不做大规模前后端重构（优先 Streamlit 内升级）。

---

## 3. 信息架构（IA）重构

建议导航改为 6 大模块：

1. **总览（Overview）**  
2. **实验严谨性（Rigor）**  
3. **算法增益拆解（Gain Decomposition）**  
4. **错误分析与 Badcase（Error X-Ray）**  
5. **方法链路与开关（Method Graph）**  
6. **复现与工程可信度（Repro & CI）**

统一交互控件（全局侧边栏）：
- `run_id`（报告批次）
- `baseline_variant`
- `compare_variant`
- `metric`
- `q_type`
- `confidence threshold`（badcase 过滤）

---

## 4. 页面级详细改造方案

## 4.1 总览页（Overview）

### 目标
用最短时间给出“项目做了什么、效果如何、结论是什么”。

### 组件设计
1. Hero 区
- 一句话定位：垂直领域 RAG 实验平台（强调可证据化）。
- 当前 run 信息：时间、样本数、baseline/variant。

2. KPI 卡片（4 大指标）
- Recall / Precision / Faithfulness / Relevance。
- 显示 `当前值 + 与 baseline 的 Δ`。

3. 总结结论卡（Top Findings）
- 自动抽取 3-5 条关键结论：
  - 哪个指标提升最大
  - 哪个 q_type 受益最大
  - 哪个模块带来关键贡献

4. 快速跳转卡
- “去看显著性”
- “去看多跳问题收益”
- “去看高风险 badcase”

### 数据来源
- `summary.json`
- `ablation_real.csv`
- `gain_by_query_type.csv`

### 验收标准
- 面试官 1 分钟内可定位当前最优组合与主要收益。

---

## 4.2 实验严谨性页（Rigor）

### 目标
证明结果不是偶然波动，而是具备统计可信度。

### 组件设计
1. 真实消融矩阵（Ablation Matrix）
- 行：variant
- 列：模块开关（hybrid/reranker/reflection/rewrite）
- 显示指标均值与排序。

2. 显著性结果表（Significance Table）
- 列：metric, mean_diff, effect_size, p_value, CI, significant。
- 支持按 p-value/效应量排序。

3. 置信区间可视化
- 每指标一条误差线（variant vs baseline）。

4. 结果解释区
- 自动提示：
  - “统计显著但效应小”
  - “不显著但趋势一致”
  - “显著且效应中等以上”

### 数据来源
- `ablation_real.csv`
- `stats_significance.csv`

### 验收标准
- 能明确回答“该改动是否显著有效”。

---

## 4.3 算法增益拆解页（Gain Decomposition）

### 目标
回答“对哪些问题类型有效”。

### 组件设计
1. 热力图
- 维度：`variant x q_type x metric`
- 色值：gain。

2. 分桶柱状图
- 固定 q_type，比较各 variant 的 gain。

3. 模块收益雷达图（可选）
- 每种 q_type 一张雷达图展示四指标变化。

4. 解释卡
- 自动提取各 q_type 的最佳模块组合与薄弱点。

### 数据来源
- `gain_by_query_type.csv`

### 验收标准
- 能指出“multi-hop 为什么更吃检索增强”。

---

## 4.4 错误分析与 Badcase 页（Error X-Ray）

### 目标
回答“失败发生在哪里、如何修复”。

### 组件设计
1. 错误分型分布
- `no_recall / bad_rank / good_retrieval_bad_generation / over_retrieval_noise / ok`
- 展示占比与趋势。

2. 高风险样例列表
- 支持按 `risk_score`、`error_label`、`q_type` 筛选。

3. 样例详情抽屉
- 展示问题、模型回答、关键指标、错误标签、建议修复策略。

4. 修复建议映射
- 每类错误对应策略：
  - no_recall -> query rewrite / hybrid weight 调整
  - bad_rank -> reranker depth/threshold
  - generation 问题 -> reflection/calibration

### 数据来源
- `error_dashboard.csv`
- `error_cases_topk.json`

### 验收标准
- 每类错误至少可展示 2 个代表样例并给出下一步策略。

---

## 4.5 方法链路与开关页（Method Graph）

### 目标
让面试官理解“系统不是黑箱拼装”，而是有机制设计。

### 组件设计
1. 可视化流程图
- Query -> Rewrite/HyDE -> Hybrid Retrieval -> Rerank -> Generation -> Reflection -> Judge。

2. 模块说明卡
- 每个模块：输入、输出、配置开关、主要影响指标。

3. 开关模拟器（只读）
- 切换 variant 后高亮对应开启模块。

4. 代价面板
- 显示近似延迟/计算成本（可先用静态估计，后续接真实profiling）。

### 数据来源
- `configs/task_v2.yaml`
- `ablation_real.csv`

### 验收标准
- 能清楚讲出每个模块存在理由与作用边界。

---

## 4.6 复现与工程可信度页（Repro & CI）

### 目标
证明项目可复现、可审计、可持续演进。

### 组件设计
1. 运行元信息
- seed、config_path、run_time、输入数据路径。

2. 一键命令区
- 展示可复制命令：task-v2 运行、smoke test。

3. 工件清单
- 列出本次 run 所有输出文件，并支持点击打开。

4. CI 状态区
- 展示 CI 工作流名称、最近一次状态（先静态，后续接 API）。

### 数据来源
- `summary.json`
- `.github/workflows/ci.yml`

### 验收标准
- 新成员可按页面说明在 15 分钟内复现实验输出。

---

## 5. 数据层与文件映射规范

### 5.1 推荐输入契约
- `summary.json`
- `ablation_real.csv`
- `stats_significance.csv`
- `gain_by_query_type.csv`
- `error_dashboard.csv`
- `error_cases_topk.json`

### 5.2 运行批次管理
建议目录：
- `code/rag/reports/task_v3/<run_id>/...`

页面支持选择 run_id 以对比不同实验轮次。

### 5.3 校验规则
页面启动前执行数据校验：
1. 必需文件是否存在。  
2. 必需列是否完整。  
3. 数值列是否可解析。  
4. baseline_variant 是否在 variant 集合中。

---

## 6. 交互与叙事策略（面试模式）

### 6.1 默认演示路径（建议）
1. 总览：项目与结论。  
2. 实验严谨性：证明有效。  
3. 增益拆解：解释为何有效。  
4. 错误分析：展示工程深度与迭代方向。  
5. 复现页：证明可信度。

### 6.2 面试模式按钮（后续实现）
- 点击后按固定顺序跳转并展示“讲解提示词”。

### 6.3 提示文案原则
- 每个图表下有一句“如何解读”说明。  
- 避免图多结论少。

---

## 7. 视觉与体验改造建议

1. 设计语言
- 保留技术感，但减少“纯炫酷”元素，优先信息密度与清晰度。  
- 强调对比色用于“提升/下降/显著”。

2. 组件风格
- 指标卡统一高度与数值格式。  
- 表格统一排序、搜索、导出。

3. 图表规范
- 所有 y 轴注明范围与单位。  
- 显著性结果强制显示 p-value 与 CI。  
- 所有图可导出 PNG/CSV（后续实现）。

4. 可用性
- 支持窄屏模式。  
- 页面加载显示数据版本和更新时间。

---

## 8. 实施里程碑（建议 2 周）

### M1（第 1-2 天）：信息架构重排
- 重构导航与全局筛选器。  
- 搭建 6 页空框架与路由。

### M2（第 3-5 天）：核心数据页落地
- 完成总览页、实验严谨性页、增益拆解页。

### M3（第 6-8 天）：错误分析与方法链路
- 完成 error x-ray + method graph。

### M4（第 9-10 天）：复现页与演示打磨
- 完成 Repro/CI 页。  
- 完成面试演示顺序与讲解文案。

### M5（第 11-12 天）：验收与回归
- 完成功能验收、数据一致性检查、演示彩排。

---

## 9. 验收标准（DoD）

1. 关键亮点可见
- Task-v2 四大亮点均有独立模块展示。

2. 结论可证
- 每个重要结论可回溯到数据文件与统计项。

3. 演示可讲
- 按默认路径可在 8 分钟完成完整讲解。

4. 复现可信
- 页面可见 seed、config、命令、输出工件。

5. 稳定性
- 缺失文件时给出明确错误，不吞异常。

---

## 10. 风险与应对

1. 数据字段不统一
- 应对：增加 strict schema 校验与缺失提示。

2. 图表太多导致认知负担
- 应对：每页只保留 1-2 个主图 + 结论卡。

3. 指标解释口径不一致
- 应对：建立统一术语字典（指标定义固定化）。

4. 演示时网络不稳定
- 应对：页面完全离线读取本地工件，避免在线依赖。

---

## 11. 交付清单（规划级）

1. `task-v3.md`（本文件）  
2. 页面改造后的 IA 草图（后续）  
3. 组件与数据映射表（后续）  
4. 演示讲稿脚本（后续）

---

## 12. 后续可选增强（v3.1+）

1. 引入 Answer Calibration 专页（置信度-准确率曲线）。  
2. 引入 Query Router 策略对比页。  
3. 增加 latency/cost profiling 自动采集。  
4. 对接 GitHub Actions API 显示实时 CI 状态。  
5. 增加“简历版摘要生成器”（自动生成项目陈述）。
