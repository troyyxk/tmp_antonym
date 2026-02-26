# Constraint-Aware RAG 交付版（V1）

## 0. 项目一句话
做一个可插拔的 `Constraint Encoder`，在 RAG 的检索阶段识别并过滤“语义相关但逻辑冲突”的文档，显著提升否定/反义约束查询下的检索忠实度。

---

## 1. 要解决的问题

### 1.1 现象
当前 Dense Retriever 在以下查询上容易失败：

- `not dirty` 检索到大量 `dirty` 相关文档
- `without peanuts` 检索到 `contains peanuts` 文档
- `under $100` 检索到 `$500` 产品文档

这些错误样本通常“主题相关”，但“约束冲突”，属于 Hard Negatives。

### 1.2 根因假设
- 检索向量空间把 `Topic Relevance` 和 `Constraint Compliance` 混在一起建模
- 反义词、否定句在分布上共享上下文，导致相似度被错误抬高
- 结果是 Top-k 里“相关但违约束”的文档比例过高，后续 LLM 无法纠正

---

## 2. 方案设计（最终版本）

### 2.1 总体架构：Dual-View Retrieval

`Encoder A`（Topic Expert）：
- 负责召回，目标是高 Recall
- 直接复用现成 embedding 模型（如 BGE / E5 / OpenAI embedding）

`Encoder B`（Constraint Expert）：
- 负责约束一致性打分，目标是高 CCR
- 训练轻量模型识别 query-doc 的逻辑一致/冲突关系

### 2.2 在线推理链路

1. `Retrieve`：用 `Encoder A` 从向量库召回 Top-100  
2. `Filter/Rerank`：用 `Encoder B` 对 Top-100 打约束一致性分数  
3. `Select`：保留一致性最高的 Top-10  
4. `Generate`：将清洗后的 Top-10 交给 LLM

### 2.3 融合公式（可落地）

给定文档 `d`，最终分数：

`Score(d) = alpha * Sim_topic(q, d) + (1 - alpha) * Sim_constraint(q, d)`

- `alpha` 初始取 `0.7`
- 同时提供硬阈值版本：若 `Sim_constraint < tau` 则直接过滤

---

## 3. 数据与训练

### 3.1 训练样本定义
训练 `Encoder B` 的三元组：

`(Query, Positive, Hard_Negative)`

- `Positive`：语义相关且满足约束
- `Hard_Negative`：语义相关但违背约束

### 3.2 数据来源（第一版）
- SNLI / MNLI（用于 contradiction 信号）
- 规则增强样本（not/without/no/under/at most 等模板）
- 小规模人工校验集（200-500 条）用于最终评测可信性

### 3.3 模型与损失
- Backbone：`distilbert-base-uncased`（或同量级模型）
- Loss：`MultipleNegativesRankingLoss` 或 `TripletLoss`
- 训练目标：拉近 `(q, pos)`，推远 `(q, hard_neg)`

---

## 4. 评估协议（必须交付的指标）

### 4.1 检索指标
- Recall@10 / Recall@100
- NDCG@10

### 4.2 约束指标（核心）
- `CCR@10`（Constraint Compliance Rate）  
  定义：Top-10 中满足约束的文档占比

### 4.3 端到端指标
- QA Faithfulness（回答是否被错误检索误导）
- Negation Failure Rate（否定类问题错误率）

### 4.4 对比基线
- BM25
- Vanilla Dense Retriever（只用 Encoder A）
- Cross-Encoder Reranker（性能上界，速度下界）

---

## 5. 交付里程碑与验收标准

## Phase 1（Week 1）PoC：证明问题存在
交付物：
- `poc_negation_gap.py`
- 一张图：vanilla embedding 在 contradiction 样本上的相似度分布

验收：
- 能复现实验并导出图表
- 明确展示“相关但冲突”样本得分偏高

## Phase 2（Week 2-3）数据管线与训练
交付物：
- `build_triplets.py`
- `train_constraint_encoder.py`
- `eval_constraint_encoder.py`

验收：
- 成功产出可用的 `Encoder B` 模型权重
- 在验证集上 `CCR` 相对 baseline 有提升

## Phase 3（Week 4）在线检索融合
交付物：
- `retrieve_then_filter.py`
- 支持 `alpha`、`tau` 两类融合策略

验收：
- 在同一数据集上，相比 vanilla retriever：
  - `CCR@10` 提升 >= 20%（相对提升）
  - `Recall@10` 下降不超过 3%

## Phase 4（Week 5-6）端到端 RAG 验证
交付物：
- `rag_eval.py`
- 失败案例分析报告（至少 20 个样本）

验收：
- Negation 类问题回答正确率明显提升
- 给出成本/延迟对比表（vs cross-encoder）

---

## 6. 工程目录建议（最小可运行）

```text
ant/
  README.md
  想法整理/
    RAG方向.md
  experiments/
    poc_negation_gap.py
    build_triplets.py
    train_constraint_encoder.py
    eval_constraint_encoder.py
    retrieve_then_filter.py
    rag_eval.py
  data/
    raw/
    processed/
  outputs/
    figures/
    checkpoints/
    reports/
```

---

## 7. 风险与应对

1) `Encoder B` 提升 CCR 但伤害 Recall  
- 应对：采用软融合 + 阈值回退策略；保留 Top-k 多样性

2) NLI 数据迁移到检索场景不稳定  
- 应对：加入检索模板增强与少量人工标注校验集

3) 与 Cross-Encoder 对比时不公平  
- 应对：统一候选池与评测集；同时报告质量与延迟

4) 论文贡献被误解为“只是反义词任务”  
- 应对：始终表述为 `Constraint Modeling for RAG`

---

## 8. 第一周行动清单（今天就能开工）

1. 固定 PoC 数据：从 SNLI/MNLI 抽取 contradiction 对  
2. 先跑 1 个现成 embedding 模型，画相似度分布图  
3. 明确 30 条真实 RAG 约束查询作为 smoke test  
4. 确认 CCR 标注规则（什么算“满足约束”）  
5. 输出第一版实验日志模板（命令、参数、结果、结论）

---

## 9. 对外表述模板（用于汇报/论文）

> 我们提出一个面向 RAG 的约束感知检索框架，通过双视角解耦建模话题相关性与约束一致性。在不替换现有向量检索系统的前提下，仅增加轻量级约束编码器即可显著降低否定/反义约束下的检索错误，并提升端到端回答忠实度。

