# Research Proposal: Constraint-Aware Disentangled Retrieval for RAG

## gemini 问答
- https://gemini.google.com/share/014bb722a364

**面向RAG系统中强负约束（Hard Negative Constraints）的解耦检索架构**

## 1. 问题背景 (Motivation & Problem Statement)

### 1.1 核心痛点

当前的 RAG（检索增强生成）系统虽然在“找相关内容”上表现出色，但在处理**否定约束（Negation）**和**反义逻辑（Antonyms）**时存在严重的“致盲”现象。

* **现象**：
* 用户 Query: "Reviews for hotels that are **not** dirty."
* 现有 Retriever (OpenAI/BGE): 检索出含有 "dirty" 的文档，导致召回了大量由于卫生差被投诉的酒店。
* 用户 Query: "Plans that do **not** include peanuts."
* 现有 Retriever: 检索出 "This dish contains peanuts."


* **根本原因 (Theoretical Failure)**：
* **分布假设的副作用**：根据 Distributional Hypothesis，语义相反的词（如 Clean/Dirty, Cheap/Expensive）往往出现在极度相似的上下文中。
* **向量纠缠 (Entanglement)**：现有的 Dense Retrieval 模型将“相关性（Topic）”和“满足性（Constraint/Polarity）”压缩在同一个向量空间中。导致“相关但矛盾”的文档（Hard Negatives）得分过高。



### 1.2 现有方案的局限

* **Cross-Encoder Reranking**：虽然用 BERT/LLM 做重排序可以解决，但在大规模或端侧场景下**推理成本过高**，延迟不可接受。
* **Prompt Engineering**：让 LLM 自己去过滤，前提是 Retriever 必须先召回正确的文档。如果 Top-k 全是错的，LLM 也无能为力。

---

## 2. 提出的方法 (Proposed Methodology)

### 2.1 核心理念：解耦检索 (Disentangled Retrieval)

我们不试图训练一个“万能”的 Encoder，而是提出一种**双视角（Dual-View）**架构，将检索任务拆解为两个子任务：

1. **Semantic Relevance (话题相关性)**：这篇文档在讲什么？（现有模型已做得很好）
2. **Constraint Compliance (约束一致性)**：这篇文档是否满足用户的逻辑/属性限制？（本项目的核心创新）

### 2.2 模型架构 (Architecture)

采用 **MoE (Mixture of Experts) 思想的轻量级双 Encoder 架构**，可作为现有 RAG 系统的**即插即用（Plug-and-Play）**组件。

* **Encoder A (Topical Expert)**:
* **角色**：负责召回。复用现有的 SOTA 模型（如 BGE-M3, OpenAI Embedding）。
* **逻辑**：`Sim(Query, Doc)` 越高越好，只要话题一致。


* **Encoder B (Constraint Expert) —— *Our Contribution***:
* **角色**：负责过滤/重排序。一个轻量级的 Transformer（如 DistilBERT）。
* **逻辑**：专门训练用于识别**逻辑冲突**。
* **空间特性**：在这个空间里，Query "Not Expensive" 与 Doc "Expensive" 距离极远（正交或负相关）。



### 2.3 融合策略 (Integration Pipeline)

该模块将位于 Vector DB 检索之后，LLM 生成之前。

或者作为 **Hard Filter**：

1. **Retrieve**: 使用 Encoder A 从向量库召回 Top-100。
2. **Rerank**: 使用 Encoder B 对 Top-100 进行逻辑校验，剔除冲突文档。
3. **Generate**: 将清洗后的 Top-10 喂给 LLM。

---

## 3. 实验计划 (Experimental Setup)

### 3.1 数据集构建 (Data Construction)

为了训练和验证 Encoder B，我们需要构建包含 **Hard Negatives** 的三元组 `(Query, Positive, Hard_Negative)`。

* **来源**：利用 NLI (Natural Language Inference) 数据集（如 MNLI, SNLI）。
* *Entailment*  Positive Sample
* *Contradiction*  Hard Negative Sample


* **类型覆盖**：
1. **Lexical Antonyms**: (Clean vs. Dirty)
2. **Negation Scope**: (with vs. without)
3. **Numeric/Logical Constraints**: (Under $100 vs. $500)



### 3.2 评估指标 (Metrics)

我们不仅仅看 Recall@K，更关注**约束遵循率**：

1. **Standard Retrieval Metrics**: NDCG@10, Recall@10（保证基础性能不掉）。
2. **Constraint Compliance Rate (CCR)**: 在 Top-10 结果中，真正满足用户逻辑约束（如“非负面”、“不含某物”）的文档占比。**（这是我们的决胜指标）**
3. **RAG Hallucination Rate**: 端到端测试，看 LLM 最终回答的准确率提升。

### 3.3 基线对比 (Baselines)

* **BM25**: 传统的稀疏检索（对否定词不敏感）。
* **Vanilla Dense Retriever**: BGE-Base / OpenAI-Ada-002。
* **Counter-fitting Embeddings**: 2016年的静态词向量后处理方法（经典对照组）。
* **Cross-Encoder Reranker**: 性能天花板，但我们要证明我们能以 **1/10 的算力**达到接近的效果。

---

## 4. 实施路线图 (Implementation Roadmap)

* **Phase 1: Proof of Concept (Week 1)**
* 编写脚本，测试当前 SOTA 模型（如 `all-MiniLM`）在 NLI Contradiction 样本上的相似度。
* **产出**：一张展示现有模型无法区分反义句子的热力图或直方图（用于论文 Intro）。


* **Phase 2: Data Pipeline & Model Setup (Week 2-3)**
* 清洗 NLI 数据，构建训练用的 Triplet。
* 搭建双 Encoder 的推理 Pipeline。


* **Phase 3: Training Constraint Encoder (Week 4-6)**
* 使用 Contrastive Loss / Triplet Loss 训练 Encoder B。
* 重点调试 Negative Sampling 的策略（如何挖掘最难的负样本）。


* **Phase 4: Evaluation & Paper Writing (Week 7-10)**
* 在 BEIR 和自建数据集上跑分。
* 撰写论文，目标投递 EMNLP Short/Long。



---

## 5. 预期挑战与应对 (Anticipated Q&A)

这里预判了 Reviewer 或导师可能提出的挑战性问题：

* **Q1: 为什么不用大模型（LLM）直接做 Rerank？**
* *A*: 成本和延迟。LLM Rerank 无法处理高并发的实时检索。我们的 Dual Encoder 是轻量级的，适合作为向量库后的第一道防线，通过低成本过滤提升下游 LLM 的准确率。


* **Q2: 专门做一个“反义词”Encoder 是否太窄（Too Narrow）？**
* *A*: 我们做的不仅仅是“反义词”，而是 **"Constraint Modeling"（约束建模）**。它涵盖了否定词、逻辑互斥和反义属性，这是目前 RAG 产生幻觉的主要原因之一，具有广泛的通用性。


* **Q3: 这种方法能兼容现有的 RAG 架构吗？**
* *A*: 完全兼容。它不需要用户重建向量索引（Re-indexing），只需要在检索后增加一个极快的 Inference 步骤，对现有系统的侵入性为零。



---

## 6. 资源需求 (Resource Requirements)

* **计算资源**: 只需要单卡 RTX 3090 或 4090 即可完成 Encoder B（基于 DistilBERT/BERT-Base）的训练。
* **数据**: 公开数据集 (MNLI, SNLI, BEIR)，无需标注成本。

---

## 7. 导师补充 (Potential Gaps check)

*(这部分留给导师填写，但我已为您预留了两个方向)*

1. **Case Studies**: 我们是否需要做一些具体的定性分析？比如展示 5 个真实的 RAG 失败案例，以及我们的模型是如何修正它们的。
2. **Domain Adaptation**: 这个模型在特定领域（如法律、医疗）的泛化能力如何？是否需要针对特定领域微调？

# 实验
## 针对 RAG 的
| Method | Type | Latency (ms) | Recall@10 (General) | Constraint Compliance % (Target) |
| --- | --- | --- | --- | --- |
| **BM25** | Sparse | **< 10** | 45.0 | 20.0 (Fail) |
| **BGE-Base** | Dense | 20 | **85.0** | 40.0 (Mixes up antonyms) |
| **Contriever** | Dense | 20 | 82.0 | 60.0 |
| **Cross-Encoder** | Rerank | 500+ (Slow) | 88.0 | **95.0 (Upper Bound)** |
| **Ours (Dual)** | Dense+MoE | **25 (Fast)** | 84.5 (Comparable) | **90.0 (High Win)** |

## 针对反义词的
- Counter-fitting (Mrkšić et al., NAACL 2016)
- antonym-sensitive embedding 
    - https://aclanthology.org/K15-1026.pdf?utm_source=chatgpt.com
- Integrating Distributional Lexical Contrast into Word Embeddings for Antonym-Synonym Distinction
    - https://arxiv.org/abs/1605.07766
- Bhav-Net: Knowledge Transfer for Cross-Lingual Antonym vs Synonym Distinction via Dual-Space Graph Transformers
    - https://www.arxiv.org/abs/2508.15792

## 两个 encoder 的组合方式
- 串联或者相乘，看看哪个好

# related paper
这是一个非常专业的文献调研问题。

要证明你的 Idea (Dual-View / Disentangled Retrieval) 是 **“Novel but Grounded”**（有创新但有依据），你需要引用三类文章：

1. **直接对标（Competitors）**：试图解决否定/约束问题，但方法不如你优雅（通常是 Prompting 或 Re-weighting）。
2. **理论支撑（Foundations）**：证明“解耦（Disentanglement）”在 NLP 里是可行的。
3. **数据与评测（Benchmarks）**：证明大家开始关注“Instruction/Constraint”了，但还没解决好。

以下是为你整理的 ACL / EMNLP / ICLR 等顶会最近三年（2023-2025）最相关的文章。

### 1. 核心对标：Negation & Constraint in Retrieval (最相关的竞品)

这几篇文章是你必须读的，它们直接处理“否定”或“指令遵循”问题。

* **Enhancing Negation Awareness in Universal Text Embeddings (arXiv 2025 / Under Review)**
* **内容**：这篇文章几乎是你的**直接前身**。它明确指出当前的 Embedding 模型（如 OpenAI, BGE）在“否定”上完全致盲。
* **它的方法**：提出了一种 **Embedding Re-weighting**（重加权）方法，在推理时动态调整某些维度的权重来捕捉否定。
* **你的攻击点**：它是“修补式”的（Post-hoc），而你是“架构式”的（End-to-End Dual Encoder）。你的方法理论上上限更高。
* **引用价值**：用来证明 Problem Statement —— “Current embeddings are negation-blind”。


* **FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions (NAACL 2025 / arXiv 2024)**
* **内容**：这是一个**Benchmark（基准）工作**。它发布了一个数据集，专门测试 Retriever 能否听懂复杂的指令（包括 "Do not retrieve..."）。
* **结论**：发现除了超大的 LLM，现有的 Dense Retrievers 在 Instruction Following 上几乎全部失败。
* **引用价值**：这是你实验部分**必须要跑的数据集**。用它的结论来支撑你“为什么要做专门的 Constraint Encoder”。


* **Instruction-following Text Embedding via Answering the Question (ACL 2024)**
* **内容**：提出把“指令”看作一个问题，训练模型去生成“答案”的 Embedding。
* **你的差异**：它还是把所有信息压在一个向量里（Single Vector），而你主张解耦（Dual Vector）。



### 2. 架构灵感：Disentangled Representation (你的理论亲戚)

这几篇文章证明了“把一个向量拆成两半用”在学术界是被认可的。

* **Disentangling Aspect and Stance via a Siamese Autoencoder for Aspect Clustering (ACL Findings 2023)**
* **内容**：这篇非常有参考价值。它做的是把“话题（Aspect）”和“立场（Stance，支持/反对）”拆开。
* **关联**：这完全对应你的“话题（Topic）”和“约束（Constraint/Polarity）”。
* **引用价值**：在 Methodology 章节引用，说“受此启发，我们将该思想迁移到了 Dense Retrieval 领域”。


* **Disentangling Questions from Query Generation for Task-Adaptive Retrieval (EMNLP Findings 2024)**
* **内容**：讨论在生成查询时如何解耦“问题意图”和“领域风格”。
* **引用价值**：证明 Disentanglement 在 Retrieval 任务中是热点方向。



### 3. 硬负采样与逻辑推理 (Hard Negatives & Logic)

* **RaDeR: Reasoning-aware Dense Retrieval Models (EMNLP 2025)**
* **内容**：提出了“Reasoning-aware”的检索，专门针对这就需要多步推理（Chain-of-Thought）的查询。
* **关联**：你的 Constraint Encoder 本质上也是一种“Logic-aware”的组件。
* **引用价值**：将你的工作拔高到“Reasoning/Logic Retrieval”的高度，而不仅仅是“反义词”。


* **TriSampler: A Better Negative Sampling Principle for Dense Retrieval (AAAI 2024)**
* **内容**：从理论上分析了怎么选负样本。
* **引用价值**：在你的 Loss Function 设计部分引用，说明你在构建 Hard Negative（反义词）时遵循了最新的采样理论。



---

### 如何串联这些文章

你可以这样说：

> “目前 ACL 和 EMNLP 的风向（如 **FollowIR, NAACL 2025**）已经非常明确地指出，现有的 Retriever 根本听不懂‘不要什么’（Negation Constraints）。
> 现有的解决方案要么是像 **ACL 2024** 那样做指令微调（但还是单向量，容易混淆），要么像 **arXiv 2025** 那样做后处理重加权（治标不治本）。
> 也就是像 **ACL 2023** 在 Stance Detection 领域做的那样，在 Retrieval 里实现真正的**架构级解耦（Structural Disentanglement）**。目前还没有人专门为 RAG 的 Hard Constraint 做过这个 Dual-View 架构，这就是我们的 Opportunity。”

### 总结表格（Markdown 格式）

你可以直接把这个表放在 Proposal 的 Related Work 里：

| Category | Paper Title | Venue | Relevance to Your Work |
| --- | --- | --- | --- |
| **Problem** | *Enhancing Negation Awareness in Universal Text Embeddings* | arXiv '25 | 证明了否定致盲问题，提出了简单的重加权Baseline。 |
| **Benchmark** | *FollowIR: Evaluating and Teaching IR Models to Follow Instructions* | NAACL '25 | 提供了包含大量 Negation/Constraint 的测试数据集。 |
| **Method** | *Disentangling Aspect and Stance via a Siamese Autoencoder* | ACL '23 | 提供了“解耦（Disentanglement）”的模型架构理论支撑。 |
| **Context** | *Instruction-following Text Embedding via Answering the Question* | ACL '24 | 代表了目前“指令遵循检索”的主流（单向量）做法，是你的主要对比对象。 |
| **Concept** | *RaDeR: Reasoning-aware Dense Retrieval Models* | EMNLP '25 | 将检索任务从单纯的“匹配”提升到了“推理/逻辑”层面。 |

这些文章都非常新，引用它们能证明你的 Idea 是站在 2025 年的前沿，而不是在炒冷饭。
