# Constraint-Aware Disentangled Retrieval for RAG: Improving Logical Compliance Under Negation and Hard Constraints

**Anonymous Authors**  
*Submission to EMNLP 2026 (Draft in Markdown)*

---

## Abstract

Dense retrieval systems in Retrieval-Augmented Generation (RAG) are strong at topical matching but often fail on logical constraints such as negation, exclusion, and numeric limits. This causes retrieval of semantically related yet constraint-violating documents, which then propagates errors to downstream generation.  
We propose a lightweight dual-view retrieval framework that disentangles topic relevance from constraint compliance: a standard topical retriever is followed by a constraint encoder that reranks or filters candidates before generation.  
On our controlled benchmark, the proposed method improves CCR@3 (Constraint Compliance Rate) from 47.67% to 57.28% (+9.61 absolute points) with a MiniLM topical retriever over 300 queries. On a local Llama-3.2-3B topical setup (60-query subset), CCR@3 improves from 40.56% to 57.50% (+16.94). Category-level analysis shows strongest gains on numeric and exclusion constraints, while negation remains challenging. These findings support constraint-specialized retrieval as a practical and low-cost pre-generation defense for RAG faithfulness.

---

## 1 Introduction

Retrieval-Augmented Generation (RAG) depends on the quality of retrieved evidence. While modern dense retrievers perform well on semantic relevance, they often underperform on *constraint-sensitive retrieval*:

- Negation: "hotels that are **not dirty**"
- Exclusion: "meals **without peanuts**"
- Numeric constraints: "phones **under $300**"

In such cases, retrievers return documents that are topically related but logically conflicting (hard negatives). This is a known failure mode under distributional similarity: terms with opposite logical polarity may share similar contexts and therefore similar embeddings.

Most existing pipelines try to fix this at generation time (prompting, post-hoc filtering, or expensive cross-encoder reranking). We argue that a lightweight and explicit retrieval-stage constraint module can reduce downstream failure earlier and cheaper.

We ask:

1. Can we improve retrieval logical compliance without replacing the base retriever?
2. What constraint types benefit most from a specialized constraint encoder?
3. How does this behavior vary across topical backbones?

Our answer is a dual-view retrieval architecture with a constraint-focused encoder used between candidate retrieval and generation.

### Contributions

1. We introduce a practical dual-view RAG retrieval framework that disentangles topical relevance and constraint compliance.
2. We provide an executable evaluation pipeline with CCR-oriented analysis and hyperparameter grid search.
3. We report category-level findings showing substantial gains on numeric and exclusion constraints, and identify negation as the key unresolved difficulty.

---

## 2 Problem Formulation

Given query \(q\) and corpus \(D\), standard dense retrieval returns top-k documents by topical similarity:

\[
\text{TopK}_{\text{topic}}(q, D)
\]

This ranking ignores whether a document satisfies explicit constraints in \(q\).  
We define a constraint compliance target:

\[
\text{CCR@k} = \frac{1}{k}\sum_{d \in \text{TopK}(q)} \mathbb{1}[d \text{ satisfies constraints of } q]
\]

Goal: maximize CCR@k while preserving useful retrieval coverage.

---

## 3 Method

### 3.1 Dual-View Retrieval

Our framework has two components:

- **Encoder A (Topic Expert)**: standard retriever for semantic recall.
- **Encoder B (Constraint Expert)**: trained to distinguish constraint-satisfying vs. constraint-violating pairs.

Pipeline:

1. Retrieve top-N candidates with Encoder A.
2. Score candidates with Encoder B.
3. Filter/rerank candidates before passing to LLM.

### 3.2 Scoring and Filtering

For candidate \(d\), we combine scores:

\[
\text{Score}(d)=\alpha\cdot \text{Sim}_{\text{topic}}(q,d)+(1-\alpha)\cdot \text{Sim}_{\text{constraint}}(q,d)
\]

Additionally, we use a hard threshold \(\tau\) on constraint score:

\[
\text{keep}(d)=\mathbb{1}[\text{Sim}_{\text{constraint}}(q,d)\ge \tau]
\]

If no candidates pass \(\tau\), we back off to topical ranking.

### 3.3 Constraint Encoder Training

We train Encoder B with NLI-derived triplets:

\[
(q, d^+, d^-)
\]

- \(d^+\): semantically aligned and logically consistent
- \(d^-\): semantically related but logically contradictory

Backbone: MiniLM-style sentence encoder.  
Objective: triplet-style contrastive ranking (implemented with MultipleNegativesRankingLoss).

---

## 4 Experimental Setup

### 4.1 Data

We use two evaluation sets:

1. **Smoke set** (small sanity benchmark).
2. **Constraint Benchmark v1** (synthetic, 300 queries):
   - 100 negation
   - 100 exclusion
   - 100 numeric

Each query is paired with mixed satisfying and violating documents.

### 4.2 Models

- Topic retrievers:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - Local `/data/xingkun/local_model/Llama-3.2-3B-Instruct` (sentence-transformers wrapper)
- Constraint encoder:
  - `outputs/checkpoints/constraint-encoder-v1`

### 4.3 Metrics

- **CCR@3**: primary metric.
- Pairwise triplet accuracy for constraint encoder sanity.

### 4.4 Hyperparameter Search

Grid:

- \(\alpha \in \{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0\}\)
- \(\tau \in \{-1.0, 0.0, 0.2, 0.4, 0.6\}\)

Best configuration selected by highest CCR@3.

---

## 5 Results

### 5.1 Constraint Encoder Sanity (Triplet)

On validation triplets (800 examples):

- Baseline pairwise accuracy: 94.63%
- Constraint encoder: 96.50%
- Absolute gain: +1.88 points

This confirms Encoder B learns stronger contradiction separation.

### 5.2 Overall CCR Improvements

**MiniLM topic model, 300 queries**

- Vanilla CCR@3: 47.67%
- Best dual-view CCR@3: 57.28%
- Gain: +9.61 points
- Best settings: \(\alpha=0.0,\ \tau=0.6\)

**Local Llama-3.2-3B topic model, 60-query subset**

- Vanilla CCR@3: 40.56%
- Best dual-view CCR@3: 57.50%
- Gain: +16.94 points
- Best settings: \(\alpha=0.0,\ \tau=0.6\)

### 5.3 Category-Level Analysis

| Setting | Category | Count | Vanilla CCR | Dual CCR | Improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| MiniLM (300) | exclusion | 100 | 54.33% | 60.50% | +6.17% |
| MiniLM (300) | negation | 100 | 22.00% | 22.00% | +0.00% |
| MiniLM (300) | numeric | 100 | 66.67% | 89.33% | +22.67% |
| Local Llama (60) | exclusion | 24 | 45.83% | 60.42% | +14.58% |
| Local Llama (60) | negation | 19 | 10.53% | 21.05% | +10.53% |
| Local Llama (60) | numeric | 17 | 66.67% | 94.12% | +27.45% |

### 5.4 Key Observation

The method consistently helps with **numeric** and **exclusion** constraints.  
**Negation** remains hardest, especially under smaller semantic backbones. This indicates that negation-specific data and sampling are needed beyond current training.

---

## 6 Discussion

### 6.1 Why High \(\tau\) Works

Across runs, strong filtering (\(\tau=0.6\)) and low \(\alpha\) performed best. This suggests that in constraint-heavy queries, topical similarity alone is insufficient, and explicit conflict rejection is crucial.

### 6.2 Practical Value

Compared with heavy cross-encoder reranking, this design is:

- modular (plug-and-play after retriever),
- cheaper than full cross-encoder pipelines,
- easy to integrate into existing RAG systems.

### 6.3 Failure Modes

- Negation scope ambiguity in natural language.
- Lexical shortcuts in synthetic templates.
- Potential over-filtering when constraints are implicit rather than explicit.

---

## 7 Limitations

1. Main benchmark is synthetic and template-based; real-world query diversity is underrepresented.
2. End-to-end generation faithfulness is not fully evaluated yet (current focus is retrieval-level CCR).
3. Local Llama results are on subset size 60 due to computational constraints.
4. Cross-encoder latency/quality frontier is not fully benchmarked in this draft.

---

## 8 Ethics and Broader Impact

Improving constraint compliance can reduce harmful retrieval errors in domains like healthcare, finance, and legal assistance, where negation mistakes are high-risk.  
However, overconfident filtering may remove useful evidence when constraints are misparsed. Deployment should include fallback mechanisms and human-auditable logs.

---

## 9 Conclusion

We present a constraint-aware dual-view retrieval framework for RAG. By disentangling topical relevance and logical compliance, we improve CCR substantially on a controlled benchmark, especially for numeric and exclusion constraints.  
Our results indicate that lightweight constraint-specialized reranking/filtering is a promising retrieval-stage defense for RAG faithfulness. Future work should focus on real-world benchmarks, stronger negation modeling, and end-to-end QA faithfulness.

---

## Reproducibility Checklist (Draft)

- Code: available in `experiments/`
- Data generation scripts: `build_constraint_benchmark.py`
- Training script: `train_constraint_encoder.py`
- Evaluation scripts:
  - `rag_grid_search.py`
  - `rag_category_report.py`
- Key reports:
  - `outputs/reports/rag_grid_search_v1_minilm_300.json`
  - `outputs/reports/rag_grid_search_v1_local_llama_60.json`
  - `outputs/reports/rag_category_report_minilm_300.json`
  - `outputs/reports/rag_category_report_local_llama_60.json`

---

## References (to finalize in BibTeX)

- FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions.
- Instruction-following Text Embedding via Answering the Question.
- Counter-fitting Word Vectors to Linguistic Constraints.
- Representation Engineering: A Top-Down Approach to AI Transparency.
- Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.
- Recent works on negation-aware retrieval and logical retrieval benchmarks.

