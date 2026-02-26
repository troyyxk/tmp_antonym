# Constraint-Aware Disentangled Retrieval for RAG  
## A Two-Stage Module for Hard Negative Constraints

**Draft Version:** 2026-02-25  
**Status:** Working Draft for EMNLP-style submission

---

## Abstract

Dense retrievers in Retrieval-Augmented Generation (RAG) systems are effective at topical relevance but often fail on logical constraints such as negation and antonymic conditions. This leads to high-scoring hard negatives, where retrieved documents are semantically related yet contradictory to user intent (e.g., retrieving peanut-containing items for queries requiring peanut-free content). We present a constraint-aware two-stage retrieval module that disentangles topical relevance and constraint compliance. Stage-1 performs standard recall with a pluggable retriever backend (lexical, local dense, or API-based dense embeddings). Stage-2 applies a dedicated constraint scorer and reranking/filtering strategy. We evaluate the module with Recall@K, Constraint Compliance Rate (CCR), and latency. On our current benchmark setup, the module consistently improves CCR from 0.667 to 1.000 and can improve Recall@3 from 0.833 to 1.000 under a local dense backend, while keeping practical latency. The system is designed for plug-and-play deployment and supports reproducible CSV/JSON report export for rapid research iteration.

---

## 1. Introduction

RAG systems rely on retrieval quality to ground generation. A recurring failure mode is **constraint blindness**: retrievers return documents that are topically relevant but violate user constraints. Typical examples include:

1. Query: “meal plans that do not include peanuts”  
Returned docs: “includes peanuts”
2. Query: “hotels that are not dirty”  
Returned docs: “dirty room”
3. Query: “phones under $500”  
Returned docs: premium devices above budget

This issue is tied to representation entanglement: topical similarity and logical consistency are compressed into a single similarity score. We address this by decomposing retrieval into two explicit decisions:

1. Is this document on-topic?
2. Does this document satisfy the constraint?

Our contribution is a practical, modular, and reproducible **constraint-aware retrieval module** for RAG that can be inserted between vector recall and generation.

---

## 2. Problem Formulation

Given a query \( q \) and corpus \( D \), retrieval typically ranks documents by topical similarity:

\[
s_t(q, d)
\]

However, we need an additional constraint compliance signal:

\[
s_c(q, d)
\]

Our final score is:

\[
s(q, d) = \alpha \cdot \hat{s}_t(q,d) + \beta \cdot s_c(q,d)
\]

where \( \hat{s}_t \) is normalized topical score and \( \alpha + \beta = 1 \).  
Optionally, a hard filter removes documents when \( s_c(q,d) < \tau \).

Goal:

1. Maintain or improve Recall@K.
2. Improve constraint faithfulness (CCR).
3. Keep latency low enough for RAG serving.

---

## 3. Method

### 3.1 Architecture

The module has two stages:

1. **Stage-1 Topical Recall**
2. **Stage-2 Constraint-aware Rerank/Filter**

This decouples relevance from logical compliance and reduces hard-negative leakage.

### 3.2 Stage-1: Pluggable Topical Retriever

We support:

1. Lexical baseline retriever.
2. Dense retriever via `sentence-transformers` (model name or local path).
3. Dense retriever via OpenAI-compatible embedding endpoint.

For dense retrieval, document and query embeddings are L2-normalized and ranked by dot product.

### 3.3 Stage-2: Constraint Scorer (Current PoC)

The current scorer is rule-based and captures:

1. Negation constraints: `not/without/no`, `不含/无`.
2. Inclusion constraints: `with/include/contain`, `包含/包括`.
3. Numeric bounds: `under/below/at most`, `低于/最多`, and lower-bound variants.

Implementation details include:

1. Basic morphological normalization.
2. Negated context handling for phrases like `peanut-free`.
3. Score clamping to \([0,1]\).

### 3.4 Fusion and Filtering

We combine normalized topical score and constraint score with fixed weights:

1. \( \alpha = 0.65 \)
2. \( \beta = 0.35 \)
3. Hard filter threshold \( \tau = 0.45 \)

---

## 4. Experimental Setup

### 4.1 Data

Current PoC benchmark includes a small but targeted hard-negative set with three query types:

1. Lexical negation (`not dirty`)
2. Attribute exclusion (`do not include peanuts`)
3. Numeric constraint (`under $500`)

### 4.2 Metrics

We report:

1. **Recall@K**: coverage of relevant documents.
2. **CCR (Constraint Compliance Rate)**: fraction of retrieved documents that satisfy constraints.
3. **Latency (ms)**: retrieval runtime per query.

### 4.3 Backends

1. `lexical` retriever
2. `bge` mode with local model path: `/data/xingkun/local_model/Llama-3.2-3B-Instruct` (CPU)

---

## 5. Results

### 5.1 Summary Results (K=3)

| Backend | Baseline Recall@3 | Baseline CCR | Baseline Latency (ms) | Constraint Recall@3 | Constraint CCR | Constraint Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lexical | 0.611 | 0.667 | 0.07 | 0.611 | 1.000 | 0.60 |
| local dense (`bge` mode) | 0.833 | 0.667 | 177.19 | 1.000 | 1.000 | 171.30 |

### 5.2 Observations

1. CCR improves consistently from 0.667 to 1.000 across tested backends.
2. Under local dense backend, Recall@3 also improves (0.833 to 1.000), indicating removal of contradictory yet high-similarity items.
3. Additional rerank/filter cost is small relative to dense retrieval latency.

---

## 6. Error Analysis

Current failure patterns:

1. Rule coverage gaps for complex compositional constraints.
2. Cases where topical retrieval introduces near-duplicates from unrelated domains that still pass weak constraints.
3. Multilingual and long-context negation scope remains underexplored.

---

## 7. Ablation and Discussion

### 7.1 Why Decoupling Works

Single-score retrieval conflates “about the same topic” with “satisfies the requirement.” Our two-stage design separates these signals, reducing logical conflicts in the final context fed to the generator.

### 7.2 Deployment Practicality

The module is backend-agnostic and can be inserted after vector recall without requiring index rebuild. This supports incremental rollout in existing RAG stacks.

### 7.3 Current Limitation

The Stage-2 scorer is still rule-based. While effective for PoC, robust generalization requires a trainable constraint encoder.

---

## 8. Future Work

1. Replace rule-based Stage-2 with a learned `Encoder B` trained on NLI-style triplets.
2. Expand evaluation to larger hard-negative benchmarks and multilingual settings.
3. Integrate with production vector DB and report end-to-end RAG faithfulness gains.
4. Compare against cross-encoder rerankers on quality/latency trade-offs.

---

## 9. Reproducibility

### 9.1 Demo

```bash
cd /home/xingkun/ant
python3 scripts/demo_constraint_rag.py \
  --topical-backend bge \
  --bge-model /data/xingkun/local_model/Llama-3.2-3B-Instruct \
  --bge-device cpu \
  --query 'phones under $500' \
  --top-k 3
```

### 9.2 Offline Evaluation and Report Export

```bash
python3 scripts/eval_retrieval_backends.py \
  --backends bge \
  --bge-model /data/xingkun/local_model/Llama-3.2-3B-Instruct \
  --bge-device cpu \
  --top-k 3 \
  --output-dir reports/paper_tables \
  --run-name local_bge_v1
```

Generated artifacts:

1. `summary_<run_name>.csv`
2. `per_query_<run_name>.csv`
3. `summary_<run_name>.json`
4. `report_<run_name>.json`

---

## 10. Ethical Considerations

Constraint-aware retrieval can improve factual and logical faithfulness, but rule-based filters may encode brittle assumptions. In high-stakes domains (medical/legal), explicit validation, dataset auditing, and human oversight are necessary.

---

## 11. Conclusion

We presented a constraint-aware, disentangled retrieval module for RAG that explicitly separates topical relevance and constraint compliance. Even with a lightweight Stage-2 scorer, results show strong gains in constraint faithfulness and competitive retrieval quality under both lexical and local dense backends. This establishes a practical foundation for a trainable constraint encoder and larger-scale evaluation.

---

## References (Draft)

1. Mrkšić et al. 2016. Counter-fitting Word Vectors to Linguistic Constraints.
2. Faruqui et al. 2015. Retrofitting Word Vectors to Semantic Lexicons.
3. Gao et al. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings.
4. Zou et al. 2023. Representation Engineering: A Top-Down Approach to AI Transparency.
5. Li et al. 2023. Inference-Time Intervention.
