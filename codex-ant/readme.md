# Constraint-Aware RAG Retriever (PoC)

这个仓库现在聚焦一个可交付方向：
`RAG 约束检索模块`，用于解决 Dense Retrieval 在否定词/反义约束上的误召回问题。

## What is implemented

- Two-stage retrieval pipeline: Stage 1 topical recall + Stage 2 constraint-aware rerank/filter.
- Stage-1 backends: lexical retriever, BGE/sentence-transformers dense retriever, OpenAI-compatible dense retriever.
- Constraint support in PoC: lexical negation (`not dirty`, `without peanuts`, `不含花生`), include constraints, numeric bounds.
- Evaluation helper: `constraint_compliance_rate` (CCR).

## Quick start

```bash
cd /home/xingkun/ant
python3 scripts/demo_constraint_rag.py
```

Run one query:

```bash
python3 scripts/demo_constraint_rag.py --query "reviews for hotels that are not dirty" --top-k 5
```

Use local BGE (dense stage-1):

```bash
pip install sentence-transformers
python3 scripts/demo_constraint_rag.py --topical-backend bge --bge-model BAAI/bge-small-en-v1.5
```

Use a local HF model directory (no download, e.g. your `/data/xingkun/local_model`):

```bash
python3 scripts/demo_constraint_rag.py \
  --topical-backend bge \
  --bge-model /data/xingkun/local_model/Llama-3.2-3B-Instruct \
  --bge-device cpu \
  --query 'phones under $500'
```

Use OpenAI-compatible embeddings (dense stage-1):

```bash
export OPENAI_API_KEY=your_key
python3 scripts/demo_constraint_rag.py --topical-backend openai --openai-model text-embedding-3-small
```

Offline evaluation (compare backends on same query set):

```bash
python3 scripts/eval_retrieval_backends.py --backends lexical --top-k 3 --show-per-query
```

Offline evaluation with local model backend:

```bash
python3 scripts/eval_retrieval_backends.py \
  --backends bge \
  --bge-model /data/xingkun/local_model/Llama-3.2-3B-Instruct \
  --bge-device cpu \
  --top-k 3
```

Export control:

```bash
# Custom report directory/name
python3 scripts/eval_retrieval_backends.py \
  --backends lexical,bge \
  --output-dir reports/paper_tables \
  --run-name local_benchmark_v1

# Disable export (console only)
python3 scripts/eval_retrieval_backends.py --backends lexical --no-export
```

By default, reports are exported to `reports/`:
- `summary_<run_name>.csv`
- `per_query_<run_name>.csv`
- `summary_<run_name>.json`
- `report_<run_name>.json`

Try all backends (available ones run, unavailable ones are skipped):

```bash
python3 scripts/eval_retrieval_backends.py --backends lexical,bge,openai --top-k 3
```

## Code layout

- `rag_constraint_retrieval/types.py`: document/result schema
- `rag_constraint_retrieval/text.py`: tokenizer + cosine similarity
- `rag_constraint_retrieval/topical.py`: stage-1 topical recall (`lexical` + `dense`)
- `rag_constraint_retrieval/embeddings.py`: pluggable embedder backends (BGE/OpenAI-compatible)
- `rag_constraint_retrieval/constraints.py`: constraint parser and scorer
- `rag_constraint_retrieval/pipeline.py`: end-to-end retrieval pipeline
- `rag_constraint_retrieval/metrics.py`: CCR metric
- `scripts/demo_constraint_rag.py`: runnable demo
- `scripts/eval_retrieval_backends.py`: offline benchmark (`Recall@K + CCR + latency`)

## Delivery path (next)

1. Expand the eval set (more hard negatives, multilingual, numeric constraints) and export CSV/JSON reports.
2. Replace rule-based scorer with trainable `Encoder B` (NLI triplet training).
3. Benchmark on hard-negative retrieval set and report `Recall@K + CCR + latency` under dense backends.

## Research notes

Original brainstorming and paper notes are preserved under:
`/home/xingkun/ant/想法整理/`
