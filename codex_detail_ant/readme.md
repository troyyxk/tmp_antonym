# Constraint-Aware RAG (Dual-View Retrieval)

这个项目实现了一个最小可运行版本的 Constraint-Aware RAG 检索链路：

- `Encoder A` 负责话题召回（Topic relevance）
- `Encoder B` 负责约束一致性（Constraint compliance）
- 在线使用 `Retrieve -> Filter/Rerank -> Generate` 思路，先召回再约束过滤

项目文档主线见：`想法整理/RAG方向.md`

## 目录结构

```text
ant/
  experiments/
    common.py
    poc_negation_gap.py
    build_triplets.py
    train_constraint_encoder.py
    eval_constraint_encoder.py
    retrieve_then_filter.py
    rag_eval.py
    rag_grid_search.py
  data/
    raw/
    processed/
  outputs/
    checkpoints/
    figures/
    reports/
  想法整理/
    RAG方向.md
```

## 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键流程（按顺序执行）

### 1) PoC：验证现有 embedding 的 negation/contradiction gap

```bash
python experiments/poc_negation_gap.py --dataset snli --split validation --max-samples 2000
```

产出：
- `outputs/reports/poc_negation_gap_summary.txt`
- `outputs/reports/poc_negation_gap_scores.csv`
- `outputs/figures/poc_negation_gap_hist.png`

### 2) 构建 triplets + smoke 数据

```bash
python experiments/build_triplets.py --dataset snli --train-max 20000 --val-max 3000
```

产出：
- `data/processed/train_triplets.jsonl`
- `data/processed/val_triplets.jsonl`
- `data/processed/smoke_eval.jsonl`
- `data/processed/demo_corpus.jsonl`

可选：构建更大规模第三轮评测集（300 queries）：

```bash
python experiments/build_constraint_benchmark.py \
  --output-file data/processed/constraint_benchmark_v1.jsonl \
  --num-negation 100 \
  --num-exclusion 100 \
  --num-numeric 100
```

### 3) 训练 Constraint Encoder

```bash
python experiments/train_constraint_encoder.py \
  --train-file data/processed/train_triplets.jsonl \
  --output-dir outputs/checkpoints/constraint-encoder-v1 \
  --epochs 1 \
  --batch-size 16
```

### 4) 离线评估（pairwise）

```bash
python experiments/eval_constraint_encoder.py \
  --eval-file data/processed/val_triplets.jsonl \
  --constraint-model outputs/checkpoints/constraint-encoder-v1
```

产出：
- `outputs/reports/constraint_eval_report.json`

### 5) 在线检索演示（retrieve + constraint rerank）

```bash
python experiments/retrieve_then_filter.py \
  --query "Find hotels that are not dirty" \
  --corpus-file data/processed/demo_corpus.jsonl \
  --constraint-model outputs/checkpoints/constraint-encoder-v1 \
  --alpha 0.7
```

### 6) RAG smoke 评估（CCR）

```bash
python experiments/rag_eval.py \
  --eval-file data/processed/smoke_eval.jsonl \
  --constraint-model outputs/checkpoints/constraint-encoder-v1 \
  --alpha 0.7 \
  --top-k 3
```

产出：
- `outputs/reports/rag_eval_report.json`

### 7) 第二轮：`alpha/tau` 网格搜索

```bash
python experiments/rag_grid_search.py \
  --eval-file data/processed/constraint_benchmark_v1.jsonl \
  --topic-model sentence-transformers/all-MiniLM-L6-v2 \
  --constraint-model outputs/checkpoints/constraint-encoder-v1 \
  --alphas=0.0,0.1,0.3,0.5,0.7,0.9,1.0 \
  --taus=-1.0,0.0,0.2,0.4,0.6 \
  --top-k 3 \
  --report-file outputs/reports/rag_grid_search_minilm.json
```

本地 Llama 主题模型版本：

```bash
python experiments/rag_grid_search.py \
  --eval-file data/processed/constraint_benchmark_v1.jsonl \
  --topic-model /data/xingkun/local_model/Llama-3.2-3B-Instruct \
  --constraint-model outputs/checkpoints/constraint-encoder-v1 \
  --alphas=0.0,0.1,0.3,0.5,0.7,0.9,1.0 \
  --taus=-1.0,0.0,0.2,0.4,0.6 \
  --top-k 3 \
  --max-queries 60 \
  --report-file outputs/reports/rag_grid_search_local_llama.json
```

## 指标解释

- `Pairwise Accuracy`: 在 `(query, positive, hard_negative)` 中，是否满足 `sim(q, pos) > sim(q, neg)`
- `CCR@k`: top-k 检索结果中满足约束的比例

## 注意事项

- 首次运行会下载 HuggingFace 数据和模型，网络较慢时耗时较长
- `constraint-model` 默认路径是 `outputs/checkpoints/constraint-encoder-v1`
- 如果显存不足，优先降低 `--batch-size` 或减少 `--train-max`
