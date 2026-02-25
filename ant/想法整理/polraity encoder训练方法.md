这是一份手把手的 **Encoder B (Constraint Encoder)** 训练实战指南。

我们将使用目前业界最成熟的 `sentence-transformers` 库。这比手写原生 PyTorch 循环要快得多，而且它内置了处理 Triplets（三元组）的最佳实践。

### 核心逻辑回顾

我们要训练的模型（DistilBERT），需要学会：

1. **看到**：`[Anchor]`（用户指令）
2. **拉近**：`[Positive]`（符合指令的文档）
3. **推远**：`[Hard Negative]`（与指令逻辑冲突/矛盾的文档）

---

### 第一步：环境准备

你需要安装这两个核心库：

```bash
pip install sentence-transformers datasets

```

---

### 第二步：数据准备 (The Secret Sauce)

这是成败的关键。我们不自己造数据，直接利用 **SNLI (Stanford Natural Language Inference)** 或 **MNLI** 数据集。

* SNLI 数据里自带了 `Premise` (前提), `Hypothesis` (假设) 和 `Label` (0:蕴含, 1:中立, 2:矛盾)。
* 我们要把它们转换成：`[Premise, Entailment_Hypothesis, Contradiction_Hypothesis]`。

新建文件 `train_encoder_b.py`，写入以下代码：

```python
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader

# 设置日志，让你看到训练进度
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_and_process_nli_data(dataset_name='snli', split='train', max_samples=10000):
    """
    读取 SNLI 数据集并转化为 (Anchor, Positive, Negative) 的三元组格式
    """
    logging.info(f"正在加载 {dataset_name} 数据集...")
    dataset = load_dataset(dataset_name, split=split)
    
    triplets = []
    # 我们需要构建一个字典来暂存数据: {premise: {label: hypothesis}}
    # 这样才能把同一个 Premise 对应的 Entailment 和 Contradiction 配对
    data_map = {}
    
    logging.info("正在处理数据构建三元组...")
    for row in dataset:
        premise = row['premise']
        hypothesis = row['hypothesis']
        label = row['label'] # 0: Entailment, 1: Neutral, 2: Contradiction
        
        if premise not in data_map:
            data_map[premise] = {}
        data_map[premise][label] = hypothesis
        
        # 只有当一个 Premise 同时拥有“蕴含(0)”和“矛盾(2)”时，我们才生成一个训练样本
        if 0 in data_map[premise] and 2 in data_map[premise]:
            # 构建 InputExample
            # 格式: texts=[Anchor, Positive, Hard_Negative]
            triplets.append(InputExample(texts=[
                premise,                    # Anchor (Query)
                data_map[premise][0],       # Positive (Entailment)
                data_map[premise][2]        # Hard Negative (Contradiction)
            ]))
            # 清理内存，这一对用完了就删掉
            del data_map[premise]
            
            if len(triplets) >= max_samples:
                break
                
    logging.info(f"构建完成，共生成 {len(triplets)} 个训练三元组。")
    return triplets

# 1. 准备训练数据
# 为了演示速度，我们只取 10,000 条。做正式实验时建议用 100,000+ 或跑全量 MNLI
train_examples = load_and_process_nli_data(dataset_name='snli', split='train', max_samples=10000)

# 2. 封装进 DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

```

---

### 第三步：模型定义 (Model Setup)

我们要用一个轻量级的 `distilbert-base-uncased` 作为底座。
关键在于：我们不需要它有多强的知识储备（那是 Encoder A 的事），我们需要它**反应快**且**懂逻辑**。

```python
# 3. 定义模型架构
model_name = 'distilbert-base-uncased'
word_embedding_model = models.Transformer(model_name, max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# 组装成 SentenceTransformer
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

logging.info(f"模型 {model_name} 加载完毕，准备训练...")

```

---

### 第四步：定义 Loss Function (核心)

我们要用 **`MultipleNegativesRankingLoss`**。

* 这是一个非常强大的 Loss。
* 当你输入 `[Anchor, Positive, Negative]` 时，它会做两件事：
1. 拉近 Anchor 和 Positive。
2. 推远 Anchor 和 Negative (这是你提供的 Hard Negative)。
3. **附赠功能**：它还会把同一个 Batch 里的其他行的样本也作为 Negative（In-batch Negatives），极大增加了训练效率。



```python
# 4. 定义 Loss
# 该 Loss 自动识别 InputExample 中的三个文本：(Anchor, Pos, Neg)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

```

---

### 第五步：开始训练 (Training Loop)

```python
# 5. 配置并开始训练
num_epochs = 1  # 演示用 1 个 epoch，正式训练建议 3-5 个
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% warmup
output_path = './output/constraint-encoder-v1'

logging.info("开始训练...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path,
    show_progress_bar=True
)

logging.info(f"训练完成！模型已保存至 {output_path}")

```

---

### 第六步：即刻验证 (Sanity Check)

训练完脚本会自动退出。现在新建一个脚本 `test_encoder.py`，看看它有没有学会“否定”和“反义”。

```python
from sentence_transformers import SentenceTransformer, util

# 加载刚才训练好的模型
model_path = './output/constraint-encoder-v1'
model = SentenceTransformer(model_path)

print("模型加载成功！开始测试 Constraint 能力...")

# 测试案例：否定逻辑
query = "I want a phone that is not expensive"
docs = [
    "This is a cheap phone",       # 应该近
    "This is an expensive phone",  # 应该远（虽然词重叠高）
    "This allows you to make calls" # 中性
]

# 编码
query_vec = model.encode(query)
doc_vecs = model.encode(docs)

# 计算相似度
scores = util.cos_sim(query_vec, doc_vecs)[0]

print(f"\nQuery: {query}")
for doc, score in zip(docs, scores):
    print(f"Score: {score:.4f} | Doc: {doc}")

# 你的预期结果：
# "cheap phone" 分数应该最高
# "expensive phone" 分数应该显著变低（哪怕它包含 expensive 这个词）

```

### 几个关键的“坑”与“技巧”

1. **显存不够怎么办？**
* 把 `batch_size` 从 16 调成 8 或 4。
* 把 `distilbert-base-uncased` 换成更小的 `prajjwal1/bert-tiny` (速度极快，但逻辑能力稍弱)。


2. **数据不够怎么办？**
* **Data Augmentation (数据增强)**：这是一个让导师眼前一亮的技巧。
* 你可以写个简单的规则：拿到一个 Positive 句子，随机插入 "not", "never" 把它变成 Negative，然后放到三元组的第三列。这样模型对 "not" 会极其敏感。


3. **怎么证明它比原来的好？**
* 在该脚本里，你可以同时加载一个原始的 `sentence-transformers/all-MiniLM-L6-v2`，对同样的 Case 跑分。
* 你会发现原始模型给 "expensive phone" 的分很高（因为它只看关键词），而你的模型分很低。**截图对比**，这就是论文里最漂亮的图表。



你可以先把这套代码跑通（大概只需要 10-20 分钟就能跑完 10000 条数据），确认环境没问题。如果你需要更进阶的“数据增强”代码，随时告诉我。