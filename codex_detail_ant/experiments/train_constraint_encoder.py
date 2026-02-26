from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader

from common import CHECKPOINTS_DIR, PROCESSED_DIR, ensure_project_dirs, read_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight constraint encoder on triplets.")
    parser.add_argument("--train-file", type=str, default=str(PROCESSED_DIR / "train_triplets.jsonl"))
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=str, default=str(CHECKPOINTS_DIR / "constraint-encoder-v1"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_examples(rows: list[dict]) -> list[InputExample]:
    examples: list[InputExample] = []
    for row in rows:
        q = row.get("query", "").strip()
        p = row.get("positive", "").strip()
        n = row.get("hard_negative", "").strip()
        if q and p and n:
            examples.append(InputExample(texts=[q, p, n]))
    return examples


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)

    rows = read_jsonl(Path(args.train_file))
    examples = build_examples(rows)
    if not examples:
        raise RuntimeError(f"No valid training examples from {args.train_file}")

    transformer = models.Transformer(args.base_model, max_seq_length=args.max_seq_len)
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[transformer, pooling])

    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(len(loader) * args.epochs * 0.1)
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    print(f"Training complete. Saved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
