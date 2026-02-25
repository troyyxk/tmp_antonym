#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_constraint_retrieval import (
    ConstraintAwareRetriever,
    DenseTopicalRetriever,
    Document,
    LexicalTopicalRetriever,
    OpenAICompatibleEmbedder,
    RetrievalConfig,
    SentenceTransformerEmbedder,
    constraint_compliance_rate,
)


def build_demo_documents() -> List[Document]:
    return [
        Document("hotel-1", "The hotel room was dirty and had a strong smell."),
        Document("hotel-2", "Clean rooms, fresh towels, and polite staff."),
        Document("hotel-3", "This hotel is spotless and quiet, great for families."),
        Document("hotel-4", "Good location but dirty bathroom and old carpet."),
        Document("hotel-5", "Guests repeatedly said the room was not dirty and felt very clean."),
        Document("meal-1", "This meal plan includes peanuts and dairy."),
        Document("meal-2", "Peanut-free vegetarian meal plan with high protein."),
        Document("meal-3", "Balanced meal plan without peanuts, suitable for allergies."),
        Document("meal-4", "This dish contains peanut sauce and sesame oil."),
        Document("phone-1", "Budget phone at $399 with solid battery life."),
        Document("phone-2", "Premium phone priced at $999 with flagship camera."),
        Document("phone-3", "Mid-range phone at $549, smooth performance."),
        Document("phone-4", "Cheap phone under $200 for light daily usage."),
    ]


def print_results(title: str, results: Iterable) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for idx, result in enumerate(results, start=1):
        print(
            f"{idx:>2}. {result.document.doc_id:<8} "
            f"topical={result.topical_score:.3f} "
            f"constraint={result.constraint_score:.3f} "
            f"final={result.final_score:.3f} "
            f"| {result.document.text}"
        )


def run_one_query(retriever: ConstraintAwareRetriever, query: str, final_k: int) -> None:
    print(f"\nQuery: {query}")
    baseline = retriever.baseline_search(query, top_k=final_k)
    constrained = retriever.search(
        query,
        RetrievalConfig(
            first_stage_k=20,
            final_k=final_k,
            topical_weight=0.65,
            constraint_weight=0.35,
            hard_filter=True,
            min_constraint_score=0.45,
        ),
    )

    print_results("Baseline (Topical only)", baseline)
    print_results("Constraint-aware (Topical + Constraint)", constrained)

    baseline_ccr = constraint_compliance_rate(
        baseline,
        min_constraint_score=0.45,
        query=query,
        scorer=retriever.constraint_scorer,
    )
    constrained_ccr = constraint_compliance_rate(
        constrained,
        min_constraint_score=0.45,
        query=query,
        scorer=retriever.constraint_scorer,
    )
    print(f"\nCCR baseline={baseline_ccr:.2%} | CCR constraint-aware={constrained_ccr:.2%}")


def build_topical_retriever(args: argparse.Namespace, documents: List[Document]):
    if args.topical_backend == "lexical":
        return LexicalTopicalRetriever(documents)

    if args.topical_backend == "bge":
        embedder = SentenceTransformerEmbedder(
            model_name=args.bge_model,
            device=args.bge_device,
        )
        return DenseTopicalRetriever(documents, embedder=embedder)

    if args.topical_backend == "openai":
        embedder = OpenAICompatibleEmbedder(
            model=args.openai_model,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
        )
        return DenseTopicalRetriever(documents, embedder=embedder)

    raise ValueError(f"Unknown backend: {args.topical_backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint-aware RAG retrieval demo")
    parser.add_argument("--query", type=str, default=None, help="Run a single query")
    parser.add_argument("--top-k", type=int, default=5, help="Final top-k results")
    parser.add_argument(
        "--topical-backend",
        type=str,
        default="lexical",
        choices=["lexical", "bge", "openai"],
        help="Stage-1 retriever backend",
    )
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--bge-device", type=str, default=None, help="e.g. cpu, cuda")
    parser.add_argument("--openai-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--openai-api-key", type=str, default=None)
    args = parser.parse_args()

    documents = build_demo_documents()
    try:
        topical_retriever = build_topical_retriever(args, documents)
    except Exception as exc:
        parser.exit(2, f"Failed to initialize backend '{args.topical_backend}': {exc}\n")

    retriever = ConstraintAwareRetriever(documents, topical_retriever=topical_retriever)

    if args.query:
        run_one_query(retriever, args.query, args.top_k)
        return

    demo_queries = [
        "reviews for hotels that are not dirty",
        "meal plans that do not include peanuts",
        "phones under $500",
    ]
    for query in demo_queries:
        run_one_query(retriever, query, args.top_k)


if __name__ == "__main__":
    main()
