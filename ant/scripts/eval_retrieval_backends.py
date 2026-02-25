#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

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


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant_doc_ids: Sequence[str]


@dataclass(frozen=True)
class BackendMetrics:
    backend: str
    baseline_recall: float
    baseline_ccr: float
    baseline_latency_ms: float
    constrained_recall: float
    constrained_ccr: float
    constrained_latency_ms: float


@dataclass(frozen=True)
class PerQueryMetrics:
    backend: str
    query: str
    relevant_doc_ids: Sequence[str]
    baseline_recall: float
    baseline_ccr: float
    baseline_latency_ms: float
    baseline_doc_ids: Sequence[str]
    constrained_recall: float
    constrained_ccr: float
    constrained_latency_ms: float
    constrained_doc_ids: Sequence[str]


@dataclass(frozen=True)
class BackendEvalResult:
    summary: BackendMetrics
    per_query: Sequence[PerQueryMetrics]


def build_documents() -> List[Document]:
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


def build_eval_cases() -> List[EvalCase]:
    return [
        EvalCase(
            query="reviews for hotels that are not dirty",
            relevant_doc_ids=["hotel-2", "hotel-3", "hotel-5"],
        ),
        EvalCase(
            query="meal plans that do not include peanuts",
            relevant_doc_ids=["meal-2", "meal-3"],
        ),
        EvalCase(
            query="phones under $500",
            relevant_doc_ids=["phone-1", "phone-4"],
        ),
    ]


def build_retriever(args: argparse.Namespace, backend: str, documents: List[Document]) -> ConstraintAwareRetriever:
    if backend == "lexical":
        return ConstraintAwareRetriever(documents, topical_retriever=LexicalTopicalRetriever(documents))

    if backend == "bge":
        embedder = SentenceTransformerEmbedder(model_name=args.bge_model, device=args.bge_device)
        return ConstraintAwareRetriever(documents, topical_retriever=DenseTopicalRetriever(documents, embedder))

    if backend == "openai":
        embedder = OpenAICompatibleEmbedder(
            model=args.openai_model,
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
        )
        return ConstraintAwareRetriever(documents, topical_retriever=DenseTopicalRetriever(documents, embedder))

    raise ValueError(f"Unsupported backend: {backend}")


def recall_at_k(results, relevant_doc_ids: Sequence[str]) -> float:
    if not relevant_doc_ids:
        return 0.0
    retrieved = {result.document.doc_id for result in results}
    hit = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved)
    return hit / len(relevant_doc_ids)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def evaluate_backend(
    backend: str,
    retriever: ConstraintAwareRetriever,
    cases: Sequence[EvalCase],
    top_k: int,
    first_stage_k: int,
    show_per_query: bool,
) -> BackendEvalResult:
    baseline_recall_scores: List[float] = []
    baseline_ccr_scores: List[float] = []
    baseline_latency_values: List[float] = []

    constrained_recall_scores: List[float] = []
    constrained_ccr_scores: List[float] = []
    constrained_latency_values: List[float] = []
    per_query_results: List[PerQueryMetrics] = []

    cfg = RetrievalConfig(
        first_stage_k=first_stage_k,
        final_k=top_k,
        topical_weight=0.65,
        constraint_weight=0.35,
        hard_filter=True,
        min_constraint_score=0.45,
    )

    for case in cases:
        t0 = time.perf_counter()
        baseline = retriever.baseline_search(case.query, top_k=top_k)
        t1 = time.perf_counter()
        constrained = retriever.search(case.query, config=cfg)
        t2 = time.perf_counter()

        baseline_recall = recall_at_k(baseline, case.relevant_doc_ids)
        constrained_recall = recall_at_k(constrained, case.relevant_doc_ids)
        baseline_ccr = constraint_compliance_rate(
            baseline,
            min_constraint_score=0.45,
            query=case.query,
            scorer=retriever.constraint_scorer,
        )
        constrained_ccr = constraint_compliance_rate(
            constrained,
            min_constraint_score=0.45,
            query=case.query,
            scorer=retriever.constraint_scorer,
        )

        baseline_recall_scores.append(baseline_recall)
        constrained_recall_scores.append(constrained_recall)
        baseline_ccr_scores.append(baseline_ccr)
        constrained_ccr_scores.append(constrained_ccr)
        baseline_latency_values.append((t1 - t0) * 1000.0)
        constrained_latency_values.append((t2 - t1) * 1000.0)

        if show_per_query:
            print(
                f"[{backend}] {case.query}\n"
                f"  baseline:   Recall@{top_k}={baseline_recall:.3f}, CCR={baseline_ccr:.3f}, latency_ms={(t1 - t0) * 1000.0:.2f}\n"
                f"  constrained: Recall@{top_k}={constrained_recall:.3f}, CCR={constrained_ccr:.3f}, latency_ms={(t2 - t1) * 1000.0:.2f}"
            )

        per_query_results.append(
            PerQueryMetrics(
                backend=backend,
                query=case.query,
                relevant_doc_ids=list(case.relevant_doc_ids),
                baseline_recall=baseline_recall,
                baseline_ccr=baseline_ccr,
                baseline_latency_ms=(t1 - t0) * 1000.0,
                baseline_doc_ids=[item.document.doc_id for item in baseline],
                constrained_recall=constrained_recall,
                constrained_ccr=constrained_ccr,
                constrained_latency_ms=(t2 - t1) * 1000.0,
                constrained_doc_ids=[item.document.doc_id for item in constrained],
            )
        )

    summary = BackendMetrics(
        backend=backend,
        baseline_recall=mean(baseline_recall_scores),
        baseline_ccr=mean(baseline_ccr_scores),
        baseline_latency_ms=mean(baseline_latency_values),
        constrained_recall=mean(constrained_recall_scores),
        constrained_ccr=mean(constrained_ccr_scores),
        constrained_latency_ms=mean(constrained_latency_values),
    )
    return BackendEvalResult(summary=summary, per_query=per_query_results)


def print_summary_table(metrics: Sequence[BackendMetrics], top_k: int) -> None:
    print("\nSummary")
    print("-------")
    header = (
        f"| Backend | Baseline Recall@{top_k} | Baseline CCR | Baseline Latency (ms) | "
        f"Constraint Recall@{top_k} | Constraint CCR | Constraint Latency (ms) |"
    )
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    print(header)
    print(divider)
    for row in metrics:
        print(
            f"| {row.backend} | {row.baseline_recall:.3f} | {row.baseline_ccr:.3f} | {row.baseline_latency_ms:.2f} | "
            f"{row.constrained_recall:.3f} | {row.constrained_ccr:.3f} | {row.constrained_latency_ms:.2f} |"
        )


def parse_backends(raw: str) -> List[str]:
    out = []
    for item in raw.split(","):
        value = item.strip().lower()
        if value:
            out.append(value)
    return out


def write_reports(
    output_dir: Path,
    run_name: str,
    top_k: int,
    summaries: Sequence[BackendMetrics],
    per_query_rows: Sequence[PerQueryMetrics],
    args: argparse.Namespace,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: List[Path] = []

    summary_csv = output_dir / f"summary_{run_name}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "backend",
                f"baseline_recall@{top_k}",
                "baseline_ccr",
                "baseline_latency_ms",
                f"constraint_recall@{top_k}",
                "constraint_ccr",
                "constraint_latency_ms",
            ]
        )
        for item in summaries:
            writer.writerow(
                [
                    item.backend,
                    f"{item.baseline_recall:.6f}",
                    f"{item.baseline_ccr:.6f}",
                    f"{item.baseline_latency_ms:.6f}",
                    f"{item.constrained_recall:.6f}",
                    f"{item.constrained_ccr:.6f}",
                    f"{item.constrained_latency_ms:.6f}",
                ]
            )
    created_files.append(summary_csv)

    per_query_csv = output_dir / f"per_query_{run_name}.csv"
    with per_query_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "backend",
                "query",
                "relevant_doc_ids",
                f"baseline_recall@{top_k}",
                "baseline_ccr",
                "baseline_latency_ms",
                "baseline_doc_ids",
                f"constraint_recall@{top_k}",
                "constraint_ccr",
                "constraint_latency_ms",
                "constraint_doc_ids",
            ]
        )
        for row in per_query_rows:
            writer.writerow(
                [
                    row.backend,
                    row.query,
                    "|".join(row.relevant_doc_ids),
                    f"{row.baseline_recall:.6f}",
                    f"{row.baseline_ccr:.6f}",
                    f"{row.baseline_latency_ms:.6f}",
                    "|".join(row.baseline_doc_ids),
                    f"{row.constrained_recall:.6f}",
                    f"{row.constrained_ccr:.6f}",
                    f"{row.constrained_latency_ms:.6f}",
                    "|".join(row.constrained_doc_ids),
                ]
            )
    created_files.append(per_query_csv)

    summary_json = output_dir / f"summary_{run_name}.json"
    summary_payload = [
        {
            "backend": item.backend,
            f"baseline_recall@{top_k}": item.baseline_recall,
            "baseline_ccr": item.baseline_ccr,
            "baseline_latency_ms": item.baseline_latency_ms,
            f"constraint_recall@{top_k}": item.constrained_recall,
            "constraint_ccr": item.constrained_ccr,
            "constraint_latency_ms": item.constrained_latency_ms,
        }
        for item in summaries
    ]
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    created_files.append(summary_json)

    report_json = output_dir / f"report_{run_name}.json"
    full_payload = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": top_k,
        "first_stage_k": args.first_stage_k,
        "backends": parse_backends(args.backends),
        "config": {
            "bge_model": args.bge_model,
            "bge_device": args.bge_device,
            "openai_model": args.openai_model,
            "openai_base_url": args.openai_base_url,
        },
        "summary": summary_payload,
        "per_query": [
            {
                "backend": row.backend,
                "query": row.query,
                "relevant_doc_ids": list(row.relevant_doc_ids),
                f"baseline_recall@{top_k}": row.baseline_recall,
                "baseline_ccr": row.baseline_ccr,
                "baseline_latency_ms": row.baseline_latency_ms,
                "baseline_doc_ids": list(row.baseline_doc_ids),
                f"constraint_recall@{top_k}": row.constrained_recall,
                "constraint_ccr": row.constrained_ccr,
                "constraint_latency_ms": row.constrained_latency_ms,
                "constraint_doc_ids": list(row.constrained_doc_ids),
            }
            for row in per_query_rows
        ],
    }
    report_json.write_text(json.dumps(full_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    created_files.append(report_json)

    return created_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval across retrieval backends")
    parser.add_argument("--backends", type=str, default="lexical,bge,openai")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--first-stage-k", type=int, default=20)
    parser.add_argument("--show-per-query", action="store_true")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-export", action="store_true")

    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--bge-device", type=str, default=None, help="e.g. cpu, cuda")
    parser.add_argument("--openai-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--openai-api-key", type=str, default=None)
    args = parser.parse_args()

    requested_backends = parse_backends(args.backends)
    if not requested_backends:
        parser.exit(2, "No backend requested.\n")

    documents = build_documents()
    cases = build_eval_cases()

    metrics: List[BackendMetrics] = []
    per_query_rows: List[PerQueryMetrics] = []
    for backend in requested_backends:
        try:
            retriever = build_retriever(args, backend, documents)
        except Exception as exc:
            print(f"[skip] backend={backend}: {exc}")
            continue

        result = evaluate_backend(
            backend=backend,
            retriever=retriever,
            cases=cases,
            top_k=args.top_k,
            first_stage_k=args.first_stage_k,
            show_per_query=args.show_per_query,
        )
        metrics.append(result.summary)
        per_query_rows.extend(result.per_query)

    if not metrics:
        parser.exit(2, "No backend initialized successfully.\n")

    print_summary_table(metrics, top_k=args.top_k)

    if not args.no_export:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        created_files = write_reports(
            output_dir=output_dir,
            run_name=run_name,
            top_k=args.top_k,
            summaries=metrics,
            per_query_rows=per_query_rows,
            args=args,
        )
        print("\nExported")
        print("--------")
        for path in created_files:
            print(path)


if __name__ == "__main__":
    main()
