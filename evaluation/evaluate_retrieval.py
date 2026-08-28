import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:
    from src.load_data import load_finqa_examples
    from src.retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from src.schemas import FinancialExample, RetrievalResult
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from src.load_data import load_finqa_examples
    from src.retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from src.schemas import FinancialExample, RetrievalResult


DEFAULT_DATASET_PATH = Path("data/FinQA/dataset/dev.json")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_LIMIT = 100
DEFAULT_MAX_K = 10
DEFAULT_CUTOFFS = (1, 3, 5, 10)


@dataclass(frozen=True)
class RetrievalEvaluation:
    example_id: str
    question: str
    gold_ids: set[str]
    retrieved_ids: list[str]
    reciprocal_rank: float


def source_ids_from_results(results: list[RetrievalResult]) -> list[str]:
    source_ids = []
    for result in results:
        source_ids.extend(result.chunk.source_ids)
    return source_ids


def hit_rate_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    return float(bool(set(retrieved_ids[:k]) & gold_ids))


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0

    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in gold_ids:
            return 1 / rank

    return 0.0


def evaluate_example(
    example: FinancialExample,
    chunks,
    index,
    metadata: dict,
    max_k: int,
    method: str,
    bm25_index,
    candidate_k: int,
    neighbor_window: int,
) -> RetrievalEvaluation:
    results, route = retrieve_with_route(
        example.question,
        chunks=chunks,
        index=index,
        metadata=metadata,
        top_k=max_k,
        method=method,
        bm25_index=bm25_index,
        candidate_k=candidate_k,
        neighbor_window=neighbor_window,
    )
    if not route.related_to_index:
        results = []

    retrieved_ids = source_ids_from_results(results)
    gold_ids = set(example.gold_evidence.keys())

    return RetrievalEvaluation(
        example_id=example.example_id,
        question=example.question,
        gold_ids=gold_ids,
        retrieved_ids=retrieved_ids,
        reciprocal_rank=reciprocal_rank(retrieved_ids, gold_ids),
    )


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def summarize_evaluations(
    evaluations: list[RetrievalEvaluation],
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    metrics = {
        "examples": float(len(evaluations)),
        "mrr": average([evaluation.reciprocal_rank for evaluation in evaluations]),
    }

    for cutoff in cutoffs:
        metrics[f"hit_rate@{cutoff}"] = average(
            [
                hit_rate_at_k(evaluation.retrieved_ids, evaluation.gold_ids, cutoff)
                for evaluation in evaluations
            ]
        )
        metrics[f"recall@{cutoff}"] = average(
            [
                recall_at_k(evaluation.retrieved_ids, evaluation.gold_ids, cutoff)
                for evaluation in evaluations
            ]
        )

    return metrics


def save_failures(
    evaluations: list[RetrievalEvaluation],
    path: str | Path,
    cutoff: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "example_id",
                "question",
                "gold_ids",
                "retrieved_ids",
            ],
        )
        writer.writeheader()

        for evaluation in evaluations:
            if hit_rate_at_k(evaluation.retrieved_ids, evaluation.gold_ids, cutoff):
                continue

            writer.writerow(
                {
                    "example_id": evaluation.example_id,
                    "question": evaluation.question,
                    "gold_ids": " ".join(sorted(evaluation.gold_ids)),
                    "retrieved_ids": " ".join(evaluation.retrieved_ids[:cutoff]),
                }
            )


def default_failures_path(method: str) -> Path:
    return DEFAULT_RESULTS_DIR / f"{method}_retrieval_failures.csv"


def print_metrics(metrics: dict[str, float | str]) -> None:
    print("Retrieval evaluation")
    print("=" * 80)
    for name, value in metrics.items():
        if isinstance(value, str):
            print(f"{name}: {value}")
        elif name == "examples":
            print(f"{name}: {int(value)}")
        else:
            print(f"{name}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against FinQA gold evidence IDs.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--neighbor-window", type=int, default=0)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "adaptive"], default="dense")
    parser.add_argument("--failure-cutoff", type=int, default=5)
    parser.add_argument("--failures-path")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    candidate_k = max(args.candidate_k, args.max_k)

    examples = load_finqa_examples(path=args.dataset_path, limit=args.limit)
    chunks, index, metadata = load_retrieval_artifacts(limit=args.limit)
    bm25_index = build_bm25_index(chunks) if args.method in {"bm25", "hybrid", "adaptive"} else None

    evaluations = [
        evaluate_example(
            example,
            chunks=chunks,
            index=index,
            metadata=metadata,
            max_k=args.max_k,
            method=args.method,
            bm25_index=bm25_index,
            candidate_k=candidate_k,
            neighbor_window=args.neighbor_window,
        )
        for example in examples
    ]

    metrics = summarize_evaluations(evaluations)
    metrics = {"method": args.method, **metrics}
    print_metrics(metrics)

    failures_path = Path(args.failures_path) if args.failures_path else default_failures_path(args.method)
    save_failures(evaluations, failures_path, cutoff=args.failure_cutoff)
    print(f"failures saved to: {failures_path}")


if __name__ == "__main__":
    main()
