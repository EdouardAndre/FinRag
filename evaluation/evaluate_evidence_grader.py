import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:
    from src.grade_documents import grade_retrieved_evidence
    from src.load_data import load_finqa_examples
    from src.retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from src.schemas import EvidenceGrade, FinancialExample, RetrievalResult
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from src.grade_documents import grade_retrieved_evidence
    from src.load_data import load_finqa_examples
    from src.retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from src.schemas import EvidenceGrade, FinancialExample, RetrievalResult


DEFAULT_DATASET_PATH = Path("data/FinQA/dataset/dev.json")
DEFAULT_RESULTS_PATH = Path("results/evidence_grader.csv")
DEFAULT_LIMIT = 100
DEFAULT_TOP_K = 10
RETRIEVAL_SCOPES = {"global", "example"}


@dataclass(frozen=True)
class EvidenceGraderEvaluation:
    example_id: str
    question: str
    gold_ids: set[str]
    retrieved_ids: list[str]
    grade: EvidenceGrade
    has_any_gold_evidence: bool
    has_all_gold_evidence: bool


def evidence_id(example_id: str, source_id: str) -> str:
    return f"{example_id}::{source_id}"


def evidence_ids_from_results(results: list[RetrievalResult]) -> list[str]:
    evidence_ids = []
    for result in results:
        evidence_ids.extend(
            evidence_id(result.chunk.example_id, source_id)
            for source_id in result.chunk.source_ids
        )
    return evidence_ids


def gold_evidence_ids(example: FinancialExample) -> set[str]:
    return {
        evidence_id(example.example_id, source_id)
        for source_id in example.gold_evidence.keys()
    }


def evaluate_example(
    example: FinancialExample,
    results: list[RetrievalResult],
) -> EvidenceGraderEvaluation:
    retrieved_ids = evidence_ids_from_results(results)
    retrieved_id_set = set(retrieved_ids)
    gold_ids = gold_evidence_ids(example)
    grade = grade_retrieved_evidence(example.question, results)

    return EvidenceGraderEvaluation(
        example_id=example.example_id,
        question=example.question,
        gold_ids=gold_ids,
        retrieved_ids=retrieved_ids,
        grade=grade,
        has_any_gold_evidence=bool(retrieved_id_set & gold_ids),
        has_all_gold_evidence=gold_ids.issubset(retrieved_id_set),
    )


def average(values: list[bool]) -> float:
    if not values:
        return 0.0

    return sum(float(value) for value in values) / len(values)


def summarize_evaluations(evaluations: list[EvidenceGraderEvaluation]) -> dict[str, float]:
    sufficient = [evaluation.grade.is_sufficient for evaluation in evaluations]
    has_any_gold = [evaluation.has_any_gold_evidence for evaluation in evaluations]
    has_all_gold = [evaluation.has_all_gold_evidence for evaluation in evaluations]

    false_sufficient = [
        evaluation.grade.is_sufficient and not evaluation.has_any_gold_evidence
        for evaluation in evaluations
    ]
    false_insufficient = [
        not evaluation.grade.is_sufficient and evaluation.has_any_gold_evidence
        for evaluation in evaluations
    ]

    return {
        "examples": float(len(evaluations)),
        "grader_sufficient_rate": average(sufficient),
        "has_any_gold_evidence_rate": average(has_any_gold),
        "has_all_gold_evidence_rate": average(has_all_gold),
        "false_sufficient_rate": average(false_sufficient),
        "false_insufficient_rate": average(false_insufficient),
    }


def save_evaluations(
    evaluations: list[EvidenceGraderEvaluation],
    path: str | Path,
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
                "is_sufficient",
                "needs_retry",
                "confidence",
                "missing_reason",
                "matched_signals",
                "missing_signals",
                "has_any_gold_evidence",
                "has_all_gold_evidence",
            ],
        )
        writer.writeheader()

        for evaluation in evaluations:
            writer.writerow(
                {
                    "example_id": evaluation.example_id,
                    "question": evaluation.question,
                    "gold_ids": " ".join(sorted(evaluation.gold_ids)),
                    "retrieved_ids": " ".join(evaluation.retrieved_ids),
                    "is_sufficient": evaluation.grade.is_sufficient,
                    "needs_retry": evaluation.grade.needs_retry,
                    "confidence": evaluation.grade.confidence,
                    "missing_reason": evaluation.grade.missing_reason or "",
                    "matched_signals": " ".join(evaluation.grade.matched_signals),
                    "missing_signals": " ".join(evaluation.grade.missing_signals),
                    "has_any_gold_evidence": evaluation.has_any_gold_evidence,
                    "has_all_gold_evidence": evaluation.has_all_gold_evidence,
                }
            )


def print_metrics(metrics: dict[str, float | str]) -> None:
    print("Evidence grader evaluation")
    print("=" * 80)
    for name, value in metrics.items():
        if isinstance(value, str):
            print(f"{name}: {value}")
        elif name == "examples":
            print(f"{name}: {int(value)}")
        else:
            print(f"{name}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristic evidence grading against gold IDs.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "adaptive"], default="dense")
    parser.add_argument("--scope", choices=sorted(RETRIEVAL_SCOPES), default="global")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=0)
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    examples = load_finqa_examples(path=args.dataset_path, limit=args.limit)
    chunks, index, metadata = load_retrieval_artifacts(limit=args.limit)
    bm25_index = build_bm25_index(chunks) if args.method in {"bm25", "hybrid", "adaptive"} else None

    evaluations = []
    for position, example in enumerate(examples, start=1):
        print(f"grading {position}/{len(examples)}: {example.example_id}")
        results, _ = retrieve_with_route(
            example.question,
            chunks=chunks,
            index=index,
            metadata=metadata,
            method=args.method,
            top_k=args.top_k,
            bm25_index=bm25_index,
            candidate_k=max(args.candidate_k, args.top_k),
            neighbor_window=args.neighbor_window,
            allowed_example_id=example.example_id if args.scope == "example" else None,
        )
        evaluations.append(evaluate_example(example, results))

    metrics = {"method": args.method, "scope": args.scope, **summarize_evaluations(evaluations)}
    print_metrics(metrics)
    save_evaluations(evaluations, args.results_path)
    print(f"grader evaluations saved to: {args.results_path}")


if __name__ == "__main__":
    main()
