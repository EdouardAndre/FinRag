import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:
    from src.generate import generate_answer
    from src.load_data import load_finqa_examples
    from src.retrieve import build_bm25_index, load_retrieval_artifacts, retrieve
    from src.schemas import FinancialExample, RAGAnswer, RetrievalResult
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from src.generate import generate_answer
    from src.load_data import load_finqa_examples
    from src.retrieve import build_bm25_index, load_retrieval_artifacts, retrieve
    from src.schemas import FinancialExample, RAGAnswer, RetrievalResult


DEFAULT_DATASET_PATH = Path("data/FinQA/dataset/dev.json")
DEFAULT_RESULTS_PATH = Path("results/answer_predictions.csv")
DEFAULT_LIMIT = 20
DEFAULT_TOP_K = 5
DEFAULT_MODEL = "mistral-small-latest"
NUMERIC_TOLERANCE = 1e-3


@dataclass(frozen=True)
class AnswerEvaluation:
    example_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    exact_match: bool
    numerical_match: bool
    citation_valid: bool
    insufficient_evidence: bool
    error: str | None
    retrieved_chunk_ids: list[str]
    cited_chunk_ids: list[str]


def normalize_answer(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[,$%]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_numbers(text: str) -> list[float]:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return [float(match.replace(",", "")) for match in matches]


def gold_numeric_candidates(gold_answer: str, gold_value: float | int | str | None) -> list[float]:
    candidates = extract_numbers(gold_answer)

    if gold_value is not None:
        numeric_gold_value = float(gold_value)
        candidates.append(numeric_gold_value)

        if "%" in gold_answer:
            candidates.append(numeric_gold_value * 100)

    return candidates


def exact_match(predicted: str, gold: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(gold)


def numbers_match(predicted_value: float, gold_value: float) -> bool:
    absolute_error = abs(predicted_value - gold_value)
    relative_tolerance = abs(gold_value) * NUMERIC_TOLERANCE
    return absolute_error <= max(NUMERIC_TOLERANCE, relative_tolerance)


def numerical_match(
    predicted: str,
    gold_answer: str,
    gold_value: float | int | str | None,
) -> bool:
    predicted_values = extract_numbers(predicted)
    gold_values = gold_numeric_candidates(gold_answer, gold_value)

    if not predicted_values or not gold_values:
        return False

    return any(
        numbers_match(predicted_value, expected_value)
        for predicted_value in predicted_values
        for expected_value in gold_values
    )


def citations_are_valid(answer: RAGAnswer, results: list[RetrievalResult]) -> bool:
    retrieved_chunk_ids = {result.chunk.chunk_id for result in results}
    return all(citation.chunk_id in retrieved_chunk_ids for citation in answer.citations)


def evaluate_answer(
    example: FinancialExample,
    answer: RAGAnswer,
    results: list[RetrievalResult],
) -> AnswerEvaluation:
    return AnswerEvaluation(
        example_id=example.example_id,
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=answer.answer,
        exact_match=exact_match(answer.answer, example.answer),
        numerical_match=numerical_match(
            answer.answer,
            example.answer,
            example.metadata.get("gold_numeric_answer"),
        ),
        citation_valid=citations_are_valid(answer, results),
        insufficient_evidence=answer.insufficient_evidence,
        error=None,
        retrieved_chunk_ids=[result.chunk.chunk_id for result in results],
        cited_chunk_ids=[citation.chunk_id for citation in answer.citations],
    )


def failed_evaluation(example: FinancialExample, error: Exception) -> AnswerEvaluation:
    return AnswerEvaluation(
        example_id=example.example_id,
        question=example.question,
        gold_answer=example.answer,
        predicted_answer="",
        exact_match=False,
        numerical_match=False,
        citation_valid=False,
        insufficient_evidence=False,
        error=str(error),
        retrieved_chunk_ids=[],
        cited_chunk_ids=[],
    )


def average(values: list[bool]) -> float:
    if not values:
        return 0.0

    return sum(float(value) for value in values) / len(values)


def summarize_evaluations(evaluations: list[AnswerEvaluation]) -> dict[str, float]:
    completed = [evaluation for evaluation in evaluations if evaluation.error is None]

    return {
        "examples": float(len(evaluations)),
        "completed": float(len(completed)),
        "exact_match": average([evaluation.exact_match for evaluation in completed]),
        "numerical_match": average([evaluation.numerical_match for evaluation in completed]),
        "citation_validity": average([evaluation.citation_valid for evaluation in completed]),
        "insufficient_evidence_rate": average(
            [evaluation.insufficient_evidence for evaluation in completed]
        ),
        "error_rate": 1 - (len(completed) / len(evaluations) if evaluations else 0.0),
    }


def save_predictions(evaluations: list[AnswerEvaluation], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "example_id",
                "question",
                "gold_answer",
                "predicted_answer",
                "exact_match",
                "numerical_match",
                "citation_valid",
                "insufficient_evidence",
                "error",
                "retrieved_chunk_ids",
                "cited_chunk_ids",
            ],
        )
        writer.writeheader()

        for evaluation in evaluations:
            writer.writerow(
                {
                    "example_id": evaluation.example_id,
                    "question": evaluation.question,
                    "gold_answer": evaluation.gold_answer,
                    "predicted_answer": evaluation.predicted_answer,
                    "exact_match": evaluation.exact_match,
                    "numerical_match": evaluation.numerical_match,
                    "citation_valid": evaluation.citation_valid,
                    "insufficient_evidence": evaluation.insufficient_evidence,
                    "error": evaluation.error or "",
                    "retrieved_chunk_ids": " ".join(evaluation.retrieved_chunk_ids),
                    "cited_chunk_ids": " ".join(evaluation.cited_chunk_ids),
                }
            )


def print_metrics(metrics: dict[str, float | str]) -> None:
    print("Answer evaluation")
    print("=" * 80)
    for name, value in metrics.items():
        if isinstance(value, str):
            print(f"{name}: {value}")
        elif name in {"examples", "completed"}:
            print(f"{name}: {int(value)}")
        else:
            print(f"{name}: {value:.4f}")


def evaluate_answers(
    examples: list[FinancialExample],
    method: str,
    top_k: int,
    candidate_k: int,
    model: str,
) -> list[AnswerEvaluation]:
    chunks, index, metadata = load_retrieval_artifacts(limit=len(examples))
    bm25_index = build_bm25_index(chunks) if method in {"bm25", "hybrid"} else None
    evaluations = []

    for position, example in enumerate(examples, start=1):
        print(f"evaluating {position}/{len(examples)}: {example.example_id}")
        try:
            results = retrieve(
                example.question,
                chunks=chunks,
                index=index,
                metadata=metadata,
                top_k=top_k,
                method=method,
                bm25_index=bm25_index,
                candidate_k=max(candidate_k, top_k),
            )
            answer = generate_answer(example.question, results, model=model)
            evaluations.append(evaluate_answer(example, answer, results))
        except Exception as error:
            evaluations.append(failed_evaluation(example, error))

    return evaluations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated RAG answers against FinQA answers.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--predictions-path", default=str(DEFAULT_RESULTS_PATH))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    examples = load_finqa_examples(path=args.dataset_path, limit=args.limit)
    evaluations = evaluate_answers(
        examples,
        method=args.method,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        model=args.model,
    )
    metrics = {"method": args.method, **summarize_evaluations(evaluations)}
    print_metrics(metrics)
    save_predictions(evaluations, args.predictions_path)
    print(f"predictions saved to: {args.predictions_path}")


if __name__ == "__main__":
    main()
