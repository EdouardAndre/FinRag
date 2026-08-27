import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from src.chunk import load_chunks
    from src.load_data import load_finqa_examples
    from src.schemas import Chunk, FinancialExample
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from src.chunk import load_chunks
    from src.load_data import load_finqa_examples
    from src.schemas import Chunk, FinancialExample


DEFAULT_DATASET_PATH = Path("data/FinQA/dataset/dev.json")
DEFAULT_CHUNKS_PATH = Path("data/chunks.json")
DEFAULT_PREDICTIONS_PATH = Path("results/answer_predictions.csv")
DEFAULT_OUTPUT_PATH = Path("results/error_analysis.csv")


@dataclass(frozen=True)
class ErrorAnalysisRow:
    example_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    failure_category: str
    gold_ids: list[str]
    retrieved_gold_ids: list[str]
    missing_gold_ids: list[str]
    retrieved_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    exact_match: bool
    numerical_match: bool
    citation_valid: bool
    insufficient_evidence: bool
    error: str


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def split_ids(value: str) -> list[str]:
    if not value:
        return []

    return value.split()


def load_predictions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def examples_by_id(path: str | Path, limit: int | None) -> dict[str, FinancialExample]:
    examples = load_finqa_examples(path=path, limit=limit)
    return {example.example_id: example for example in examples}


def chunks_by_id(path: str | Path) -> dict[str, Chunk]:
    chunks = load_chunks(path)
    return {chunk.chunk_id: chunk for chunk in chunks}


def retrieved_gold_ids_for_example(
    row: dict[str, str],
    example: FinancialExample,
    chunk_lookup: dict[str, Chunk],
) -> list[str]:
    gold_ids = set(example.gold_evidence.keys())
    retrieved_gold_ids = []

    for chunk_id in split_ids(row.get("retrieved_chunk_ids", "")):
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None or chunk.example_id != example.example_id:
            continue

        for source_id in chunk.source_ids:
            if source_id in gold_ids:
                retrieved_gold_ids.append(source_id)

    return sorted(set(retrieved_gold_ids))


def classify_failure(
    row: dict[str, str],
    gold_ids: list[str],
    retrieved_gold_ids: list[str],
) -> str:
    exact_match = parse_bool(row.get("exact_match", "False"))
    numerical_match = parse_bool(row.get("numerical_match", "False"))
    citation_valid = parse_bool(row.get("citation_valid", "False"))
    insufficient_evidence = parse_bool(row.get("insufficient_evidence", "False"))
    error = row.get("error", "")
    predicted_answer = row.get("predicted_answer", "")

    if error:
        return "pipeline_error"

    if exact_match or numerical_match:
        return "correct"

    if not citation_valid:
        return "citation_issue"

    if not retrieved_gold_ids:
        return "retrieval_failure"

    if len(retrieved_gold_ids) < len(gold_ids):
        return "partial_retrieval"

    if insufficient_evidence or not predicted_answer or predicted_answer.lower() in {"none", "insufficient evidence"}:
        return "over_abstention"

    return "calculation_or_generation_error"


def analyze_prediction(
    row: dict[str, str],
    example_lookup: dict[str, FinancialExample],
    chunk_lookup: dict[str, Chunk],
) -> ErrorAnalysisRow:
    example = example_lookup[row["example_id"]]
    gold_ids = sorted(example.gold_evidence.keys())
    retrieved_gold_ids = retrieved_gold_ids_for_example(row, example, chunk_lookup)
    missing_gold_ids = sorted(set(gold_ids) - set(retrieved_gold_ids))
    failure_category = classify_failure(row, gold_ids, retrieved_gold_ids)

    return ErrorAnalysisRow(
        example_id=row["example_id"],
        question=row["question"],
        gold_answer=row["gold_answer"],
        predicted_answer=row["predicted_answer"],
        failure_category=failure_category,
        gold_ids=gold_ids,
        retrieved_gold_ids=retrieved_gold_ids,
        missing_gold_ids=missing_gold_ids,
        retrieved_chunk_ids=split_ids(row.get("retrieved_chunk_ids", "")),
        cited_chunk_ids=split_ids(row.get("cited_chunk_ids", "")),
        exact_match=parse_bool(row.get("exact_match", "False")),
        numerical_match=parse_bool(row.get("numerical_match", "False")),
        citation_valid=parse_bool(row.get("citation_valid", "False")),
        insufficient_evidence=parse_bool(row.get("insufficient_evidence", "False")),
        error=row.get("error", ""),
    )


def analyze_predictions(
    predictions: list[dict[str, str]],
    example_lookup: dict[str, FinancialExample],
    chunk_lookup: dict[str, Chunk],
) -> list[ErrorAnalysisRow]:
    rows = []

    for prediction in predictions:
        if prediction["example_id"] not in example_lookup:
            continue

        rows.append(analyze_prediction(prediction, example_lookup, chunk_lookup))

    return rows


def save_error_analysis(rows: list[ErrorAnalysisRow], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "example_id",
                "failure_category",
                "question",
                "gold_answer",
                "predicted_answer",
                "gold_ids",
                "retrieved_gold_ids",
                "missing_gold_ids",
                "retrieved_chunk_ids",
                "cited_chunk_ids",
                "exact_match",
                "numerical_match",
                "citation_valid",
                "insufficient_evidence",
                "error",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "example_id": row.example_id,
                    "failure_category": row.failure_category,
                    "question": row.question,
                    "gold_answer": row.gold_answer,
                    "predicted_answer": row.predicted_answer,
                    "gold_ids": " ".join(row.gold_ids),
                    "retrieved_gold_ids": " ".join(row.retrieved_gold_ids),
                    "missing_gold_ids": " ".join(row.missing_gold_ids),
                    "retrieved_chunk_ids": " ".join(row.retrieved_chunk_ids),
                    "cited_chunk_ids": " ".join(row.cited_chunk_ids),
                    "exact_match": row.exact_match,
                    "numerical_match": row.numerical_match,
                    "citation_valid": row.citation_valid,
                    "insufficient_evidence": row.insufficient_evidence,
                    "error": row.error,
                }
            )


def print_summary(rows: list[ErrorAnalysisRow]) -> None:
    counts = Counter(row.failure_category for row in rows)

    print("Error analysis")
    print("=" * 80)
    print(f"examples: {len(rows)}")
    for category, count in counts.most_common():
        print(f"{category}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorize RAG answer failures using predictions and gold evidence.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--chunks-path", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--predictions-path", default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions_path)
    example_lookup = examples_by_id(args.dataset_path, limit=args.limit)
    chunk_lookup = chunks_by_id(args.chunks_path)

    rows = analyze_predictions(predictions, example_lookup, chunk_lookup)
    save_error_analysis(rows, args.output_path)
    print_summary(rows)
    print(f"error analysis saved to: {args.output_path}")


if __name__ == "__main__":
    main()
