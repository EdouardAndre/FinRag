import json
from pathlib import Path
from typing import Any

try:
    from .schemas import FinancialExample
except ImportError:
    from schemas import FinancialExample


DEFAULT_DEV_PATH = Path("data/FinQA/dataset/dev.json")


def load_raw_finqa(path: str | Path = DEFAULT_DEV_PATH):
    dataset_path = Path(path)
    with dataset_path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_finqa_example(raw_example: dict[str, Any]):
    qa = raw_example["qa"]

    return FinancialExample(
        example_id=raw_example["id"],
        document_id=raw_example["filename"],
        question=qa["question"],
        answer=qa["answer"],
        pre_text=raw_example["pre_text"],
        table=raw_example["table"],
        post_text=raw_example["post_text"],
        gold_evidence=qa.get("gold_inds", {}),
        gold_program=qa.get("program"),
        metadata={
            "gold_numeric_answer": qa.get("exe_ans"),
            "answer_explanation": qa.get("explanation"),
            "annotated_table_rows": qa.get("ann_table_rows", []),
            "annotated_text_rows": qa.get("ann_text_rows", []),
        },
    )


def load_finqa_examples(
    path: str | Path = DEFAULT_DEV_PATH,
    limit: int | None = None,
):
    raw_examples = load_raw_finqa(path)
    if limit is not None:
        raw_examples = raw_examples[:limit]

    return [parse_finqa_example(raw_example) for raw_example in raw_examples]


def describe_example(example: FinancialExample) -> str:
    return "\n".join(
        [
            f"example_id: {example.example_id}",
            f"document_id: {example.document_id}",
            f"question: {example.question}",
            f"answer: {example.answer}",
            f"pre_text paragraphs: {len(example.pre_text)}",
            f"table rows: {len(example.table)}",
            f"post_text paragraphs: {len(example.post_text)}",
            f"gold evidence ids: {list(example.gold_evidence.keys())}",
            f"gold program: {example.gold_program}",
        ]
    )


if __name__ == "__main__":
    examples = load_finqa_examples(limit=5)
    print(f"loaded examples: {len(examples)}")
    print()
    print(describe_example(examples[0]))
