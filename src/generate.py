import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .embed import create_mistral_client
    from .retrieve import build_bm25_index, load_retrieval_artifacts, retrieve
    from .schemas import Citation, RAGAnswer, RetrievalResult
except ImportError:
    from embed import create_mistral_client
    from retrieve import build_bm25_index, load_retrieval_artifacts, retrieve
    from schemas import Citation, RAGAnswer, RetrievalResult


DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TOP_K = 5
DEFAULT_LIMIT = 100


SYSTEM_PROMPT = """You answer financial questions using only the supplied evidence.

Rules:
- Cite every factual claim using a chunk_id from the evidence.
- For numerical questions, show the calculation.
- Do not use outside knowledge.
- If the evidence is insufficient, set insufficient_evidence to true.
- Never invent a citation.
- Return valid JSON only.
"""


def format_evidence(results: list[RetrievalResult]) -> str:
    evidence_blocks = []

    for result in results:
        chunk = result.chunk
        evidence_blocks.append(
            "\n".join(
                [
                    f"[CHUNK {chunk.chunk_id}]",
                    f"type: {chunk.chunk_type}",
                    f"section: {chunk.section}",
                    f"source_ids: {', '.join(chunk.source_ids)}",
                    "content:",
                    chunk.content,
                ]
            )
        )

    return "\n\n".join(evidence_blocks)


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    return f"""Question:
{question}

Evidence:
{format_evidence(results)}

Return JSON with exactly these fields:
{{
  "answer": "string",
  "citations": [
    {{
      "chunk_id": "string",
      "quote": "short exact quote from the cited chunk"
    }}
  ],
  "calculation": "string or null",
  "insufficient_evidence": true or false
}}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise

        return json.loads(match.group(0))


def parse_rag_answer(payload: dict[str, Any]) -> RAGAnswer:
    citations = [
        Citation(
            chunk_id=normalize_generated_chunk_id(str(citation.get("chunk_id", ""))),
            quote=str(citation.get("quote", "")),
        )
        for citation in payload.get("citations", [])
    ]

    calculation = payload.get("calculation")
    if calculation is not None:
        calculation = str(calculation)

    return RAGAnswer(
        answer=str(payload.get("answer", "")),
        citations=citations,
        calculation=calculation,
        insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
    )


def normalize_generated_chunk_id(chunk_id: str) -> str:
    normalized = chunk_id.strip()
    if normalized.startswith("CHUNK "):
        return normalized.removeprefix("CHUNK ").strip()

    return normalized


def validate_citations(answer: RAGAnswer, results: list[RetrievalResult]) -> None:
    retrieved_chunk_ids = {result.chunk.chunk_id for result in results}

    invalid_citations = [
        citation.chunk_id
        for citation in answer.citations
        if citation.chunk_id not in retrieved_chunk_ids
    ]

    if invalid_citations:
        raise ValueError(f"Generated citations were not in retrieved evidence: {invalid_citations}")


def generate_answer(
    question: str,
    results: list[RetrievalResult],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> RAGAnswer:
    client = create_mistral_client()
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, results)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    payload = extract_json_object(content)
    answer = parse_rag_answer(payload)
    validate_citations(answer, results)
    return answer


def run_rag(
    question: str,
    method: str = "dense",
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = 10,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
) -> tuple[RAGAnswer, list[RetrievalResult]]:
    chunks, index, metadata = load_retrieval_artifacts(limit=limit)
    bm25_index = build_bm25_index(chunks) if method in {"bm25", "hybrid"} else None
    results = retrieve(
        question,
        chunks=chunks,
        index=index,
        metadata=metadata,
        top_k=top_k,
        method=method,
        bm25_index=bm25_index,
        candidate_k=candidate_k,
    )
    answer = generate_answer(question, results, model=model)
    return answer, results


def print_answer(answer: RAGAnswer) -> None:
    print(json.dumps(asdict(answer), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded answers from retrieved FinQA evidence.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-evidence", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    answer, results = run_rag(
        args.query,
        method=args.method,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        limit=args.limit,
        model=args.model,
    )

    print_answer(answer)

    if args.show_evidence:
        print()
        print("Retrieved evidence")
        print("=" * 80)
        for result in results:
            print(f"{result.rank}. {result.chunk.chunk_id} score={result.score}")
            print(result.chunk.content)
            print()


if __name__ == "__main__":
    main()
