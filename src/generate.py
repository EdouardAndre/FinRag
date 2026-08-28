import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .calculation import execute_calculation_steps
    from .embed import create_mistral_client
    from .grade_documents import grade_retrieved_evidence
    from .query_rewrite import rewrite_query_for_retrieval
    from .retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from .router import RetrievalRoute
    from .schemas import CalculationStep, Citation, EvidenceGrade, RAGAnswer, RetrievalResult
except ImportError:
    from calculation import execute_calculation_steps
    from embed import create_mistral_client
    from grade_documents import grade_retrieved_evidence
    from query_rewrite import rewrite_query_for_retrieval
    from retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from router import RetrievalRoute
    from schemas import CalculationStep, Citation, EvidenceGrade, RAGAnswer, RetrievalResult


DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TOP_K = 5
DEFAULT_LIMIT = 100
CORRECTIVE_METHOD = "hybrid"
CORRECTIVE_TOP_K = 10
CORRECTIVE_CANDIDATE_K = 30
CORRECTIVE_NEIGHBOR_WINDOW = 1


SYSTEM_PROMPT = """You answer financial questions using only the supplied evidence.

Rules:
- Cite every factual claim using a chunk_id from the evidence.
- For numerical questions, return calculation_steps as executable arithmetic.
- Use only these operations in calculation_steps: add, subtract, multiply, divide, exp, greater.
- Use numeric operands from the evidence or previous step references like "#0".
- Use answer_scale "auto" unless a specific display conversion is needed.
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
  "calculation_steps": [
    {{
      "operation": "add|subtract|multiply|divide|exp|greater",
      "arguments": ["number or #step_reference", "number or #step_reference"]
    }}
  ],
  "answer_unit": "raw|percent|percentage_points|million|billion|dollars|shares or null",
  "answer_scale": "auto|raw|ratio_to_percent",
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

    calculation_steps = [
        CalculationStep(
            operation=str(step.get("operation", "")),
            arguments=list(step.get("arguments", [])),
        )
        for step in payload.get("calculation_steps", [])
        if isinstance(step, dict)
    ]

    return RAGAnswer(
        answer=str(payload.get("answer", "")),
        citations=citations,
        calculation=calculation,
        insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
        calculation_steps=calculation_steps,
        answer_unit=optional_string(payload.get("answer_unit")),
        answer_scale=optional_string(payload.get("answer_scale")),
    )


def canonicalize_answer_citations(answer: RAGAnswer, results: list[RetrievalResult]) -> RAGAnswer:
    chunk_ids = {result.chunk.chunk_id for result in results}
    chunks_by_source_id: dict[str, list[str]] = {}
    for result in results:
        for source_id in result.chunk.source_ids:
            chunks_by_source_id.setdefault(source_id, []).append(result.chunk.chunk_id)

    citations = []
    for citation in answer.citations:
        chunk_id = citation.chunk_id
        if chunk_id not in chunk_ids:
            matching_chunk_ids = chunks_by_source_id.get(chunk_id, [])
            if len(matching_chunk_ids) == 1:
                chunk_id = matching_chunk_ids[0]

        citations.append(Citation(chunk_id=chunk_id, quote=citation.quote))

    return RAGAnswer(
        answer=answer.answer,
        citations=citations,
        calculation=answer.calculation,
        insufficient_evidence=answer.insufficient_evidence,
        calculation_steps=answer.calculation_steps,
        answer_unit=answer.answer_unit,
        answer_scale=answer.answer_scale,
        executed_answer=answer.executed_answer,
        execution_error=answer.execution_error,
    )


def optional_string(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)


def apply_deterministic_calculation(answer: RAGAnswer) -> RAGAnswer:
    if answer.insufficient_evidence or not answer.calculation_steps:
        return answer

    execution = execute_calculation_steps(
        answer.calculation_steps,
        answer_unit=answer.answer_unit,
        answer_scale=answer.answer_scale,
    )
    if execution.error is not None:
        return RAGAnswer(
            answer=answer.answer,
            citations=answer.citations,
            calculation=answer.calculation,
            insufficient_evidence=answer.insufficient_evidence,
            calculation_steps=answer.calculation_steps,
            answer_unit=answer.answer_unit,
            answer_scale=answer.answer_scale,
            executed_answer=None,
            execution_error=execution.error,
        )

    return RAGAnswer(
        answer=execution.formatted_answer or answer.answer,
        citations=answer.citations,
        calculation=execution.trace or answer.calculation,
        insufficient_evidence=answer.insufficient_evidence,
        calculation_steps=answer.calculation_steps,
        answer_unit=answer.answer_unit,
        answer_scale=answer.answer_scale,
        executed_answer=execution.formatted_answer,
        execution_error=None,
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
    answer = canonicalize_answer_citations(answer, results)
    answer = apply_deterministic_calculation(answer)
    validate_citations(answer, results)
    return answer


def merge_retrieval_results(
    primary_results: list[RetrievalResult],
    corrective_results: list[RetrievalResult],
    top_k: int,
) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}

    for result in primary_results + corrective_results:
        chunk_id = result.chunk.chunk_id
        existing = merged.get(chunk_id)
        if existing is None or result.score > existing.score:
            merged[chunk_id] = result

    ranked_results = sorted(merged.values(), key=lambda result: result.score, reverse=True)[:top_k]
    return [
        RetrievalResult(rank=rank, score=result.score, chunk=result.chunk)
        for rank, result in enumerate(ranked_results, start=1)
    ]


def retrieve_and_grade(
    question: str,
    chunks,
    index,
    metadata: dict,
    method: str,
    top_k: int,
    bm25_index,
    candidate_k: int,
    neighbor_window: int,
    use_evidence_grader: bool,
    return_evidence_grade: bool,
    rewrite_model: str,
    allowed_example_id: str | None = None,
    rerank: bool = False,
) -> tuple[list[RetrievalResult], RetrievalRoute, EvidenceGrade | None]:
    results, route = retrieve_with_route(
        question,
        chunks=chunks,
        index=index,
        metadata=metadata,
        top_k=top_k,
        method=method,
        bm25_index=bm25_index,
        candidate_k=candidate_k,
        neighbor_window=neighbor_window,
        allowed_example_id=allowed_example_id,
        rerank=rerank,
    )
    evidence_grade = (
        grade_retrieved_evidence(question, results)
        if use_evidence_grader or return_evidence_grade
        else None
    )
    if not route.related_to_index and evidence_grade is None:
        evidence_grade = grade_retrieved_evidence(question, results)

    if (
        use_evidence_grader
        and route.related_to_index
        and evidence_grade is not None
        and evidence_grade.needs_retry
    ):
        rewritten_query = rewrite_query_for_retrieval(question, model=rewrite_model)
        corrected_results, corrected_route = retrieve_with_route(
            rewritten_query,
            chunks=chunks,
            index=index,
            metadata=metadata,
            top_k=max(top_k, CORRECTIVE_TOP_K),
            method=CORRECTIVE_METHOD,
            bm25_index=bm25_index,
            candidate_k=max(candidate_k, CORRECTIVE_CANDIDATE_K),
            neighbor_window=max(neighbor_window, CORRECTIVE_NEIGHBOR_WINDOW),
            allowed_example_id=allowed_example_id,
            rerank=True,
        )
        merged_results = merge_retrieval_results(
            primary_results=results,
            corrective_results=corrected_results,
            top_k=max(top_k, CORRECTIVE_TOP_K),
        )
        corrected_grade = grade_retrieved_evidence(question, merged_results)

        if corrected_grade.confidence >= evidence_grade.confidence:
            return merged_results, corrected_route, corrected_grade

    return results, route, evidence_grade


def run_rag(
    question: str,
    method: str = "dense",
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = 10,
    neighbor_window: int = 0,
    limit: int = DEFAULT_LIMIT,
    model: str = DEFAULT_MODEL,
    use_evidence_grader: bool = False,
    return_evidence_grade: bool = False,
    allowed_example_id: str | None = None,
    rerank: bool = False,
) -> tuple[RAGAnswer, list[RetrievalResult], RetrievalRoute, EvidenceGrade | None]:
    chunks, index, metadata = load_retrieval_artifacts(limit=limit)
    bm25_index = build_bm25_index(chunks) if method in {"bm25", "hybrid", "adaptive"} else None
    if use_evidence_grader and bm25_index is None:
        bm25_index = build_bm25_index(chunks)

    results, route, evidence_grade = retrieve_and_grade(
        question,
        chunks,
        index,
        metadata,
        method=method,
        top_k=top_k,
        bm25_index=bm25_index,
        candidate_k=candidate_k,
        neighbor_window=neighbor_window,
        use_evidence_grader=use_evidence_grader,
        return_evidence_grade=return_evidence_grade,
        rewrite_model=model,
        allowed_example_id=allowed_example_id,
        rerank=rerank,
    )

    if not route.related_to_index:
        return (
            RAGAnswer(
                answer="Insufficient evidence",
                citations=[],
                calculation=None,
                insufficient_evidence=True,
            ),
            results,
            route,
            evidence_grade or grade_retrieved_evidence(question, results),
        )

    answer = generate_answer(question, results, model=model)
    return answer, results, route, evidence_grade


def print_answer(answer: RAGAnswer) -> None:
    print(json.dumps(asdict(answer), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded answers from retrieved FinQA evidence.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "adaptive"], default="dense")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--neighbor-window", type=int, default=0)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-evidence", action="store_true")
    parser.add_argument("--show-route", action="store_true")
    parser.add_argument("--use-evidence-grader", action="store_true")
    parser.add_argument("--show-grade", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    answer, results, route, evidence_grade = run_rag(
        args.query,
        method=args.method,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        neighbor_window=args.neighbor_window,
        limit=args.limit,
        model=args.model,
        use_evidence_grader=args.use_evidence_grader,
        return_evidence_grade=args.show_grade,
        rerank=args.rerank,
    )

    if args.show_route:
        print(f"route_method: {route.method}")
        print(f"route_top_k: {route.top_k}")
        print(f"route_candidate_k: {route.candidate_k}")
        print(f"route_neighbor_window: {route.neighbor_window}")
        print(f"route_related_to_index: {route.related_to_index}")
        print(f"route_signals: {route.signals}")
        print(f"route_reason: {route.reason}")
        print()

    if args.show_grade and evidence_grade is not None:
        print("evidence_grade:")
        print(json.dumps(asdict(evidence_grade), indent=2))
        print()

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
