import argparse
import json
import re
from collections import Counter
from dataclasses import asdict

from dotenv import load_dotenv

try:
    from .retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from .router import detect_question_signals
    from .schemas import EvidenceGrade, RetrievalResult
except ImportError:
    from retrieve import build_bm25_index, load_retrieval_artifacts, retrieve_with_route
    from router import detect_question_signals
    from schemas import EvidenceGrade, RetrievalResult


DEFAULT_LIMIT = 100
DEFAULT_TOP_K = 10
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}
DOMAIN_STOPWORDS = {
    "amount",
    "average",
    "change",
    "company",
    "difference",
    "fiscal",
    "million",
    "percentage",
    "total",
    "year",
}


def grade_retrieved_evidence(
    question: str,
    results: list[RetrievalResult],
) -> EvidenceGrade:
    if not results:
        return EvidenceGrade(
            is_sufficient=False,
            needs_retry=True,
            confidence=0.0,
            missing_reason="No chunks were retrieved.",
            missing_signals=["retrieved_evidence"],
        )

    question_signals = detect_question_signals(question)
    cohesive_results = dominant_example_results(results)
    evidence_text = "\n".join(result.chunk.content for result in cohesive_results)
    matched_signals = []
    missing_signals = []
    score = 0.0

    if has_metric_overlap(question, evidence_text):
        matched_signals.append("metric_overlap")
        score += 0.25
    else:
        missing_signals.append("metric_overlap")

    if years_are_supported(question, evidence_text):
        matched_signals.append("year_support")
        score += 0.20
    else:
        missing_signals.append("year_support")

    if numeric_evidence_is_supported(question_signals, evidence_text):
        matched_signals.append("numeric_support")
        score += 0.20
    else:
        missing_signals.append("numeric_support")

    if table_evidence_is_supported(question_signals, cohesive_results):
        matched_signals.append("table_support")
        score += 0.15
    else:
        missing_signals.append("table_support")

    if retrieved_chunks_are_cohesive(results):
        matched_signals.append("document_cohesion")
        score += 0.15
    else:
        missing_signals.append("document_cohesion")

    if results[0].score:
        matched_signals.append("ranked_evidence")
        score += 0.05

    confidence = min(score, 1.0)
    is_sufficient = confidence >= 0.65 and not has_blocking_missing_signal(
        question_signals,
        missing_signals,
    )

    return EvidenceGrade(
        is_sufficient=is_sufficient,
        needs_retry=not is_sufficient,
        confidence=confidence,
        missing_reason=build_missing_reason(missing_signals) if not is_sufficient else None,
        matched_signals=matched_signals,
        missing_signals=missing_signals,
    )


def dominant_example_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    top_results = results[: min(5, len(results))]
    example_counts = Counter(result.chunk.example_id for result in top_results)
    dominant_example_id, _ = example_counts.most_common(1)[0]
    return [
        result
        for result in results
        if result.chunk.example_id == dominant_example_id
    ]


def has_metric_overlap(question: str, evidence_text: str) -> bool:
    question_terms = content_terms(question)
    evidence_terms = content_terms(evidence_text)
    return bool(question_terms & evidence_terms)


def content_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower())
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in STOPWORDS and token not in DOMAIN_STOPWORDS
    }


def years_are_supported(question: str, evidence_text: str) -> bool:
    question_years = set(extract_years(question))
    if not question_years:
        return True

    evidence_years = set(extract_years(evidence_text))
    return bool(question_years & evidence_years)


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def numeric_evidence_is_supported(question_signals: list[str], evidence_text: str) -> bool:
    if "calculation" not in question_signals and "table_metric" not in question_signals:
        return True

    return bool(re.search(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?", evidence_text))


def table_evidence_is_supported(
    question_signals: list[str],
    results: list[RetrievalResult],
) -> bool:
    if "calculation" not in question_signals and "table_metric" not in question_signals:
        return True

    return any(result.chunk.chunk_type == "table" for result in results)


def retrieved_chunks_are_cohesive(results: list[RetrievalResult]) -> bool:
    top_results = results[: min(5, len(results))]
    example_counts = Counter(result.chunk.example_id for result in top_results)
    _, count = example_counts.most_common(1)[0]
    return count >= 2 or len(top_results) == 1


def has_blocking_missing_signal(
    question_signals: list[str],
    missing_signals: list[str],
) -> bool:
    if "calculation" in question_signals or "table_metric" in question_signals:
        return "numeric_support" in missing_signals

    return False


def build_missing_reason(missing_signals: list[str]) -> str:
    if not missing_signals:
        return "Evidence confidence was below the sufficiency threshold."

    return "Missing or weak: " + ", ".join(missing_signals) + "."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade whether retrieved FinQA evidence is sufficient.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "adaptive"], default="dense")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=0)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    chunks, index, metadata = load_retrieval_artifacts(limit=args.limit)
    bm25_index = build_bm25_index(chunks) if args.method in {"bm25", "hybrid", "adaptive"} else None
    results, route = retrieve_with_route(
        args.query,
        chunks=chunks,
        index=index,
        metadata=metadata,
        method=args.method,
        top_k=args.top_k,
        bm25_index=bm25_index,
        candidate_k=args.candidate_k,
        neighbor_window=args.neighbor_window,
    )
    grade = grade_retrieved_evidence(args.query, results)

    print("Route")
    print("=" * 80)
    print(json.dumps(asdict(route), indent=2))
    print()
    print("Evidence grade")
    print("=" * 80)
    print(json.dumps(asdict(grade), indent=2))


if __name__ == "__main__":
    main()
