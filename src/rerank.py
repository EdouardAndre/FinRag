import re
from collections import Counter

try:
    from .router import detect_question_signals
    from .schemas import RetrievalResult
except ImportError:
    from router import detect_question_signals
    from schemas import RetrievalResult


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


def rerank_results(
    question: str,
    results: list[RetrievalResult],
    top_k: int,
) -> list[RetrievalResult]:
    if not results:
        return []

    question_terms = content_terms(question)
    question_years = set(extract_years(question))
    question_numbers = set(extract_numbers(question))
    question_signals = set(detect_question_signals(question))
    example_scores = score_examples(results)

    scored_results = [
        (
            rerank_score(
                result=result,
                question_terms=question_terms,
                question_years=question_years,
                question_numbers=question_numbers,
                question_signals=question_signals,
                example_scores=example_scores,
            ),
            result,
        )
        for result in deduplicate_results(results)
    ]
    scored_results.sort(key=lambda item: item[0], reverse=True)

    return [
        RetrievalResult(rank=rank, score=score, chunk=result.chunk)
        for rank, (score, result) in enumerate(scored_results[:top_k], start=1)
    ]


def rerank_score(
    result: RetrievalResult,
    question_terms: set[str],
    question_years: set[str],
    question_numbers: set[str],
    question_signals: set[str],
    example_scores: dict[str, float],
) -> float:
    content = result.chunk.content.lower()
    content_tokens = set(tokenize(content))
    content_years = set(extract_years(content))
    content_numbers = set(extract_numbers(content))

    term_overlap = jaccard(question_terms, content_terms(content))
    year_support = 1.0 if not question_years or question_years & content_years else 0.0
    number_support = 1.0 if question_numbers and question_numbers & content_numbers else 0.0
    table_bonus = 1.0 if result.chunk.chunk_type == "table" and question_signals & {"calculation", "table_metric"} else 0.0
    exact_token_overlap = len(set(tokenize(" ".join(question_terms))) & content_tokens)
    example_cohesion = example_scores.get(result.chunk.example_id, 0.0)

    return (
        reciprocal_rank_signal(result.rank) * 0.30
        + normalize_positive_score(result.score) * 0.15
        + term_overlap * 0.25
        + year_support * 0.10
        + number_support * 0.05
        + table_bonus * 0.05
        + min(exact_token_overlap / 5, 1.0) * 0.05
        + min(example_cohesion, 1.0) * 0.05
    )


def score_examples(results: list[RetrievalResult]) -> dict[str, float]:
    counts = Counter(result.chunk.example_id for result in results[: min(len(results), 20)])
    return {
        example_id: count / max(len(results[: min(len(results), 20)]), 1)
        for example_id, count in counts.items()
    }


def deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    deduplicated = {}
    for result in results:
        existing = deduplicated.get(result.chunk.chunk_id)
        if existing is None or result.rank < existing.rank:
            deduplicated[result.chunk.chunk_id] = result

    return list(deduplicated.values())


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z-]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"\$?\d+(?:\.\d+)?%?|[a-zA-Z]+(?:\.[a-zA-Z]+)?", text.lower())


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", text.replace("$", ""))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def reciprocal_rank_signal(rank: int) -> float:
    return 1 / max(rank, 1)


def normalize_positive_score(score: float) -> float:
    if score <= 0:
        return 0.0

    return min(score, 1.0)
