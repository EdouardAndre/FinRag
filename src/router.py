from dataclasses import dataclass, field
import re


SUPPORTED_RETRIEVAL_METHODS = {"dense", "bm25", "hybrid"}
DEFAULT_ADAPTIVE_TOP_K = 10
DEFAULT_ADAPTIVE_CANDIDATE_K = 20

CALCULATION_TERMS = {
    "average",
    "change",
    "decline",
    "decrease",
    "difference",
    "growth",
    "increase",
    "percentage",
    "ratio",
    "total",
}
TABLE_TERMS = {
    "amount",
    "balance",
    "cash",
    "cost",
    "expense",
    "income",
    "liability",
    "margin",
    "million",
    "net",
    "percent",
    "rate",
    "revenue",
    "sales",
    "shares",
    "tax",
    "transaction",
    "volume",
}
OUT_OF_SCOPE_TERMS = {
    "breaking news",
    "current stock price",
    "real time",
    "right now",
    "stock price today",
    "weather",
}


@dataclass(frozen=True)
class RetrievalRoute:
    related_to_index: bool
    method: str
    top_k: int
    candidate_k: int
    neighbor_window: int
    reason: str
    signals: list[str] = field(default_factory=list)


def route_question(question: str) -> RetrievalRoute:
    signals = detect_question_signals(question)

    if "out_of_scope" in signals:
        return RetrievalRoute(
            related_to_index=False,
            method="dense",
            top_k=0,
            candidate_k=0,
            neighbor_window=0,
            reason="Question appears to need information outside the indexed reports.",
            signals=signals,
        )

    if "calculation" in signals or "table_metric" in signals:
        return RetrievalRoute(
            related_to_index=True,
            method="dense",
            top_k=DEFAULT_ADAPTIVE_TOP_K,
            candidate_k=DEFAULT_ADAPTIVE_CANDIDATE_K,
            neighbor_window=1,
            reason="Numeric or table-like question; use dense retrieval plus neighbors for calculation context.",
            signals=signals,
        )

    if "exact_term" in signals:
        return RetrievalRoute(
            related_to_index=True,
            method="hybrid",
            top_k=DEFAULT_ADAPTIVE_TOP_K,
            candidate_k=DEFAULT_ADAPTIVE_CANDIDATE_K,
            neighbor_window=0,
            reason="Question contains exact terms or dates; combine dense and lexical retrieval.",
            signals=signals,
        )

    return RetrievalRoute(
        related_to_index=True,
        method="dense",
        top_k=DEFAULT_ADAPTIVE_TOP_K,
        candidate_k=DEFAULT_ADAPTIVE_CANDIDATE_K,
        neighbor_window=0,
        reason="Default report question route; dense retrieval is the strongest current baseline.",
        signals=signals,
    )


def detect_question_signals(question: str) -> list[str]:
    normalized = question.lower()
    signals = []

    if contains_any_phrase(normalized, OUT_OF_SCOPE_TERMS):
        signals.append("out_of_scope")

    if contains_any_word(normalized, CALCULATION_TERMS) or has_arithmetic_language(normalized):
        signals.append("calculation")

    if contains_any_word(normalized, TABLE_TERMS):
        signals.append("table_metric")

    if has_year(normalized) or has_quoted_phrase(question) or has_uppercase_ticker_like_term(question):
        signals.append("exact_term")

    if has_number(normalized):
        signals.append("number")

    return signals


def contains_any_phrase(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def contains_any_word(text: str, words: set[str]) -> bool:
    tokens = set(re.findall(r"[a-z]+", text))
    return bool(tokens & words)


def has_arithmetic_language(text: str) -> bool:
    return bool(re.search(r"\b(divided by|minus|plus|per|from .+ to|between .+ and)\b", text))


def has_year(text: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}\b", text))


def has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def has_quoted_phrase(text: str) -> bool:
    return bool(re.search(r"['\"].+['\"]", text))


def has_uppercase_ticker_like_term(text: str) -> bool:
    return bool(re.search(r"\b[A-Z]{2,5}\b", text))
