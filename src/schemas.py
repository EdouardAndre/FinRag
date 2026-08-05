from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FinancialExample:
    example_id: str
    document_id: str
    question: str
    answer: str
    pre_text: list[str]
    table: list[list[str]]
    post_text: list[str]
    gold_evidence: dict[str, str] = field(default_factory=dict)
    gold_program: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
