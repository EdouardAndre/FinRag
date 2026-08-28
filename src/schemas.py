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


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    example_id: str
    document_id: str
    chunk_type: str
    section: str
    content: str
    source_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    chunk: Chunk


@dataclass(frozen=True)
class EvidenceGrade:
    is_sufficient: bool
    needs_retry: bool
    confidence: float
    missing_reason: str | None
    matched_signals: list[str] = field(default_factory=list)
    missing_signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class CalculationStep:
    operation: str
    arguments: list[Any]


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    citations: list[Citation]
    calculation: str | None
    insufficient_evidence: bool
    calculation_steps: list[CalculationStep] = field(default_factory=list)
    answer_unit: str | None = None
    answer_scale: str | None = None
    executed_answer: str | None = None
    execution_error: str | None = None
