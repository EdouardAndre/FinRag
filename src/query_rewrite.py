import argparse
import json
import re
from typing import Any

from dotenv import load_dotenv

try:
    from .embed import create_mistral_client
except ImportError:
    from embed import create_mistral_client


DEFAULT_REWRITE_MODEL = "mistral-small-latest"
REWRITE_SYSTEM_PROMPT = """You rewrite financial questions into compact retrieval queries.

Rules:
- Preserve company names, dates, metrics, units, and calculation intent.
- Remove filler words.
- Do not answer the question.
- Do not add facts that are not in the question.
- Return valid JSON only.
"""


QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "does",
    "for",
    "from",
    "if",
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
    "when",
    "which",
    "who",
    "with",
}


def rewrite_query_for_retrieval(
    question: str,
    model: str = DEFAULT_REWRITE_MODEL,
) -> str:
    try:
        return rewrite_query_with_llm(question, model=model)
    except Exception:
        return rewrite_query_heuristically(question)


def rewrite_query_with_llm(
    question: str,
    model: str = DEFAULT_REWRITE_MODEL,
) -> str:
    client = create_mistral_client()
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": build_rewrite_prompt(question)},
        ],
        temperature=0.0,
        max_tokens=128,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    payload = extract_json_object(content)
    rewritten_query = str(payload.get("query", "")).strip()

    if not rewritten_query:
        return rewrite_query_heuristically(question)

    return rewritten_query


def build_rewrite_prompt(question: str) -> str:
    return f"""Question:
{question}

Return JSON with exactly this field:
{{
  "query": "compact retrieval query, 25 words or fewer"
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


def rewrite_query_heuristically(question: str) -> str:
    parts = [
        *extract_quoted_phrases(question),
        *extract_years(question),
        *extract_number_mentions(question),
        *extract_content_terms(question),
    ]
    rewritten = " ".join(deduplicate(parts))

    if not rewritten:
        return question.strip()

    return rewritten


def extract_quoted_phrases(text: str) -> list[str]:
    return [match.strip() for match in re.findall(r"['\"]([^'\"]+)['\"]", text)]


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def extract_number_mentions(text: str) -> list[str]:
    return re.findall(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?", text)


def extract_content_terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z&.-]*", text.lower())
    return [
        token.strip(".-")
        for token in tokens
        if len(token.strip(".-")) > 2 and token not in QUESTION_STOPWORDS
    ]


def deduplicate(items: list[str]) -> list[str]:
    seen = set()
    deduplicated = []

    for item in items:
        normalized = item.lower().strip()
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        deduplicated.append(item)

    return deduplicated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a FinQA question into a compact retrieval query.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default=DEFAULT_REWRITE_MODEL)
    parser.add_argument("--heuristic", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.heuristic:
        print(rewrite_query_heuristically(args.query))
    else:
        print(rewrite_query_for_retrieval(args.query, model=args.model))


if __name__ == "__main__":
    main()
