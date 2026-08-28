import math
import re
from dataclasses import dataclass
from typing import Any

try:
    from .schemas import CalculationStep
except ImportError:
    from schemas import CalculationStep


SUPPORTED_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
}
SUPPORTED_ANSWER_SCALES = {"auto", "raw", "ratio_to_percent"}
SUPPORTED_ANSWER_UNITS = {
    None,
    "",
    "raw",
    "number",
    "percent",
    "percentage",
    "percentage_points",
    "million",
    "millions",
    "billion",
    "billions",
    "dollar",
    "dollars",
    "share",
    "shares",
}


@dataclass(frozen=True)
class CalculationExecution:
    value: float | str | None
    formatted_answer: str | None
    trace: str | None
    error: str | None = None


def execute_calculation_steps(
    steps: list[CalculationStep],
    answer_unit: str | None = None,
    answer_scale: str | None = None,
) -> CalculationExecution:
    if not steps:
        return CalculationExecution(
            value=None,
            formatted_answer=None,
            trace=None,
            error="No calculation steps were provided.",
        )

    results: list[float | str] = []
    trace_lines = []

    try:
        for index, step in enumerate(steps):
            operation = normalize_operation(step.operation)
            if operation not in SUPPORTED_OPERATIONS:
                raise ValueError(f"Unsupported operation: {step.operation}")

            if len(step.arguments) != 2:
                raise ValueError(f"{operation} expects exactly two arguments.")

            left = resolve_operand(step.arguments[0], results)
            right = resolve_operand(step.arguments[1], results)
            result = apply_operation(operation, left, right)
            results.append(result)
            trace_lines.append(
                f"#{index} = {operation}({format_operand(left)}, {format_operand(right)}) = {format_operand(result)}"
            )

        final_value = results[-1]
        formatted_answer = format_executed_answer(final_value, answer_unit, answer_scale)
        return CalculationExecution(
            value=final_value,
            formatted_answer=formatted_answer,
            trace="\n".join(trace_lines),
        )
    except Exception as error:
        return CalculationExecution(
            value=None,
            formatted_answer=None,
            trace="\n".join(trace_lines) if trace_lines else None,
            error=str(error),
        )


def normalize_operation(operation: str) -> str:
    return operation.strip().lower()


def resolve_operand(operand: Any, previous_results: list[float | str]) -> float | str:
    if isinstance(operand, int | float):
        return float(operand)

    text = str(operand).strip()
    if not text:
        raise ValueError("Empty calculation operand.")

    if text.startswith("#"):
        result_index = int(text.removeprefix("#"))
        try:
            return previous_results[result_index]
        except IndexError as error:
            raise ValueError(f"Step reference {text} does not exist.") from error

    if text.startswith("const_"):
        text = text.removeprefix("const_")

    return parse_number(text)


def parse_number(text: str) -> float:
    normalized = text.lower().strip()
    normalized = normalized.replace("$", "").replace(",", "").replace("%", "")

    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1]

    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        raise ValueError(f"Could not parse numeric operand: {text}")

    return float(match.group(0))


def apply_operation(operation: str, left: float | str, right: float | str) -> float | str:
    if operation == "greater":
        return "yes" if float(left) > float(right) else "no"

    left_value = float(left)
    right_value = float(right)

    if operation == "add":
        return left_value + right_value
    if operation == "subtract":
        return left_value - right_value
    if operation == "multiply":
        return left_value * right_value
    if operation == "divide":
        if math.isclose(right_value, 0.0):
            raise ValueError("Cannot divide by zero.")
        return left_value / right_value
    if operation == "exp":
        return left_value**right_value

    raise ValueError(f"Unsupported operation: {operation}")


def format_executed_answer(
    value: float | str,
    answer_unit: str | None = None,
    answer_scale: str | None = None,
) -> str:
    if isinstance(value, str):
        return value

    normalized_unit = normalize_optional_label(answer_unit)
    normalized_scale = normalize_optional_label(answer_scale) or "auto"

    if normalized_unit not in SUPPORTED_ANSWER_UNITS:
        normalized_unit = "raw"
    if normalized_scale not in SUPPORTED_ANSWER_SCALES:
        normalized_scale = "raw"

    display_value = value
    should_scale_percent = normalized_unit in {"percent", "percentage"} and abs(display_value) <= 1
    if normalized_scale == "ratio_to_percent" and should_scale_percent:
        display_value *= 100
    elif normalized_scale == "auto" and should_scale_percent:
        display_value *= 100

    number = format_number(display_value)
    if normalized_unit in {"percent", "percentage"}:
        return f"{number}%"
    if normalized_unit == "percentage_points":
        return f"{number} percentage points"
    if normalized_unit in {"million", "millions"}:
        return f"{number} million"
    if normalized_unit in {"billion", "billions"}:
        return f"{number} billion"
    if normalized_unit in {"dollar", "dollars"}:
        return f"${number}"
    if normalized_unit in {"share", "shares"}:
        return f"{number} shares"

    return number


def normalize_optional_label(label: str | None) -> str | None:
    if label is None:
        return None

    return label.strip().lower().replace(" ", "_").replace("-", "_")


def format_number(value: float) -> str:
    rounded = round(value, 5)
    if math.isclose(rounded, round(rounded)):
        return str(int(round(rounded)))

    return f"{rounded:.5f}".rstrip("0").rstrip(".")


def format_operand(value: float | str) -> str:
    if isinstance(value, str):
        return value

    return format_number(value)
