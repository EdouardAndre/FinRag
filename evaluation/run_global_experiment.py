import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_PATH = Path("data/FinQA/dataset/dev.json")
DEFAULT_RESULTS_DIR = Path("results/global_experiment")
RETRIEVAL_METHODS = ("dense", "bm25", "hybrid", "adaptive")


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    return_code: int
    metrics: dict[str, str]
    stdout: str
    stderr: str


def run_command(name: str, command: list[str]) -> CommandResult:
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)

    return CommandResult(
        name=name,
        command=command,
        return_code=completed.returncode,
        metrics=parse_metrics(completed.stdout),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_metrics(output: str) -> dict[str, str]:
    metrics = {}
    for line in output.splitlines():
        if ": " not in line:
            continue

        name, value = line.split(": ", 1)
        if name and value:
            metrics[name.strip()] = value.strip()

    return metrics


def save_summary(results: list[CommandResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted(
        {
            metric_name
            for result in results
            for metric_name in result.metrics.keys()
        }
    )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["name", "return_code", "command", *metric_names],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result.name,
                    "return_code": result.return_code,
                    "command": " ".join(result.command),
                    **result.metrics,
                }
            )


def save_manifest(results: list[CommandResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": result.name,
            "command": result.command,
            "return_code": result.return_code,
            "metrics": result.metrics,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        for result in results
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_index_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "src/embed.py",
        "--dataset-path",
        str(args.dataset_path),
        "--limit",
        str(args.index_limit),
        "--batch-size",
        str(args.batch_size),
        "--rebuild-chunks",
    ]
    return command


def retrieval_command(args: argparse.Namespace, method: str, rerank: bool) -> list[str]:
    command = [
        args.python,
        "evaluation/evaluate_retrieval.py",
        "--dataset-path",
        str(args.dataset_path),
        "--method",
        method,
        "--limit",
        str(args.eval_limit),
        "--max-k",
        str(args.max_k),
        "--candidate-k",
        str(args.candidate_k),
        "--scope",
        "global",
        "--failures-path",
        str(args.results_dir / f"{method}_{label_rerank(rerank)}_retrieval_failures.csv"),
    ]
    if rerank:
        command.append("--rerank")

    return command


def answer_command(args: argparse.Namespace, method: str, rerank: bool) -> list[str]:
    command = [
        args.python,
        "evaluation/evaluate_answers.py",
        "--dataset-path",
        str(args.dataset_path),
        "--method",
        method,
        "--limit",
        str(args.answer_limit),
        "--top-k",
        str(args.max_k),
        "--candidate-k",
        str(args.candidate_k),
        "--scope",
        "global",
        "--predictions-path",
        str(args.results_dir / f"{method}_{label_rerank(rerank)}_answer_predictions.csv"),
    ]
    if rerank:
        command.append("--rerank")
    if args.use_evidence_grader:
        command.append("--use-evidence-grader")

    return command


def label_rerank(rerank: bool) -> str:
    return "reranked" if rerank else "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible global FinRag comparison.")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--index-limit", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--answer-limit", type=int, default=20)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--run-answers", action="store_true")
    parser.add_argument("--use-evidence-grader", action="store_true")
    parser.add_argument("--include-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if not args.skip_index:
        results.append(run_command("build_index", build_index_command(args)))

    for method in RETRIEVAL_METHODS:
        results.append(
            run_command(
                name=f"{method}_baseline_retrieval",
                command=retrieval_command(args, method=method, rerank=False),
            )
        )
        if args.include_rerank:
            results.append(
                run_command(
                    name=f"{method}_reranked_retrieval",
                    command=retrieval_command(args, method=method, rerank=True),
                )
            )

    if args.run_answers:
        for method in RETRIEVAL_METHODS:
            results.append(
                run_command(
                    name=f"{method}_baseline_answers",
                    command=answer_command(args, method=method, rerank=False),
                )
            )
            if args.include_rerank:
                results.append(
                    run_command(
                        name=f"{method}_reranked_answers",
                        command=answer_command(args, method=method, rerank=True),
                    )
                )

    save_summary(results, args.results_dir / "summary.csv")
    save_manifest(results, args.results_dir / "manifest.json")
    print(f"summary saved to: {args.results_dir / 'summary.csv'}")
    print(f"manifest saved to: {args.results_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
