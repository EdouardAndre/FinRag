import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from dotenv import load_dotenv

try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client.sdk import Mistral

try:
    from .chunk import build_chunks_for_dataset, load_chunks, save_chunks
    from .load_data import load_finqa_examples
    from .schemas import Chunk
except ImportError:
    from chunk import build_chunks_for_dataset, load_chunks, save_chunks
    from load_data import load_finqa_examples
    from schemas import Chunk


DEFAULT_CHUNKS_PATH = Path("data/chunks.json")
DEFAULT_INDEX_PATH = Path("data/index.faiss")
DEFAULT_METADATA_PATH = Path("data/index_metadata.json")
DEFAULT_MODEL = "mistral-embed"
DEFAULT_BATCH_SIZE = 32
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT_MS = 60_000


def batch_items(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def create_mistral_client(api_key: str | None = None) -> Mistral:
    resolved_api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not resolved_api_key:
        raise ValueError("MISTRAL_API_KEY is required to create embeddings.")

    return Mistral(api_key=resolved_api_key)


def embed_texts(
    texts: list[str],
    client: Mistral,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> np.ndarray:
    embeddings: list[list[float]] = []
    batches = list(batch_items(texts, batch_size))

    for batch_index, batch in enumerate(batches, start=1):
        print(f"embedding batch {batch_index}/{len(batches)} ({len(batch)} texts)")

        for attempt in range(1, retry_attempts + 1):
            try:
                response = client.embeddings.create(
                    model=model,
                    inputs=batch,
                    timeout_ms=timeout_ms,
                )
                break
            except Exception:
                if attempt == retry_attempts:
                    raise

                # computing wait time for exponential backoff
                sleep_seconds = 2 ** (attempt - 1)
                print(
                    f"batch {batch_index} failed on attempt {attempt}; "
                    f"retrying in {sleep_seconds}s"
                )
                time.sleep(sleep_seconds)

        embeddings.extend(item.embedding for item in response.data)

    return np.asarray(embeddings, dtype="float32")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    normalized = vectors.copy()
    faiss.normalize_L2(normalized)
    return normalized


def build_faiss_index(vectors: np.ndarray, metric: str = "cosine") -> faiss.Index:
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {vectors.shape}.")

    if metric == "cosine":
        indexed_vectors = normalize_vectors(vectors)
        index = faiss.IndexFlatIP(indexed_vectors.shape[1])
    elif metric == "l2":
        indexed_vectors = vectors
        index = faiss.IndexFlatL2(indexed_vectors.shape[1])
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    index.add(indexed_vectors)
    return index


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))


def load_faiss_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def save_index_metadata(
    chunks: list[Chunk],
    vectors: np.ndarray,
    path: str | Path,
    model: str,
    metric: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model,
        "metric": metric,
        "embedding_dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def build_embedding_index(
    chunks: list[Chunk],
    client: Mistral,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    metric: str = "cosine",
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> tuple[faiss.Index, np.ndarray]:
    texts = [chunk.content for chunk in chunks]
    vectors = embed_texts(
        texts,
        client=client,
        model=model,
        batch_size=batch_size,
        retry_attempts=retry_attempts,
        timeout_ms=timeout_ms,
    )
    index = build_faiss_index(vectors, metric=metric)
    return index, vectors


def build_and_save_index(
    chunks: list[Chunk],
    index_path: str | Path = DEFAULT_INDEX_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    metric: str = "cosine",
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> None:
    client = create_mistral_client()
    index, vectors = build_embedding_index(
        chunks,
        client=client,
        model=model,
        batch_size=batch_size,
        metric=metric,
        retry_attempts=retry_attempts,
        timeout_ms=timeout_ms,
    )
    save_faiss_index(index, index_path)
    save_index_metadata(chunks, vectors, metadata_path, model=model, metric=metric)


def ensure_chunks(
    chunks_path: str | Path,
    dataset_path: str | Path,
    limit: int,
    rebuild_chunks: bool,
) -> list[Chunk]:
    chunks_file = Path(chunks_path)

    # If chunks already exist we thus avoid rebuilding them once more
    if chunks_file.exists() and not rebuild_chunks:
        return load_chunks(chunks_file)

    examples = load_finqa_examples(path=dataset_path, limit=limit)
    chunks = build_chunks_for_dataset(examples)
    save_chunks(chunks, chunks_file)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mistral embeddings and a FAISS index for FinQA chunks.")
    parser.add_argument("--dataset-path", default="data/FinQA/dataset/dev.json")
    parser.add_argument("--chunks-path", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--rebuild-chunks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    chunks = ensure_chunks(
        chunks_path=args.chunks_path,
        dataset_path=args.dataset_path,
        limit=args.limit,
        rebuild_chunks=args.rebuild_chunks,
    )

    print(f"chunks: {len(chunks)}")
    print(f"chunks path: {args.chunks_path}")

    if args.dry_run:
        print("dry run: skipped Mistral embedding request and FAISS index creation")
        return

    build_and_save_index(
        chunks,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        model=args.model,
        batch_size=args.batch_size,
        metric=args.metric,
        retry_attempts=args.retry_attempts,
        timeout_ms=args.timeout_ms,
    )

    print(f"index path: {args.index_path}")
    print(f"metadata path: {args.metadata_path}")


if __name__ == "__main__":
    main()
