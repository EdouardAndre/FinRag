import json
import argparse
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

try:
    from .embed import ensure_chunks, create_mistral_client, normalize_vectors
    from .schemas import Chunk, RetrievalResult
except ImportError:
    from embed import ensure_chunks, create_mistral_client, normalize_vectors
    from schemas import Chunk, RetrievalResult


DATASET_PATH = Path("data/FinQA/dataset/dev.json")
CHUNKS_PATH = Path("data/chunks.json")
INDEX_PATH = Path("data/index.faiss")
METADATA_PATH = Path("data/index_metadata.json")
DEFAULT_LIMIT = 100
DEFAULT_TOP_K = 5
DEFAULT_MODEL = "mistral-embed"


def load_index_metadata(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_retrieval_artifacts(
    chunks_path: str | Path = CHUNKS_PATH,
    dataset_path: str | Path = DATASET_PATH,
    index_path: str | Path = INDEX_PATH,
    metadata_path: str | Path = METADATA_PATH,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[Chunk], faiss.Index, dict]:
    chunks = ensure_chunks(
        chunks_path=chunks_path,
        dataset_path=dataset_path,
        limit=limit,
        rebuild_chunks=False,
    )
    index = faiss.read_index(str(index_path))
    metadata = load_index_metadata(metadata_path)
    validate_artifact_alignment(chunks, index, metadata)
    return chunks, index, metadata


def validate_artifact_alignment(
    chunks: list[Chunk],
    index: faiss.Index,
    metadata: dict,
) -> None:
    if len(chunks) != index.ntotal:
        raise ValueError(f"Chunk count {len(chunks)} does not match FAISS index size {index.ntotal}.")

    if metadata.get("chunk_count") != len(chunks):
        raise ValueError("Index metadata chunk_count does not match loaded chunks.")

    chunk_ids = metadata.get("chunk_ids", [])
    if chunk_ids and chunk_ids[0] != chunks[0].chunk_id:
        raise ValueError("Index metadata appears out of sync with chunks.json.")


def embed_query(
    query: str,
    model: str = DEFAULT_MODEL,
    metric: str = "cosine",
) -> np.ndarray:
    client = create_mistral_client()
    response = client.embeddings.create(
        model=model,
        inputs=[query],
    )

    query_vector = np.array(
        [response.data[0].embedding],
        dtype=np.float32,
    )

    if metric == "cosine":
        return normalize_vectors(query_vector)

    return query_vector


def search_index(
    index: faiss.Index,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    return index.search(query_vector, k=top_k)

def build_retrieval_results(
    scores: np.ndarray,
    indices: np.ndarray,
    chunks: list[Chunk],
) -> list[RetrievalResult]:
    results = []

    for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
        if index == -1:
            continue

        chunk = chunks[index]
        results.append(
            RetrievalResult(
                rank=rank,
                score=float(score),
                chunk=chunk,
            )
        )

    return results


def retrieve(
    query: str,
    chunks: list[Chunk],
    index: faiss.Index,
    metadata: dict,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    query_vector = embed_query(
        query,
        model=metadata.get("model", DEFAULT_MODEL),
        metric=metadata.get("metric", "cosine"),
    )
    scores, indices = search_index(index, query_vector, top_k=top_k)
    return build_retrieval_results(scores, indices, chunks)


def print_retrieval_results(results: list[RetrievalResult]) -> None:
    for result in results:
        chunk = result.chunk
        print("=" * 80)
        print(f"rank:       {result.rank}")
        print(f"score:      {result.score}")
        print(f"chunk_id:   {chunk.chunk_id}")
        print(f"type:       {chunk.chunk_type}")
        print(f"section:    {chunk.section}")
        print(f"source_ids: {chunk.source_ids}")
        print()
        print(chunk.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense retrieval over the FinQA FAISS index.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser.parse_args()

def main() -> None:
    load_dotenv()
    args = parse_args()

    chunks, index, metadata = load_retrieval_artifacts(
        limit=args.limit,
    )
    results = retrieve(args.query, chunks, index, metadata, top_k=args.top_k)
    print_retrieval_results(results)

if __name__ == "__main__":
    main()
