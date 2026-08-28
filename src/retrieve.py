import json
import argparse
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

try:
    from .embed import ensure_chunks, create_mistral_client, normalize_vectors
    from .router import SUPPORTED_RETRIEVAL_METHODS, RetrievalRoute, route_question
    from .schemas import Chunk, RetrievalResult
except ImportError:
    from embed import ensure_chunks, create_mistral_client, normalize_vectors
    from router import SUPPORTED_RETRIEVAL_METHODS, RetrievalRoute, route_question
    from schemas import Chunk, RetrievalResult


DATASET_PATH = Path("data/FinQA/dataset/dev.json")
CHUNKS_PATH = Path("data/chunks.json")
INDEX_PATH = Path("data/index.faiss")
METADATA_PATH = Path("data/index_metadata.json")
DEFAULT_LIMIT = 100
DEFAULT_TOP_K = 5
DEFAULT_MODEL = "mistral-embed"
DEFAULT_RRF_K = 60
DEFAULT_NEIGHBOR_WINDOW = 0
ADAPTIVE_METHOD = "adaptive"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\$?\d+(?:\.\d+)?%?|[a-zA-Z]+(?:\.[a-zA-Z]+)?", text.lower())


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


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenized_chunks = [tokenize(chunk.content) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def search_bm25(
    query: str,
    bm25_index: BM25Okapi,
    chunks: list[Chunk],
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    scores = bm25_index.get_scores(tokenize(query))
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [
        RetrievalResult(
            rank=rank,
            score=float(scores[index]),
            chunk=chunks[index],
        )
        for rank, index in enumerate(ranked_indices, start=1)
    ]

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


def neighbor_key(chunk: Chunk) -> tuple[str, str, int] | None:
    if chunk.chunk_type == "table":
        row_index = chunk.metadata.get("row_index")
    else:
        row_index = chunk.metadata.get("row_index")

    if row_index is None:
        return None

    return (chunk.example_id, chunk.section, int(row_index))


def build_neighbor_lookup(chunks: list[Chunk]) -> dict[tuple[str, str, int], Chunk]:
    lookup = {}

    for chunk in chunks:
        key = neighbor_key(chunk)
        if key is not None:
            lookup[key] = chunk

    return lookup


def expand_with_neighbors(
    results: list[RetrievalResult],
    chunks: list[Chunk],
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> list[RetrievalResult]:
    if neighbor_window <= 0:
        return results

    lookup = build_neighbor_lookup(chunks)
    expanded_chunks: dict[str, tuple[float, Chunk]] = {}

    for result in results:
        chunk = result.chunk
        expanded_chunks.setdefault(chunk.chunk_id, (result.score, chunk))

        key = neighbor_key(chunk)
        if key is None:
            continue

        example_id, section, row_index = key
        for offset in range(-neighbor_window, neighbor_window + 1):
            if offset == 0:
                continue

            neighbor = lookup.get((example_id, section, row_index + offset))
            if neighbor is None:
                continue

            expanded_chunks.setdefault(neighbor.chunk_id, (result.score, neighbor))

    return [
        RetrievalResult(rank=rank, score=score, chunk=chunk)
        for rank, (score, chunk) in enumerate(expanded_chunks.values(), start=1)
    ]


def retrieve_dense(
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


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalResult]],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for result in ranking:
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + result.rank)
            chunks_by_id[chunk_id] = result.chunk

    ranked_chunk_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    return [
        RetrievalResult(
            rank=rank,
            score=scores[chunk_id],
            chunk=chunks_by_id[chunk_id],
        )
        for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)
    ]


def retrieve_hybrid(
    query: str,
    chunks: list[Chunk],
    index: faiss.Index,
    metadata: dict,
    bm25_index: BM25Okapi,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = 10,
) -> list[RetrievalResult]:
    dense_results = retrieve_dense(
        query,
        chunks=chunks,
        index=index,
        metadata=metadata,
        top_k=candidate_k,
    )
    bm25_results = search_bm25(
        query,
        bm25_index=bm25_index,
        chunks=chunks,
        top_k=candidate_k,
    )
    return reciprocal_rank_fusion([dense_results, bm25_results], top_k=top_k)


def retrieve(
    query: str,
    chunks: list[Chunk],
    index: faiss.Index,
    metadata: dict,
    top_k: int = DEFAULT_TOP_K,
    method: str = "dense",
    bm25_index: BM25Okapi | None = None,
    candidate_k: int = 10,
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> list[RetrievalResult]:
    if method == "dense":
        results = retrieve_dense(query, chunks=chunks, index=index, metadata=metadata, top_k=top_k)
        return expand_with_neighbors(results, chunks, neighbor_window=neighbor_window)

    if method == "bm25":
        resolved_bm25_index = bm25_index or build_bm25_index(chunks)
        results = search_bm25(query, resolved_bm25_index, chunks, top_k=top_k)
        return expand_with_neighbors(results, chunks, neighbor_window=neighbor_window)

    if method == "hybrid":
        resolved_bm25_index = bm25_index or build_bm25_index(chunks)
        results = retrieve_hybrid(
            query,
            chunks=chunks,
            index=index,
            metadata=metadata,
            bm25_index=resolved_bm25_index,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        return expand_with_neighbors(results, chunks, neighbor_window=neighbor_window)

    raise ValueError(f"Unsupported retrieval method: {method}")


def resolve_retrieval_route(
    query: str,
    method: str,
    top_k: int,
    candidate_k: int,
    neighbor_window: int,
) -> RetrievalRoute:
    if method == ADAPTIVE_METHOD:
        return route_question(query)

    if method not in SUPPORTED_RETRIEVAL_METHODS:
        raise ValueError(f"Unsupported retrieval method: {method}")

    return RetrievalRoute(
        related_to_index=True,
        method=method,
        top_k=top_k,
        candidate_k=max(candidate_k, top_k),
        neighbor_window=neighbor_window,
        reason="Explicit retrieval settings were provided.",
        signals=[],
    )


def retrieve_with_route(
    query: str,
    chunks: list[Chunk],
    index: faiss.Index,
    metadata: dict,
    method: str = "dense",
    top_k: int = DEFAULT_TOP_K,
    bm25_index: BM25Okapi | None = None,
    candidate_k: int = 10,
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> tuple[list[RetrievalResult], RetrievalRoute]:
    route = resolve_retrieval_route(
        query=query,
        method=method,
        top_k=top_k,
        candidate_k=candidate_k,
        neighbor_window=neighbor_window,
    )

    if not route.related_to_index:
        return [], route

    results = retrieve(
        query,
        chunks=chunks,
        index=index,
        metadata=metadata,
        top_k=route.top_k,
        method=route.method,
        bm25_index=bm25_index,
        candidate_k=route.candidate_k,
        neighbor_window=route.neighbor_window,
    )
    return results, route


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
    parser = argparse.ArgumentParser(description="Run retrieval over the FinQA chunk index.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--neighbor-window", type=int, default=DEFAULT_NEIGHBOR_WINDOW)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "adaptive"], default="dense")
    parser.add_argument("--show-route", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    chunks, index, metadata = load_retrieval_artifacts(
        limit=args.limit,
    )
    bm25_index = build_bm25_index(chunks) if args.method in {"bm25", "hybrid", "adaptive"} else None
    results, route = retrieve_with_route(
        args.query,
        chunks,
        index,
        metadata,
        top_k=args.top_k,
        method=args.method,
        bm25_index=bm25_index,
        candidate_k=args.candidate_k,
        neighbor_window=args.neighbor_window,
    )
    if args.show_route:
        print(f"route_method: {route.method}")
        print(f"route_top_k: {route.top_k}")
        print(f"route_candidate_k: {route.candidate_k}")
        print(f"route_neighbor_window: {route.neighbor_window}")
        print(f"route_related_to_index: {route.related_to_index}")
        print(f"route_signals: {route.signals}")
        print(f"route_reason: {route.reason}")
        print()
    print_retrieval_results(results)


if __name__ == "__main__":
    main()
