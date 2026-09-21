import unittest

from src.retrieve import expand_with_neighbors, resolve_retrieval_route
from src.router import route_question
from src.schemas import Chunk, RetrievalResult


def table_chunk(row_index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc::table::{row_index}",
        example_id="doc",
        document_id="doc.pdf",
        chunk_type="table",
        section="table",
        content=str(row_index),
        source_ids=[f"table_{row_index}"],
        metadata={"row_index": row_index},
    )


class NeighborExpansionTests(unittest.TestCase):
    def test_seed_results_stay_ahead_of_neighbors(self) -> None:
        chunks = [table_chunk(index) for index in range(1, 7)]
        seeds = [
            RetrievalResult(rank=1, score=0.9, chunk=chunks[2]),
            RetrievalResult(rank=2, score=0.8, chunk=chunks[4]),
        ]

        expanded = expand_with_neighbors(seeds, chunks, neighbor_window=1)

        self.assertEqual(
            [result.chunk.chunk_id for result in expanded],
            [
                "doc::table::3",
                "doc::table::5",
                "doc::table::2",
                "doc::table::4",
                "doc::table::6",
            ],
        )


class AdaptiveCandidateTests(unittest.TestCase):
    def test_adaptive_route_uses_thirty_candidates_by_default(self) -> None:
        self.assertEqual(route_question("what is the average revenue?").candidate_k, 30)

    def test_adaptive_route_honors_larger_candidate_pool(self) -> None:
        route = resolve_retrieval_route(
            query="what is the average revenue?",
            method="adaptive",
            top_k=10,
            candidate_k=50,
            neighbor_window=0,
        )

        self.assertEqual(route.candidate_k, 50)


if __name__ == "__main__":
    unittest.main()
