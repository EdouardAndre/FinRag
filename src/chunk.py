import json
from dataclasses import asdict
from pathlib import Path

try:
    from .schemas import Chunk, FinancialExample
except ImportError:
    from schemas import Chunk, FinancialExample

def get_header(table: list[list[str]]) -> list[str]:
    res = []
    for i in table[0]:
        res.append(i)
    return res

def table_to_chunk(table: list[list[str]], header: list[str], index: int) -> str:
    res = "table row. "
    i = 0
    while i < len(header):
        res += header[i] + ": " + table[index][i]
        i += 1
        if (i != len(header)):
            res += "; "
    return res

def table_chunking(table: list[list[str]], example_id: str, document_id: str) -> list[Chunk]:
    header = get_header(table)
    res = []
    for i in range(1, len(table)):
        row  = table_to_chunk(table, header, i)
        chunk = Chunk(
            chunk_id= example_id + "::table::" + str(i),
            example_id=example_id,
            document_id=document_id,
            chunk_type= "table",
            section="table",
            content=row,
            source_ids=["table_" + str(i)],
            metadata={"row_index":i, "header":header},
        )
        res.append(chunk)
    return res

def text_chunking(rows: list[str], example_id: str, document_id: str, section: str, index=int) -> list[Chunk]:
    res = []
    for i in range(len(rows)):
        row = rows[i].strip()
        if row == "":
            continue
        chunk = Chunk(
            chunk_id=example_id + "::" + section + "::" + str(index + i),
            example_id=example_id,
            document_id=document_id,
            chunk_type="text",
            section=section,
            content=row,
            source_ids=["text_" + str(index + i)],
            metadata={"text_index": index+i, "row_index": i}
        )
        res.append(chunk)
    return res

def create_chunks(data_point: FinancialExample) -> list[Chunk]:
    chunks = []

    chunks.extend(
        text_chunking(
            data_point.pre_text,
            data_point.example_id,
            data_point.document_id,
            section="pre_text",
            index=0,
        )
    )

    chunks.extend(
        table_chunking(
            data_point.table,
            data_point.example_id,
            data_point.document_id,
        )
    )

    chunks.extend(
        text_chunking(
            data_point.post_text,
            data_point.example_id,
            data_point.document_id,
            section="post_text",
            index=len(data_point.pre_text),
        )
    )

    return chunks


def format_table_row(table: list[list[str]], header: list[str], row_index: int) -> str:
    return table_to_chunk(table, header, row_index)


def build_table_chunks(table: list[list[str]], example_id: str, document_id: str) -> list[Chunk]:
    return table_chunking(table, example_id, document_id)


def build_text_chunks(
    rows: list[str],
    example_id: str,
    document_id: str,
    section: str,
    start_index: int,
) -> list[Chunk]:
    return text_chunking(rows, example_id, document_id, section, start_index)


def build_chunks_for_example(example: FinancialExample) -> list[Chunk]:
    return create_chunks(example)


def build_chunks_for_dataset(examples: list[FinancialExample]) -> list[Chunk]:
    chunks = []
    for example in examples:
        chunks.extend(build_chunks_for_example(example))
    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(chunk) for chunk in chunks], file, indent=2)


def load_chunks(path: str | Path) -> list[Chunk]:
    input_path = Path(path)

    with input_path.open(encoding="utf-8") as file:
        chunk_dicts = json.load(file)

    return [Chunk(**chunk_dict) for chunk_dict in chunk_dicts]


def pretty_print_chunks(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        print("=" * 80)
        print(f"chunk_id:   {chunk.chunk_id}")
        print(f"example_id: {chunk.example_id}")
        print(f"document_id:{chunk.document_id}")
        print(f"type:       {chunk.chunk_type}")
        print(f"section:    {chunk.section}")
        print(f"source_ids: {chunk.source_ids}")
        print(f"metadata:   {chunk.metadata}")
        print()
        print("content:")
        print(chunk.content)
        print()


if __name__ == "__main__":
    from load_data import load_finqa_examples

    examples = load_finqa_examples(limit=10)

    for example in examples:
        chunks = create_chunks(example)
        gold_ids = set(example.gold_evidence.keys())

        matched = [
            chunk
            for chunk in chunks
            if gold_ids & set(chunk.source_ids)
        ]

        print(example.example_id, gold_ids, len(matched))
