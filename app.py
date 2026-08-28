from dataclasses import asdict

import streamlit as st
from dotenv import load_dotenv

from src.generate import run_rag


DEFAULT_QUESTION = "what is the average payment volume per transaction for american express?"


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="FinRag", layout="wide")
    st.title("FinRag")

    with st.sidebar:
        method = st.selectbox("Retrieval method", ["dense", "bm25", "hybrid", "adaptive"], index=0)
        top_k = st.slider("Top k", min_value=1, max_value=20, value=10)
        candidate_k = st.slider("Candidate k", min_value=top_k, max_value=60, value=max(30, top_k))
        neighbor_window = st.slider("Neighbor window", min_value=0, max_value=2, value=0)
        use_evidence_grader = st.checkbox("Use evidence grader", value=False)
        rerank = st.checkbox("Rerank candidates", value=True)
        show_evidence = st.checkbox("Show retrieved evidence", value=True)
        model = st.text_input("Generation model", value="mistral-small-latest")

    question = st.text_area("Question", value=DEFAULT_QUESTION, height=90)
    run_button = st.button("Run RAG", type="primary")

    if not run_button:
        return

    if not question.strip():
        st.warning("Enter a question first.")
        return

    with st.spinner("Retrieving evidence and generating an answer..."):
        try:
            answer, results, route, grade = run_rag(
                question.strip(),
                method=method,
                top_k=top_k,
                candidate_k=candidate_k,
                neighbor_window=neighbor_window,
                model=model,
                use_evidence_grader=use_evidence_grader,
                return_evidence_grade=True,
                rerank=rerank,
            )
        except Exception as error:
            st.error(str(error))
            return

    answer_col, route_col = st.columns([2, 1])
    with answer_col:
        st.subheader("Answer")
        if answer.insufficient_evidence:
            st.warning(answer.answer)
        else:
            st.success(answer.answer)

        if answer.calculation:
            st.subheader("Calculation")
            st.code(answer.calculation)

        if answer.execution_error:
            st.error(f"Execution error: {answer.execution_error}")

        if answer.citations:
            st.subheader("Citations")
            for citation in answer.citations:
                st.markdown(f"- `{citation.chunk_id}`: {citation.quote}")

    with route_col:
        st.subheader("Route")
        st.json(asdict(route))

        if grade is not None:
            st.subheader("Evidence Grade")
            st.json(asdict(grade))

        st.subheader("Execution")
        st.json(
            {
                "executed_answer": answer.executed_answer,
                "answer_unit": answer.answer_unit,
                "answer_scale": answer.answer_scale,
                "calculation_steps": [asdict(step) for step in answer.calculation_steps],
            }
        )

    if show_evidence:
        st.subheader("Retrieved Evidence")
        for result in results:
            with st.expander(f"#{result.rank} {result.chunk.chunk_id} ({result.chunk.chunk_type})"):
                st.caption(f"score: {result.score}")
                st.caption(f"source_ids: {', '.join(result.chunk.source_ids)}")
                st.write(result.chunk.content)


if __name__ == "__main__":
    main()
