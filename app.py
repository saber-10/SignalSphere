import streamlit as st

from agents.graph import get_graph


st.set_page_config(
    page_title="SignalSphere",
    page_icon="📡",
)

st.title("📡 SignalSphere")
st.caption("AI-powered Signals & Systems Tutor")

graph = get_graph()

question = st.text_area(
    "Ask your question"
)

if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    state = {
        "question": question,
        "agent_trace": [],
    }

    result = graph.invoke(state)

    st.divider()

    st.subheader("📖 Answer")
    st.write(result["answer"])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Verification")
        st.write("Verified:", result.get("verified", "N/A"))
        st.write("Confidence:", result.get("answer_confidence", "N/A"))

    with col2:
        st.subheader("🧠 Parser")
        st.write("Topics:", result.get("topics", []))
        st.write("Problem Type:", result.get("problem_type", "N/A"))
        st.write("Strategy:", result.get("strategy", "N/A"))

    st.divider()

    st.subheader("📚 Retrieved Chunks")

    for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):

        with st.expander(f"Chunk {i} • {chunk['source']}"):

            if "reranker_score" in chunk:
                st.write("Reranker Score:", chunk["reranker_score"])

            if "retrievers" in chunk:
                st.write("Retrieved By:", ", ".join(chunk["retrievers"]))

            st.write(chunk["text"])

    st.divider()

    st.subheader("⚙ Agent Trace")

    st.json(result.get("agent_trace", []))