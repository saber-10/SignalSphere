import streamlit as st

from agents.graph import get_graph


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SignalSphere",
    page_icon="📡",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    "What is the difference between the Laplace Transform and the Z-Transform?",
    "Derive the region of convergence for a causal LTI system.",
    "How would you design a filter to remove 60 Hz noise from a signal?",
    "What is convolution?",
]


@st.cache_resource(show_spinner=False)
def load_graph():
    """Build the LangGraph pipeline once and cache it across reruns."""
    return get_graph()


def confidence_label(confidence):
    """Map a numeric confidence score to a colored status label."""
    if confidence is None or confidence == "N/A":
        return "⚪ Unknown"
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return "⚪ Unknown"
    if score >= 0.7:
        return f"🟢 High ({score:.2f})"
    if score >= 0.4:
        return f"🟡 Moderate ({score:.2f})"
    return f"🔴 Low ({score:.2f})"


# ---------------------------------------------------------------------------
# Sidebar — context, not clutter
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 SignalSphere")
    st.caption("Multi-agent RAG tutor for Signals & Systems")

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "Parser → Router → Retriever "
        "(FAISS + BM25) → Reranker → Solver → Verifier"
    )

    st.markdown("---")
    st.markdown("**Try an example**")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["question_input"] = q


# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------
st.title("📡 SignalSphere")
st.caption("Ask a Signals & Systems question and get a grounded, verified answer.")

question = st.text_area(
    "Your question",
    key="question_input",
    placeholder="e.g. What is the region of convergence for a causal LTI system?",
    height=100,
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    graph = load_graph()
    state = {"question": question, "agent_trace": []}

    with st.spinner("Retrieving sources and generating a verified answer..."):
        try:
            result = graph.invoke(state)
        except Exception as exc:  # surface pipeline errors instead of a raw crash
            st.error(f"Something went wrong while processing your question: {exc}")
            st.stop()

    st.session_state["result"] = result
    st.session_state["last_question"] = question

# ---------------------------------------------------------------------------
# Results — tabbed so a single screenshot stays compact
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state["result"]

    st.divider()
    st.caption(f"Question: *{st.session_state['last_question']}*")

    tab_answer, tab_verify, tab_sources, tab_trace = st.tabs(
        ["📖 Answer", "✅ Verification", "📚 Sources", "⚙️ Agent Trace"]
    )

    with tab_answer:
        st.write(result.get("answer", "No answer generated."))

    with tab_verify:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Verification**")
            verified = result.get("verified", "N/A")
            st.write("Verified:", "✅ Yes" if verified is True else verified)
            st.write("Confidence:", confidence_label(result.get("answer_confidence")))

        with col2:
            st.markdown("**Query Understanding**")
            st.write("Topics:", ", ".join(result.get("topics", [])) or "N/A")
            st.write("Problem Type:", result.get("problem_type", "N/A"))
            st.write("Strategy:", result.get("strategy", "N/A"))

    with tab_sources:
        chunks = result.get("retrieved_chunks", [])
        if not chunks:
            st.info("No source chunks were retrieved for this question.")
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"Chunk {i} • {chunk.get('source', 'unknown source')}"):
                meta_col1, meta_col2 = st.columns(2)
                if "reranker_score" in chunk:
                    meta_col1.metric("Reranker Score", f"{chunk['reranker_score']:.3f}")
                if "retrievers" in chunk:
                    meta_col2.write("Retrieved by: " + ", ".join(chunk["retrievers"]))
                st.write(chunk.get("text", ""))

    with tab_trace:
        st.json(result.get("agent_trace", []))