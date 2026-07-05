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

    state = {
        "question": question
    }

    result = graph.invoke(state)

    st.subheader("Answer")

    st.write(result["answer"])