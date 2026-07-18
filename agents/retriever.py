from agents.states import AgentState

from knowledge.retriever import Retriever
from agents.utils import build_context


class RetrieverAgent:

    def __init__(self):

        self.retriever = Retriever()

    def retrieve(self,state: AgentState,) -> AgentState:
        question = state["question"]
        retrieved_chunks = self.retriever.retrieve(question = question, top_k = 5)
        context = build_context(retrieved_chunks)
        state["context"] = context
        state["retrieved_chunks"] = retrieved_chunks

        state.setdefault("agent_trace", []).append({
            "agent": "RetrieverAgent",
            "status": "completed"})
        

        return state