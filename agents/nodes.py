from typing import NotRequired, TypedDict, List, Dict, Any
from agents.prompts import create_solver_prompt
from agents.states import AgentState
from knowledge.vector_store import get_retriever
from config import Config
from langchain_ollama import ChatOllama
from knowledge.vector_store import retrieve


class AgentNodes:
    def __init__(self):
        self.index, self.metadata, self.embedding_model = get_retriever()
        self.llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2,
)
    
    def solver_node(self, state: AgentState) -> AgentState:
        try:
            question = state["question"]

            retrieved_chunks = retrieve(
                query = question, 
                index = self.index,
                model = self.embedding_model,
                metadata = self.metadata,
                top_k = 5
            )

            context = ""

            for chunk in retrieved_chunks:
                context += (
                    f"Source: {chunk['source']}\n"
                    f"{chunk['text']}\n\n"
                )

            messages = create_solver_prompt(context = context, question = question)

            answer = self.llm.invoke(messages)

            state["retrieved_chunks"] = retrieved_chunks
            state["context"] = context
            state["answer"] = answer.content

            return state
        
        except Exception as e:
            state["answer"] = f"An error occurred: {str(e)}"
            return state
    
    
