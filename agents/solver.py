from typing import NotRequired, TypedDict, List, Dict, Any
from agents.prompts import create_solver_prompt
from agents.states import AgentState
from knowledge.vector_store import get_retriever
from config import Config
from langchain_ollama import ChatOllama
from knowledge.vector_store import retrieve
from knowledge.retriever import Retriever


class SolverAgent:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2,
)
    
    def solve(self, state: AgentState) -> AgentState:
        try:
            question = state["question"]
            context = state["context"]


            messages = create_solver_prompt(context = context, question = question)

            answer = self.llm.invoke(messages)

            
            state["answer"] = answer.content

            state.setdefault("agent_trace", []).append({
                "agent": "SolverAgent",
                "status": "completed"
            })

            return state
        
        except Exception as e:
            state["answer"] = f"An error occurred: {str(e)}"
            return state