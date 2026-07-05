import json

from langchain_ollama import ChatOllama

from config import Config
from agents.states import AgentState
from agents.prompts import create_verifier_prompt


class VerifierAgent:
    def __init__(self):
        self.llm = ChatOllama(model = Config.MODEL, temperature = 0)

    def verify(self, state: AgentState) -> AgentState:
        try:
            question = state["question"]
            context = state["context"]
            answer = state["answer"]

            messages = create_verifier_prompt(question = question, context = context, answer = answer)

            response = self.llm.invoke(messages)

            result = json.loads(response.content)

            state["verified"] = result["verified"]
            state["answer_confidence"] = result["confidence"]
            state["verification_notes"] = result["notes"]

            state.setdefault("agent_trace", []).append({
                "agent": "verifier",
                "status": "completed",
                "verified": result["verified"],
                "confidence": result["confidence"],
            }
            )

            return state
        except Exception as e:
            state["verified"] = False
            state["answer_confidence"] = 0.0
            state["verification_notes"] = str(e)
            state.setdefault("agent_trace", []).append({
                "agent": "verifier",
                "status": "failed",
                "error": str(e),
            }
            )
            raise state