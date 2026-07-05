from agents.states import AgentState
from agents.enums import ProblemType, Strategy

ROUTING_TABLE = {
    ProblemType.CONCEPT.value: Strategy.SOLVER,
    ProblemType.NUMERICAL.value: Strategy.SOLVER_WITH_MATH,
    ProblemType.DERIVATION.value: Strategy.SOLVER,
    ProblemType.COMPARISON.value: Strategy.SOLVER,
    ProblemType.APPLICATION.value: Strategy.SOLVER,
}


class RouterAgent:
    def __init__(self):
        pass
    def route(self, state: AgentState) -> AgentState:

        problem_type = state.get(
            "problem_type",
            ProblemType.UNKNOWN.value,
        )

        strategy = ROUTING_TABLE.get(
            problem_type,
            Strategy.UNKNOWN,
        )

        state["strategy"] = strategy.value

        state.setdefault("agent_trace", []).append(
            {
                "agent": "router",
                "problem_type": problem_type,
                "strategy": strategy.value,
                "status": "completed",
            }
        )

        return state