
from langgraph.graph import StateGraph, START, END

from agents.states import AgentState
from agents.parser import ParserAgent
from agents.retriever import RetrieverAgent
from agents.router import RouterAgent
from agents.solver import SolverAgent
from agents.verifier import VerifierAgent

def build_graph():
    """
    Build and compile the LangGraph workflow.

    Current Workflow:

    START
      ↓
    Solver
      ↓
     END
    """

    builder = StateGraph(AgentState)

    parser = ParserAgent()
    retriever = RetrieverAgent()
    router = RouterAgent()
    solver = SolverAgent()
    verifier = VerifierAgent()





    builder.add_node(
    "parser",
    parser.parse,
    )

    builder.add_node(
        "retriever",
        retriever.retrieve
        )

    builder.add_node(
        "router",
        router.route,
    )

    builder.add_node(
        "solver",
        solver.solve,
    )

    builder.add_node(
        "solver_math",
        solver.solve
    )

    builder.add_node(
        "verifier",
        verifier.verify,
    )

    builder.add_edge(
    START,
    "parser",
)

    builder.add_edge(
    "parser",
    "retriever",
    )

    builder.add_edge(
    "retriever",
    "router",
)

    builder.add_conditional_edges(
        "router",
        route_after_router,
    )  

    builder.add_edge(
        "solver",
        "verifier"
    )
    builder.add_edge(
        "verifier",
        END,
    )

    return builder.compile()


_GRAPH = None

def route_after_router(state: AgentState) -> str:
    strategy = state["strategy"]
    if strategy == "solver":
        return "solver"
    elif strategy == "solver_math":
        return "solver_math"
    return "solver"

def get_graph():

    global _GRAPH

    if _GRAPH is None:
        _GRAPH = build_graph()

    return _GRAPH