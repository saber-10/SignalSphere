from typing import TypedDict, NotRequired, List, Dict, Any


class AgentState(TypedDict):
    # User Input
    question: str

    # Parser Output
    topics: NotRequired[List[str]]
    problem_type: NotRequired[str]
    requires_math: NotRequired[bool]
    topic_confidence: NotRequired[float]
    problem_confidence: NotRequired[float]

    # Router Output
    strategy: NotRequired[str]

    # Retriever
    retrieved_chunks: NotRequired[List[Dict[str, Any]]]
    context: NotRequired[str]

    # Solver
    answer: NotRequired[str]

    # verification
    verified: NotRequired[bool]

    verification_notes: NotRequired[str]

    answer_confidence: NotRequired[float]

    # Debug
    agent_trace: NotRequired[List[Dict[str, Any]]]