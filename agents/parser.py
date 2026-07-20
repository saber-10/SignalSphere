from agents.enums import ProblemType
from typing import Tuple

from agents.states import AgentState
TOPIC_KEYWORDS = {

    "laplace_transform": [
        "laplace",
        "inverse laplace",
        "roc",
    ],

    "fourier_transform": [
        "fourier",
        "dtft",
        "dft",
        "fft",
    ],

    "convolution": [
        "convolution",
        "convolve",
    ],

    "sampling": [
        "sampling",
        "nyquist",
        "aliasing",
    ],

    "z_transform": [
        "z transform",
        "z-transform",
    ],

    "lti_systems": [
        "lti",
        "linear time invariant",
        "time invariant",
    ],

    "signals": [
        "signal",
        "continuous time",
        "discrete time",
    ],

    "state_space": [
        "state space",
        "state-space",
        "state variable",
        "state vector",
        "system matrix",
        "matrix exponential",
        "controllability",
        "observability",
        "similarity transformation",
    ],

    "convolution": [
        "convolution",
        "impulse response",
        "impulse",
        "cascade",
        "parallel",
        "feedback",
        "integrator",
        "accumulator",
    ],

    "fourier_series": [
        "fourier series",
        "ctfs",
        "dtfs",
        "harmonics",
        "dirichlet",
        "parseval",
    ],

    "dft": [
        "dft",
        "fft",
        "fast fourier transform",
        "twiddle",
        "zero padding",
        "frequency resolution",
        "circular convolution",
    ]
}

def detect_topics(question:str) -> tuple[list[str], float]:
    question = question.lower()
    detected_topics = set()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question:
                detected_topics.add(topic)
    detected_topics = sorted(detected_topics)
    confidence = min(len(detected_topics) / 2, 1.0)
    return detected_topics, confidence




CONCEPT_KEYWORDS = {
    "what",
    "why",
    "explain",
    "define",
    "describe",
}

NUMERICAL_KEYWORDS = {
    "find",
    "calculate",
    "compute",
    "evaluate",
    "solve",
}

DERIVATION_KEYWORDS = {
    "derive",
    "prove",
}

COMPARISON_KEYWORDS = {
    "compare",
    "difference between",
    "versus",
    "vs",
}

APPLICATION_KEYWORDS = {
    "application",
    "importance",
    "real world",
}




def detect_problem_type(question: str) -> Tuple[ProblemType, float]:

    question = question.lower()

    scores = {
        ProblemType.CONCEPT: 0,
        ProblemType.NUMERICAL: 0,
        ProblemType.DERIVATION: 0,
        ProblemType.COMPARISON: 0,
        ProblemType.APPLICATION: 0,
    }

    question = question.lower()

    for word in COMPARISON_KEYWORDS:

        if word in CONCEPT_KEYWORDS:
            scores[ProblemType.CONCEPT] += 1

        if word in NUMERICAL_KEYWORDS:
            scores[ProblemType.NUMERICAL] += 1

        if word in DERIVATION_KEYWORDS:
            scores[ProblemType.DERIVATION] += 1

        if word in COMPARISON_KEYWORDS:
            scores[ProblemType.COMPARISON] += 1

        if word in APPLICATION_KEYWORDS:
            scores[ProblemType.APPLICATION] += 1

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        return ProblemType.UNKNOWN, 0.0

    total_score = sum(scores.values())
    confidence = best_score / total_score

    return best_type, confidence

class ParserAgent:
    def __init__(self):
        pass

    def parse(self, state: AgentState) -> AgentState:
        question = state["question"]

        topics, topic_confidence = detect_topics(question)
        problem_type, problem_confidence = detect_problem_type(question)

        requires_math = problem_type in {ProblemType.NUMERICAL, ProblemType.DERIVATION}

        state["topics"] = topics
        state["problem_type"] = problem_type.value
        state["requires_math"] = requires_math
        state["parser_confidence"] = min(topic_confidence, problem_confidence)

        state.setdefault("agent_trace", []).append({
            "agent": "parser",
            "status": "completed",
            "topics": topics,
            "problem_type": problem_type.value,
            "requires_math": requires_math,})
        
        return state




