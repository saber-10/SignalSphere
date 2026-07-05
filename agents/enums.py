from enum import Enum

class ProblemType(Enum):
    CONCEPT = "concept"
    NUMERICAL = "numerical"
    DERIVATION = "derivation"
    COMPARISON = "comparison"
    APPLICATION = "application"
    UNKNOWN = "unknown"

class Strategy(Enum):
    SOLVER = "solver"
    SOLVER_WITH_MATH = "solver_with_math"
    OCR = "ocr"
    UNKNOWN = "unknown"
    