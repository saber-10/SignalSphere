EXPANSION_MAP = {
    "lt": "Laplace Transform",
    "ft": "Fourier Transform",
    "roc": "Region of Convergence",
    "lti": "Linear Time Invariant",
    "ct": "Continuous Time",
    "dt": "Discrete Time",
    "dtft": "Discrete Time Fourier Transform",
    "fft": "Fast Fourier Transform",
    "dft": "Discrete Fourier Transform",
    "zt": "Z Transform",
}


class QueryExpander:

    def __init__(self):
        self.expansion_map = EXPANSION_MAP

    def expand(self,question: str,) -> str:
        expanded = question
        lower = question.lower()

        for short, full in self.expansion_map.items():
            if short in lower:
                expanded += f" {full}"
        return expanded