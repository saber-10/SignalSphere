from typing import List, Dict, Any

def build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    sections = []

    for chunk in retrieved_chunks:
        sections.append(
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(sections)
