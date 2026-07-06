from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, question: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        
        pairs = [(question, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        chunk_scores = list(
            zip(chunks, scores)
        )
        chunk_scores = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        chunk_scores = chunk_scores[:top_k]
        results = []
        for chunk, score in chunk_scores:
            new_chunk = chunk.copy()
            new_chunk["reranker_score"] = float(score)
            results.append(new_chunk)
        
        return results