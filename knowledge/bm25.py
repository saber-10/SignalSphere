import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Retriever:
    def __init__(self, metadata):
        tokenized_corpus = [chunk['text'].lower().split() for chunk in metadata]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.metadata = metadata

    def retrieve(self, question, top_k=5):
        query = question.lower().split()
        scores = self.bm25.get_scores(query)
        indices = np.argsort(scores)[::-1]

        results = []

        top_indices = indices[:top_k]
        for idx in top_indices:
            chunk = self.metadata[idx].copy()
            chunk['score'] = scores[idx]
            results.append(chunk)

        return results