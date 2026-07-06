from typing import List, Dict, Any
from knowledge.vector_store import get_retriever, retrieve as faiss_retrieve
from knowledge.bm25 import BM25Retriever


class Retriever:
    def __init__(self):
        self.index, self.metadata, self.embedding_model = get_retriever()
        self.bm25 = BM25Retriever(self.metadata)
    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        
        faiss_results = faiss_retrieve(query = question, index = self.index, metadata = self.metadata, model = self.embedding_model, top_k = top_k)
        bm25_results = self.bm25.retrieve(question, top_k = top_k)  
        
        return self.merge_results(faiss_results, bm25_results)
    def merge_results(self, faiss_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = {}
        
        for chunk in faiss_results:
            key = chunk["text"]
            new_chunk = chunk.copy()
            new_chunk["retrievers"] = ["faiss"]
            merged[key] = new_chunk

        for chunk in bm25_results:
            key = chunk["text"]
            if key in merged:
                merged[key]["retrievers"].append("bm25")
                merged[key]["score"] = max(merged[key]["score"], chunk["score"])
            else:
                new_chunk = chunk.copy()
                new_chunk["retrievers"] = ["bm25"]
                merged[key] = new_chunk
        return list(merged.values())
       


