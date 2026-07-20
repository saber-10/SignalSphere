from typing import List, Dict, Any
from knowledge.vector_store import get_retriever, retrieve as faiss_retrieve
from knowledge.bm25 import BM25Retriever
from knowledge.reranker import CrossEncoderReranker


class Retriever:
    def __init__(self):
        self.index, self.metadata, self.embedding_model = get_retriever()
        self.bm25 = BM25Retriever(self.metadata)
        self.reranker = CrossEncoderReranker()
    def retrieve(self, question: str, top_k: int = 5, mode: str = "hybrid_reranker") -> List[Dict[str, Any]]:
        if mode == "faiss":
            return faiss_retrieve(query = question, index = self.index, metadata = self.metadata, model = self.embedding_model, top_k = top_k)
        elif mode == "bm25":
            return self.bm25.retrieve(question, top_k=top_k)
        else:
            faiss_results = faiss_retrieve(query = question, index = self.index, metadata = self.metadata, model = self.embedding_model, top_k = top_k)
            bm25_results = self.bm25.retrieve(question, top_k = top_k)
            merged = self.merge_results(faiss_results=faiss_results, bm25_results=bm25_results)

        if mode == "hybrid":
            return merged
        elif mode == "hybrid_reranker":
            return self.reranker.rerank(question, merged, top_k=5) 


          
        
        return self.merge_results(faiss_results, bm25_results)
    def merge_results(self,faiss_results: List[Dict[str, Any]],bm25_results: List[Dict[str, Any]],) -> List[Dict[str, Any]]:

        K = 60
        merged = {}
        # FAISS contribution
        for rank, chunk in enumerate(faiss_results, start=1):
            key = chunk["text"]
            if key not in merged:
                new_chunk = chunk.copy()
                new_chunk["retrievers"] = ["faiss"]
                new_chunk["rrf_score"] = 0.0
                merged[key] = new_chunk

            merged[key]["rrf_score"] += 1 / (K + rank)

        # BM25 contribution
        for rank, chunk in enumerate(bm25_results, start=1):
            key = chunk["text"]
            if key not in merged:
                new_chunk = chunk.copy()
                new_chunk["retrievers"] = ["bm25"]
                new_chunk["rrf_score"] = 0.0
                merged[key] = new_chunk
            else:
                merged[key]["retrievers"].append("bm25")

            merged[key]["rrf_score"] += 1 / (K + rank)

        return sorted(merged.values(),key=lambda x: x["rrf_score"],reverse=True)
       


