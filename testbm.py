from knowledge.vector_store import get_retriever
from knowledge.bm25 import BM25Retriever

_, metadata, _ = get_retriever()

bm25 = BM25Retriever(metadata)

results = bm25.retrieve("What is Laplace Transform?")

for i, chunk in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Score:", chunk["score"])
    print("Source:", chunk["source"])
    print(chunk["text"][:200])