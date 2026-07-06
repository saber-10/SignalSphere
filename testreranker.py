from knowledge.retriever import Retriever
from knowledge.reranker import CrossEncoderReranker

retriever = Retriever()
reranker = CrossEncoderReranker()

chunks = retriever.retrieve(
    "What is Laplace Transform?"
)

results = reranker.rerank(
    "What is Laplace Transform?",
    chunks,
)
print("Number of merged chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(i, chunk["source"], chunk["retrievers"])
    
for chunk in results:
    print("=" * 80)
    print("Score:", chunk["reranker_score"])
    print("Source:", chunk["source"])
    print(chunk["text"][:200])