from knowledge.retriever import Retriever

retriever = Retriever()

results = retriever.retrieve(
    "What is Laplace Transform?"
)

for r in results:

    print(r["source"])

    print(r["score"])

    print(r["text"][:100])

    print("-"*50)

print(results[0])