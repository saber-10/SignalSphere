import json
from knowledge.retriever import Retriever


def reciprocal_rank(sources, expected):
    """
    Computes reciprocal rank for one query.
    """
    for i, src in enumerate(sources):
        if src in expected:
            return 1 / (i + 1)
    return 0.0


retriever = Retriever()

with open("evaluation/questions.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)["eval_set"]

modes = [
    "bm25",
    "faiss",
    "hybrid",
    "hybrid_reranker",
]

for mode in modes:

    correct1 = 0
    correct3 = 0
    correct5 = 0
    mrr = 0

    failures = []

    total = len(dataset)

    print("\n" + "=" * 60)
    print(f"Evaluating : {mode.upper()}")
    print("=" * 60)

    for sample in dataset:

        question = sample["question"]

        expected = {
            doc.replace(".txt", "")
            for doc in sample["expected_documents"]
        }

        retrieved = retriever.retrieve(
            question,
            top_k=5,
            mode=mode,
        )

        sources = [chunk["source"] for chunk in retrieved]

        if any(src in expected for src in sources[:1]):
            correct1 += 1

        if any(src in expected for src in sources[:3]):
            correct3 += 1

        if any(src in expected for src in sources[:5]):
            correct5 += 1

        else:
            failures.append({
                "question": question,
                "expected": list(expected),
                "retrieved": sources,
            })

        mrr += reciprocal_rank(
            sources,
            expected,
        )

    print(f"Questions : {total}")
    print(f"Recall@1 : {correct1/total:.2%}")
    print(f"Recall@3 : {correct3/total:.2%}")
    print(f"Recall@5 : {correct5/total:.2%}")
    print(f"MRR       : {mrr/total:.3f}")

    with open(
        f"evaluation/failures_{mode}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            failures,
            f,
            indent=4,
        )