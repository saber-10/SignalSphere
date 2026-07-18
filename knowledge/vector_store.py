import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import re

_faiss = None
_SentenceTransformer = None

def _get_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError(
                "Faiss is not installed. Please install it with `pip install faiss-cpu` or `pip install faiss-gpu`."
            )
    return _faiss

def _get_st():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformers is not installed. Please install it with `pip install sentence-transformers`."
            )
    return _SentenceTransformer

chunk_size = 900
chunk_overlap = 150
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
KB_DIR = Path(__file__).parent / "text_documents"
INDEX_PATH = Path(__file__).parent / "vector_store" / "index.faiss"

def _split_large_text(text: str, chunk_size: int, overlap: int):
    """Split oversized text into overlapping chunks at sentence boundaries."""
    chunks = []

    while len(text) > chunk_size:
        split = text.rfind("\n", 0, chunk_size)

        if split == -1:
            split = text.rfind(". ", 0, chunk_size)

        if split == -1:
            split = chunk_size

        chunks.append(text[:split].strip())

        text = text[max(0, split - overlap):].strip()

    if text:
        chunks.append(text)

    return chunks


def _chunk_text(text: str, source: str) -> List[Dict]:
    chunks = []

    # Remove decorative header
    header_end = text.find("SECTION 1")
    if header_end != -1:
        text = text[header_end:]

    # Split into sections while keeping section titles
    sections = re.split(r"(?=SECTION\s+\d+:)", text)

    for section in sections:

        section = section.strip()

        if not section:
            continue

        if len(section) <= chunk_size:

            chunks.append({
                "text": section,
                "source": source
            })

        else:

            for piece in _split_large_text(
                section,
                chunk_size,
                chunk_overlap,
            ):
                chunks.append({
                    "text": piece,
                    "source": source
                })

    return chunks




def load_knowledge_base() -> List[Dict[str, Any]]:
    knowledge_base = []
    for fpath in sorted(KB_DIR.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8")
        chunks = _chunk_text(text, source = fpath.stem)
        knowledge_base.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {fpath.name}")
    return knowledge_base

def build_index(force_rebuild: bool = False):
    faiss = _get_faiss()
    st = _get_st()

    meta_path = INDEX_PATH.with_suffix(".json")

    if not force_rebuild and INDEX_PATH.exists() and meta_path.exists():
        # Load existing index and metadata
        index = faiss.read_index(str(INDEX_PATH))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model = st(EMBEDDING_MODEL)
        print(f"Loaded existing index from {INDEX_PATH} with {index.ntotal} vectors.")
        return index, metadata, model
    
    # if we reach here, we need to build the index from scratch

    print("Building new index...")
    chunks = load_knowledge_base()
    texts = [chunk["text"] for chunk in chunks]

    model = st(EMBEDDING_MODEL)
    embeddings = model.encode(texts,  show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Using Inner Product for cosine similarity
    index.add(embeddings)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f , indent=2)
    
    print(f"Built new index with {index.ntotal} vectors and saved to {INDEX_PATH}.")
    return index, chunks, model


def retrieve(query: str, index, metadata: List[Dict], model, top_k: int = 5) -> List[Dict]:
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0<= idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

    return results

_SINGLETON: tuple | None = None

def get_retriever():
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = build_index()
    return _SINGLETON



if __name__ == "__main__":
    print("Testing vector store...")

    index, metadata, model = build_index(force_rebuild=True)

    print(f"\n✅ Success!")
    print(f"Vectors in index: {index.ntotal}")
    print(f"Chunks loaded: {len(metadata)}")

    results = retrieve(
        "What is Laplace Transform?",
        index,
        metadata,
        model,
        top_k=3,
    )

    print("\nTop Results:\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i}")
        print("Score:", round(result["score"], 4))
        print("Source:", result["source"])
        print(result["text"][:200], "...")
        print("-" * 60)