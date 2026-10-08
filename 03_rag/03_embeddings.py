"""
03_embeddings.py
================
Module: RAG - Vector Embeddings & Similarity Computation

What are Embeddings?
An embedding model converts text into a high-dimensional dense vector of numbers
(e.g., a list of 1536 floating-point values for OpenAI's `text-embedding-3-small`).
Texts with similar semantic meanings will have vectors that point in nearly the
same direction in vector space.

Key Concepts:
1. `embed_query(text)`: Generates an embedding for a single search string.
2. `embed_documents(list_of_texts)`: Generates embeddings for multiple documents in batch.
3. Vector Dimensions & Norms:
   - Dimensions: Number of numerical features (e.g. 1536 for OpenAI small, 384 for MiniLM).
   - L2 Norm: Magnitude of the vector (OpenAI vectors are pre-normalized to 1.0).
4. Cosine Similarity:
   Calculates the cosine of the angle between two vectors:
   cos(theta) = (A . B) / (||A|| * ||B||)
   - 1.0 = Semantically identical.
   - 0.0 = Completely unrelated.
5. `CacheBackedEmbeddings`:
   Caches embeddings on disk or in key-value stores to prevent redundant, expensive API calls.

Prerequisites:
- OPENAI_API_KEY set in `.env`
- Optional: `sentence-transformers` / `langchain-huggingface` for local embeddings
"""

import tempfile
import numpy as np
from dotenv import load_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.storage import LocalFileStore

# In LangChain 0.3+, CacheBackedEmbeddings is in langchain.embeddings
try:
    from langchain.embeddings import CacheBackedEmbeddings
except ImportError:
    from langchain_classic.embeddings.cache import CacheBackedEmbeddings

load_dotenv()


# ============================================================================
# 1. OPENAI EMBEDDINGS BASICS
# ============================================================================

def basic_embed():
    """
    Demonstrates generating an embedding vector for a single query using OpenAI.
    """
    print("\n--- 1. Basic Single Query Embedding ---")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text = "What is the capital of France?"

    vector = embeddings.embed_query(text)

    print(f"Text: '{text}'")
    print(f"Total Vector Dimensions: {len(vector)}")
    print(f"First 5 Float Values: {vector[:5]}")
    # Compute Euclidean norm: OpenAI vectors are normalized to unit length (1.0)
    print(f"Vector L2 Norm: {np.linalg.norm(vector):.4f}")


def batch_embed():
    """
    Demonstrates batch embedding with `embed_documents`.
    Batching sends multiple texts in a single HTTP payload, drastically
    reducing API overhead and latency.
    """
    print("\n--- 2. Batch Embedding Multiple Documents ---")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    texts = [
        "What is the day today?",
        "We have been in the meeting for ten minutes.",
        "Always start my mornings with coffee.",
    ]

    batch_vectors = embeddings.embed_documents(texts)

    print(f"Total documents embedded: {len(batch_vectors)}")
    for i, (text, vec) in enumerate(zip(texts, batch_vectors)):
        print(f"  Doc {i + 1}: '{text}' -> Dim: {len(vec)}, Norm: {np.linalg.norm(vec):.2f}")


# ============================================================================
# 2. LOCAL HUGGINGFACE EMBEDDINGS (OPEN SOURCE)
# ============================================================================

def huggingface_demo():
    """
    Demonstrates local embeddings using open-source HuggingFace models.
    Runs entirely on your local machine with zero external API fees.
    Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
    """
    print("\n--- 3. Local HuggingFace Embeddings (all-MiniLM-L6-v2) ---")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text = "We are going to win the championship finals."

        vector = embeddings.embed_query(text)
        print(f"Local Embedding Dimensions: {len(vector)} (Compact 384-d vector)")
        print(f"First 5 Values: {vector[:5]}")
    except Exception as e:
        print(f"HuggingFace demo skipped (install langchain-huggingface sentence-transformers): {e}")


# ============================================================================
# 3. COSINE SIMILARITY CALCULATION
# ============================================================================

def cosine_similarity(a: list, b: list) -> float:
    """Computes the cosine similarity between two vectors a and b."""
    vec_a = np.array(a)
    vec_b = np.array(b)
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def similarity_demo():
    """
    Demonstrates semantic similarity retrieval using vector math.
    Finds the most semantically relevant documents for a user query.
    """
    print("\n--- 4. Semantic Similarity Search via Cosine Distance ---")

    docs = [
        "C# is widely used in game development and Unity.",
        "She will be home tomorrow evening.",
        "Python is a versatile programming language for data science and AI.",
        "JavaScript and TypeScript are dominant in web applications.",
        "We will be arriving tomorrow at your home.",
    ]

    query = "What programming languages exist?"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Embed query and documents
    query_vector = embeddings.embed_query(query)
    doc_vectors = embeddings.embed_documents(docs)

    scores = []
    for doc, vec in zip(docs, doc_vectors):
        sim = cosine_similarity(query_vector, vec)
        scores.append((doc, sim))

    # Sort descending by similarity score
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: '{query}'\n")
    print("Ranked Results (Highest similarity to lowest):")
    for rank, (doc, score) in enumerate(scores, 1):
        print(f"  #{rank} [Score: {score:.4f}] {doc}")


# ============================================================================
# 4. CACHED EMBEDDINGS (AVOID REPEATED API CALLS)
# ============================================================================

def cache_backed_embeddings_demo():
    """
    Demonstrates `CacheBackedEmbeddings`.
    Wraps an embedding model with a persistent key-value store (e.g. LocalFileStore).
    If a document has already been embedded, its vector is fetched instantly from disk.
    """
    print("\n--- 5. Cache-Backed Embeddings (Cost & Speed Optimization) ---")

    with tempfile.TemporaryDirectory() as cache_dir:
        store = LocalFileStore(cache_dir)
        base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=base_embeddings,
            document_embedding_cache=store,
            namespace=base_embeddings.model,
        )

        sample_texts = ["Hello world", "Learning LangChain is exciting", "Hello world"]

        print("First pass (calls embedding API and writes to cache):")
        vectors_1 = cached_embedder.embed_documents(sample_texts)
        print(f"Embedded {len(vectors_1)} items.")

        print("\nSecond pass (retrieved from local cache instantly with 0 API calls):")
        vectors_2 = cached_embedder.embed_documents(sample_texts)
        print(f"Retrieved {len(vectors_2)} items from cache.")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run Vector Embedding demos.
    """
    print("=" * 60)
    print("LANGCHAIN EMBEDDINGS & SIMILARITY SEARCH")
    print("=" * 60)

    basic_embed()
    batch_embed()
    similarity_demo()
    cache_backed_embeddings_demo()
    # huggingface_demo()


if __name__ == "__main__":
    main()
