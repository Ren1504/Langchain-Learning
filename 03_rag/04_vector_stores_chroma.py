"""
04_vector_stores_chroma.py
==========================
Module: RAG - Vector Stores with Chroma DB

Vector stores are specialized databases engineered for storing high-dimensional
embeddings and performing ultra-fast similarity searches (Approximate Nearest
Neighbors / ANN).

Chroma is a lightweight, open-source embedding database that runs embedded
(in-process) without needing a separate server daemon.

Techniques covered in this module:
1. Creating & Persisting a Chroma collection from `Document` objects.
2. Standard Similarity Search (`similarity_search(query, k=...)`).
3. Similarity Search with Distance Scores (`similarity_search_with_score(query, k=...)`).
   (Lower score in Chroma L2/cosine distance = closer/more similar).
4. Maximum Marginal Relevance (MMR) Search:
   Retrieves documents that are both relevant to the query AND diverse from each other,
   avoiding redundant identical results.
5. Metadata Filtering:
   Filtering candidates by metadata attributes (e.g. `{"topic": "database"}`).
6. Updating and Deleting vectors by ID.

Prerequisites:
- Requires `langchain-chroma`, `chromadb`, and OPENAI_API_KEY in `.env`.
"""

from pathlib import Path
import tempfile
from dotenv import load_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Embedding model used for vector indexing
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Persistent directory inside 03_rag folder
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"

# Sample knowledge base documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview", "doc_id": "1"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview", "doc_id": "2"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database", "doc_id": "3"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate, grounded LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture", "doc_id": "4"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals", "doc_id": "5"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database", "doc_id": "6"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook AI Research.",
        metadata={"source": "faiss_docs", "topic": "database", "doc_id": "7"},
    ),
    Document(
        page_content="Pinecone is a managed cloud vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database", "doc_id": "8"},
    ),
]


# ============================================================================
# 1. INITIALIZING & QUERYING CHROMA
# ============================================================================

def chroma_basics():
    """
    Demonstrates creating an in-memory Chroma vector store and querying top-k results.
    """
    print("\n--- 1. Chroma Initialization & Basic Similarity Search ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Chroma.from_documents embeds and indexes the documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "What is LangChain?"
        results = vectorstore.similarity_search(query, k=2)

        print(f"Query: '{query}'")
        print(f"Top {len(results)} Matches:")
        for i, doc in enumerate(results, 1):
            print(f"  #{i} [{doc.metadata.get('source')}]: {doc.page_content}")


# ============================================================================
# 2. SIMILARITY SEARCH WITH DISTANCE SCORES
# ============================================================================

def similarity_with_scores():
    """
    Demonstrates `similarity_search_with_score`.
    Returns tuples of `(Document, distance_score)`.
    For Chroma with cosine/L2 distance:
    - Distance closer to 0 = High semantic similarity.
    - Distance larger = Less similar.
    """
    print("\n--- 2. Similarity Search With Distance Scores ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "Tell me about vector databases and indexing."
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print(f"Query: '{query}'")
        for doc, score in results_with_scores:
            print(f"  [Distance Score: {score:.4f}] {doc.page_content}")


# ============================================================================
# 3. MAXIMUM MARGINAL RELEVANCE (MMR) SEARCH
# ============================================================================

def mmr_search_demo():
    """
    Demonstrates Maximum Marginal Relevance (MMR).
    Standard similarity search often returns 3 chunks that say the exact same thing.
    MMR penalizes redundancy:
    - `fetch_k`: Number of initial candidate documents to fetch (e.g. 10).
    - `k`: Number of final diverse documents to return (e.g. 3).
    - `lambda_mult`: 0.0 = Maximum diversity, 1.0 = Maximum relevance (standard search).
    """
    print("\n--- 3. Maximum Marginal Relevance (MMR) Search ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "database"

        print("Standard Similarity Search (May cluster around one specific concept):")
        standard_results = vectorstore.similarity_search(query, k=3)
        for d in standard_results:
            print(f"  - {d.page_content}")

        print("\nMMR Search (Actively selects diverse facets of the topic):")
        mmr_results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=6, lambda_mult=0.5)
        for d in mmr_results:
            print(f"  - {d.page_content}")


# ============================================================================
# 4. METADATA FILTERING
# ============================================================================

def metadata_filtering_demo():
    """
    Demonstrates filtering candidates using metadata conditions before or during retrieval.
    This lets you narrow search results to specific users, dates, or topics.
    """
    print("\n--- 4. Metadata Filtering ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        query = "tools for AI"

        # Restrict search only to documents where topic == "database"
        filtered_results = vectorstore.similarity_search(
            query,
            k=2,
            filter={"topic": "database"},
        )

        print(f"Query: '{query}' with filter topic='database':")
        for d in filtered_results:
            print(f"  Topic: {d.metadata['topic']} | Content: {d.page_content}")


# ============================================================================
# 5. UPDATING & DELETING DOCUMENTS IN CHROMA
# ============================================================================

def update_and_delete_demo():
    """
    Demonstrates adding, updating, and deleting vectors by ID.
    Essential for production applications where documents change or expire.
    """
    print("\n--- 5. Document Management: IDs, Updates & Deletion ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma(
            collection_name="demo_collection",
            embedding_function=embeddings_model,
            persist_directory=tmpdir,
        )

        doc1 = Document(page_content="Old version of product documentation.", metadata={"v": 1})
        ids = vectorstore.add_documents([doc1], ids=["doc_alpha"])
        print(f"Inserted document with ID: {ids[0]}")

        # Update: Overwrite document content using the same ID
        updated_doc = Document(page_content="New 2026 version of product documentation.", metadata={"v": 2})
        vectorstore.update_document(document_id="doc_alpha", document=updated_doc)
        print("Updated document 'doc_alpha'.")

        result = vectorstore.similarity_search("product", k=1)
        print(f"After Update: '{result[0].page_content}'")

        # Delete by ID
        vectorstore.delete(ids=["doc_alpha"])
        print("Deleted document 'doc_alpha'.")
        remaining = vectorstore.similarity_search("product", k=1)
        print(f"Remaining matching docs after delete: {len(remaining)}")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run Chroma DB vector store demos.
    """
    print("=" * 60)
    print("CHROMA VECTOR DATABASE: INDEXING, RETRIEVAL & FILTERING")
    print("=" * 60)

    chroma_basics()
    similarity_with_scores()
    mmr_search_demo()
    metadata_filtering_demo()
    update_and_delete_demo()


if __name__ == "__main__":
    main()
