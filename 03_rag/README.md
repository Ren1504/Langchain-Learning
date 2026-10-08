# 03 - Retrieval-Augmented Generation (RAG)

This folder contains the complete RAG curriculum, progressing from raw document ingestion to advanced multi-strategy retrieval.

## Curriculum & Architecture

```
Ingestion Pipeline:
Raw Documents ──> [01_document_loaders] ──> [02_text_splitters] ──> [03_embeddings] ──> [04_vector_stores_chroma]
                                                                                               │
Query Pipeline:                                                                                ▼
User Question ──> [05_basic_rag_pipeline] / [06_advanced_rag] <──────────────────────── Vector Index
                               │
                               ▼
                        Grounded Answer
```

## Detailed File Breakdown

| File | Topics & Techniques |
| :--- | :--- |
| [`01_document_loaders.py`](file:///d:/Flutter/LangChain-Learning/03_rag/01_document_loaders.py) | • `Document` data model (`page_content`, `metadata`)<br>• `TextLoader`, `WebBaseLoader`, `DirectoryLoader`<br>• Memory-efficient `lazy_load()` generators<br>• `PyPDFLoader` loading [`data/langchain_demo.pdf`](file:///d:/Flutter/LangChain-Learning/03_rag/data/langchain_demo.pdf) |
| [`02_text_splitters.py`](file:///d:/Flutter/LangChain-Learning/03_rag/02_text_splitters.py) | • `RecursiveCharacterTextSplitter` hierarchy (`\n\n`, `\n`, ` `, `""`)<br>• Chunk size vs chunk overlap trade-offs<br>• `TokenTextSplitter` (token-boundary alignment)<br>• `MarkdownHeaderTextSplitter` (preserves document outline)<br>• Python code splitter (`Language.PYTHON`) |
| [`03_embeddings.py`](file:///d:/Flutter/LangChain-Learning/03_rag/03_embeddings.py) | • OpenAI embeddings (`text-embedding-3-small`)<br>• Local HuggingFace embeddings (`all-MiniLM-L6-v2`)<br>• Vector dimensions, norms, and Cosine Similarity math<br>• `CacheBackedEmbeddings` with `LocalFileStore` |
| [`04_vector_stores_chroma.py`](file:///d:/Flutter/LangChain-Learning/03_rag/04_vector_stores_chroma.py) | • In-memory and persistent Chroma vector stores<br>• Similarity search with distance scores<br>• Maximum Marginal Relevance (MMR) for diversity<br>• Metadata filtering (`filter={"topic": "database"}`)<br>• Vector ID CRUD (adding, updating, deleting) |
| [`05_basic_rag_pipeline.py`](file:///d:/Flutter/LangChain-Learning/03_rag/05_basic_rag_pipeline.py) | • Canonical LCEL RAG chain (`{"context": retriever \| format_docs, "question": RunnablePassthrough()}`)<br>• Source citations and document attribution<br>• Hallucination prevention & graceful fallback<br>• Typed Pydantic structured output RAG<br>• Real-world Dorothy retail store FAQ exercise |
| [`06_advanced_rag.py`](file:///d:/Flutter/LangChain-Learning/03_rag/06_advanced_rag.py) | • **Multi-Query Retriever**: Synthetic query expansion<br>• **Contextual Compression**: `LLMChainExtractor` noise reduction<br>• **Hybrid Search**: `EnsembleRetriever` combining BM25 keyword search + Chroma vector search<br>• **Parent Document Retriever**: Child chunk vector indexing + Parent chunk LLM context<br>• End-to-end advanced pipeline |

## Data & Storage Folders

- [`data/langchain_demo.pdf`](file:///d:/Flutter/LangChain-Learning/03_rag/data/langchain_demo.pdf): Sample PDF for testing loaders and indexing.
- `chroma_db/`: Persistent local Chroma database storage directory.

## How to Run

```bash
python 03_rag/01_document_loaders.py
python 03_rag/02_text_splitters.py
python 03_rag/03_embeddings.py
python 03_rag/04_vector_stores_chroma.py
python 03_rag/05_basic_rag_pipeline.py
python 03_rag/06_advanced_rag.py
```
