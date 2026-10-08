# 🦜️🔗 LangChain & LangGraph Learning Roadmap

A structured, hands-on learning repository designed to take you from foundational LangChain Expression Language (LCEL) concepts to advanced RAG and autonomous multi-actor LangGraph agent workflows.

---

## 📂 Repository Architecture

```
LangChain-Learning/
│
├── 📁 01_core_concepts/            # 1. Fundamentals, LCEL, Prompts & Parsers
│   ├── README.md                   # Core concepts overview & cheatsheet
│   ├── 01_basics_and_parsers.py    # Models, prompts, streaming, batching, Pydantic schemas
│   └── 02_runnable_patterns.py     # RunnableParallel, Sequence, Passthrough, Branch
│
├── 📁 02_memory/                   # 2. Conversational Memory & Persistence
│   ├── README.md                   # Memory strategies comparison table & trade-offs
│   ├── 01_conversation_memory.py   # In-memory buffer, token trimming, window, summary & SQLite
│   └── chat_history.db             # Local SQLite database for persistent session storage
│
├── 📁 03_rag/                      # 3. Retrieval-Augmented Generation (RAG)
│   ├── README.md                   # RAG architecture overview & pipeline guide
│   ├── 01_document_loaders.py      # Text, web scraping, directory batching, lazy loading, PDF
│   ├── 02_text_splitters.py        # Recursive, token-based, Markdown header, Python code chunking
│   ├── 03_embeddings.py            # Vector math, OpenAI vs HuggingFace, cosine similarity, cache
│   ├── 04_vector_stores_chroma.py  # Chroma DB CRUD, similarity with score, MMR search, filtering
│   ├── 05_basic_rag_pipeline.py    # Canonical LCEL RAG, source citations, structured FAQ bot
│   ├── 06_advanced_rag.py          # Multi-Query, Contextual Compression, Hybrid BM25, Parent Doc
│   ├── 📁 data/
│   │   └── langchain_demo.pdf      # Sample PDF document for loader testing
│   └── 📁 chroma_db/               # Persistent Chroma vector index storage
│
├── 📁 04_langgraph/                # 4. Stateful Multi-Actor Agent Orchestration
│   ├── README.md                   # LangGraph architecture & agent cycles guide
│   ├── 01_graph_basics.py          # StateGraph, TypedDict schemas, reducers (operator.add)
│   ├── 02_conversation_graph.py    # Sentiment classifier node -> Adaptive tone responder node
│   ├── 03_conditional_routing.py   # Dynamic branching, multi-path routing, quality review loops
│   ├── 04_cycles_and_loops.py      # Self-correcting code agent loop with syntax error feedback
│   ├── 05_human_in_the_loop.py     # State checkpointing, breakpoints, pause & approval flow
│   └── 📁 assets/                  # Mermaid graph diagrams and rendered PNG topologies
│       ├── basic_graph.png
│       ├── conditional_graph.png
│       ├── conditional_looping_graph.png
│       └── graph.png
│
├── .env                            # Environment variables (API keys)
├── requirements.txt                # Python package dependencies
└── README.md                       # Master Learning Syllabus (this file)
```

---

## 🚀 Setup & Installation

### 1. Python Environment
Ensure you have Python 3.10+ installed. Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory with your provider API keys:

```env
# Required for OpenAI models and embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Required for Google Gemini models
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: LangSmith Tracing & Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langchain-learning
```

---

## 🗺️ Step-by-Step Learning Curriculum

### Phase 1: Core Concepts & LCEL
Start here to understand how LangChain models and chains communicate.
- [`01_core_concepts/01_basics_and_parsers.py`](file:///d:/Flutter/LangChain-Learning/01_core_concepts/01_basics_and_parsers.py):
  - Initializing models via `init_chat_model` and `ChatGoogleGenerativeAI`.
  - The pipe syntax `prompt | model | parser`.
  - Batching (`.batch()`) and real-time token streaming (`.stream()`).
  - Few-shot prompting and Pydantic structured output parsing.
- [`01_core_concepts/02_runnable_patterns.py`](file:///d:/Flutter/LangChain-Learning/01_core_concepts/02_runnable_patterns.py):
  - `RunnableParallel` for running tasks concurrently.
  - `RunnablePassthrough` & `.assign()` for passing state and enriching dictionaries.
  - `RunnableBranch` for dynamic intent-based routing.

### Phase 2: Conversational Memory
Learn how to maintain dialogue context in stateless LLMs.
- [`02_memory/01_conversation_memory.py`](file:///d:/Flutter/LangChain-Learning/02_memory/01_conversation_memory.py):
  - `InMemoryChatMessageHistory` with `RunnableWithMessageHistory`.
  - Multi-session isolation via `session_id`.
  - Token trimming with `trim_messages` to prevent exceeding model context windows.
  - Sliding window buffer (keeping the last $K$ turns).
  - Rolling summary memory (compressing older dialogue with an LLM while keeping recent turns verbatim).
  - True persistence with SQLite (`SQLChatMessageHistory`).

### Phase 3: Retrieval-Augmented Generation (RAG)
Master how to connect LLMs with custom external data.
- [`03_rag/01_document_loaders.py`](file:///d:/Flutter/LangChain-Learning/03_rag/01_document_loaders.py): Ingestion from text, web pages, directories, and PDFs.
- [`03_rag/02_text_splitters.py`](file:///d:/Flutter/LangChain-Learning/03_rag/02_text_splitters.py): Splitting strategies, chunk overlap, Markdown headers, and code splitters.
- [`03_rag/03_embeddings.py`](file:///d:/Flutter/LangChain-Learning/03_rag/03_embeddings.py): Dense vector math, cosine similarity, dimensions, and disk caching.
- [`03_rag/04_vector_stores_chroma.py`](file:///d:/Flutter/LangChain-Learning/03_rag/04_vector_stores_chroma.py): Chroma DB indexing, similarity search with score, MMR diversity search, and metadata filtering.
- [`03_rag/05_basic_rag_pipeline.py`](file:///d:/Flutter/LangChain-Learning/03_rag/05_basic_rag_pipeline.py): Canonical LCEL RAG chain, source citations, hallucination prevention, and structured customer service FAQ bots.
- [`03_rag/06_advanced_rag.py`](file:///d:/Flutter/LangChain-Learning/03_rag/06_advanced_rag.py): Multi-Query generation, Contextual Compression, Hybrid Search (BM25 + Chroma), and Parent Document Retriever.

### Phase 4: LangGraph Agent Workflows
Build autonomous, stateful multi-actor systems with self-correction and human control.
- [`04_langgraph/01_graph_basics.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/01_graph_basics.py): `StateGraph`, `TypedDict` schemas, and reducers (`operator.add`, `add_messages`).
- [`04_langgraph/02_conversation_graph.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/02_conversation_graph.py): Multi-node sentiment classifier and adaptive persona responder. Exporting graph diagrams to Mermaid and PNG.
- [`04_langgraph/03_conditional_routing.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/03_conditional_routing.py): Conditional edges, intent routers, 2D decision matrices, and quality evaluation loops.
- [`04_langgraph/04_cycles_and_loops.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/04_cycles_and_loops.py): Self-correcting agents with compiler error feedback loops and recursion safeguards.
- [`04_langgraph/05_human_in_the_loop.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/05_human_in_the_loop.py): Checkpointing with `MemorySaver`, pause breakpoints with `interrupt_before`, state inspection, modification, and resumption.

---

## 🏃 Running Examples

Every script includes a dedicated `if __name__ == "__main__":` entrypoint with pre-configured examples ready to run:

```bash
# Core Concepts
python 01_core_concepts/01_basics_and_parsers.py
python 01_core_concepts/02_runnable_patterns.py

# Memory
python 02_memory/01_conversation_memory.py

# RAG
python 03_rag/01_document_loaders.py
python 03_rag/02_text_splitters.py
python 03_rag/03_embeddings.py
python 03_rag/04_vector_stores_chroma.py
python 03_rag/05_basic_rag_pipeline.py
python 03_rag/06_advanced_rag.py

# LangGraph
python 04_langgraph/01_graph_basics.py
python 04_langgraph/02_conversation_graph.py
python 04_langgraph/03_conditional_routing.py
python 04_langgraph/04_cycles_and_loops.py
python 04_langgraph/05_human_in_the_loop.py
```
