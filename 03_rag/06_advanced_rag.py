"""
06_advanced_rag.py
==================
Module: RAG - Advanced Retrieval Strategies

Why Naive RAG Fails in Production:
1. Phrasing Sensitivity: If a user writes an unusual query or uses synonyms,
   dense vector search can fail to match the indexed documents.
2. Irrelevant Noise: Chunks often contain 80% background details and only 20%
   actual answers. Feeding entire chunks wastes prompt tokens and causes hallucinations.
3. Keyword Blindness: Vector search is great for semantic ideas, but struggles
   with exact acronyms, IDs, or rare terms (e.g., "ACID", "CVE-2024-1234").
4. Chunk Size Dilemma: Small chunks retrieve accurately but lack broad context;
   large chunks provide full context but dilute vector accuracy.

Advanced Retrieval Techniques Demonstrated:
1. `MultiQueryRetriever`: Uses an LLM to generate 3 alternative query variations,
   retrieves documents for all variations, and takes the union of results.
2. `ContextualCompressionRetriever` with `LLMChainExtractor`: Uses an LLM to extract
   and compress ONLY the relevant sentences from retrieved chunks before sending to the prompt.
3. `EnsembleRetriever` (Hybrid Search): Combines keyword search (`BM25Retriever`)
   with dense semantic search (`Chroma`) using Reciprocal Rank Fusion (RRF).
4. `ParentDocumentRetriever`: Splits documents into small child chunks (for pinpoint
   vector search) but stores and returns the parent chunk (for complete context).
5. `advanced_rag_chain`: End-to-end combination of Multi-Query and Contextual Compression.

Prerequisites:
- OPENAI_API_KEY in `.env`
- Requires `rank_bm25` for BM25 retriever: `pip install rank-bm25`
"""

import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Safe imports for retrievers across langchain versions
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever, ParentDocumentRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.storage import InMemoryStore
except ImportError:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, ParentDocumentRetriever
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    from langchain_classic.storage import InMemoryStore

from langchain_community.retrievers import BM25Retriever

load_dotenv()

# Configure logging to observe multi-query generation in the terminal
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Models
llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Benchmark Documents
INFO_BURIED = [
    Document(
        page_content="""ACME AI SOLUTIONS - COMPANY OVERVIEW & TECH STACK
Founded in 2018 in San Francisco, ACME AI Solutions helps enterprises deploy generative AI systems.
Our backend stack runs Python and FastAPI, orchestrated on AWS EKS with Kubernetes and Istio.
Our transactional database is PostgreSQL, caching runs on Redis, and vector search runs on Pinecone.
For LLM workflows, our engineers use LangChain and LangGraph.
LangChain provides prompt templates, output parsers, and tool interfaces.
LangGraph adds stateful graph orchestration, cyclical self-correction, and human-in-the-loop approvals.
Employee benefits include comprehensive health coverage, unlimited PTO, and a 401(k) with 4% match.""",
        metadata={"source": "acme_overview.pdf", "topic": "company_docs"},
    )
]

TECH_DOCS = [
    Document(
        page_content="Python is a high-level language popular for data science, machine learning, and automation.",
        metadata={"topic": "programming", "lang": "python"},
    ),
    Document(
        page_content="JavaScript and TypeScript power frontend web apps using modern frameworks like React and Next.js.",
        metadata={"topic": "programming", "lang": "javascript"},
    ),
    Document(
        page_content="PostgreSQL is an ACID compliant relational database supporting JSON and pgvector extensions.",
        metadata={"topic": "database", "type": "relational"},
    ),
    Document(
        page_content="Pinecone, Chroma, and Qdrant are vector databases optimized for nearest neighbor embedding search.",
        metadata={"topic": "database", "type": "vector"},
    ),
    Document(
        page_content="LangChain is a framework for chaining prompts, models, and retrieval components.",
        metadata={"topic": "ai", "framework": "langchain"},
    ),
    Document(
        page_content="LangGraph extends LangChain to build stateful multi-actor agent loops with cycles and human review.",
        metadata={"topic": "ai", "framework": "langgraph"},
    ),
]


def create_base_vectorstore() -> Chroma:
    """Helper: Indexes test documents into Chroma."""
    return Chroma.from_documents(
        documents=INFO_BURIED + TECH_DOCS,
        embedding=embeddings,
    )


# ============================================================================
# 1. MULTI-QUERY RETRIEVER (QUERY EXPANSION)
# ============================================================================

def multi_query_demo():
    """
    Overcomes prompt phrasing sensitivity.
    Generates 3 synthetic variations of the user's question, retrieves documents
    for each, and combines their unique results.
    """
    print("\n--- 1. Multi-Query Retriever (Query Expansion) ---")

    vectorstore = create_base_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_model,
    )

    query = "What technologies does ACME use for storing vectors and relational data?"
    print(f"Original Query: '{query}'")

    docs = multi_retriever.invoke(query)
    print(f"\nTotal Unique Documents Retrieved: {len(docs)}")
    for i, d in enumerate(docs, 1):
        print(f"  #{i}: {d.page_content[:120]}...")


# ============================================================================
# 2. CONTEXTUAL COMPRESSION (NOISE REDUCTION)
# ============================================================================

def contextual_compression_demo():
    """
    Demonstrates `ContextualCompressionRetriever` with `LLMChainExtractor`.
    Instead of passing the entire raw chunk into the prompt, the compressor
    uses an LLM call to strip out everything except sentences directly relevant
    to the query.
    """
    print("\n--- 2. Contextual Compression with LLMChainExtractor ---")

    vectorstore = create_base_vectorstore()
    compressor = LLMChainExtractor.from_llm(llm_model)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    )

    query = "What benefits does ACME offer to its employees?"
    print(f"Query: '{query}'")

    print("\n[A] WITHOUT Compression (Full 800-character chunk):")
    raw_docs = vectorstore.as_retriever(search_kwargs={"k": 1}).invoke(query)
    print(f"Length: {len(raw_docs[0].page_content)} chars")
    print(f"Content:\n{raw_docs[0].page_content[:150]}...")

    print("\n[B] WITH Compression (Only the extracted relevant facts):")
    compressed_docs = compression_retriever.invoke(query)
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Compressed Content:\n{doc.page_content.strip()}")


# ============================================================================
# 3. ENSEMBLE / HYBRID SEARCH (BM25 KEYWORDS + DENSE VECTORS)
# ============================================================================

def ensemble_hybrid_search():
    """
    Demonstrates Hybrid Search combining:
    - BM25Retriever: Sparse keyword search (great for exact matches, technical acronyms like ACID).
    - Chroma: Dense semantic vector search (great for conceptual meaning).
    Combined using Reciprocal Rank Fusion (RRF) with configurable weights.
    """
    print("\n--- 3. Ensemble / Hybrid Search (BM25 + Chroma) ---")

    vectorstore = create_base_vectorstore()

    # Keyword retriever
    bm25 = BM25Retriever.from_documents(TECH_DOCS)
    bm25.k = 2

    # Dense vector retriever
    semantic = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Hybrid ensemble: 40% keyword + 60% semantic
    ensemble = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.4, 0.6],
    )

    test_queries = [
        "ACID transactions",  # Keyword heavy
        "tools for building stateful reasoning loops",  # Semantic concept
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        res = ensemble.invoke(q)
        print(f"Top Hybrid Result: {res[0].page_content}")


# ============================================================================
# 4. PARENT DOCUMENT RETRIEVER (CHILD MATCHING -> PARENT CONTEXT)
# ============================================================================

def parent_document_retriever_demo():
    """
    Demonstrates `ParentDocumentRetriever`.
    Splits long documents into:
    - Small Child Chunks (e.g. 150 chars) -> Stored in Vector Store for sharp vector search.
    - Large Parent Chunks (e.g. 800 chars) -> Stored in Docstore (InMemoryStore).
    
    When a query matches a child chunk, the retriever fetches the entire parent
    chunk, giving the LLM complete surrounding context!
    """
    print("\n--- 4. Parent Document Retriever ---")

    long_article = Document(
        page_content="""
Complete Architecture Guide for Autonomous AI Agents

Section 1: Foundations
Autonomous agents combine LLMs with planning, memory, and tool invocation.
Chains are suitable for single-shot pipelines, but multi-step workflows require state persistence.

Section 2: Orchestration Frameworks
LangChain provides prompt templates, loaders, and model integrations.
LangGraph adds stateful multi-actor graphs, enabling cyclical execution and human approval.
CrewAI focuses on multi-agent role-playing teams collaborating on projects.

Section 3: Production Hardening
Observability with LangSmith provides distributed tracing and latency metrics.
Token budgets and rate-limiting safeguard production stability.""",
        metadata={"source": "agent_guide.md"},
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=embeddings,
    )
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents([long_article])

    query = "What does LangGraph add to agent workflows?"
    print(f"Query: '{query}'")

    results = retriever.invoke(query)
    print(f"Parent Document Retrieved (Length: {len(results[0].page_content)} chars):")
    print(results[0].page_content.strip())


# ============================================================================
# 5. END-TO-END ADVANCED RAG PIPELINE
# ============================================================================

def advanced_rag_chain_demo():
    """
    Combines Multi-Query expansion with Contextual Compression inside an LCEL RAG chain.
    """
    print("\n--- 5. End-to-End Advanced RAG Pipeline ---")

    vectorstore = create_base_vectorstore()
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm_model,
    )
    compressor = LLMChainExtractor.from_llm(llm_model)

    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever,
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following compressed context.
Cite specific technologies mentioned.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )

    query = "What frameworks does ACME use for LLM development?"
    print(f"Query: '{query}'")
    answer = rag_chain.invoke(query)
    print(f"Answer:\n{answer}")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run Advanced RAG demos.
    """
    print("=" * 60)
    print("ADVANCED RAG: MULTI-QUERY, COMPRESSION, HYBRID & PARENT RETRIEVAL")
    print("=" * 60)

    multi_query_demo()
    contextual_compression_demo()
    ensemble_hybrid_search()
    parent_document_retriever_demo()
    advanced_rag_chain_demo()


if __name__ == "__main__":
    main()
