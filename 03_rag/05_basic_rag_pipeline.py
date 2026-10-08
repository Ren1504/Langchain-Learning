"""
05_basic_rag_pipeline.py
========================
Module: RAG - End-to-End Retrieval-Augmented Generation (RAG) Pipeline

Retrieval-Augmented Generation (RAG) grounds LLM responses on your private,
domain-specific knowledge base without fine-tuning.

Standard RAG Architecture:
                    ┌────────────────────────┐
                    │ Raw Knowledge Document │
                    └───────────┬────────────┘
                                │ Split (RecursiveCharacterTextSplitter)
                    ┌───────────▼────────────┐
                    │ Document Text Chunks   │
                    └───────────┬────────────┘
                                │ Embed & Index
                    ┌───────────▼────────────┐
                    │ Chroma Vector Database │
                    └───────────┬────────────┘
                                │
   User Query ──> [Retriever] ──┴──> Top-K Relevant Context Chunks
                                               │
                                               ▼
         [Prompt: Context + Question] ──> [LLM] ──> Grounded Answer

Techniques demonstrated in this file:
1. `demo_basic_rag`: Canonical LCEL RAG chain with `RunnablePassthrough`.
2. `rag_with_sources`: Source attribution — formatting citations directly into responses.
3. `rag_with_fallback`: Strict grounding — instructing the model to say "I don't know" to prevent hallucinations.
4. `structured_rag`: Emitting typed Pydantic responses (`answer`, `confidence`, `sources_used`, `follow_up`).
5. `retail_faq_exercise`: Real-world FAQ bot for an e-commerce retail store (Dorothy Retail).

Prerequisites:
- OPENAI_API_KEY set in `.env`
"""

import tempfile
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

load_dotenv()

# Embedding and Chat Models
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
    temperature=0.3,  # Low temperature for factual precision
)

# Reference Knowledge Base Document
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components
1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.
2. **Prompts**: Templates for structuring inputs to language models.
3. **Chains**: Sequences of calls to models and other components.
4. **Agents**: Systems that use LLMs to determine which actions to take.
5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph
LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop workflows
- Persistence

## Pricing
LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started
Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


def create_kb() -> Chroma:
    """Helper: Indexes the knowledge base into an in-memory Chroma vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledgebase.md"})
    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store


# ============================================================================
# 1. CANONICAL LCEL RAG CHAIN
# ============================================================================

def demo_basic_rag():
    """
    Demonstrates the standard LCEL RAG pattern.
    
    The dictionary:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    simultaneously:
    - Passes the user question to the retriever, then formats retrieved docs into a string.
    - Passes the raw user question unchanged via `RunnablePassthrough()`.
    """
    print("\n--- 1. Canonical LCEL RAG Pipeline ---")

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:
{context}

Question: {question}

Make sure to answer concisely. If you do not know the answer from the context, respond with "I don't know"."""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
        "Who is the president of USA?",  # Not in context -> triggers refusal
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}\nA: {answer}\n" + "-" * 40)


# ============================================================================
# 2. RAG WITH SOURCE ATTRIBUTION / CITATIONS
# ============================================================================

def rag_with_sources():
    """
    Demonstrates including source metadata in the context so the model can cite
    the exact files or paragraphs used in its answer.
    """
    print("\n--- 2. RAG with Source Citations ---")

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context. Always cite which sources and sections you used.
{context}

Question: {question}

Answer (including citations):"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            formatted.append(f"[Source {i}: {src}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "What are the core components of LangChain?"
    print(f"Query: {query}")
    answer = rag_chain.invoke(query)
    print(f"Answer:\n{answer}")


# ============================================================================
# 3. STRUCTURED OUTPUT RAG (PYDANTIC)
# ============================================================================

def structured_rag():
    """
    Demonstrates generating structured JSON/Pydantic output from a RAG pipeline.
    Instead of unstructured free text, the LLM outputs a typed schema with:
    - `answer`: Main textual reply
    - `confidence`: 'high', 'medium', or 'low'
    - `sources_used`: List of referenced documentation sections
    - `follow_up`: Suggested next query for the user
    """
    print("\n--- 3. Structured Output RAG (Pydantic Schema) ---")

    class RAGResponse(BaseModel):
        answer: str = Field(description="Direct, factual answer to the question.")
        confidence: str = Field(description="Confidence level: high, medium, or low.")
        sources_used: List[str] = Field(description="List of specific sources or sections referenced.")
        follow_up: str = Field(description="A helpful follow-up question the user might ask next.")

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
Context:
{context}

Question: {question}

Provide your response according to the requested schema."""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    result: RAGResponse = rag_chain.invoke("What is LangGraph?")

    print(f"Answer:     {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources:    {result.sources_used}")
    print(f"Follow-up:  {result.follow_up}")


# ============================================================================
# 4. PRACTICAL EXERCISE: E-COMMERCE FAQ BOT (DOROTHY RETAIL)
# ============================================================================

def retail_faq_exercise():
    """
    Hands-on exercise: Building a customer service FAQ bot for 'Dorothy Retail'.
    Indexes a multi-topic store policy document and answers complex customer queries.
    """
    print("\n--- 4. Practical Exercise: E-Commerce Store FAQ Bot ---")

    policy_doc = """Dorothy is an online retail company that sells a variety of products through its website. Customers can browse available products, place orders online, and receive their purchases at their chosen delivery address.

Dorothy usually processes orders within 1–2 business days after the order is placed. Once shipped, standard delivery typically takes between 3–5 business days. Delivery times may vary depending on location and holidays.

Customers can track orders using the tracking number emailed upon dispatch. If an order hasn't arrived within the expected window, contact support with your order number.

Dorothy accepts returns for eligible products within 30 days of delivery. Items must be unused and in their original packaging. Damaged or defective items should be reported immediately for replacement or refund."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=30)
    doc = Document(page_content=policy_doc, metadata={"source": "Dorothy_Store_Policies.md"})
    chunks = splitter.split_documents([doc])

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    class CustomerServiceResponse(BaseModel):
        answer: str = Field(description="Helpful answer to customer question.")
        confidence: str = Field(description="Confidence level: high, medium, or low.")
        sources_used: List[str] = Field(description="Referenced policy sections.")
        follow_up: str = Field(description="Suggested next question.")

    structured_llm = llm.with_structured_output(CustomerServiceResponse)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful customer support bot for Dorothy Retail.
Use only the following store policy details:
{details}

Customer Question: {question}"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    faq_chain = (
        {"details": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    test_queries = [
        "What is the company's return window?",
        "How many days does standard delivery take?",
    ]

    for q in test_queries:
        res = faq_chain.invoke(q)
        print(f"Customer: {q}")
        print(f"Bot:      {res.answer}")
        print(f"Followup: {res.follow_up}\n" + "-" * 40)


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run RAG pipeline demos.
    """
    print("=" * 60)
    print("END-TO-END RAG PIPELINES & GROUNDED ANSWERING")
    print("=" * 60)

    demo_basic_rag()
    rag_with_sources()
    structured_rag()
    retail_faq_exercise()


if __name__ == "__main__":
    main()
