from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(documents=SAMPLE_DOCS,
                                            embedding=embeddings_model,
                                            persist_directory=tmpdir)

        print(f"Vector Store Created {vectorstore._collection.count()} and persisted")

        query = "What is Langchain"
        results = vectorstore.similarity_search(query,k=2)

        print(f"Top 2 results for {query} :")
        for i , doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

def similarity_search_score():
    with tempfile.TemporaryDirectory() as tmpdir:

        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,embedding=embeddings_model,persist_directory=tmpdir
        )

        query = 'Explain vector stores.'
        results_with_scores = vectorstore.similarity_search_with_score(query,k=3)

        print(f"Top 3 results for {query} :")
        for i , (doc,score) in enumerate(results_with_scores):
            print(f"Result {i+1}: {doc.page_content} Score: {score:.4f}(Source: {doc.metadata['source']})")


def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding= embeddings_model, persist_directory=tmpdir
        )

        query = "What datanases are available"

        results = vectorstore.similarity_search(query,k=5)
        print(f"results without metadata filtering {query}")
        print(f"Top 5 results for {query} :")
        for i , doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

        filter_criteria = {"topic":"database"}
        filter_results = vectorstore.similarity_search(query,k=5,filter=filter_criteria)
        print(f"Top 5 filtered results for {query} :")
        for i , doc in enumerate(filter_results):
            print(f"Result {i+1}: {doc.page_content}(Source: {doc.metadata['source']})")

def persistent():
    persist_dir = "./chroma_db/"

    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        persist_directory=persist_dir,
    )

    original_count = vectorstore._collection.count()
    print(f"Persisted vector store with {original_count} documents.")
    print(f"Vector store persisted at: {persist_dir}")

    # simulate restart - load from disk
    del vectorstore

    reloaded = Chroma(
        embedding_function=embeddings_model,
        persist_directory=persist_dir,
    )

    reloaded_count = reloaded._collection.count()
    print(f"Reloaded vector store with {reloaded_count} documents.")

    # verify search still works
    results = reloaded.similarity_search("LangChain", k=2)
    print(f"Search result: {results[0].page_content[:50]}...")

def as_retriever():

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=tmpdir,
        )

        # basic retriever usage
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        # use retriever to get relevant documents
        docs = retriever.invoke("How do I build AI applications?")

        print("Retriever results:")
        for i, doc in enumerate(docs):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )

        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 5},  # fetch 5 docs and return 3 diverse
        )
        mmr_docs = mmr_retriever.invoke("vector databases and embeddings")
        print("\nMMR Retriever results:")
        for i, doc in enumerate(mmr_docs):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )

def exercise():
    splitter = RecursiveCharacterTextSplitter(chunk_size = 50,chunk_overlap= 20)

    chunks = splitter.split_documents(SAMPLE_DOCS)
    persist_dir = "./chroma_db/"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_dir,
    )

    retriever = vectorstore.as_retriever(
            search_type = 'similarity',
            search_kwargs={"k":3}
        )

    return retriever

def main():
    # chroma_basics()
    # similarity_search_score()
    # metadata_filtering()
    # persistent()
    # as_retriever()
    retriever = exercise()
    docs = retriever.invoke("What is langchain")
    for doc in docs:
        print(doc.page_content)
        print("---")



if __name__ == "__main__":
    main()
        