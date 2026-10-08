"""
01_document_loaders.py
======================
Module: RAG (Retrieval-Augmented Generation) - Document Loaders

Document loaders are the entry point of the RAG ingestion pipeline. They extract
raw data from diverse sources (plain text, markdown, PDFs, websites, databases)
and standardize them into LangChain `Document` objects.

A LangChain `Document` consists of two primary attributes:
1. `page_content` (str): The actual extracted textual data.
2. `metadata` (dict): Contextual metadata such as file path, page number, URL, etc.

Loaders covered in this module:
- `TextLoader`: Loads standard text files from disk.
- `WebBaseLoader`: Scrapes and extracts text from web pages via BeautifulSoup.
- `DirectoryLoader`: Batches and loads files matching a glob pattern from a folder.
- `lazy_load()`: Memory-efficient generator streaming documents one-by-one.
- `PyPDFLoader`: Extracts text and page metadata from PDF files.

Prerequisites:
- Requires `pypdf`, `beautifulsoup4`
"""

import os
import tempfile
import warnings
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document

# Suppress minor deprecation warnings for cleaner educational output
warnings.simplefilter("ignore", DeprecationWarning)
load_dotenv()

# Path to the sample PDF document in the data folder
DEFAULT_PDF_PATH = Path(__file__).parent / "data" / "langchain_demo.pdf"


# ============================================================================
# 1. UNDERSTANDING THE DOCUMENT DATA MODEL
# ============================================================================

def doc_structure():
    """
    Examines the foundational LangChain `Document` class.
    Every loader, text splitter, and vector store operates on `Document` objects.
    """
    print("\n--- 1. LangChain Document Data Structure ---")

    doc = Document(
        page_content="LangChain is an orchestration framework for AI applications.",
        metadata={"source": "manual_entry.txt", "author": "Alice", "page": 1}
    )

    print(f"Document Object: {doc}")
    print(f"  Content:  {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")


# ============================================================================
# 2. TEXT LOADER
# ============================================================================

def load_txt():
    """
    Demonstrates loading a local text file using `TextLoader`.
    """
    print("\n--- 2. TextLoader Demonstration ---")

    # Create a temporary file to demonstrate loading cleanly without external dependencies
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write("LangChain Document Loaders\nLine 2: Standardizing unstructured data.\nLine 3: Ready for indexing.")
        tempfile_path = tmp_file.name

    try:
        loader = TextLoader(tempfile_path, encoding="utf-8")
        docs = loader.load()

        print(f"Loaded {len(docs)} document(s):")
        for doc in docs:
            print(f"  Source: {doc.metadata.get('source')}")
            print(f"  Content:\n{doc.page_content}")
    finally:
        if os.path.exists(tempfile_path):
            os.remove(tempfile_path)


# ============================================================================
# 3. WEB SCRAPING LOADER
# ============================================================================

def web_loader():
    """
    Demonstrates scraping web pages into documents using `WebBaseLoader`.
    Uses BeautifulSoup under the hood to strip HTML tags and extract text.
    """
    print("\n--- 3. WebBaseLoader Demonstration ---")

    target_url = "https://www.scrapethissite.com/pages/"
    print(f"Fetching: {target_url} ...")

    try:
        loader = WebBaseLoader(target_url, bs_kwargs={"parse_only": None})
        docs = loader.load()

        print(f"Successfully loaded {len(docs)} document(s).")
        print(f"  Source URL: {docs[0].metadata.get('source')}")
        print(f"  Total Characters: {len(docs[0].page_content)}")
        print(f"  Preview:\n{docs[0].page_content[:250]}...\n")
    except Exception as e:
        print(f"Web loading skipped or failed (network dependent): {e}")


# ============================================================================
# 4. DIRECTORY LOADER & LAZY LOADING
# ============================================================================

def lazy_loader():
    """
    Demonstrates batch loading an entire directory of files.
    - `DirectoryLoader`: Recursively scans a directory with a glob pattern (e.g. `*.txt`).
    - `lazy_load()`: Yields documents lazily as an iterator rather than loading
      all files into memory at once. Essential for gigabyte-scale datasets.
    """
    print("\n--- 4. DirectoryLoader & Lazy Generator Loading ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a few sample documents inside the temporary directory
        for i in range(4):
            path = Path(tmpdir) / f"doc_{i}.txt"
            path.write_text(f"Document #{i}: Sample text demonstrating DirectoryLoader batching.", encoding="utf-8")

        loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

        print("Streaming documents with lazy_load():")
        for doc in loader.lazy_load():
            print(f"  Loaded: {doc.metadata.get('source')} -> Content: '{doc.page_content.strip()}'")


# ============================================================================
# 5. PDF DOCUMENT LOADER
# ============================================================================

def pdf_loader(pdf_path: Path = DEFAULT_PDF_PATH):
    """
    Demonstrates loading a PDF file page-by-page using `PyPDFLoader`.
    Each page in the PDF corresponds to one LangChain `Document` object
    with metadata containing `page: 0`, `page: 1`, etc.
    """
    print(f"\n--- 5. PyPDFLoader Demonstration ({pdf_path.name}) ---")

    if not pdf_path.exists():
        print(f"PDF file not found at: {pdf_path}")
        return

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    print(f"Total Pages Loaded: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"\n[Page {i + 1}] Metadata: {doc.metadata}")
        print(f"Content Preview: {doc.page_content[:150].strip()}...")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run Document Loader demos.
    """
    print("=" * 60)
    print("LANGCHAIN DOCUMENT LOADERS")
    print("=" * 60)

    doc_structure()
    load_txt()
    lazy_loader()
    pdf_loader()
    # Uncomment to test online web scraping:
    # web_loader()


if __name__ == "__main__":
    main()
