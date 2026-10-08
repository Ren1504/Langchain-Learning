"""
02_text_splitters.py
====================
Module: RAG - Text Chunking & Splitting Strategies

Why do we split documents into chunks?
1. LLM Context Window Constraints: Passing an entire 100-page book in a single prompt
   is cost-prohibitive and suffers from the "lost in the middle" phenomenon.
2. Semantic Search Granularity: Smaller, focused chunks produce sharper, more accurate
   vector embeddings compared to large multi-topic documents.

Key Concepts:
- `chunk_size`: Maximum number of characters/tokens per chunk.
- `chunk_overlap`: Number of characters/tokens shared between adjacent chunks.
  Overlap prevents sentences or ideas from being cut awkwardly at chunk boundaries.

Splitters covered in this module:
1. `RecursiveCharacterTextSplitter`: LangChain's recommended general-purpose splitter.
   Recursively tries split points: `["\\n\\n", "\\n", " ", ""]` to keep paragraphs intact.
2. `CharacterTextSplitter`: Splits strictly on a single separator.
3. `TokenTextSplitter`: Splits based on token count (using tiktoken) rather than character length.
4. `MarkdownHeaderTextSplitter`: Preserves structural markdown hierarchy (# Header, ## Subheader).
5. Code Splitters: Language-aware splitting (e.g. `Language.PYTHON`) that preserves functions and classes intact.
"""

from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_core.documents import Document

load_dotenv()

# Sample academic essay text
SAMPLE_TEXT = """In academic writing, readers expect each paragraph to have a sentence or two that captures its main point. They’re often called “topic sentences,” though many writing instructors prefer to call them “key sentences.” There are at least two downsides of the phrase “topic sentence.” First, it makes it seem like the paramount job of that sentence is simply to announce the topic of the paragraph. Second, it makes it seem like the topic sentence must always be a single grammatical sentence. Calling it a “key sentence” reminds us that it expresses the central idea of the paragraph. And sometimes a question or a two-sentence construction functions as the key.

Key sentences in academic writing do two things. First, they establish the main point that the rest of the paragraph supports. Second, they situate each paragraph within the sequence of the argument, a task that requires transitioning from the prior paragraph. Consider these two examples:[2]"""

# Sample Python code
SAMPLE_CODE = """def quicksort(arr):
    # Base case: arrays with 0 or 1 element are already sorted
    if len(arr) <= 1:
        return arr
    
    # Selecting the middle element as the pivot
    pivot = arr[len(arr) // 2]
    
    # Partitioning the array into three components
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Recursively sort left and right, then combine
    return quicksort(left) + middle + quicksort(right)

# Example usage:
data = [3, 6, 8, 10, 1, 2, 1]
print("Sorted Array:", quicksort(data))
"""


# ============================================================================
# 1. RECURSIVE CHARACTER TEXT SPLITTER
# ============================================================================

def recursive_splitter():
    """
    Demonstrates `RecursiveCharacterTextSplitter`.
    It attempts to split on paragraphs (`\\n\\n`), then lines (`\\n`),
    then words (` `), and finally individual characters (`""`).
    This preserves semantic structure better than any other naive splitter.
    """
    print("\n--- 1. RecursiveCharacterTextSplitter ---")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Original Text Length: {len(SAMPLE_TEXT)} characters")
    print(f"Total Chunks Generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i + 1}] (Length: {len(chunk)} chars):\n\"{chunk}\"")


def chunk_size_comparison():
    """
    Compares the impact of different chunk sizes (200 vs 500 vs 1000).
    - Smaller chunks: High precision retrieval, but may lack surrounding context.
    - Larger chunks: Rich context, but harder to match specific questions accurately.
    """
    print("\n--- 2. Comparing Chunk Sizes ---")

    sizes = [150, 300, 600]

    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 5,
        )
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"Chunk Size {size} (Overlap {size // 5}): {len(chunks)} chunk(s) produced.")


def overlap_demonstration():
    """
    Demonstrates why chunk overlap matters.
    Without overlap, a critical keyword or sentence might be halved at the boundary.
    With overlap, the context is duplicated at the seam.
    """
    print("\n--- 3. Chunk Overlap Demonstration ---")

    test_sentence = "The quick brown fox jumps over the lazy dog. " * 5

    splitter_no_overlap = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
    splitter_with_overlap = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=25)

    print("Without Overlap (Seam transitions):")
    chunks_a = splitter_no_overlap.split_text(test_sentence)
    for i, c in enumerate(chunks_a[:2]):
        print(f"  Chunk {i + 1}: ...{c[-25:]}")

    print("\nWith Overlap (Shared context across boundary):")
    chunks_b = splitter_with_overlap.split_text(test_sentence)
    for i, c in enumerate(chunks_b[:2]):
        print(f"  Chunk {i + 1}: ...{c[-25:]}")


# ============================================================================
# 2. TOKEN-BASED TEXT SPLITTING
# ============================================================================

def token_splitter():
    """
    Demonstrates `TokenTextSplitter`.
    LLMs reason in tokens (BPE subwords), not characters.
    1 token ≈ 4 characters or 0.75 words in English.
    TokenTextSplitter ensures chunks match exact model token limits.
    """
    print("\n--- 4. Token-Based Text Splitter ---")

    try:
        splitter = TokenTextSplitter(chunk_size=40, chunk_overlap=10)
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"Token-based chunks produced: {len(chunks)}")
        print(f"Chunk 1: '{chunks[0]}'")
    except Exception as e:
        print(f"TokenTextSplitter note (requires tiktoken): {e}")


# ============================================================================
# 3. STRUCTURED MARKDOWN SPLITTING
# ============================================================================

def markdown_splitter():
    """
    Demonstrates `MarkdownHeaderTextSplitter`.
    Instead of blind character cuts, it parses markdown headers (#, ##, ###)
    and attaches the section headers directly into the chunk's metadata!
    """
    print("\n--- 5. MarkdownHeaderTextSplitter ---")

    markdown_doc = """
# LangChain Architecture

LangChain connects LLMs with external tools and data.

## Core Models
Models include Chat Models and Text Embeddings.

### Chat Models
Chat models take messages and return messages.

## Memory System
Memory tracks conversational history between turns.
"""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_chunks = markdown_splitter.split_text(markdown_doc)

    print(f"Markdown Sections Extracted: {len(md_chunks)}")
    for chunk in md_chunks:
        print(f"Metadata: {chunk.metadata}")
        print(f"Content:  '{chunk.page_content.strip()}'\n")


# ============================================================================
# 4. CODE-AWARE SPLITTING (PYTHON)
# ============================================================================

def code_splitter():
    """
    Demonstrates syntax-aware splitting for programming languages.
    When splitting code, we don't want to cut inside a function body or class.
    `Language.PYTHON` splits along function, class, and statement definitions.
    """
    print("\n--- 6. Code-Aware Python Splitter ---")

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=120,
        chunk_overlap=20,
    )

    code_chunks = python_splitter.split_text(SAMPLE_CODE)
    print(f"Code Chunks Created: {len(code_chunks)}")
    for i, chunk in enumerate(code_chunks):
        print(f"\n--- Code Chunk {i + 1} ---")
        print(chunk)


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run Text Splitter demos.
    """
    print("=" * 60)
    print("LANGCHAIN TEXT SPLITTERS & CHUNKING STRATEGIES")
    print("=" * 60)

    recursive_splitter()
    chunk_size_comparison()
    overlap_demonstration()
    markdown_splitter()
    code_splitter()


if __name__ == "__main__":
    main()
