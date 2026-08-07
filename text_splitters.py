from langchain_text_splitters import RecursiveCharacterTextSplitter , CharacterTextSplitter , TokenTextSplitter , MarkdownHeaderTextSplitter , Language
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()


text = """In academic writing, readers expect each paragraph to have a sentence or two that captures its main point. They’re often called “topic sentences,” though many writing instructors prefer to call them “key sentences.” There are at least two downsides of the phrase “topic sentence.” First, it makes it seem like the paramount job of that sentence is simply to announce the topic of the paragraph. Second, it makes it seem like the topic sentence must always be a single grammatical sentence. Calling it a “key sentence” reminds us that it expresses the central idea of the paragraph. And sometimes a question or a two-sentence construction functions as the key.

Key sentences in academic writing do two things. First, they establish the main point that the rest of the paragraph supports. Second, they situate each paragraph within the sequence of the argument, a task that requires transitioning from the prior paragraph. Consider these two examples:[2]"""

code_text = """ def quicksort(arr):
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
# Output: [1, 1, 2, 3, 6, 8, 10]
                            """

def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10,
        chunk_overlap = 5,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)
    print(len(text))
    print(len(chunks))
    print([len(c) for c in chunks])
    print(chunks[0][:30])

def chunk_size_comparison():
    sizes = [200,500,1000 ]

    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = size, chunk_overlap=size//5
        )

        chunks = splitter.split_text(text)
        print(f"Size {size}: {len(chunks)} chunks")

def overlap():
    text = "The quick brown fox jumps over the lazy dog. " * 10

    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

    chunk = no_overlap.split_text(text)
    chunk_overlap = with_overlap.split_text(text)

    print(f"Chunk 1 end: ...{chunk[0][-20:]}")
    print(f"Chunk 2 start: ...{chunk[1][:20]}")

    print(f"Chunk 1 end (overlap): ...{chunk_overlap[0][-20:]}")
    print(f"Chunk 2 start (overlap): ...{chunk_overlap[1][:20]}")

def markdown_splitters():
    headers = [
        ("#","h1"),
        ("##","h2"),
        ("###","h3")
    ]


    

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    chunks = splitter.split_text(text)

    print(len(chunks))

def code_splitters():
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                   chunk_size=500,
                                                                   chunk_overlap=50)

    chunks = python_splitter.split_text(code_text)

def pdf_splitter():

    loader = PyPDFLoader("./langchain_demo.pdf")
    docs = loader.load()

    print(f"loaded docs {len(docs)} docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=50)

    split_docs = splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk: {split_docs[0].metadata}")
    print(f"\nfirst chunk content: {split_docs[0].page_content[:100]}...")
    print(f"\nLast chunk content: {split_docs[-1].metadata}")


def main():
    # recursive_splitter()
    # chunk_size_comparison()
    # overlap()
    # markdown_splitters()
    pdf_splitter()
if __name__ == "__main__":
    main()