import os
import tempfile
from langchain_community.document_loaders import TextLoader , WebBaseLoader , DirectoryLoader , PyPDFLoader
from langchain_core.documents import Document
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

def load_txt():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(b"Tempfile here.\nLine 2\nJust a demonstration")
        tempfile_path = tmp_file.name

    try:
        loader = TextLoader(tempfile_path)
        docs = loader.load()

        for doc in docs:
            print("__________________________")
            print(doc)
            print(doc.page_content)
            print("__________________________")

    finally:
        os.remove(tempfile_path)

def web_loader():
    loader = WebBaseLoader("https://www.scrapethissite.com/pages/",bs_kwargs={"parse_only":None})
    docs = loader.load()

    print(f"Loaded{len(docs)}")
    print(f"source:{docs[0].metadata.get('source','N/A')}")
    print(f"Content length:{len(docs[0].page_content)}")
    print(f"Preview:{docs[0].page_content[:200]}....")

def lazy_loader():
    with tempfile.TemporaryDirectory() as tmpdir:

        for i in range(5):
            path = Path(tmpdir) / f"doc{i}.txt"
            path.write_text(f"This is document{i}.It contains sample content")

        loader = DirectoryLoader(tmpdir,glob="*.txt",loader_cls=TextLoader)

        for doc in loader.lazy_load():
            print(f"Doc preview: {doc.page_content[:50]}....")

def doc_structure():
    doc = Document(
        page_content="sample text",
        metadata = {"source":"sample.txt"}
    )

    print(f"doc content {doc.page_content}")

def pdf_loader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} Document(s) from PDF")

    for i , doc in enumerate(docs):
        print(f"Doc {i} Content preview {doc.page_content[:100]}...")


def main():
    # load_txt()
    # web_loader()
    # lazy_loader()
    # doc_structure()
    pdf_loader("./langchain_demo.pdf")

if __name__ == "__main__":
    main()