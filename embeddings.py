from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_community.storage import LocalFileStore
import tempfile
import numpy as np

load_dotenv()

def huggingface():

    embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

    text = "We are going to win the championship finals"

    embedding = embeddings.embed_query(text)
    print(f"Embedding for single text: {embedding}")

    print(len(embedding))

    embeds = embeddings.embed_documents(["This is text one","it is text two"])



def basic_embed():

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text = "What is the day today"
    embed = embeddings.embed_query(text)

    print(f"Vector dimensions: {len(embed)}")
    print(f"First 5 values: {embed[:5]}")
    print(f"Vector norm: {np.linalg.norm(embed):.4f}")
    print("-"*20)
    print(embed)

def batch_embed():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text = ["What is the day today","We have been in there for ten minutes","Always start my mornings with coffee"]

    batch_embed = embeddings.embed_documents(text)

    for i , embed in enumerate(batch_embed):
            print(f"Vector dimensions {i+1}: {len(embed)}")
            print(f"First 5 values {i+1}: {embed[:5]}")
            print(f"Vector norm {i+1}: {np.linalg.norm(embed):.4f}")
            print("-"*20)

def similarity():
    docs = ["C# is use in game development",
             "She will be home tomorrow",
             "Python is a programming language",
             "Javascript is a programming languge",
             "We will be arriving tomorrow in your home"]

    query = "What programming languages exist?"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1,vec2):
         return np.dot(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarites = [cosine_similarity(query_vector,doc_vec) for doc_vec in doc_vector]

    ranked_docs = sorted(zip(docs,similarites),key=lambda x:x[1],reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity: ")
    for doc,score in ranked_docs:
         print(f"{score:.4f}:{doc}")

    """Ranked by similarity: 
0.4428:Python is a programming language
0.3975:Javascript is a programming languge
0.3514:C# is use in game development
0.0291:We will be arriving tomorrow in your home
-0.0190:She will be home tomorrow"""

def embedding_caching():
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(directory = tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
             underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small"),
             document_embedding_cache = store,
             namespace = "exercise"
        )

        text = "When are they coming home?"

        print("API call")
        vector1 = cached_embeddings.embed_documents([text])
        print(f"Embedded{len(vector1)} documents")

        vector2 = cached_embeddings.embed_documents([text])
        print(f"Embedded{len(vector2)} documents")

        vector3 = cached_embeddings.embed_documents([text])
        print(f"Embedded{len(vector3)} documents")

     

     
def main():
    # basic_embed()
    # batch_embed()
    # similarity()
    embedding_caching()

if __name__ == "__main__":
    main()
