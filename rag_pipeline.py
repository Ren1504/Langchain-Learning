from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()

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
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""



embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            temperature = 0.3
        )

def create_kb():
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50)
    doc = Document(page_content=KNOWLEDGE_BASE,metadata ={"source":"langchain_knowledgebase.md"})

    chunks = splitter.split_documents([doc])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp()
    )

    return vector_store


def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type = "similarity")
    llm = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        temperature = 0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context
        {context}
        Question: {question}
        Make sure to answer in concise manner if you don't know the answer just say "I dont know" """
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context":retriever | format_docs ,
         "question":RunnablePassthrough()}| prompt | llm | StrOutputParser()
    )

    print("Basic RAG\n")
    print("*"*20,"\n")

    questions = ["What is Langchain?","Who is the president of USA?","What is LangGraph used for?","Who created Langchain?"]

    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"Q:{question} || A:{answer}\n")
    

def rag_with_sources():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type = "similarity", search_kwargs={'k':2}
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the context including with sources that you have used
        {context}
        Question: {question}
        Make sure to answer in concise manner if you don't know the answer just say "I dont know"
         Answer (include sources) """
    )

    def format_docs_with_sources(docs):
        formatted = []

        for i, doc in enumerate(docs):
            source = doc.metadata.get( 'source','unknown')
            formatted.append(f"[{i+1} {source}:\n{doc.page_content}]")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context":retriever | format_docs_with_sources ,
         "question":RunnablePassthrough()}| prompt | llm | StrOutputParser()
    ) 

    answer = rag_chain.invoke("What are the core components of Langchain?")
    print(f"A: {answer}")

def rag_with_fallback():
    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs = {"k":2})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context, if the answer is not present in the context , respond with "I dont know"
Context:{context}
queston:{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context":retriever | format_docs ,
         "question":RunnablePassthrough()}| prompt | llm | StrOutputParser()
    ) 

    questions = [
        "What is the pricing for Langsmith?",
        "How do I deploy Langchain to AWS?",
        "What is the stockprice of OpenAI?"
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q:{q}\nA:{answer}")

def structured_rag():
    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs = {"k":3})

    class RAGResponse(BaseModel):
        answer: str = Field(description="Answer")
        confidence: str = Field(description="high medium low")
        sources_used: List[str] = Field("List of sources")
        follow_up: str = Field("suggested follow up question")

    structured_llm = llm.with_structured_output(RAGResponse)
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context, if the answer is not present in the context , respond with "I dont know"
Context:{context}
queston:{question}

Provide with a structured response
"""
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context":retriever | format_docs ,
         "question":RunnablePassthrough()}| prompt | structured_llm
    ) 

    result = rag_chain.invoke("What is LangGraph?")

    print(f"A:{result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}")

 

    

def main():
    # demo_basic_rag()

    """Basic RAG

******************** 

Q:What is Langchain? || A:LangChain is a framework for developing applications powered by language models, created by Harrison Chase in October 2022. It supports various components like models, prompts, chains, agents, and memory for building language model applications.

Q:Who is the president of USA? || A:I don't know.

Q:What is LangGraph used for? || A:LangGraph is used for building stateful, multi-actor applications with features like state management, cycles and loops, human-in-the-loop, and persistence.

Q:Who created Langchain? || A:LangChain was created by Harrison Chase."""

    # rag_with_sources()

"""A: The core components of LangChain are:

1. **Models**: Supports various LLM providers including OpenAI, Anthropic, and local models.
2. **Prompts**: Templates for structuring inputs to language models.
3. **Chains**: Sequences of calls to models and other components.
4. **Agents**: Systems that use LLMs to determine which actions to take.
5. **Memory**: Components for persisting state between chain/agent calls."""

# rag_with_fallback()
"""Q:What is the pricing for Langsmith?
A:LangSmith has a free tier and paid plans starting at $39/month.
Q:How do I deploy Langchain to AWS?
A:I don't know.
Q:What is the stockprice of OpenAI?
A:I dont know"""

# structured_rag()

"""A:LangGraph is a library designed for building stateful, multi-actor applications. It includes key features such as state management, cycles and loops, human-in-the-loop capabilities, and persistence for maintaining state between chain and agent calls.
Confidence: high
Sources: ['LangGraph description in the context']
Follow-up: Would you like to know more about its features or how to use it?"""

def exercise(Doc):

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    llm = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            temperature = 0.3
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size = 50,chunk_overlap= 20)
    doc = Document(page_content=Doc,metadata ={"source":"Dorothy.md"})

    chunks = splitter.split_documents([doc])
    persist_dir = "./chroma_db/"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_dir,
    )

    retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k":3})
    class RAGResponse(BaseModel):
        answer: str = Field(description="Answer")
        confidence: str = Field(description="high medium low")
        sources_used: List[str] = Field(description="List of sources")
        follow_up: str = Field(description="suggested follow up question")

    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_template("""
You are a FAQ bot that Answers customer's questions based on the given company details 
{details} and answer the question {question} based on the given details
answer in a structured format
""")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"details":retriever | format_docs ,
         "question":RunnablePassthrough()}| prompt | structured_llm
    ) 

    questions = ["What is the name of the company?",
                 "How long to return products?",
                 "How many days the products get delivered?",
                 "What does the company do?",
                 "What applies for a product to be returned?",
                 "How can one buy products?"

    ]

    for question in questions:
        result = rag_chain.invoke(question)
        print(f"Q:{question}\n")
        print(f"A:{result.answer}\nConfidence{result.confidence}\nFollow-up:{result.follow_up}\n{result.sources_used}")

doc = """Dorothy is an online retail company that sells a variety of products through its website. Customers can browse available products, place orders online, and receive their purchases at their chosen delivery address. Dorothy aims to provide a simple shopping experience with clear information about products, orders, payments, and delivery.

Dorothy usually processes orders within 1–2 business days after the order is placed. Once an order has been processed and shipped, standard delivery typically takes between 3–5 business days. Delivery times may vary depending on the customer's location, weekends, public holidays, weather conditions, and other circumstances that may affect the courier service.

Customers can check the status of their order using the tracking information provided by Dorothy after the package has been shipped. If an order has not arrived within the expected delivery period, customers should contact Dorothy's customer support team with their order number so that the delivery can be investigated.

Dorothy accepts returns for eligible products within 30 days of delivery. Items must generally be unused and returned in their original condition and packaging. Some products may not be eligible for return because of their nature or specific company policies. Customers should review Dorothy's return policy before sending an item back.

If a customer receives a damaged, incorrect, or defective product, they should contact Dorothy's support team as soon as possible and provide the order number along with relevant details or photographs. Dorothy will review the issue and, where appropriate, arrange a replacement, refund, or another suitable solution.

Customers can contact Dorothy's customer support team for questions about orders, shipping, returns, refunds, or products. When contacting support about an existing order, customers should provide their order number so that the support team can locate the relevant order and provide accurate assistance."""

exercise(doc)

"""Q:What is the name of the company?

A:The name of the company is Dorothy.
Confidencehigh
Follow-up:What types of products does Dorothy sell?
[]
Q:How long to return products?

A:You can return eligible products within 30 days of delivery.
Confidencehigh
Follow-up:What items are considered eligible for return?
[]
Q:How many days the products get delivered?

A:The products are delivered within 3 to 5 business days after the order is processed. Please note that delivery times may vary depending on the specific circumstances of your order.
Confidencehigh
Follow-up:What is the return policy for eligible products?
[]
Q:What does the company do?

A:Dorothy is an online retail company that specializes in selling a diverse range of products. The company focuses on providing customers with a variety of items across different categories, ensuring a convenient shopping experience through its online platform.
Confidencehigh
Follow-up:What types of products does Dorothy offer?
[]
Q:What applies for a product to be returned?

A:To determine if a product is eligible for return at Dorothy, please consider the following criteria:

1. **Condition of the Product**: The item must be unused, in its original packaging, and in the same condition as when it was received.

2. **Return Window**: Returns must be initiated within the specified return period (e.g., 30 days from the date of purchase).

3. **Non-Returnable Items**: Certain products may be marked as non-returnable, such as personalized items, intimate apparel, or perishable goods.

4. **Proof of Purchase**: A valid receipt or proof of purchase is required to process the return.

5. **Return Authorization**: Customers may need to obtain a return authorization before sending items back to ensure proper processing.

Please ensure that all these conditions are met before sending an item back to ensure a smooth return process.
Confidencehigh
Follow-up:What is the process for initiating a return?
['Company Return Policy Documentation']
Q:How can one buy products?

A:To buy products from our company, please follow these steps:

1. **Browse Products**: Visit our website and navigate to the products section. You can explore various categories to find what you need.

2. **Select a Product**: Click on the product you are interested in to view its details, including price, specifications, and availability.

3. **Add to Cart**: If you wish to purchase the product, click the 'Add to Cart' button. You can continue shopping or proceed to checkout.

4. **Review Your Cart**: Once you are ready to purchase, click on the cart icon to review your selected items. Ensure that the quantities and product details are correct.

5. **Proceed to Checkout**: Click on the 'Checkout' button to start the payment process.

6. **Enter Shipping Information**: Fill in your shipping address where you would like the products to be delivered.

7. **Choose Payment Method**: Select your preferred payment method (credit card, PayPal, etc.) and enter the necessary payment details.

8. **Confirm Order**: Review all the information provided, and if everything is correct, click on the 'Place Order' button to finalize your purchase.

9. **Receive Confirmation**: After placing your order, you will receive a confirmation email with the details of your purchase and estimated delivery time.

10. **Track Your Order**: You can track the status of your order through the website or the confirmation email you received.

If you have any questions during the process, feel free to reach out to our customer service for assistance!
Confidencehigh
Follow-up:What payment methods do you accept?
[]"""

if __name__ == "__main__":
    main()
