"""
02_runnable_patterns.py
=======================
Module: Core Concepts - Advanced Runnable & LCEL Composition Patterns

This module demonstrates key architectural patterns in LangChain Expression
Language (LCEL) using core Runnable primitives:
1. `RunnableSequence` (`|` operator): Linear chaining where output of one step feeds the next.
2. `RunnableParallel`: Concurrent execution of multiple independent runnables on the same input.
3. `RunnablePassthrough`: Forwarding the raw input unchanged or augmenting it using `.assign()`.
4. `RunnableLambda`: Wrapping standard Python functions into runnable components.
5. `RunnableBranch`: Conditional branching based on classification or rules.

Prerequisites:
- Valid GOOGLE_API_KEY in `.env` (or change model_provider to 'openai' with OPENAI_API_KEY).
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)

# Load environment variables
load_dotenv()

# Initialize the chat model
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
)


# ============================================================================
# 1. LINEAR CHAIN (RunnableSequence)
# ============================================================================

def basic_chain():
    """
    Demonstrates a basic linear sequence:
    Input Dict -> ChatPromptTemplate -> ChatModel -> StrOutputParser -> Result String
    """
    print("\n--- 1. Linear Sequence Pattern ---")

    prompt = ChatPromptTemplate.from_template("Summarize the following into one sentence: {text}")
    parser = StrOutputParser()

    chain = prompt | model | parser

    text_input = (
        "A video game is an interactive electronic form of play where players use a controller "
        "or device to make choices, solve problems, and influence visual and auditory events "
        "on a screen over time."
    )

    result = chain.invoke({"text": text_input})
    print(f"Original Text: {text_input}\n")
    print(f"Summary: {result}")


# ============================================================================
# 2. PARALLEL EXECUTION (RunnableParallel)
# ============================================================================

def parallel_chain():
    """
    Demonstrates executing multiple runnables concurrently on the same input.
    
    Architecture:
                  ┌─> Summarize Chain ──> summary
    Input Text ───┼─> Keyword Chain   ──> keyword
                  └─> Sentiment Chain ──> sentiment
    
    All three sub-chains run concurrently, cutting total response latency.
    """
    print("\n--- 2. Parallel Runnable Execution Pattern ---")

    # Define prompts for 3 different extraction goals
    summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")
    keyword_prompt = ChatPromptTemplate.from_template("Extract 5 key terms/keywords from: {text}")
    sentiment_prompt = ChatPromptTemplate.from_template("What is the sentiment of the given text: {text}")

    parser = StrOutputParser()

    # RunnableParallel maps dictionary keys to sub-runnables
    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | parser,
        keywords=keyword_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )

    text = (
        "Artificial intelligence is computer technology built to perform tasks that normally require "
        "human intelligence. Instead of following strict, manual rules written by programmers, AI looks "
        "at massive amounts of data, finds patterns, and learns how to solve problems, recognize speech, "
        "or create novel content on its own."
    )

    results = analysis_chain.invoke({"text": text})

    print(f"Summary:\n{results['summary']}\n")
    print(f"Keywords:\n{results['keywords']}\n")
    print(f"Sentiment:\n{results['sentiment']}\n")


# ============================================================================
# 3. PASSTHROUGH & LAMBDAS (RunnablePassthrough & RunnableLambda)
# ============================================================================

def passthrough_chain():
    """
    Demonstrates using `RunnablePassthrough` and `RunnableLambda` to assemble
    a context-augmented QA pipeline.
    
    - `RunnableLambda`: Turns a regular Python function into a runnable component.
    - `RunnablePassthrough`: Keeps the user's original query untouched while
      retrieving context in parallel.
    """
    print("\n--- 3. Passthrough & Custom Lambda Functions ---")

    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n\n"
        "Answer the question strictly based on the context."
    )

    # Simulated retrieval function
    def fake_retriever(input_dict):
        # In a real system, this would query a vector store like Chroma
        return "LangChain simplifies building LLM applications by connecting models, prompts, and memory."

    # Assemble pipeline:
    # 1. Parallel step gathers context via fake_retriever and preserves the question via RunnablePassthrough.
    # 2. Lambda formats the dictionary keys cleanly for the prompt.
    # 3. Prompt -> Model -> Parser.
    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(
            lambda x: {
                "context": x["context"],
                "question": x["question"]["question"],
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke({"question": "What does LangChain do?"})
    print(f"Answer: {result}")


# ============================================================================
# 4. CONDITIONAL BRANCHING (RunnableBranch & .assign())
# ============================================================================

def chain_branching():
    """
    Demonstrates dynamic routing using `RunnableBranch` and `RunnablePassthrough.assign()`.
    
    Workflow:
    1. Classifier step identifies if the input is about a 'movie', 'tv show', or 'neither'.
    2. `.assign(category=...)` injects the classification result into the state dict.
    3. `RunnableBranch` inspects the state and routes to the appropriate specialist chain.
    """
    print("\n--- 4. Dynamic Conditional Branching Pattern ---")

    # Specialized persona prompts
    movie_prompt = ChatPromptTemplate.from_template("You are a movie expert. Answer: {input}")
    series_prompt = ChatPromptTemplate.from_template("You are a TV show expert. Answer: {input}")
    general_prompt = ChatPromptTemplate.from_template(
        "I specialize only in movies and TV shows. I cannot assist with: {input}"
    )

    # Intent classifier
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify the following subject as 'movie', 'tv show', or 'neither': {input}\n"
        "Return ONLY one word: 'movie', 'tv show', or 'neither'."
    )
    classifier_chain = classifier_prompt | model | StrOutputParser()

    # RunnableBranch format:
    # (condition_callable_1, runnable_1),
    # (condition_callable_2, runnable_2),
    # default_runnable
    full_chain = (
        RunnablePassthrough.assign(category=classifier_chain)
        | RunnableBranch(
            (lambda x: "movie" in x["category"].lower(), movie_prompt | model | StrOutputParser()),
            (lambda x: "tv show" in x["category"].lower(), series_prompt | model | StrOutputParser()),
            general_prompt | model | StrOutputParser(),
        )
    )

    test_queries = [
        "Avengers: Infinity War release and plot",
        "Breaking Bad season 5 climax",
        "How do I change the oil in a car?",
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        response = full_chain.invoke({"input": query})
        print(f"Routed Response: {response}\n" + "-" * 50)


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run each LCEL pattern demo.
    """
    print("=" * 60)
    print("LANGCHAIN LCEL & RUNNABLE PATTERNS")
    print("=" * 60)

    basic_chain()
    parallel_chain()
    passthrough_chain()
    chain_branching()


if __name__ == "__main__":
    main()
