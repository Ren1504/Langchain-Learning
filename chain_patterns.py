from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableSequence , RunnableLambda , RunnablePassthrough

load_dotenv()

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
)
def basic_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following into one sentence: {text}")
    parser = StrOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({"text":"A video game is an interactive electronic form of play where players use a controller or device to make choices, solve problems, and influence visual and auditory events on a screen over time."})

    print(result)

def parallel_chain():
    """sdsd"""

    summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")
    keyword_prompt = ChatPromptTemplate.from_template("Extract 5 keywords in the follwoing text: {text}")
    sentiment_prompt = ChatPromptTemplate.from_template("What is the sentiment for teh given text {text}")

    parser = StrOutputParser()

    #parallel execution
    analysis_chain = RunnableParallel(
       summary =  summarize_prompt | model | parser,
       keyword =  keyword_prompt | model | parser,
       sentiment =  sentiment_prompt | model | parser
    )

    text = """computer technology built to do tasks that normally need human thinking. Instead of following strict, step-by-step rules written by a programmer,
      AI looks at massive amounts of data, finds patterns, and learns how to solve problems,
        recognize speech, or create content on its own."""

    results = analysis_chain.invoke({"text":text})
    print(results)
    print("*"*20)
    print(f"Summary:{results['summary']}")
    print("*"*20)
    print(f"Keyword:{results['keyword']}")
    print("*"*20)
    print(f"Sentiment:{results['sentiment']}")
    print("*"*20)

def passthrough_chain():

    prompt = ChatPromptTemplate.from_template("" \
    "Original question: {question}\n" \
    "Context: {context}\n\n" \
    "Answer the question based on the context")

    def fake_retriever(input_dict):
        return "I am learning so much in LangChain now"

    prompt = ChatPromptTemplate.from_template(
        """Original question: {question}
    Context: {context}

    Answer the question based on the context."""
    )

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

    result = chain.invoke({"question":"What am I learning now"})
    print(f"Answer: {result}")


def main():
    # basic_chain()
    # parallel_chain()
    passthrough_chain()

if __name__ == "__main__":
    main()