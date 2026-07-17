from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer in one sentence: {question}")
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"question":"Who is the chief minister of TamilNadu> "})
    print(f"Response : {result}")

    return result
def main():
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # response = llm.invoke("Say 'Setup Complete!' in one word")
    # print(response)

    demo_basic_chain()

if __name__ == "__main__":
    main()
