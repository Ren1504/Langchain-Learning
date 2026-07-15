from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
# from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

print(f"langchain-core version: {core_version}")
# print(f"langgraph version: {lg_version}")

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke("Say 'Setup Complete!' in one word")
    print(response)

if __name__ == "__main__":
    main()
