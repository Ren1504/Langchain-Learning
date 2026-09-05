from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph , START , END
from typing_extensions import Annotated , TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , AIMessage , BaseMessage, SystemMessage
import operator
from langgraph.graph import add_messages
from dotenv import load_dotenv

load_dotenv()

class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage],operator.add]
    sentiment:str
    response_count :int


def visualize_graph(graph:StateGraph):
    # Visualize the graph using Mermaid syntax
    mermaid_graph = graph.draw_mermaid()
    print("Mermaid Graph visualization:")
    print(mermaid_graph)

    # Save the graph as a PNG file
    png_bytes = graph.draw_mermaid_png()
    with open("basic_graph.png", "wb") as f:
        f.write(png_bytes)
    print("Graph saved as graph.png")

def create_conversation_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def sentiment_node(state:ConversationState) -> dict:
        last_message = state["messages"][-1] if state["messages"] else None

        response = llm.invoke([SystemMessage(content=f"Classify sentiment as positive, negative, or neutral: {state['messages']}"),
                               HumanMessage(content=f"Last message: {last_message}")])
        return {"sentiment": response.content.lower().strip()}

    def generate_response_node(state:ConversationState) -> dict:
        sentiment = state["sentiment"]
        last_message = state["messages"][-1] if state["messages"] else None

        system_prompts = {
            "positive": "Respond enthusiastically and build on the positive sentiment.",
            "negative": "Respond empathetically and address the negative sentiment.",
            "neutral": "Respond informatively and maintain a neutral tone."
        }

        prompt = system_prompts.get(sentiment, system_prompts["neutral"])

        response = llm.invoke([SystemMessage(content=prompt),
                               HumanMessage(content=f"Last message: {last_message}")])

        return {
            "messages": [f"AI: {response.content}"],
            "response_count": state["response_count"] + 1
        }

    graph = StateGraph(ConversationState)
    graph.add_node("sentiment_node", sentiment_node)
    graph.add_node("generate_response_node", generate_response_node)
    graph.add_edge(START, "sentiment_node")
    graph.add_edge("sentiment_node", "generate_response_node")
    graph.add_edge("generate_response_node", END)

    app = graph.compile()

    return app

def main():
    app = create_conversation_graph()

    test_messages = [
        HumanMessage(content="I love this product!"),
        HumanMessage(content="This is the worst experience I've ever had."),
        HumanMessage(content="Can you tell me more about this feature?")
    ]

    for message in test_messages:
        result = app.invoke({"messages": [message], "sentiment": "", "response_count": 0})
        print("-" * 50)
        print(f"Final_result: {result['messages']}, Sentiment: {result['sentiment']}, Response Count: {result['response_count']}\n")

if __name__ == "__main__":
    main()