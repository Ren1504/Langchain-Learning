"""
02_conversation_graph.py
========================
Module: LangGraph - Conversational Routing & Graph Visualization

This module demonstrates building a 2-stage conversational intelligence pipeline:
1. Stage 1 (`sentiment_node`): Analyzes the sentiment of incoming messages
   (classified as 'positive', 'negative', or 'neutral').
2. Stage 2 (`generate_response_node`): Dynamically crafts a persona response adapted
   to the detected sentiment (enthusiastic, empathetic, or neutral).

Graph Architecture:
               ┌────────────────┐
               │    [START]     │
               └───────┬────────┘
                       ▼
             ┌───────────────────┐
             │  sentiment_node   │  (Classifies tone)
             └─────────┬─────────┘
                       ▼
         ┌───────────────────────────┐
         │  generate_response_node   │  (Adapts persona & responds)
         └─────────────┬─────────────┘
                       ▼
                ┌──────────────┐
                │    [END]     │
                └──────────────┘

Visualizing LangGraph:
- `graph.draw_mermaid()`: Generates ASCII / Mermaid markdown diagrams.
- `graph.draw_mermaid_png()`: Renders and saves graphical diagrams to disk (`assets/`).

Prerequisites:
- OPENAI_API_KEY in `.env`
"""

import operator
from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Assets directory for graph images
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class ConversationState(TypedDict):
    # Appends new messages using operator.add
    messages: Annotated[list[BaseMessage], operator.add]
    sentiment: str
    response_count: int


# ============================================================================
# 2. GRAPH VISUALIZATION UTILITY
# ============================================================================

def visualize_graph(app_graph):
    """
    Renders Mermaid syntax and exports a PNG image of the graph topology.
    """
    print("\n--- Mermaid Graph Syntax ---")
    try:
        mermaid_syntax = app_graph.draw_mermaid()
        print(mermaid_syntax)

        png_bytes = app_graph.draw_mermaid_png()
        output_file = ASSETS_DIR / "basic_graph.png"
        output_file.write_bytes(png_bytes)
        print(f"Graph diagram successfully saved to: {output_file}")
    except Exception as e:
        print(f"Note: PNG rendering requires graphviz/mermaid online rendering: {e}")


# ============================================================================
# 3. GRAPH NODES & WORKFLOW CREATION
# ============================================================================

def create_conversation_graph():
    """
    Constructs and compiles the sentiment-adaptive conversation graph.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Node 1: Sentiment Classifier
    def sentiment_node(state: ConversationState) -> dict:
        last_message = state["messages"][-1] if state["messages"] else None

        prompt = (
            f"Classify the sentiment of the user message strictly as 'positive', 'negative', or 'neutral'.\n"
            f"User message: {last_message.content if last_message else ''}"
        )
        response = llm.invoke([
            SystemMessage(content="You are a sentiment classifier. Output only one word: positive, negative, or neutral."),
            HumanMessage(content=prompt)
        ])

        detected_sentiment = response.content.lower().strip()
        print(f"  [sentiment_node]: Detected sentiment -> '{detected_sentiment}'")
        return {"sentiment": detected_sentiment}

    # Node 2: Adaptive Tone Response Generator
    def generate_response_node(state: ConversationState) -> dict:
        sentiment = state.get("sentiment", "neutral")
        last_message = state["messages"][-1] if state["messages"] else None

        # Adaptive system instructions matching the customer's mood
        tone_instructions = {
            "positive": "Respond enthusiastically and celebrate the customer's positive experience!",
            "negative": "Respond with deep empathy, apologize sincerely, and provide a helpful resolution.",
            "neutral": "Respond informatively, clearly, and maintain a polite, professional tone."
        }

        system_instruction = tone_instructions.get(sentiment, tone_instructions["neutral"])

        response = llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=last_message.content if last_message else "")
        ])

        return {
            "messages": [response],
            "response_count": state["response_count"] + 1
        }

    # Assemble StateGraph
    graph = StateGraph(ConversationState)

    graph.add_node("sentiment_node", sentiment_node)
    graph.add_node("generate_response_node", generate_response_node)

    # Linear flow
    graph.add_edge(START, "sentiment_node")
    graph.add_edge("sentiment_node", "generate_response_node")
    graph.add_edge("generate_response_node", END)

    app = graph.compile()
    return app


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Test the adaptive conversation graph across diverse sentiment scenarios.
    """
    print("=" * 60)
    print("LANGGRAPH CONVERSATION GRAPH & SENTIMENT ROUTING")
    print("=" * 60)

    app = create_conversation_graph()

    # Visualize and save PNG
    visualize_graph(app.get_graph())

    test_messages = [
        HumanMessage(content="I absolutely love this product! It made my team 10x faster!"),
        HumanMessage(content="This is the worst experience I've ever had. Nothing works properly."),
        HumanMessage(content="Can you explain how to export data to CSV?"),
    ]

    for msg in test_messages:
        print(f"\nUser Query: \"{msg.content}\"")
        initial_state = {
            "messages": [msg],
            "sentiment": "",
            "response_count": 0,
        }
        result = app.invoke(initial_state)

        bot_reply = result["messages"][-1].content
        print(f"Detected Tone: {result['sentiment']}")
        print(f"Adaptive Bot Reply: \"{bot_reply}\"")
        print("-" * 50)


if __name__ == "__main__":
    main()
