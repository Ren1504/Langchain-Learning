"""
01_graph_basics.py
==================
Module: LangGraph - StateGraph Fundamentals & Reducers

Why LangGraph?
While standard LangChain chains (LCEL) are strictly linear (Directed Acyclic Graphs),
real-world agentic systems require:
1. Stateful execution: Maintaining structured data across multiple actors/steps.
2. Loops and Cycles: Self-correction, iterative refinement, and multi-turn reasoning.
3. Human-in-the-Loop: Pausing execution for human inspection or approval.
4. Fault tolerance & Persistence: Checkpointing graph state to disk or database.

Key Primitives in LangGraph:
- `StateGraph(StateType)`: The graph definition, parameterized by a State schema (usually TypedDict).
- `Nodes`: Python functions that take the current `state` and return an update dictionary.
- `Edges`: Direct links connecting nodes:
  - `START`: Virtual entry node.
  - `END`: Virtual exit node.
  - `graph.add_edge("node_a", "node_b")`: Normal direct edge.
- State Reducers (`Annotated`):
  By default, a node overwrites the state key.
  Using `Annotated[T, reducer_fn]` changes update behavior:
  - `Annotated[list, operator.add]`: Appends new list items to existing items.
  - `Annotated[list, add_messages]`: Appends messages and automatically matches/deduplicates message IDs.

Prerequisites:
- Requires `langgraph`, `langchain-core`
- OPENAI_API_KEY in `.env`
"""

import operator
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages

load_dotenv()


# ============================================================================
# 1. UNDERSTANDING GRAPH STATE SCHEMAS & REDUCERS
# ============================================================================

# Pattern A: Default Overwrite State
# When a node returns {"step": 2}, the old value of step is completely replaced.
class SimpleState(TypedDict):
    input: str
    output: str
    step: int


# Pattern B: Accumulator State using operator.add
# When a node returns {"messages": ["new msg"], "count": 1}:
# - "new msg" is appended to the existing list.
# - count is incremented by 1 (old_count + 1).
class AccumulatingState(TypedDict):
    messages: Annotated[list[str], operator.add]
    count: Annotated[int, operator.add]


# Pattern C: Message State with LangGraph's add_messages reducer
# Specifically built for chat history.
# Handles message updates, ID tracking, and appending AIMessage/HumanMessage.
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Pattern D: Multi-Step Sequential State
class MultiStepState(TypedDict):
    input: str
    step_one_output: str
    analyzed: str
    enhanced: str
    final: str


# ============================================================================
# 2. BASIC TWO-NODE QUESTION & ANSWER GRAPH
# ============================================================================

class ExerciseLangGraphState(TypedDict):
    input: str
    questions: list[str]
    answer: str


def demo_two_node_graph():
    """
    Demonstrates a minimal 2-node graph:
    [START] ──> [questions_node] ──> [answer_node] ──> [END]
    
    1. Node 1 generates 3 diagnostic questions about a topic.
    2. Node 2 answers one of those questions.
    """
    print("\n--- 1. Basic Two-Node Sequential Graph ---")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Node 1: Generates questions
    def questions_node(state: ExerciseLangGraphState) -> dict:
        prompt = f"Generate 3 short questions about the following topic: {state['input']}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"questions": [q.strip() for q in response.content.split("\n") if q.strip()]}

    # Node 2: Answers one question
    def answer_node(state: ExerciseLangGraphState) -> dict:
        prompt = f"Answer any one of the following questions in a single sentence:\n{state['questions']}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"answer": response.content}

    # Assemble Graph
    workflow = StateGraph(ExerciseLangGraphState)

    # Register nodes
    workflow.add_node("questions_node", questions_node)
    workflow.add_node("answer_node", answer_node)

    # Connect edges
    workflow.add_edge(START, "questions_node")
    workflow.add_edge("questions_node", "answer_node")
    workflow.add_edge("answer_node", END)

    # Compile the graph into an executable runnable
    app = workflow.compile()

    # Invoke with initial state
    topic = "Python Programming"
    print(f"Invoking graph with input topic: '{topic}'...")
    result = app.invoke({"input": topic, "questions": [], "answer": ""})

    print(f"\nGenerated Questions:")
    for q in result["questions"]:
        print(f"  - {q}")
    print(f"\nFinal Answer:\n  {result['answer']}")


# ============================================================================
# 3. ACCUMULATOR REDUCER DEMO (operator.add)
# ============================================================================

def demo_accumulator_reducers():
    """
    Demonstrates how `Annotated[list[str], operator.add]` accumulates data
    across multiple independent nodes without manual list concatenation.
    """
    print("\n--- 2. Accumulator Reducer Demonstration (operator.add) ---")

    def node_a(state: AccumulatingState) -> dict:
        return {"messages": ["Step A executed"], "count": 1}

    def node_b(state: AccumulatingState) -> dict:
        return {"messages": ["Step B executed"], "count": 1}

    workflow = StateGraph(AccumulatingState)
    workflow.add_node("step_a", node_a)
    workflow.add_node("step_b", node_b)

    workflow.add_edge(START, "step_a")
    workflow.add_edge("step_a", "step_b")
    workflow.add_edge("step_b", END)

    app = workflow.compile()

    # Initial state with empty list and count 0
    result = app.invoke({"messages": [], "count": 0})

    print(f"Messages list accumulated: {result['messages']}")
    print(f"Total count summed: {result['count']}")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run LangGraph basic demos.
    """
    print("=" * 60)
    print("LANGGRAPH BASICS: STATES, NODES, EDGES & REDUCERS")
    print("=" * 60)

    demo_two_node_graph()
    demo_accumulator_reducers()


if __name__ == "__main__":
    main()
