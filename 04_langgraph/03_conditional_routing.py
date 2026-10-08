"""
03_conditional_routing.py
=========================
Module: LangGraph - Dynamic Branching & Conditional Edges

In linear pipelines, execution moves unconditionally from Node A to Node B.
In agentic architectures, the graph must make runtime routing decisions:
- "Is this user query a question, a statement, or a command?"
- "Is this customer issue urgent and complex, or normal and simple?"
- "Is the drafted answer high quality (score >= 8), or does it need another revision?"

Key Concept: `graph.add_conditional_edges(source_node, routing_function, path_map)`
1. `source_node`: The node whose output determines the next step.
2. `routing_function`: A Python function inspecting the state and returning a key (e.g. "urgent_complex").
3. `path_map`: A dictionary mapping keys to destination node names:
   `{"urgent_complex": "senior_team_node", "normal_simple": "standard_node"}`.

Patterns Demonstrated:
1. `basic_routing`: 3-way intent routing (Question vs Statement vs Command).
2. `multipath_routing`: 2D multi-criteria decision matrix (Urgency x Complexity -> 4 specialized handler nodes).
3. `conditional_loop`: Iterative quality improvement loop using conditional termination.

Prerequisites:
- OPENAI_API_KEY in `.env`
"""

from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Asset paths
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. BASIC INTENT ROUTING (QUESTION / STATEMENT / COMMAND)
# ============================================================================

class RouterState(TypedDict):
    query: str
    query_type: str
    response: str


def basic_routing():
    """
    Demonstrates routing a user query to one of three specialized handlers
    based on intent classification.
    """
    print("\n--- 1. Basic 3-Way Intent Routing ---")

    def classify_query(state: RouterState) -> dict:
        prompt = (
            f"Classify this input strictly as 'question', 'statement', or 'command'.\n"
            f"Return ONLY one word: question, statement, or command.\n"
            f"Input: {state['query']}"
        )
        response = llm.invoke(prompt)
        q_type = response.content.lower().strip()
        print(f"  [classify_query]: Classified as -> '{q_type}'")
        return {"query_type": q_type}

    def handle_question(state: RouterState) -> dict:
        ans = llm.invoke(f"Provide a clear, direct answer to: {state['query']}")
        return {"response": f"[Question Handler]: {ans.content}"}

    def handle_statement(state: RouterState) -> dict:
        comment = llm.invoke(f"Provide a thoughtful remark regarding: {state['query']}")
        return {"response": f"[Statement Handler]: {comment.content}"}

    def handle_command(state: RouterState) -> dict:
        action = llm.invoke(f"Confirm execution of this instruction: {state['query']}")
        return {"response": f"[Command Handler]: {action.content}"}

    # Router function returning the branch key
    def route_by_query_type(state: RouterState) -> Literal["question", "statement", "command"]:
        q_type = state.get("query_type", "statement")
        if "question" in q_type:
            return "question"
        elif "command" in q_type:
            return "command"
        return "statement"

    graph = StateGraph(RouterState)

    graph.add_node("classify_query", classify_query)
    graph.add_node("handle_question", handle_question)
    graph.add_node("handle_statement", handle_statement)
    graph.add_node("handle_command", handle_command)

    graph.add_edge(START, "classify_query")

    # Conditional Branching
    graph.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "question": "handle_question",
            "statement": "handle_statement",
            "command": "handle_command",
        }
    )

    graph.add_edge("handle_question", END)
    graph.add_edge("handle_statement", END)
    graph.add_edge("handle_command", END)

    app = graph.compile()

    test_queries = [
        "What is the airspeed velocity of an unladen swallow?",
        "I really enjoy learning LangChain and LangGraph.",
        "Generate a summary of the quarterly report immediately.",
    ]

    for q in test_queries:
        print(f"\nInput: \"{q}\"")
        res = app.invoke({"query": q, "query_type": "", "response": ""})
        print(f"Output: {res['response']}")


# ============================================================================
# 2. MULTI-PATH DECISION MATRIX (URGENCY x COMPLEXITY)
# ============================================================================

class TaskState(TypedDict):
    task: str
    urgency: str
    complexity: str
    handler: str
    result: str


def multipath_routing():
    """
    Demonstrates a 2D multi-criteria decision matrix:
    - Urgency: 'urgent' vs 'normal'
    - Complexity: 'complex' vs 'simple'
    
    Routes to 4 specialized departments:
    1. Urgent + Complex -> Senior SWAT Team
    2. Urgent + Simple  -> Rapid Response Desk
    3. Normal + Complex -> Specialized Tier-2 Engineer
    4. Normal + Simple  -> Automated Standard Queue
    """
    print("\n--- 2. Multi-Path Decision Matrix (Urgency x Complexity) ---")

    def analyze_task(state: TaskState) -> dict:
        urgency_res = llm.invoke(f"Is this urgent or normal? Return only 'urgent' or 'normal': {state['task']}")
        complexity_res = llm.invoke(f"Is this complex or simple? Return only 'complex' or 'simple': {state['task']}")
        return {
            "urgency": urgency_res.content.lower().strip(),
            "complexity": complexity_res.content.lower().strip(),
        }

    def urgent_complex_handler(state: TaskState) -> dict:
        return {"handler": "Senior SWAT Team", "result": "Escalated immediately to Senior Staff."}

    def urgent_simple_handler(state: TaskState) -> dict:
        return {"handler": "Rapid Response Desk", "result": "Dispatched to on-call support representative."}

    def normal_complex_handler(state: TaskState) -> dict:
        return {"handler": "Specialist Tier-2", "result": "Scheduled for deep technical triage."}

    def normal_simple_handler(state: TaskState) -> dict:
        return {"handler": "Standard Queue", "result": "Queued for routine batch execution."}

    def route_task(state: TaskState) -> str:
        is_urgent = "urgent" in state["urgency"]
        is_complex = "complex" in state["complexity"]

        if is_urgent and is_complex:
            return "urgent_complex"
        elif is_urgent:
            return "urgent_simple"
        elif is_complex:
            return "normal_complex"
        else:
            return "normal_simple"

    graph = StateGraph(TaskState)

    graph.add_node("analyze", analyze_task)
    graph.add_node("urgent_complex", urgent_complex_handler)
    graph.add_node("urgent_simple", urgent_simple_handler)
    graph.add_node("normal_complex", normal_complex_handler)
    graph.add_node("normal_simple", normal_simple_handler)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_task,
        {
            "urgent_complex": "urgent_complex",
            "urgent_simple": "urgent_simple",
            "normal_complex": "normal_complex",
            "normal_simple": "normal_simple",
        }
    )

    for handler_node in ["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
        graph.add_edge(handler_node, END)

    app = graph.compile()

    tasks = [
        "Production database is locked up and payment gateway is rejecting transactions!",
        "Fix minor typo in README file on line 12",
        "Redesign complete architectural schema from MongoDB to PostgreSQL",
    ]

    for t in tasks:
        res = app.invoke({"task": t, "urgency": "", "complexity": "", "handler": "", "result": ""})
        print(f"\nTask: \"{res['task']}\"")
        print(f"Classification: Urgency={res['urgency']} | Complexity={res['complexity']}")
        print(f"Routed Handler: {res['handler']} -> Action: {res['result']}")


# ============================================================================
# 3. CONDITIONAL QUALITY REVISION LOOP
# ============================================================================

class QualityState(TypedDict):
    content: str
    quality_score: int
    feedback: str
    final_content: str
    iteration: int


def conditional_loop():
    """
    Demonstrates iterative refinement using conditional loops:
    [START] ──> [evaluate] ──(score >= 8 or iter >= 3?)──> [finalize] ──> [END]
                     ▲                                          │
                     │                 (score < 8)              │
                     └───────────── [improve] ◄─────────────────┘
    """
    print("\n--- 3. Iterative Quality Refinement Loop ---")

    def evaluate_quality(state: QualityState) -> dict:
        prompt = (
            f"Rate the literary quality and clarity of this sentence from 1 to 10 (reply with ONLY the integer):\n"
            f"Sentence: \"{state['content']}\""
        )
        response = llm.invoke(prompt)
        try:
            score = int("".join(filter(str.isdigit, response.content)))
        except Exception:
            score = 6

        print(f"  [evaluate_quality] Iteration #{state['iteration']} - Quality Score: {score}/10")
        return {"quality_score": score}

    def improve_content(state: QualityState) -> dict:
        prompt = (
            f"Improve this sentence to make it more elegant, vivid, and polished:\n"
            f"Current: \"{state['content']}\""
        )
        response = llm.invoke(prompt)
        improved = response.content.strip().strip('"')
        print(f"  [improve_content] Polished draft: \"{improved}\"")
        return {
            "content": improved,
            "iteration": state["iteration"] + 1,
        }

    def finalize_content(state: QualityState) -> dict:
        return {
            "final_content": state["content"],
            "feedback": f"Approved after {state['iteration']} iterations with score {state['quality_score']}/10",
        }

    # Conditional router
    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        if state["quality_score"] >= 8 or state["iteration"] >= 3:
            return "finalize"
        return "improve"

    graph = StateGraph(QualityState)

    graph.add_node("evaluate", evaluate_quality)
    graph.add_node("improve", improve_content)
    graph.add_node("finalize", finalize_content)

    graph.add_edge(START, "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {
        "improve": "improve",
        "finalize": "finalize",
    })
    graph.add_edge("improve", "evaluate")  # Loop back!
    graph.add_edge("finalize", END)

    app = graph.compile()

    initial_content = "We make software that works okay."
    print(f"Initial Draft: \"{initial_content}\"")

    result = app.invoke({
        "content": initial_content,
        "quality_score": 0,
        "feedback": "",
        "final_content": "",
        "iteration": 0,
    })

    print(f"\nFinal Approved Content: \"{result['final_content']}\"")
    print(f"Resolution Details: {result['feedback']}")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run conditional routing demos.
    """
    print("=" * 60)
    print("LANGGRAPH CONDITIONAL EDGES & DYNAMIC CONTROL FLOW")
    print("=" * 60)

    basic_routing()
    multipath_routing()
    conditional_loop()


if __name__ == "__main__":
    main()
