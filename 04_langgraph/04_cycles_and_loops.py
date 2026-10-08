"""
04_cycles_and_loops.py
======================
Module: LangGraph - Cycles, Iterative Loops & Self-Correcting Agents

Why are Cycles Crucial?
Standard software pipelines are strictly feed-forward: if step 2 produces a syntax
error or hallucination, the program crashes or emits garbage.
In agentic AI, cycles allow the agent to:
1. Generate a candidate solution (e.g. Python code, SQL query, JSON payload).
2. Validate the output using deterministic code execution or unit tests.
3. If errors occur, route execution back to the LLM with the compiler error trace!
4. Loop iteratively until tests pass or the recursion threshold is hit.

Guarding Against Infinite Loops:
- Always track an `iteration` counter in the state.
- Set a hard `max_iterations` cutoff.
- In LangGraph, configure `recursion_limit` on `.invoke(..., {"recursion_limit": 10})`
  as a safety ceiling.

Architecture:
               ┌───────────────────────┐
               │        [START]        │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │     generate_code     │ ◄────────────────┐
               └───────────┬───────────┘                  │
                           ▼                              │ If syntax error
               ┌───────────────────────┐                  │ & iteration < max
               │     validate_code     │                  │
               └───────────┬───────────┘                  │
                           ▼                              │
                     [should_continue] ───────────────────┘
                           │
                           │ If success OR iteration >= max
                           ▼
               ┌───────────────────────┐
               │       finalize        │ ──> [END]
               └───────────────────────┘

Prerequisites:
- OPENAI_API_KEY in `.env`
"""

import operator
from typing import Literal
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# Model: Low temperature (0.0) for deterministic, precise code generation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class CodeGenState(TypedDict):
    task: str
    code: str
    errors: Annotated[list[str], operator.add]  # Accumulates compiler errors
    iteration: int
    max_iterations: int
    success: bool


# ============================================================================
# 2. SELF-CORRECTING CODE GENERATION AGENT
# ============================================================================

def self_correcting_code():
    """
    Demonstrates a self-correcting agent loop that generates Python code,
    tests its syntax using Python's native `compile()` parser, and iterates
    with error feedback if any syntax errors are encountered.
    """
    print("\n--- Self-Correcting Code Generation Agent ---")

    def code_clean(code: str) -> str:
        """Strips markdown code fences (```python ... ```) from LLM output."""
        code = code.strip()
        if code.startswith("```"):
            lines = code.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines).strip()
        return code

    # Node 1: Code Generator (Generates initial draft or applies fixes)
    def generate_code(state: CodeGenState) -> dict:
        current_iter = state["iteration"]

        if current_iter == 0:
            prompt = (
                f"Write a standalone Python function for this task: {state['task']}\n"
                f"Return ONLY valid Python code with NO explanation and NO markdown fences."
            )
        else:
            latest_error = state["errors"][-1] if state["errors"] else "Unknown error"
            prompt = (
                f"The following Python code failed syntax validation:\n\n{state['code']}\n\n"
                f"Compiler Error:\n{latest_error}\n\n"
                f"Fix the code. Return ONLY valid Python code with NO explanation."
            )

        print(f"  [generate_code] Generating code (Attempt #{current_iter + 1})...")
        response = llm.invoke(prompt)
        clean_code = code_clean(response.content)

        return {
            "code": clean_code,
            "iteration": current_iter + 1,
        }

    # Node 2: Validator (Checks Python syntax deterministically)
    def validate_code(state: CodeGenState) -> dict:
        code_to_test = state["code"]
        print("  [validate_code] Compiling generated code with Python parser...")

        try:
            # Deterministic syntax check
            compile(code_to_test, "<agent_script>", "exec")
            print("  >>> Compilation Succeeded! Valid Python code.")
            return {"success": True}
        except SyntaxError as e:
            err_msg = f"SyntaxError on line {e.lineno}: {e.msg}"
            print(f"  >>> Compilation Failed: {err_msg}")
            return {
                "errors": [err_msg],
                "success": False,
            }
        except Exception as e:
            err_msg = f"Error: {e}"
            print(f"  >>> Compilation Error: {err_msg}")
            return {
                "errors": [err_msg],
                "success": False,
            }

    # Conditional Router: Decides whether to retry or finalize
    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        if state["success"]:
            return "end"
        if state["iteration"] >= state["max_iterations"]:
            print("  >>> Hit maximum iteration limit. Terminating loop.")
            return "end"
        print("  >>> Routing back to [generate_code] with compiler feedback.")
        return "generate"

    def finalize(state: CodeGenState) -> dict:
        return state

    # Assemble Graph
    workflow = StateGraph(CodeGenState)

    workflow.add_node("generate", generate_code)
    workflow.add_node("validate", validate_code)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges("validate", should_continue, {
        "generate": "generate",
        "end": "finalize",
    })
    workflow.add_edge("finalize", END)

    app = workflow.compile()

    task_description = "Write a recursive function to check if a number is an Armstrong number."
    print(f"Goal: {task_description}\n")

    result = app.invoke({
        "task": task_description,
        "code": "",
        "errors": [],
        "iteration": 0,
        "max_iterations": 3,
        "success": False,
    })

    print("\n" + "=" * 50)
    print("FINAL AGENT RESULT:")
    print("=" * 50)
    print(f"Task:               {result['task']}")
    print(f"Total Iterations:   {result['iteration']}")
    print(f"Validated Success:  {result['success']}")
    print(f"Final Python Code:\n\n{result['code']}")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run self-correcting cycle demo.
    """
    print("=" * 60)
    print("LANGGRAPH CYCLES & SELF-CORRECTING AGENT LOOPS")
    print("=" * 60)

    self_correcting_code()


if __name__ == "__main__":
    main()
