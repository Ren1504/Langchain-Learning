"""
05_human_in_the_loop.py
=======================
Module: LangGraph - Human-in-the-Loop (HITL) Workflows & Checkpointing

Why Human-in-the-Loop?
Autonomous AI agents are capable of high-speed reasoning and tool execution.
However, in production enterprise systems, sensitive operations MUST NOT be executed
without human oversight. Examples include:
- Executing financial refunds or transfers.
- Deleting customer accounts or database tables.
- Sending external emails or publishing public social media posts.

How LangGraph Implements HITL:
1. State Checkpointing (`MemorySaver`):
   Saves snapshots of the entire graph state at each superstep.
2. Breakpoints (`interrupt_before` or `interrupt_after`):
   Pauses graph execution automatically before entering sensitive nodes.
3. State Inspection (`app.get_state(config)`):
   Allows a human reviewer to inspect the proposed action, arguments, and reasoning.
4. State Modification (`app.update_state(config, updates)`):
   The human can edit or override the agent's drafted action before approval.
5. Resumption (`app.invoke(None, config)`):
   Resumes graph execution from the exact paused checkpoint.

Architecture:
               ┌───────────────────────┐
               │        [START]        │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │      draft_action     │ (Agent drafts refund or email)
               └───────────┬───────────┘
                           ▼
                  [PAUSE / BREAKPOINT]   ◄── Human reviews, approves, or edits
                           │
                           ▼ (On Resume)
               ┌───────────────────────┐
               │    execute_action     │ (Dispatches sensitive transaction)
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │        [END]          │
               └───────────────────────┘

Prerequisites:
- Requires `langgraph`
"""

from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class RefundState(TypedDict):
    customer_id: str
    amount: float
    reason: str
    draft_status: str
    admin_notes: str
    executed: bool


# ============================================================================
# 2. HUMAN-IN-THE-LOOP AGENT IMPLEMENTATION
# ============================================================================

def human_in_the_loop_demo():
    """
    Demonstrates the full lifecycle of a Human-in-the-Loop workflow:
    1. Agent drafts a financial refund.
    2. Execution automatically pauses before `execute_refund`.
    3. Human operator inspects the proposed refund and adds notes / adjusts amount.
    4. Execution resumes to complete the transaction.
    """
    print("\n--- Human-in-the-Loop (HITL) Refund Workflow ---")

    # Node 1: AI Agent drafts the proposed action
    def draft_refund(state: RefundState) -> dict:
        print(f"  [draft_refund]: Agent proposing refund of ${state['amount']} for customer {state['customer_id']}...")
        return {
            "draft_status": "PENDING_HUMAN_APPROVAL",
            "admin_notes": "Awaiting supervisor sign-off.",
        }

    # Node 2: Critical sensitive action that requires human approval
    def execute_refund(state: RefundState) -> dict:
        print(f"\n  >>> [execute_refund NODE TRIGGERED] <<<")
        print(f"  Processing payment gateway refund of ${state['amount']} to {state['customer_id']}...")
        print(f"  Approval Notes: \"{state['admin_notes']}\"")
        return {
            "draft_status": "EXECUTED",
            "executed": True,
        }

    # 1. Initialize in-memory checkpointer
    # (In production, replace with MongoDBSaver, PostgresSaver, or SqliteSaver)
    memory = MemorySaver()

    # 2. Build the graph
    workflow = StateGraph(RefundState)

    workflow.add_node("draft_refund", draft_refund)
    workflow.add_node("execute_refund", execute_refund)

    workflow.add_edge(START, "draft_refund")
    workflow.add_edge("draft_refund", "execute_refund")
    workflow.add_edge("execute_refund", END)

    # 3. Compile with checkpointer AND interrupt_before
    # Execution will strictly PAUSE immediately before running `execute_refund`!
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["execute_refund"],
    )

    # 4. Each conversation/user session needs a unique thread_id for state tracking
    config = {"configurable": {"thread_id": "transaction_tx_9082"}}

    # Initial request
    initial_input = {
        "customer_id": "cust_4512",
        "amount": 250.00,
        "reason": "Damaged goods upon arrival",
        "draft_status": "DRAFT",
        "admin_notes": "",
        "executed": False,
    }

    print("\nStep 1: Running graph up to the breakpoint...")
    for event in app.stream(initial_input, config=config):
        print(f"  Event: {event}")

    # 5. Inspect the current paused state
    current_state = app.get_state(config)
    print("\nStep 2: Inspecting Paused State at Breakpoint:")
    print(f"  Next Node Scheduled: {current_state.next}")
    print(f"  Current State Values: {current_state.values}")

    # 6. Human Review & Editing:
    # Supervisor reviews the claim and decides to approve with a goodwill adjustment
    print("\nStep 3: Human Supervisor Reviews & Updates State:")
    print("  Supervisor modifies: adjusted amount to $200.00 and added authorization note.")

    app.update_state(
        config,
        {
            "amount": 200.00,
            "admin_notes": "Approved by Supervisor: Partial store credit applied, adjusted to $200.",
        },
    )

    # Verify updated state
    updated_state = app.get_state(config)
    print(f"  Updated State Values: {updated_state.values}")

    # 7. Resuming Execution
    # Passing `None` resumes the graph from the paused checkpoint
    print("\nStep 4: Resuming Graph Execution...")
    for event in app.stream(None, config=config):
        print(f"  Event: {event}")

    # 8. Final state check
    final_state = app.get_state(config)
    print("\nStep 5: Final Completed State:")
    print(f"  Executed:     {final_state.values['executed']}")
    print(f"  Final Status: {final_state.values['draft_status']}")
    print(f"  Next Nodes:   {final_state.next} (Empty list indicates graph reached END)")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run Human-in-the-Loop demo.
    """
    print("=" * 60)
    print("LANGGRAPH HUMAN-IN-THE-LOOP (HITL) WORKFLOWS")
    print("=" * 60)

    human_in_the_loop_demo()


if __name__ == "__main__":
    main()
