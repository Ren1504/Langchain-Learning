# 04 - LangGraph: Stateful Multi-Actor Agent Orchestration

LangGraph is a specialized library for building stateful, multi-actor applications and autonomous AI agents with cycles, conditional branching, and human oversight.

## Curriculum & Key Concepts

| File | Core Concepts Covered |
| :--- | :--- |
| [`01_graph_basics.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/01_graph_basics.py) | • `StateGraph` and `TypedDict` state schemas<br>• State Reducers: Overwrite vs `operator.add` vs `add_messages`<br>• Nodes, edges, `START`, `END`, `.compile()`, and `.invoke()`<br>• Multi-step pipeline generation |
| [`02_conversation_graph.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/02_conversation_graph.py) | • Dynamic conversational pipelines: Sentiment classification -> Adaptive tone response<br>• Exporting diagrams: Mermaid markdown syntax and PNG images saved to [`assets/`](file:///d:/Flutter/LangChain-Learning/04_langgraph/assets/) |
| [`03_conditional_routing.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/03_conditional_routing.py) | • Dynamic control flow with `add_conditional_edges`<br>• Intent routing (Question / Statement / Command)<br>• 2D decision matrix (Urgency x Complexity -> 4 specialized handler nodes)<br>• Iterative quality evaluation and refinement loops |
| [`04_cycles_and_loops.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/04_cycles_and_loops.py) | • Agent loops and cycles for self-correction<br>• Self-healing code generator: Generates code -> Deterministic syntax check with `compile()` -> Error feedback loop<br>• Guardrails against infinite recursion (`recursion_limit`, `max_iterations`) |
| [`05_human_in_the_loop.py`](file:///d:/Flutter/LangChain-Learning/04_langgraph/05_human_in_the_loop.py) | • **Human-in-the-Loop (HITL)** architecture<br>• State checkpointing with `MemorySaver`<br>• Breakpoints with `interrupt_before`<br>• State inspection (`get_state`) and modification (`update_state`)<br>• Resuming execution from checkpoints |

## Assets & Visualizations

The [`assets/`](file:///d:/Flutter/LangChain-Learning/04_langgraph/assets/) directory contains generated visual topologies of the graphs:
- [`basic_graph.png`](file:///d:/Flutter/LangChain-Learning/04_langgraph/assets/basic_graph.png)
- [`conditional_graph.png`](file:///d:/Flutter/LangChain-Learning/04_langgraph/assets/conditional_graph.png)
- [`conditional_looping_graph.png`](file:///d:/Flutter/LangChain-Learning/04_langgraph/assets/conditional_looping_graph.png)
- [`graph.png`](file:///d:/Flutter/LangChain-Learning/04_langgraph/assets/graph.png)

## How to Run

```bash
python 04_langgraph/01_graph_basics.py
python 04_langgraph/02_conversation_graph.py
python 04_langgraph/03_conditional_routing.py
python 04_langgraph/04_cycles_and_loops.py
python 04_langgraph/05_human_in_the_loop.py
```
