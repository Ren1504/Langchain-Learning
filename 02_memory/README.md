# 02 - Conversational Memory Strategies

This folder explores how to give memory to stateless Large Language Models.

## Core Concepts & Patterns

| Pattern | Description | Token Cost | Context Retention |
| :--- | :--- | :--- | :--- |
| **In-Memory Buffer** | Appends every turn to a list. | Linear growth (unbounded) | 100% until context window limit is reached |
| **Multi-Session Isolation** | Isolates dialogues by `session_id` using `RunnableWithMessageHistory`. | Independent per user | Separated per user session |
| **Token Trimming** | Uses `trim_messages` to fit strictly within a token budget while protecting system prompts. | Bounded & predictable | Drops oldest turns |
| **Windowed Memory** | Retains only the last $K$ message pairs. | Fixed & bounded | Forgets early dialogue beyond the window |
| **Summary Memory** | Employs an LLM to maintain a rolling summary of older dialogue while keeping recent turns verbatim. | Constant + small LLM summarization cost | Compresses old context without losing key facts |
| **Persistent SQLite** | Uses `SQLChatMessageHistory` to store conversations in an SQLite database file. | Based on loading strategy | Durable across server restarts |

## Files

- [`01_conversation_memory.py`](file:///d:/Flutter/LangChain-Learning/02_memory/01_conversation_memory.py): All 6 memory patterns demonstrated with runnable code and in-depth comments.
- `chat_history.db`: SQLite database generated when running the persistent memory demos.

## How to Run

```bash
python 02_memory/01_conversation_memory.py
```
