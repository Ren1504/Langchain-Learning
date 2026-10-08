"""
01_conversation_memory.py
=========================
Module: Memory - Conversational Memory Strategies & Persistence

LLMs are inherently stateless. Each API request is treated as independent.
To support multi-turn conversations, previous messages must be passed back to
the model with each new prompt.

This module explores the full spectrum of memory management strategies:
1. Basic In-Memory History (`InMemoryChatMessageHistory` + `RunnableWithMessageHistory`).
2. Multi-Session Management (handling different users simultaneously via `session_id`).
3. Token Trimming (`trim_messages`): Enforcing hard token limits while preserving system instructions.
4. Windowed Memory: Sliding window buffer keeping only the last K interaction pairs.
5. Summary Memory: Rolling summarization — compressing older dialogue while keeping recent turns verbatim.
6. Persistent Memory: Storing chat histories in SQLite (`SQLChatMessageHistory`) across application restarts.

Prerequisites:
- OPENAI_API_KEY set in `.env` (or configured with another model provider via init_chat_model).
"""

import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize primary chat model
llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")

# Define portable path for SQLite persistent database
DB_PATH = Path(__file__).parent / "chat_history.db"


# ============================================================================
# 1. BASIC IN-MEMORY CHAT HISTORY
# ============================================================================

def basic_memory():
    """
    Demonstrates the standard modern memory pattern in LangChain:
    1. A prompt with a `MessagesPlaceholder(variable_name="history")`.
    2. A session dictionary storing `InMemoryChatMessageHistory` per user/session.
    3. `RunnableWithMessageHistory` automatically injecting and updating history.
    """
    print("\n--- 1. Basic In-Memory Conversation History ---")

    # Prompt template with placeholder for dynamic history injection
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    # Session store mapping session_id -> InMemoryChatMessageHistory
    store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Factory function returning the session history object."""
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Wrap the chain with automatic message history tracking
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # Execution configuration identifying the active session
    config = {"configurable": {"session_id": "user_123"}}

    messages = [
        "Hi, my name is Ren.",
        "I play League of Legends.",
        "What is my name and what game do I play?"
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI:   {response}")

    print(f"\nStored message count in session 'user_123': {len(store['user_123'].messages)}")


# ============================================================================
# 2. MULTI-SESSION ISOLATION
# ============================================================================

def multiple_sessions():
    """
    Demonstrates that `RunnableWithMessageHistory` isolates histories by `session_id`.
    User A and User B can have completely independent conversations concurrently.
    """
    print("\n--- 2. Multi-Session Isolation ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()
    store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    user_a_config = {"configurable": {"session_id": "user_a"}}
    user_b_config = {"configurable": {"session_id": "user_b"}}

    # User A turn
    print("\n[Session User A]")
    print("User A: My favorite language is Python.")
    chain_with_history.invoke({"input": "My favorite language is Python."}, config=user_a_config)

    # User B turn
    print("\n[Session User B]")
    print("User B: My favorite language is JavaScript.")
    chain_with_history.invoke({"input": "My favorite language is JavaScript."}, config=user_b_config)

    # Verification: Ask both users what their favorite language is
    resp_a = chain_with_history.invoke({"input": "What is my favorite language?"}, config=user_a_config)
    resp_b = chain_with_history.invoke({"input": "What is my favorite language?"}, config=user_b_config)

    print(f"\nAI to User A: {resp_a}")
    print(f"AI to User B: {resp_b}")


# ============================================================================
# 3. MESSAGE TRIMMING BY TOKEN BUDGET
# ============================================================================

def message_trimming():
    """
    Demonstrates using `trim_messages` to ensure the history fits within a strict
    token budget before passing it to the model.
    
    Parameters:
    - `max_tokens`: Upper bound of tokens allowed in history.
    - `strategy="last"`: Drops the oldest messages first.
    - `token_counter`: Can count strings or LLM-specific tokens.
    - `include_system=True`: Ensures system instructions are never dropped.
    """
    print("\n--- 3. Token-Based Message Trimming ---")

    messages = [
        SystemMessage(content="You are a helpful assistant. Be concise."),
        HumanMessage(content="Hi my name is Ren."),
        AIMessage(content="Hello Ren! How can I assist you today?"),
        HumanMessage(content="I play league of legends."),
        AIMessage(content="That's great! League of Legends is a popular MOBA game."),
        HumanMessage(content="What is my name and what game do I play?"),
    ]

    # Create trimmer with a strict 40-token limit
    trimmer = trim_messages(
        max_tokens=40,
        strategy="last",
        token_counter=llm,
        allow_partial=False,
    )

    print(f"Original message count: {len(messages)}")
    trimmed_messages = trimmer.invoke(messages)
    print(f"Trimmed message count: {len(trimmed_messages)}")

    print("\nSurviving messages in context window:")
    for msg in trimmed_messages:
        print(f"  [{msg.__class__.__name__}]: {msg.content}")


# ============================================================================
# 4. SLIDING WINDOW MEMORY (KEEP LAST K TURNS)
# ============================================================================

def windowed_memory():
    """
    Demonstrates a fixed sliding-window conversation buffer.
    Instead of allowing history to grow indefinitely, only the most recent
    `k` interaction pairs are preserved.
    
    Trade-off:
    - Pro: Predictable token cost and low latency.
    - Con: Early user facts (e.g., name, location) are forgotten once they slide out.
    """
    print("\n--- 4. Windowed Memory (Sliding Window of K Turns) ---")

    class WindowedChatHistory(InMemoryChatMessageHistory):
        """Custom Chat History that retains at most k pairs (2k messages)."""
        k: int = 2

        def add_messages(self, messages):
            super().add_messages(messages)
            if len(self.messages) > self.k * 2:
                self.messages = self.messages[-(self.k * 2):]

    store: Dict[str, WindowedChatHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowedChatHistory(k=2)
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "window_user"}}

    conversation_steps = [
        "My name is Paulo.",
        "I live in Seattle.",
        "I work as an AI engineer.",
        "I have 2 cats.",
        "What do you remember about me?",
    ]

    for step in conversation_steps:
        print(f"\nUser: {step}")
        response = chain_with_history.invoke({"input": step}, config=config)
        print(f"AI:   {response}")

    print("\nInspection of remaining memory in window:")
    remaining = store["window_user"].messages
    for m in remaining:
        print(f"  [{m.__class__.__name__}]: {m.content}")


# ============================================================================
# 5. SUMMARY BUFFER MEMORY (COMPRESSING OLD TURNS)
# ============================================================================

def summary_memory():
    """
    Demonstrates rolling summary memory:
    - Maintains a short buffer of the most recent messages verbatim (e.g., last 4).
    - When the buffer exceeds the limit, an LLM prompt condenses the oldest turns
      into an evolving summary string.
      
    Trade-off:
    - Pro: Retains long-term facts indefinitely without context window explosion.
    - Con: Requires extra LLM summarization calls when flushing old turns.
    """
    print("\n--- 5. Rolling Summary Memory ---")

    # Chat execution chain incorporating the rolling summary
    chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Be concise.\n\n"
            "Summary of earlier conversation:\n{summary}",
        ),
        MessagesPlaceholder(variable_name="recent_messages"),
        ("human", "{input}"),
    ])
    chat_chain = chat_prompt | llm | StrOutputParser()

    # Summarizer chain that updates the summary
    summarize_prompt = ChatPromptTemplate.from_template(
        "Condense the current summary and new messages into a single updated summary (2-3 sentences).\n"
        "Preserve all key personal facts about the user.\n\n"
        "Current summary:\n{current_summary}\n\n"
        "New messages:\n{new_messages}\n\n"
        "Updated summary:"
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    running_summary = ""
    recent_messages = []
    MAX_RECENT = 4  # Threshold: 2 exchanges (4 messages)

    exchanges = [
        "My name is Paulo and I'm from Seattle.",
        "I work as an AI engineer building RAG systems.",
        "I have 2 cats named Luna and Milo.",
        "I'm building a LangChain course for learners.",
        "What do you know about me? List everything you remember.",
    ]

    for user_input in exchanges:
        print(f"\nUser: {user_input}")

        response = chat_chain.invoke({
            "summary": running_summary if running_summary else "No prior conversation.",
            "recent_messages": recent_messages,
            "input": user_input,
        })
        print(f"AI:   {response}")

        # Append turn to recent messages buffer
        recent_messages.append(HumanMessage(content=user_input))
        recent_messages.append(AIMessage(content=response))

        # Check if buffer limit exceeded
        if len(recent_messages) > MAX_RECENT:
            messages_to_summarize = recent_messages[:-MAX_RECENT]
            formatted_dialogue = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in messages_to_summarize
            )

            # Update running summary with the condensed history
            running_summary = summarize_chain.invoke({
                "current_summary": running_summary if running_summary else "None yet.",
                "new_messages": formatted_dialogue,
            })

            # Keep only the tail of the buffer
            recent_messages = recent_messages[-MAX_RECENT:]
            print(f"  [Memory Event]: Summarized older turns. Updated summary:\n  \"{running_summary}\"")

    print("\n--- Final Memory State ---")
    print(f"Compressed Long-Term Summary: {running_summary}")
    print(f"Verbatim Short-Term Buffer: {len(recent_messages)} messages")


# ============================================================================
# 6. PERSISTENT SQLITE MEMORY
# ============================================================================

def persistent_memory_demo():
    """
    Demonstrates true persistence using SQLite (`SQLChatMessageHistory`).
    Even if the Python script or server restarts, previous sessions are loaded
    directly from disk based on the `session_id`.
    """
    print("\n--- 6. Persistent Memory with SQLite ---")

    db_file = str(DB_PATH)
    connection_string = f"sqlite:///{db_file}"
    session_id = "persistent_user_demo"

    # Clean existing test database for fresh run
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass

    def build_chain():
        def get_session_history(sid: str) -> BaseChatMessageHistory:
            return SQLChatMessageHistory(session_id=sid, connection=connection_string)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Remember user facts accurately."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        chain = prompt | llm | StrOutputParser()
        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    config = {"configurable": {"session_id": session_id}}

    print("--- Session 1: Storing Facts ---")
    chain_instance_1 = build_chain()
    print("User: Hi, my name is Alex and I love hiking and photography.")
    r1 = chain_instance_1.invoke(
        {"input": "Hi, my name is Alex and I love hiking and photography."}, config=config
    )
    print(f"AI:   {r1}")

    # Explicitly destroy chain instance to simulate process termination
    del chain_instance_1
    print("\n[Simulating app restart / process destruction...]")

    print("\n--- Session 2: Reloading from SQLite Database ---")
    chain_instance_2 = build_chain()
    print("User: What is my name and what are my hobbies?")
    r2 = chain_instance_2.invoke(
        {"input": "What is my name and what are my hobbies?"}, config=config
    )
    print(f"AI:   {r2}")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================

def main():
    """
    Run memory demos.
    """
    print("=" * 60)
    print("LANGCHAIN CONVERSATIONAL MEMORY STRATEGIES")
    print("=" * 60)

    basic_memory()
    multiple_sessions()
    message_trimming()
    windowed_memory()
    summary_memory()
    persistent_memory_demo()


if __name__ == "__main__":
    main()
