from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
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
from typing import Dict
from dotenv import load_dotenv
import os

load_dotenv()

llm = init_chat_model(model="gpt-4o-mini",model_provider="openai")

def basic_memory():
    llm = init_chat_model(model="gpt-4o-mini",model_provider="openai")

    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assitant. Be concise"),
        MessagesPlaceholder(variable_name="history"),
        ("human","{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    store: Dict[str,InMemoryChatMessageHistory] = {}

    def get_session_history(session_id:str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable":{"session_id":"user_123"}}

    messages = ["Hi my name is Ren",
                "I play league of legends",
                "What's my name and What game do I play?"]

    print  ("\nConversation")

    for msg in messages:
        response = chain_with_history.invoke({"input":msg},config=config)
        print(f"\nUser: {msg}")
        print(f"AIL {response}")

    print(f"\nStored History ({len(store['user_123'].messages)}) messages")

    for msg in store["user_123"].messages:
        role = "Human" if isinstance(msg,HumanMessage) else "AI"
        print(f" {role}: {msg.content[:50]}...")

def multiple_sessions():
    llm = init_chat_model(model="gpt-4o-mini",model_provider="openai")

    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assitant. Be concise"),
        MessagesPlaceholder(variable_name="history"),
        ("human","{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    store: Dict[str,InMemoryChatMessageHistory] = {}

    def get_session_history(session_id:str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    user_a_config = {"configurable":{"session_id":"user_a"}}
    user_b_config = {"configurable":{"session_id":"user_b"}}

    print("\nConversation with User A")
    print("User A: My favorite language is Python")
    chain_with_history.invoke({"input":"My favorite language is Python"},config=user_a_config)
    response_a = chain_with_history.invoke({"input":"What's my favorite language?"},config=user_a_config)
    print(f"AI: {response_a}")

    print("\nConversation with User B")
    chain_with_history.invoke({"input":"My favorite language is JavaScript"},config=user_b_config)
    print("User B: My favorite language is JavaScript")
    response_b = chain_with_history.invoke({"input":"What's my favorite language?"},config=user_b_config)
    print(f"AI: {response_b}")

def message_trimming():

    messages =[
        SystemMessage(content="You are a helpful assistant. Be concise"),
        HumanMessage(content="Hi my name is Ren"), 
        AIMessage(content="Hello Ren! How can I assist you today?"),
        HumanMessage(content="I play league of legends"),
        AIMessage(content="That's great! League of Legends is a popular game. How can I help you with it?"),
        HumanMessage(content="What's my name and What game do I play?"),
    ]

    trimmer = trim_messages(max_tokens=50, strategy="last",token_counter=llm,allow_partial=False)

    print("\nOriginal Messages:")
    for msg in messages:
        print(f" {msg.__class__.__name__}: {msg.content[:50]}...")  
    print("\nTrimmed Messages:")
    trimmed_messages = trimmer.invoke(messages)
    for msg in trimmed_messages:
        print(f" {msg.__class__.__name__}: {msg.content[:50]}...")

def windowed_memory():
    """Implement sliding window memory manually."""

    print("=" * 60)
    print("WINDOWED MEMORY (Keep Last K)")
    print("Fixed-size conversation window")
    print("=" * 60)

    class WindowedChatHistory(InMemoryChatMessageHistory):
        """Chat history that keeps only last k message pairs."""

        k: int = 3  # Pydantic field - number of exchange pairs to keep

        def add_messages(self, messages):
            super().add_messages(messages)
            # Keep only last k pairs (2k messages: human + ai)
            if len(self.messages) > self.k * 2:
                self.messages = self.messages[-(self.k * 2) :]

    store: Dict[str, WindowedChatHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowedChatHistory(k=2)
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "windowed_test"}}

    # Simulate a conversation with more than 2 pairs
    exchanges = [
        "My name is Paulo",
        "I live in Seattle",
        "I work as an AI engineer",
        "I have 2 cats",
        "What do you remember about me?",
    ]

    print("\nConversation with k=2 window:")
    for i, msg in enumerate(exchanges, 1):
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

        # Show window state after each exchange so students SEE it sliding
        history = store["windowed_test"].messages
        print(f"  [Window: {len(history)} msgs] ", end="")
        facts_in_memory = [
            m.content[:40] for m in history if isinstance(m, HumanMessage)
        ]
        print(f"Remembers: {facts_in_memory}")

    # Final state - show what survived and what was lost
    print("\n" + "=" * 60)
    print("RESULT: Window only kept last 2 exchanges!")
    print("Lost: name (Paulo), city (Seattle), AND job (AI engineer)")
    print("Kept: cats + the 'remember' question")
    print(
        "This is the tradeoff: fixed memory = predictable cost, but older context is lost."
    )

def summary_memory():

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Be concise.\n\n"
                "Summary of earlier conversation:\n{summary}",
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            ("human", "{input}"),
        ]
    )

    chat_chain = chat_prompt | llm | StrOutputParser()

    # The summarization prompt: compress messages into a running summary
    summarize_prompt = ChatPromptTemplate.from_template(
        "Condense the current summary and new messages into a single updated summary "
        "(2-3 sentences). Preserve all key facts about the user.\n\n"
        "Current summary:\n{current_summary}\n\n"
        "New messages:\n{new_messages}\n\n"
        "Updated summary:"
    )

    summarize_chain = summarize_prompt | llm | StrOutputParser()

    # --- State ---
    running_summary = ""  # starts empty
    recent_messages = []  # full message objects
    MAX_RECENT = 4  # keep last 4 messages (2 exchanges) before summarizing

    # --- Conversation ---
    exchanges = [
        "My name is Paulo and I'm from Seattle",
        "I work as an AI engineer building RAG systems",
        "I have 2 cats named Luna and Milo",
        "I'm building a LangChain course for Udemy",
        "What do you know about me? List everything.",
    ]

    print(f"\nConfig: keep last {MAX_RECENT} messages, summarize the rest\n")

    for user_input in exchanges:
        print(f"User: {user_input}")

        # 1. Call the LLM with summary + recent messages + new input
        response = chat_chain.invoke(
            {
                "summary": (
                    running_summary if running_summary else "No prior conversation."
                ),
                "recent_messages": recent_messages,
                "input": user_input,
            }
        )
        print(f"AI: {response}")

        # 2. Add this exchange to recent messages
        recent_messages.append(HumanMessage(content=user_input))
        recent_messages.append(AIMessage(content=response))

        # 3. If recent messages exceed limit, summarize the oldest ones
        if len(recent_messages) > MAX_RECENT:
            # Take the oldest messages that will be summarized away
            messages_to_summarize = recent_messages[:-MAX_RECENT]
            formatted = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in messages_to_summarize
            )

            # Update the running summary
            running_summary = summarize_chain.invoke(
                {
                    "current_summary": (
                        running_summary if running_summary else "None yet."
                    ),
                    "new_messages": formatted,
                }
            )

            # Keep only the most recent messages
            recent_messages = recent_messages[-MAX_RECENT:]

            print(
                f"  >>> Summarized! Compressed {len(messages_to_summarize)} old messages"
            )
            print(f"  >>> Summary: {running_summary}")
            print(f"  >>> Recent buffer: {len(recent_messages)} messages")
        print()

    # --- Final state ---
    print("=" * 60)
    print("FINAL MEMORY STATE")
    print("=" * 60)
    print(f"\nRunning summary (compressed old context):\n  {running_summary}")
    print(f"\nRecent messages kept verbatim ({len(recent_messages)}):")
    for msg in recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}")
    print("\nKey insight: ALL facts preserved (name, city, job, cats, course)")
    print("But token cost stays bounded -- old messages are compressed, not deleted!")


def exercise_persistent_memory():
    """Exercise: implement a persistent memory store using SQLite."""

    db_path = "./chat_history.db"

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SQLChatMessageHistory(session_id=session_id, connection=f"sqlite:///{db_path}")


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be concise"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "persistent_test"}}

    test_messages = [
        "Hi, my name is Alex.",
        "I enjoy hiking and photography.",
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

    if os.path.exists(db_path):
        os.remove(db_path)
    

def persistent_memory_demo():
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    import sqlite3
    import os

    db_path = "./chat_history.db"
    connection_string = f"sqlite:///{db_path}"
    session_id = "persistent_user"
    if os.path.exists(db_path):
        os.remove(db_path)


    def build_chain():
        llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            return SQLChatMessageHistory( session_id=session_id, connection=connection_string)

        prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a helpful assistant. Remember user preferences and facts.",
                        ),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}"),
                    ]
                )

        chain = prompt | llm | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )


    config = {"configurable": {"session_id": session_id}}

    chain_v1 = build_chain()

    run1_messages = [
        "Hi, my name is Alex.",
        "I enjoy hiking and photography.",
    ]

    for msg in run1_messages:
        response = chain_v1.invoke({"input": msg}, config=config)
        print(f"User: {msg}")
        print(f"AI: {response}\n")

    del chain_v1  # Simulate ending the session

    chain_v2 = build_chain()
    
    recall_questions = [
            "What's my name?",
            "What theme do I prefer?",
            "What programming language do I prefer?",
            "How do I like my responses?",
        ]
    
    for msg in recall_questions:
            print(f"User: {msg}")
            response = chain_v2.invoke({"input": msg}, config=config)
            print(f"AI:   {response}\n")
    
    del chain_v2
    
    # =====================================================
    # FINAL: Show total messages accumulated
    # =====================================================
    print("--- FINAL DATABASE STATE ---\n")
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM message_store")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"Total messages in DB after both runs: {count}")
    print("Key insight: The second chain had ZERO in-memory history.")
    print("Everything was loaded from SQLite -- true persistence!")
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)





def main():
    # basic_memory()
    # multiple_sessions()
    # message_trimming()
    # windowed_memory()
    # summary_memory()
    # exercise_persistent_memory()
    persistent_memory_demo()


if __name__ == "__main__":
    main()