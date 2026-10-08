# 01 - LangChain Core Concepts

This folder covers the essential fundamentals of LangChain and the LangChain Expression Language (LCEL).

## Learning Roadmap

| File | Key Concepts Covered |
| :--- | :--- |
| [`01_basics_and_parsers.py`](file:///d:/Flutter/LangChain-Learning/01_core_concepts/01_basics_and_parsers.py) | • Model initialization with `init_chat_model` and `ChatGoogleGenerativeAI`<br>• Basic LCEL pipe syntax (`prompt \| model \| parser`)<br>• Execution strategies: `invoke()`, `batch()`, and streaming with `stream()`<br>• Message types: `SystemMessage`, `HumanMessage`, `AIMessage`<br>• Prompt templating with `ChatPromptTemplate` and `FewShotChatMessagePromptTemplate`<br>• Schema inspection & Structured Output (`PydanticOutputParser`, `.with_structured_output()`) |
| [`02_runnable_patterns.py`](file:///d:/Flutter/LangChain-Learning/01_core_concepts/02_runnable_patterns.py) | • `RunnableSequence`: Linear chaining<br>• `RunnableParallel`: Concurrent multi-task execution<br>• `RunnablePassthrough`: Passing data untouched through the pipeline<br>• `RunnableLambda`: Transforming custom Python functions into runnables<br>• `RunnableBranch`: Conditional routing and state assignment (`.assign()`) |

## Quick Start

1. Ensure your `.env` file is populated:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```
2. Run any script directly:
   ```bash
   python 01_core_concepts/01_basics_and_parsers.py
   python 01_core_concepts/02_runnable_patterns.py
   ```
