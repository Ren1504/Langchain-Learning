

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph , START , END
from typing_extensions import Annotated , TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , AIMessage , BaseMessage
import operator
from langgraph.graph import add_messages
from dotenv import load_dotenv

load_dotenv()

class SimpleState(TypedDict):
    input: str
    output: str
    step: int

class AccumulatingState(TypedDict):
    messages: Annotated[list[str],operator.add]
    count: Annotated[int,operator.add]

class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

class MultiStepState(TypedDict):
    input:str
    step_one_output:str
    analyzed:str
    enhanced:str
    final:str

class ExerciseLangraph(TypedDict):
    input:str
    questions:list[str]
    answer:str

def exercise():
    llm = ChatOpenAI(model= "gpt-4o-mini", temperature=0)

    def questions_node(state:ExerciseLangraph) -> dict:
        response = llm.invoke([HumanMessage(content=f"Generate 3questions for the following topic: {state['input']}")])
        return {"questions": response.content.split("\n")}

    def answer_node(state:ExerciseLangraph) -> str:
        response = llm.invoke([HumanMessage(content=f"Answer any one of the following questions in a single sentence: {state['questions']}")])
        return {"answer": response.content}

    graph = StateGraph(ExerciseLangraph)
    graph.add_node("questions_node", questions_node)
    graph.add_node("answer_node", answer_node)
    graph.add_edge(START, "questions_node")
    graph.add_edge("questions_node", "answer_node")
    graph.add_edge("answer_node", END)

    app = graph.compile()
    result = app.invoke({"input":"Python programming"})
    print("Exercise Graph Result: ")
    print(f"Input: {result['input']}")
    print(f"Questions: {result['questions']}")
    print(f"Answer: {result['answer']}")

    visualize_graph(app.get_graph())

def multi_node_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def analyze_node(state:MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Analyze the following input: {state['input']}")])
        return {"analyzed": response.content}

    def enhance_node(state:MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Enhance the following input: {state['analyzed']}")])
        return {"enhanced": response.content}

    def final_node(state:MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Finalize the following input: {state['enhanced']}")])
        return {"final": response.content}

    graph = StateGraph(MultiStepState)
    graph.add_node("analyze_node", analyze_node)
    graph.add_node("enhance_node", enhance_node)
    graph.add_node("final_node", final_node)
    graph.add_edge(START, "analyze_node")
    graph.add_edge("analyze_node", "enhance_node")
    graph.add_edge("enhance_node", "final_node")
    graph.add_edge("final_node", END)

    app = graph.compile()
    result = app.invoke({"input":"League of Legends"})

    print("Multi Node Graph Result: ")
    print(f"Input: {result['input']}")
    print(f"Analyzed: {result['analyzed']}")
    print(f"Enhanced: {result['enhanced']}")
    print(f"Final: {result['final']}")

    visualize_graph(app.get_graph())

def message_state():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def chat_node(state:MessageState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    graph = StateGraph(MessageState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    app = graph.compile()

    visualize_graph(app.get_graph())

    initial_messages = [HumanMessage(content="Hello! How are you?")]
    result = app.invoke({"messages": [HumanMessage(content="Can you say good morning in French?")]})
    print("Message State Result: ", result)


def visualize_graph(graph:StateGraph):
    # Visualize the graph using Mermaid syntax
    mermaid_graph = graph.draw_mermaid()
    print("Mermaid Graph visualization:")
    print(mermaid_graph)

    # Save the graph as a PNG file
    png_bytes = graph.draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Graph saved as graph.png")

def simple_graph():

    def process(state:SimpleState) -> dict:
        return {"output": state["input"].upper() , "step": state["step"] + 1}

    graph = StateGraph(SimpleState)

    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    #compiling graph
    app = graph.compile()

    # print("Mermaid Graph visualization")
    # print(app.get_graph().draw_mermaid())

    # png_byts = app.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_byts)
    # print("Graph saved as graph.png")

    result = app.invoke({"input": "hello world","output": "","step": 0})
    print("Simple Graph Result: ", result)
    print(f"Input: {result['input']}, Output: {result['output']}, Step: {result['step']}")

def accumulating_state():
    def step_one(state:AccumulatingState) -> dict:
        return {"messages": ["Step 1 completed"], "count": 1}

    def step_two(state:AccumulatingState) -> dict:
        return {"messages": ["Step 2 completed"], "count": 1}

    graph = StateGraph(AccumulatingState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)
    app = graph.compile()

    visualize_graph(app.get_graph())

    result = app.invoke({"messages": ["initial message"], "count": 0})
    print("Accumulating State Result: ", result)


def main():
    # simple_graph()
    # message_state()
    # accumulating_state()
    # multi_node_graph()
    exercise()
if __name__ == "__main__":
    main()