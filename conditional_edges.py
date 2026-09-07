from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import operator
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gpt-4o-mini", temperature=0.7)

class RouterState(TypedDict):
    query:str
    query_type:str
    response:str

class QualityState(TypedDict):
    content:str
    quality_score:float
    feedback:str
    final_content:str
    iteration:int

class TaskState(TypedDict):
    task:str
    urgency:str
    complexity:str
    handler:str
    result:str

def multipath_routing():
    def analyze_task(state:TaskState) -> dict:
        urgency_response = llm.invoke(
            f"Is this task urgent? {state['task']} Reply with only 'urgent' or 'normal'"
        )

        complexity_response = llm.invoke(
            f"Is this task complex? {state['task']} Reply with only 'complex' or 'simple'"
        )

        return{
            "urgency":urgency_response.content.lower().strip(),
            "complexity":complexity_response.content.lower().strip()
        }

    def urgent_complex_handler(state:TaskState) ->dict:
        return{
            "handler":"Senior Team",
            "result":"Escalated to senior team for action"
        }

    def urgent_simple_handler(state:TaskState) -> dict:
        return{
            "handler":"Quick Response",
            "result":"Handled immediatley by available agent"
        }

    def normal_complex_handler(state:TaskState) -> dict:
            return{
            "handler":"Specialist",
            "result":"Assigned to a Specialist"
        }

    def normal_simple_handler(state:TaskState) -> dict:
            return{
            "handler":"Standard",
            "result":"Interns are working"
        }

    def route_task(state:TaskState) -> str:
        is_urgent = "urgent" == state["urgency"]
        is_complex = "complex" == state["complexity"]

        if is_urgent and is_complex:
            return "urgent_complex"
        elif is_urgent:
            return "urgent_simple"
        elif is_complex:
            return "normal_complex"
        else:
            return "normal_simple"

    graph = StateGraph(TaskState)

    graph.add_node("analyze",analyze_task)
    graph.add_node("urgent_simple",urgent_simple_handler)
    graph.add_node("urgent_complex",urgent_complex_handler)
    graph.add_node("normal_simple",normal_simple_handler)
    graph.add_node("normal_complex",normal_complex_handler)
    graph.add_node("route_task",route_task)

    graph.add_edge(START,"analyze")
    graph.add_conditional_edges("analyze",route_task,
                                {
                                    "urgent_simple":"urgent_simple",
                                    "urgent_complex":"urgent_complex",
                                    "normal_simple":"normal_simple",
                                    "normal_complex":"normal_complex"
                                })

    for node in ["normal_complex","urgent_complex","normal_simple","urgent_simple"]:
        graph.add_edge(node,END)

    app = graph.compile()

    tasks =  [
        "Server is down, need immediate assistance",
        "Need to change button on line 2321",
        "Migrate the database from MongoDB to PostgreSQL",
        "Change the line 30 in the documentation for the application"
    ]

    for task in tasks:
        result = app.invoke({"task":task})
        print(f"Task:{result['task']}")
        print(f"Urgency:{result['urgency']}")
        print(f"Handler:{result['handler']}")
        print(f"Result:{result['result']}")
        print("--"*30)


def conditional_loop():

    def evaluate_quality(state:QualityState) -> dict:
        response = llm.invoke(
            f"Rate this content quality from 1-10. Reply with just the number\n"
            f"Content:{state['content']}"
        )

        try:
            score = int(response.content.strip())
        except:
            score = 5

        return {"quality_score":score}

    def improve_content(state:QualityState) -> dict:
        response = llm.invoke(
            f"Improve this content to be more engaging and clear:\n\n {state['content']}"
        )

        return{
            "content": response.content,
            "iteration": state['iteration']+1
        }

    def finalize_content(state:QualityState) -> dict:
        return{
            "final_content":state['content'],
            "feedback":f"Approved afer {state['iteration']} iterations with score of {state['quality_score']}"}


    def should_continue(state:QualityState) -> Literal["improve","finalize"]:
        if state["quality_score"] >= 11:
            return "finalize"
        elif state["iteration"] >=3:
            return "finalize"
        else:
            return "improve"

    graph = StateGraph(QualityState)

    graph.add_node("evaluate",evaluate_quality)
    graph.add_node("improve",improve_content)
    graph.add_node("finalize",finalize_content)

    graph.add_edge(START,"evaluate")
    graph.add_conditional_edges("evaluate",should_continue,
                                {"improve":"improve","finalize":"finalize"}
                                )
    graph.add_edge("improve","evaluate")
    graph.add_edge("finalize",END)

    app = graph.compile()

    print("Conditional_looping example")

    result = app.invoke({
        "content":"We are",
        "quality_score":0,
        "feedback":"",
        "final_content":"",
        "iteration":0
    })

    print("\nFinal Output")
    print(f"content: {result['content']}")
    print(f"quality_score: {result['quality_score']}")
    print(f"feedback: {result['feedback']}")
    print(f"final_content: {result['final_content']}")
    print(f"iterations: {result['iteration']}")

    visualize_graph(app.get_graph())


    return app
    
    
def visualize_graph(graph:StateGraph):
    # Visualize the graph using Mermaid syntax
    mermaid_graph = graph.draw_mermaid()
    print("Mermaid Graph visualization:")
    print(mermaid_graph)

    # Save the graph as a PNG file
    png_bytes = graph.draw_mermaid_png()
    with open("multipath_looping_graph.png", "wb") as f:
        f.write(png_bytes)
    print("Graph saved as conditional_graph.png")

def basic_routing():
    
    def classify_query(state:RouterState) -> dict:
        response = llm.invoke(
            f"Classify this query as 'question' or 'statement' or 'command'. Reply with only one word: {state['query']}.")

        return {"query_type": response.content.lower().strip()}

    def handle_question(state:RouterState) -> dict:
        response = llm.invoke(
            f"Answer this question: {state['query']}. Reply with a concise answer.")

        return {"response": f"Question: {state['query']}\nAnswer: {response.content}"}

    def handle_statement(state:RouterState) -> dict:
        response = llm.invoke(
            f"Provide a brief comment on this statement: {state['query']}. Reply with a concise comment.")

        return {"response": f"Statement: {state['query']}\nComment: {response.content}"}

    def handle_command(state:RouterState) -> dict:
        response = llm.invoke(
            f"Provide a brief response to this command: {state['query']}. Reply with a concise response.")

        return {"response": f"Command: {state['query']}\nResponse: {response.content}"}


    def route_by_query_type(state:RouterState) -> Literal["question", "statement", "command"]:
        return state["query_type"]

    graph = StateGraph(RouterState)

    graph.add_node("classify_query", classify_query)
    graph.add_node("handle_question", handle_question)
    graph.add_node("handle_statement", handle_statement)
    graph.add_node("handle_command", handle_command)

    graph.add_edge(START,"classify_query")
    graph.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {
            "question": "handle_question",
            "statement": "handle_statement",
            "command": "handle_command"
        }
    )

    graph.add_edge("handle_question", END)
    graph.add_edge("handle_statement", END)
    graph.add_edge("handle_command", END)






def main():
    # basic_routing()

    # app = basic_routing()
    # questions = [
    #     "What is the capital of France?",
    #     "I love programming in Python.",
    #     "Please summarize the latest news."
    # ]

    # for question in questions:
    #     result = app.invoke({"query": question})
    #     print(f"Input: {question}")
    #     print(f"Query Type: {result['query_type']}")
    #     print(f"Response: {result['response']}")
    #     print("-" * 50)

    # visualize_graph(app.get_graph())

    # conditional_loop()

    multipath_routing()

if __name__ == "__main__":
    main()