from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph , START , END
from typing_extensions import Annotated , TypedDict , Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , AIMessage , BaseMessage , SystemMessage
import operator
from langgraph.graph import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gpt-4o-mini",temperature = 0.0)

class CodeGenState(TypedDict):
    task:str
    code:str
    errors:Annotated[list[str],operator.add]
    iteration:int
    max_iterations:int
    success:bool


def self_correcting_code():

    def code_clean(code):
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code=code[6:]

        return code

    def generate_code(state:CodeGenState) -> dict:
        if state["iteration"] == 0:
            prompt = f"Write a python program for the given task: {state['task']}\nReturn only the code"

        else:
            prompt = f"Fix this Python code:\n{state['code']}\n\nErrors:\n{state['errors']},Return only the corrected code "

        response = llm.invoke(prompt)
        code = response.content.strip()

        code = code_clean(code)

        return {"code":code,"iteration":int(state["iteration"])+1}

    def validate_code(state:CodeGenState) -> dict:
        code = state["code"]

        try:
            compile(code,"<string>","exec")
            return {"success":True}

        except SyntaxError as e:
            return{
                "errors":[f"SyntaxError:{e}"],
                "success":False
            }

        except Exception as e:
            return{
                "errors":[f"Error:{e}"],
                "success":False
            }


    def should_continue(state:CodeGenState) -> Literal["generate","end"]:
        if state["success"]:
            return "end"

        elif state["max_iterations"] <= state['iteration']:
            return "end"

        else:
            return "generate"


    def finalize(state:CodeGenState) -> dict:
        return state

    graph = StateGraph(CodeGenState)

    graph.add_node("generate",generate_code)
    graph.add_node("validate",validate_code)
    graph.add_node("finalize",finalize)

    graph.add_edge(START,"generate")
    graph.add_edge("generate","validate")
    graph.add_conditional_edges("validate",should_continue,{
        "generate":"generate",
        "end":"finalize"
    })
    graph.add_edge("finalize",END)

    app = graph.compile()

    result = app.invoke({
        "task":"Write a function to find Armstrong number recursively",
        "code":"",
        "errors":[],
        "iteration":0,
        "max_iterations":3,
        "success":False
    })


    print(f"Task: {result['task']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Success:{result['success']}")
    print(f"Final code:\n{result['code']}")


def main():
    self_correcting_code()


if __name__ == "__main__":
    main()


