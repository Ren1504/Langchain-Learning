from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate , FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser ,JsonOutputParser, PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage , SystemMessage
from pydantic import BaseModel , Field
load_dotenv()


def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer in one sentence: {question}")    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [{"question":"Who is the prime minister of India"},{"question":"Who won game of the year in 2020?"}]

    # result = chain.invoke({"question":"Who is the chief minister of TamilNadu> "}) #single question
    results = chain.batch(inputs)
    
    for text in zip(inputs,results):
        print(f"Input: {text[0]['question']} => Output: {text[1]}")
    # return results

def demo_streaming():
    prompt = ChatPromptTemplate.from_template("Write a haiku about {topic}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    print("Streaming Output")

    for chunk in chain.stream({"topic":"nature"}):
        print(chunk,end="",flush=True)
    print()
                                     
def demo_schema_insepection():

    prompt = ChatPromptTemplate.from_template("Write a haiku about {topic}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input:{input_schema},output:{output_schema}")

def product_chain():
    prompt = ChatPromptTemplate.from_template("Write a product tagline for this product: {product}")
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0.7)
    parser = StrOutputParser()

    product = "Game boy"

    chain = prompt | model | parser

    result = chain.invoke(product)

    print(result)

def new_way():
    model = init_chat_model("gemini-2.5-flash",temperature = 0.7,max_tokens = 1500)

def demo_chat_model():
    chat_model = init_chat_model(
        model='gemini-2.5-flash',
        model_provider= 'google_genai',
        temperature = 0.7,
        streaming = True,
        max_retries = 3
    )

    response = chat_model.invoke("What is the capital of France")
    print(f"Response:{response.content}")

    return chat_model

def demo_message():
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature = 0.7)

    messages = [
        SystemMessage(content="Answer every questions in Aussie accent"),
        HumanMessage(content="How was the match yesterday" )
    ]

    response = model.invoke(messages)
    print(f"Response: {response.content}")
    messages.append(response)
    messages.append(HumanMessage("What did I ask you before?"))
    response = model.invoke(messages)

    print(f"followup resposne: {response.content}")

def exercise_multi_model():
    models = ["gemini-2.5-flash","gemini-3.6-flash"]

    for model_name in models:
        chat_model = init_chat_model(
        model= model_name,
        model_provider= 'google_genai',
        temperature = 0.7,
        streaming = True,
        max_retries = 3
        )

        response = chat_model.invoke("Write a sentence about pyramids")

        print(f"{model_name} : {response.content}")

#___________________________________________________________________________________________________

def prompt_messages():
    prompt = ChatPromptTemplate.from_template("Recommend me a game of {genre} about {topic}")
    messages = prompt.format_messages(genre = "Adventure",topic = "Monsters")
    print(messages)

def multi_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a translator that translates {input} to {output}"
            ),
            ("human","Translate this {text}")
        ]
    )

    messages = prompt.format_messages(
        input = "English",
        output = "Tamil",
        text = "I am going to the market"
    )

    model = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0.7
    )

    # response = model.invoke(messages)
    # print(response.content)
    print("*"*10)
    print(messages)

def fewShot():

    examples = [
        {
            "input":"happy","output":"sad"
        },
        {
            "input":"big","output":"small"
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human","{input}"),
            ("ai","{output}")
        ]
    )

    prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples = examples
    )

    model = init_chat_model(model = 'gemini-2.5-flash', model_provider= "google_genai",temperature = 0.7)
    response = model.invoke(prompt.format_messages(input="happy"))

    print(response.content)

def outputParsers():

    class Person(BaseModel):
        name: str = Field(description = "Name of the person")
        age: int = Field(description="Age of the person")
        occupation: str = Field(description="Job of the person")

    class MovieReview(BaseModel):
        title: str = Field("name of the movie")
        review: str = Field("review of the movie")
        rating: float = Field("rating of the movie out of 10")


    
    # parser = JsonOutputParser()
    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template("Return a JSON object with 'name' , 'age' and 'occupation' for {description}").partial(format_instructions=parser.get_format_instructions())

    model = init_chat_model(model="gemini-2.5-flash",model_provider = "google_genai",temperature = 0)

    structured_model = model.with_structured_output(MovieReview)

    chain = prompt | model | parser

    # response = chain.invoke({"description":"Marie is a 22 year old who is working as a Marketing intern in Microsoft"})
    # print(type(response))
    # print(response)

    result = structured_model.invoke("Review: odyssey is a great film by Nolan , would give a solid 9.5/10")
    print(result)

def main():
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # response = llm.invoke("Say 'Setup Complete!' in one word")
    # print(response)

    # demo_basic_chain()
    # demo_streaming()
    # demo_schema_insepection()
    # product_chain()
    # demo_chat_model()
    # demo_message()
    # exercise_multi_model()
    # prompt_messages()
    # multi_prompt()
    # fewShot()

    outputParsers()

if __name__ == "__main__":
    main()
