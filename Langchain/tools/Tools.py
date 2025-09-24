from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun,ShellTool
from langchain_core.tools import tool,StructuredTool
from pydantic import BaseModel,Field

load_dotenv()

#this lecture is all about tools ,there are two types of tools 
# 1: builtin tools -> like wikipedia etc.
# 2: Custom tools -> custom according to usecase.

HF_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#builtin tools
Google_search = DuckDuckGoSearchRun() 
terminal_tool = ShellTool()
Gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
Gpt_model = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)


# result = terminal_tool.invoke("whoami")
# resut1 = Google_search.invoke("what is the current news of weather?")


# how to make custom tool in langchain.
@tool
def multiply(a:int,b:int):
    "this function will take two number and return their multiplication"
    return a*b

# response=multiply.invoke({"a": 12, "b": 13})
# print(response)



# Using StructuredTool and Pydantic.

class Multiplication(BaseModel):
    a: int = Field(..., description="The first number")
    b: int = Field(..., description="The second number")
    c:int  = Field(..., description="The third number") 

def multiply(a: int, b: int,c:int) -> int:
    return a * b * c
structured_tool = StructuredTool.from_function(
    func=multiply,
    name="multiplication",
    description="Multiplication of two numbers",
    args_schema=Multiplication
)
# response = structured_tool.invoke({"a": 12, "b": 13,"c":1})
# print(response)


@tool
def mul(a:int,b:int)->int:
    "this function will take two arguments and return their products"
    return a*b

@tool
def add(a:int,b:int)->int:
    "this function will take two arguments and return their addition"
    return a+b


class MathToolkit:
    def get_tools(self):
        return [mul,add]

toolkit  = MathToolkit()
tools = toolkit.get_tools()
for tool in tools:
    print(f"{tool.name}====> {tool.description}")