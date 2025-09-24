from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun,ShellTool
from langchain_core.tools import tool,StructuredTool
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage

load_dotenv()

#this lecture is all about tools_calling ,there are two types of tools 

HF_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#builtin tools
Google_search = DuckDuckGoSearchRun() 
terminal_tool = ShellTool()
Gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
Gpt_model = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)



#how tools calling work???
# tools are nothing but function but these are special function they can intract it with the llms.
# how to connect these tools with llms?
# how llms call the tools?


@tool 
def multiply(a:int,b:int)->int:
    "given two number a and b this tool return their products"
    return a*b

#tool binding
model_with_tool=Gemini_model.bind_tools([multiply])

query = HumanMessage(content="can you multiply two with three?")
messages=[query]


#call the llms
response = model_with_tool.invoke(messages)
messages.append(response)


tool_result = multiply.invoke(response.tool_calls[0])
messages.append(tool_result)


final_result =  model_with_tool.invoke(messages)
print(final_result)
