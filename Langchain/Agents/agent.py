from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
from langchain_core.tools import tool
from typing import Annotated
import requests
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import AgentExecutor,create_react_agent
from langchain import hub

load_dotenv()


HF_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EC_API_KEY = os.getenv("EC_API_KEY")
Weather_api_key = os.getenv("Weather_api_key")


Google_search = DuckDuckGoSearchRun() 
terminal_tool = ShellTool()
Gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Gpt_model = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY)




@tool
def weather(city_name):
    "this is function for getting the current weather based on city it will provide the complete info about the weather of that city"
    try:
        response = requests.get(f"http://api.weatherapi.com/v1/current.json?key={Weather_api_key}&q={city_name}&aqi=no")
        if response.status_code == 200:
            return response.content
        else:
            return response.status_code,response.content
    except Exception as e:
        return e

#what is ai agents which problem solved by ai agents
# pull is nothing but predefined prompt
# creat react agent is nothing but multistep process -> reasoning ,action with the help of tool.
# agent executor is the guy who is responsible to run the agent.

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=Gemini_model,
    tools=[Google_search,weather],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[Google_search,weather],
    verbose=True
)


result = agent_executor.invoke({"input":"what is the capital of pakistan and what is the current weather of chicago usa?"})
print(result)