from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)


messges = [
    SystemMessage(content="you are a helpfull assistant"),
    HumanMessage(content="Tell me sonething funny about machine learning.")
]

result = model.invoke(messges)
messges.append(AIMessage(content=result.content))
print(messges)