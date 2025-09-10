from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
messages = [
        SystemMessage(content="You are a helpful funny assistant Chatbot.")
    ]
while True:
    query = input("user: ").lower()
    messages.append(HumanMessage(content=query))
    if query == 'exit':
        break
    res=model.invoke(messages)
    print("AI: ",res.content)
    messages.append(AIMessage(content=res.content))

print(messages)