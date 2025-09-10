from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
import os


# message placeholder is nothing but a list of message to provide the llm while
# inferenace so the llm know what exactly was the preivous chat
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)

history=[]
chat_template = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful customer expert"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{query}")
])

# load chat history
with open("history.txt") as f:
    history.extend(f.readlines())

prompt = chat_template.invoke({'history':history,'query':"When i will get my refund please"})
res=model.invoke(prompt)
print(res.content)