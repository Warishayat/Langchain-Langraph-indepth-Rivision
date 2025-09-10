from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)


st.title("Research Assistant Tool")
# this is the static prompt that we are getting from the user.
prompt=st.text_input("Enter the topic you want to research about")
if st.button("submit"):
    res=model.invoke(prompt)
    st.write(res.content)