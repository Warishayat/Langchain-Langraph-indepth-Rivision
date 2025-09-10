from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# instead of static prompt we can make template for prompt

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)

st.title("Research Assistant")
st.header("Dynamic prompt")
topic = st.selectbox("Select title", ["Research assistant"])
paper = st.selectbox("Select Paper", ["Attention is all you need", "word2vec"])

if st.button("submit"):
    prompt = load_prompt("prompt.json")
    final_prompt = prompt.invoke({"topic_input": topic, "paper_name": paper})
    res = model.invoke(final_prompt)
    st.write(res.content)  