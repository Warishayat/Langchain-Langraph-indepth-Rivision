import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,Literal,Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


# string output parser
# json output parser
# paydentic output-parser


LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=1,
    max_new_tokens=6,
    task="text-generation"
)

model = ChatHuggingFace(llm=LLM)

model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
str_parser = StrOutputParser()

template1 = PromptTemplate(
    template = "write a detail report on the {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "write a 5 lines summary of the given {text}",
    input_variables=["text"]
)

chain = template1 | model2 | str_parser | template2 | model2 | str_parser

response = chain.invoke("neural network")

print(response)