#output parser help us to get the structure response from the llm
# while there are some model which provide us the flexibility of
# structured_output response in built.while for that model which are
# not providng the response in structure formate for that we will
# use output-parser although we can help with both types of models


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
# structured output parser
# paydentic output-parser


# string is simple output parser it take response from llm and convery
# it into string ,we got alot of meta data from the llm ,then we  print res.content
# to filter the data by this parser we can directly use this parser

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=1,
    max_new_tokens=6,
    task="text-generation"
)

template1 = PromptTemplate(
    template = "write a detail report on the {topic}",

    input_variables=["topic"]

)

template2 = PromptTemplate(
    template = "write a 5 lines summary of the given {text}",
    input_variables=["text"]
)

res=template1.invoke({"topic":"quantum physics"})


model = ChatHuggingFace(llm=LLM)
result=model.invoke(res)

res2 = template2.invoke({"text":result.content})

result2 = model.invoke(res2)

print(result2.content)