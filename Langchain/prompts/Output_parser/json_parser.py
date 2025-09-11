import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,Literal,Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=1,
    max_new_tokens=6,
    task="text-generation"
)

model = ChatHuggingFace(llm=LLM)

model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template="gave me the age,name and city and dob of the fictional person\n{formate_instruction}",
    input_variables=[],
    partial_variables={"formate_instruction":parser.get_format_instructions()}
)

prompt = template1.format()

chain = template1 | model2 | parser
res=chain.invoke(prompt)
print(res)