import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,Literal,Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
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


class Person(BaseModel):
    name:str = Field(description="Name of the person")
    age:int = Field(description="Age of the person")
    city :str =Field(description="The ncity of the person who person live")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "genrate the name,age and city of the fictional {place} person \n {formate_instruction}",
    input_variables=["person"],
    partial_variables ={"formate_instruction":parser.get_format_instructions()}
)

input = template.invoke({"place":"karachi"})

# chain = template | model2 | parser

# res=chain.invoke(input)
# print(res)
# print(type(res))

print(template)