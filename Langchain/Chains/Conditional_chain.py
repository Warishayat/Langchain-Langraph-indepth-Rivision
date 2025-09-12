import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda #lamda functio ko runnable m convert krta hae
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model_01 = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7,api_key=GROQ_API_KEY)
model_02 = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)

# Problem Statment
# user will provide the review
# we will ananlyze the sentiment of the review positve ,negative or neutral
# if positive --->thanks forkind words
# if negative ----> Sorry for inconvience we will connect csr and we will resolve it
# neutral ------> Happy to see you here again.

class Sentiment_analysis(BaseModel):
    sentiment : Literal["positive","negative"] = Field(description="classify the review into positive or negative sentiment only")

p_parser = PydanticOutputParser(pydantic_object=Sentiment_analysis)
parser = StrOutputParser()

#template to get the classify review
template = PromptTemplate(
    template = "Classify the user product review in the positive or negative \n{review}\n{formate_instruction}",
    input_variables=["review"],
    partial_variables={"formate_instruction":p_parser.get_format_instructions()}
)
classifier_chain = template | model_01 | p_parser
# res=classifier_chain.invoke(c_template)
# res_dic = dict(res)

prompt2 = PromptTemplate(
    template="write an approprivate reply to this  positive ->{positive} feedback",
    input_variables=["positive"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate reply to this negative {negative} feedback",
    input_variables=["negative"]
)
#now create the branch chain
branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model_02 | parser),
    (lambda x:x.sentiment=='negative',prompt3 | model_02 | parser),
    RunnableLambda(lambda  x: "Could not find setiment")
) 

final_chain = classifier_chain |  branch_chain

res=final_chain.invoke({"review":"THIS PHONE IS REALLY REALLY NICE."})
print(res)