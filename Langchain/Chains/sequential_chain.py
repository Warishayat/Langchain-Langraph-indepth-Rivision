import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# This is all about chains and why its is important.
# Application is always divided into steps.
# there ares oem steps
# # 1: GENRATE PROMPT FOR USER
# # 2: LLM - > PASS PROMPT TO THW LLMS
# # 3: PARSER - > TO PARSE THE RESPONSE.

## to connect all these steps and make pipelines from chains.
## its work as a linkedlist output of one node is work as the
## input of the next node.

## by suing chain we can creat differnt diffrent workflows.

## SEQUENTIAL CHAIN
## PARALELL CHAIN
## CONDITION HAINS

model = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7,api_key=GROQ_API_KEY)
# lets start with the simple chain
template = PromptTemplate(
    template = "genrate 5 intresting fact about {topic}",

    input_variables=["topic"]
)

final_prompt = template.invoke({"topic":"quantum physics"})

parser = StrOutputParser()



chain = template | model | parser     #its called langchain_expression_language

result = chain.invoke(final_prompt)



#lets get example 2 and it is also sequential chain
# user will pass topic
# sent it to llm with detail report 
# detail report from llm
# sent again to llm with with prompt get the important pointers

template_00 = PromptTemplate(
    template="genrate a detail report on this {topic}",
    input_variables=["topic"]
)

template_01 = PromptTemplate(
    template="genrate a 5 important fact from this {report}",
    input_variables=["report"]
)

chain_02 = template_00 | model | parser | template_01 | model | parser

print("Graph of this workflow")
chain_02.get_graph().print_ascii()
print()
print()
temp_var = template_00.invoke({"topic":"black hole"})
result = chain_02.invoke(temp_var)
print("The result of the example 02")
print()
print()
print(result)