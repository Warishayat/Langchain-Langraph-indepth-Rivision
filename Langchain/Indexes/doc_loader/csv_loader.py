from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

loader = CSVLoader(r"Social_Network_Ads.csv")
docs = loader.load()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)


prompt = PromptTemplate(
    template="""
    You are a helpful assistant. Use the following context to answer the user's question.

    Context:
    {text}

    Question:
    {query}

    Answer in a clear and concise manner.
    """,
    input_variables=["query", "text"]
)
parser = StrOutputParser()

chain = prompt | model | parser

res = chain.invoke({
    "query": "What is the age of this person",
    "text": docs[0].page_content
})
print(res)
print('Actual Response')
print(docs[0].page_content)
print(docs[0].metadata)
