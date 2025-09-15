import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from project import build_vectorstore
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
groq = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7,api_key=GROQ_API_KEY)
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vector_store = build_vectorstore(pdf_path="university.pdf",chunk_overlap=18,chunk_size=180,embedding_model=embedding_model)
retriver = vector_store.as_retriever(kwargs={"k":3})


prompt = PromptTemplate(
    template = """
        You are a helpful student assistant.

        Answer the student's question **only** from the given context.
        Do NOT copy the full context. Just extract the relevant fact.

        Question:
        {query}

        Context:
        {context}
        Answer in 1–2 sentences, concise and clear.
        If not found in context, say: "Not available in the university data."
        """,
    input_variables = ["query","context"]
)


parser = StrOutputParser()

rag_chain = {"context":retriver,"query":RunnablePassthrough()} | prompt | groq | parser

query = "Does the university offer Quantum Physics?"
result = rag_chain.invoke(query)
print("final_result")
print()
print(result)
