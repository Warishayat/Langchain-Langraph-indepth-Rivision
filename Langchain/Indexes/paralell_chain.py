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
from langchain.schema.runnable import RunnablePassthrough,RunnableParallel
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

final_template = PromptTemplate(
    template = """
    The following are answers from two different models.

    Gemini Answer:
    {gemini_chain_response}

    Groq Answer:
    {grok_chain_response}

    Please merge them into one concise, best possible answer for the student query.
    """,
   input_variables=["gemini_chain_response","grok_chain_response"]
)


parser = StrOutputParser()

paralell_chain = RunnableParallel(
    {
    "gemini_chain_response" : {"context":retriver,"query": RunnablePassthrough()}| prompt | gemini| parser,
    "grok_chain_response" : {"context":retriver,"query":RunnablePassthrough()} | prompt | groq | parser
    }
)


merge_chain = final_template | gemini | parser


final_chain = paralell_chain | merge_chain
res=final_chain.invoke("who is machine learning teacher?")
print(res)

# res=final_chain.invoke("who teach machine learning?")
# print("the final answer is:")
# print(res)