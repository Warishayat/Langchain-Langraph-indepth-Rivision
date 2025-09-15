# there are mutiple type of retriver
# 1: wikipedia artical
# 2: vector store
# 3: Archeive Retriver
# Alg retriver alg alg treeky se search krte haen.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import WikipediaRetriever
import warnings
warnings.filterwarnings('ignore')
import os



#1: Wikipedia retriver
load_dotenv()
retriver = WikipediaRetriever(top_k_results=3,lang='en')
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embedding_mod = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat_model = ChatGoogleGenerativeAI(model="Gemini-2.5-flash",api_key=GOOGLE_API_KEY)



# query = "history of england?"
# res=retriver.invoke(query)
# print(res)




#load pdf
loader = PyPDFLoader("purposal.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
docs = splitter.split_documents(docs)
print(docs[0])

vectorStore = Chroma(
    embedding_function = embedding_mod,
    collection_name = "pinecone_store",
    persist_directory="Chroma_db"
)

vectorStore.add_documents(docs)

search  =  vectorStore.as_retriever(kwargs={"k:2"})
response=search.invoke("file structure?")
print("Retriver")
print("Answer is:")
print(response)
