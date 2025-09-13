from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
import warnings
warnings.filterwarnings('ignore')
import os





load_dotenv()
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embedding_mod = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat_model = ChatGoogleGenerativeAI(model="Gemini-2.5-flash",api_key=GOOGLE_API_KEY)

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

search = vectorStore.similarity_search(query="what is the file structure?",k=2)

print("search",search)
