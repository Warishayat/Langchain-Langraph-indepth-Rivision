from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.docstore.in_memory import InMemoryDocstore
import warnings
import faiss
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

embedding_dim = len(embedding_mod.embed_query("hello world"))

index = faiss.IndexFlatL2(embedding_dim)

Vector_Store = FAISS(
    embedding_function=embedding_mod,
    docstore = InMemoryDocstore(),
    index=index,
    index_to_docstore_id = {}
)

Vector_Store.add_documents(docs)

search = Vector_Store.similarity_search(query="about page",k=1)
print("search")
print(search[0].page_content)

