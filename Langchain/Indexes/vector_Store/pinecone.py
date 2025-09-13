from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os


load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_mod = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#pinecone_store_name
pinecone = Pinecone(api_key=pinecone_api_key)
index_name = "pinecone-store"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

#load pdf
loader = PyPDFLoader("purposal.pdf")
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = splitter.split_documents(docs)


vector_store = PineconeVectorStore.from_documents(
    docs,
    embedding_mod,
    index_name=index_name
)


results = vector_store.similarity_search(
    "file structure",
    k=2,
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")