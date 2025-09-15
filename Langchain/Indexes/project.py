import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


groq = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7,api_key=GROQ_API_KEY)
gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")



def build_vectorstore(pdf_path: str, embedding_model, chunk_size: int = 180, chunk_overlap: int = 18):
    """
    Loads a PDF, splits it into chunks, creates embeddings, 
    and stores them in a FAISS vectorstore.

    Args:
        pdf_path (str): Path to the PDF file.
        embedding_model: Hugging Face or OpenAI embedding model.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        FAISS: A vectorstore object containing document embeddings.
    """

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)

    # Store in FAISS
    vector_store = FAISS.from_documents(
        split_docs,
        embedding=embedding_model
    )

    return vector_store
