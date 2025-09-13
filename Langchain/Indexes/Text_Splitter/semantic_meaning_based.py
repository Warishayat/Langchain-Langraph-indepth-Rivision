import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# these text splitter are experimental


# Load environment variables
load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Embedding model
embedding_mod = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Semantic Chunker
text_splitter = SemanticChunker(
    embedding_mod,
    breakpoint_type="standard_deviation",
    breakpoint_threshold_amount=1
)

print("all GOOD")
