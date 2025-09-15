
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

#it is kind of exoerimenting

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

embedding_mod = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = SemanticChunker(
    embedding_mod,breakpoint_type="standard_deviation",
    breakpoint_threshold_amount=1
)


print("all GOOD")


#sematic meaning based --->in which we are not deciding the based on text
# or words weather we split based on semantic meaning
# they are using sliding widow approach where the similarity is less between
# the sentense mean there are some diffrent sentenses