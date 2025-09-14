#lets make some advance search. mmr 
# mmr jab search krta h tu wo try karta to bring something diffrent from each other instead
# of getting the same result.

# simple retriver kabhi kbhar same data de detay hn
# jiski wjah se redundancy barh jati hae.
# aur time aur cost ka waste hojta hae.

# isi problem ko solve karta hae mmr 
# yea krta kia h similar docuemnt uthata h query se
# but wo document ik dusry se diffrent huty haen.
# 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores import FAISS
import warnings
warnings.filterwarnings('ignore')
import os



#mmr retriver
load_dotenv()
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embedding_mod = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chat_model = ChatGoogleGenerativeAI(model="Gemini-2.5-flash",api_key=GOOGLE_API_KEY)


loader = PyPDFLoader("purposal.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
docs = splitter.split_documents(docs)

retriver = FAISS.from_documents(
    embedding=embedding_mod,
    documents=docs
)
search = retriver.as_retriever(
    kwargs={"k":3},
    search_type="mmr",

)

query = "about page"
res=search.invoke(query)
print(res[0].page_content)
