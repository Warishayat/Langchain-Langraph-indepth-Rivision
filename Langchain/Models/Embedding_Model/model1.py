import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

# in this notebook i will dicuss the embeddings model
# we have two type of embeddings model 
# 1: open source 
# 2: Close Source like gemini and openai embeddings model etc.

# but we will go with opensource

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# temprature 
# if you want the same answer for same question then 
# temp should be zero and if you want the diffrent answer
# for the same question you temprature should be more then
# 0.7

# i will use google letast embedding model "google/embeddinggemma-300m"

text = [
    "hello my name is waris hayat abbasi",
    "software enginner work at goolge",
    "dairy forms people are creative minds"
]

embedding_mod = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
res=embedding_mod.embed_documents(text)
print(res)