import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
load_dotenv()


# there are two way to use hugging face model
# 1: is Uaing inference api
# 2: download and use it.

# but in our case our laptop doesnt have gpu so we will go with infernece and all 
# the thinf we look for in this video with api and infernce with that.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# temprature 
# if you want the same answer for same question then 
# temp should be zero and if you want the diffrent answer
# for the same question you temprature should be more then
# 0.7

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=1,
    max_new_tokens=6,
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
res = model.invoke("Tell me something funny about hospital?")
print(res.content)