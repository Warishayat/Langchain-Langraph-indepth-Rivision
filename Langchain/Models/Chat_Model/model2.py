import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# temprature 
# if you want the same answer for same question then 
# temp should be zero and if you want the diffrent answer
# for the same question you temprature should be more then
# 0.7

model = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7,api_key=GROQ_API_KEY)
result = model.invoke("Tell me something funny about AI and machine learning?")
print(result.content)