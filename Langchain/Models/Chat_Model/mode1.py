import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# temprature 
# if you want the same answer for same question then 
# temp should be zero and if you want the diffrent answer
# for the same question you temprature should be more then
# 0.7

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY,temperature=0)
res = model.invoke("what is the capital of paksitan?")
print(res.content)