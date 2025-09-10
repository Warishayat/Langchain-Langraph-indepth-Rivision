import os 
from dotenv import load_dotenv
from langchain_community.llms import google_palm

load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# These models are kind of obselete you will found
# at the langchain webiste but theys pecificalyy mentioned
# used chatmodel