from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from classes import MessageState
import requests
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()

groq_api_key = os.getenv("groq_api_key")
Weather_api_ey = os.getenv('Weather_api_key')
Model=ChatGroq(model='openai/gpt-oss-20b',api_key=groq_api_key)




@tool
def city_weather(city_name: str):
    """To find the weather of spevific you need to call this and return the weather of particualr city weather."""
    try:
        response = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key={Weather_api_ey}&q={city_name}&aqi=no"
        )
        if response.status_code == 200:
            data = response.json()
            location = data["location"]["name"]
            temp_c = data["current"]["temp_c"]
            condition = data["current"]["condition"]["text"]
            return f"The weather in {location} is {condition} with {temp_c}°C."
        else:
            return f"Error {response.status_code}: {response.content}"
    except Exception as e:
        return str(e)


google_search = DuckDuckGoSearchRun()

list_tools = [google_search,city_weather]
