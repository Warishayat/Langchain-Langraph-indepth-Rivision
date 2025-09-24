from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
from langchain_core.tools import tool
from typing import Annotated
import requests
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()


HF_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EC_API_KEY = os.getenv("EC_API_KEY")


Google_search = DuckDuckGoSearchRun() 
terminal_tool = ShellTool()
Gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Gpt_model = ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY)


# tool 1: conversion factor fetcher
@tool
def currency_conversion(base_currency: str, target_currency: str) -> float:
    """Fetch the currency conversion factor between base and target currency."""
    url = f"https://v6.exchangerate-api.com/v6/{EC_API_KEY}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"API error: {response.status_code} - {response.text}"
    try:
        data = response.json()
    except Exception:
        return f"Error: Could not parse JSON. Response was: {response.text}"

    return data.get("conversion_rate", "Error: conversion_rate not found")


# tool 2: apply conversion
@tool
def convert(base_currency_value: int, conversion_rate: float) -> float:
    """Given a currency conversion rate this function will calculate the target currency from the given base value"""
    return base_currency_value * conversion_rate



model = Gemini_model.bind_tools([currency_conversion, convert])


messages = []
query = HumanMessage(content="What is the conversion factor between USD and PKR, and can you convert 10 USD to PKR?")
messages.append(query)


while True:
    ai_message = model.invoke(messages)
    messages.append(ai_message)

    if not getattr(ai_message, "tool_calls", None):
        print("=== FINAL ANSWER ===")
        print(ai_message.content)
        break

    for call in ai_message.tool_calls:
        if call["name"] == "currency_conversion":
            result = currency_conversion.invoke(call["args"])
        elif call["name"] == "convert":
            result = convert.invoke(call["args"])
        else:
            result = f"Unknown tool: {call['name']}"

        messages.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )

print()
print(messages)
