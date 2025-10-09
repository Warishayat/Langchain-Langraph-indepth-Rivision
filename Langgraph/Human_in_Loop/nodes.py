from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from classes import MessageState
from langgraph.types import Interrupt,Command
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from tools import google_search,city_weather

load_dotenv()


groq_api_key = os.getenv("groq_api_key")

Model=ChatGroq(model='openai/gpt-oss-20b',api_key=groq_api_key)
bind_tools_model = Model.bind_tools([google_search,city_weather])



#post genrate node
def genrated_post(state: MessageState):
    print("[model] Generating content")
    feedback = state['human_feedback'][-1] if state['human_feedback'] else "No feedback provided."
    prompt = f"""
        You are a helpful LinkedIn post writer.

        Your task:
        - Write or revise a LinkedIn post based on the user's query and feedback.
        - If there is **no feedback**, create a post directly from the query.
        - If there **is feedback**, revise the post carefully using that feedback.

        Guidelines:
        - Keep the post concise, clear, and professional.
        - Match the tone requested by the user (e.g., funny, motivational, informative, etc.).
        - Avoid any sexual, political, or religiously offensive content.
        - If the query or feedback includes such sensitive topics, respond politely:
        "I'm sorry, but I can’t assist with that kind of content."

        Inputs:
        - User Query: {state['query']}
        - Human Feedback: {feedback}
    """
    msg = [
    SystemMessage(content='you are an expert LinkedIn post writer'),
    HumanMessage(content=prompt)
    ]
    response = Model.invoke(msg)
    genrated_linkedin_post = response.content
    return {
        'genrated_post': [AIMessage(content=genrated_linkedin_post)]
    }


def linkedin_publication(state:MessageState):
    'node to upload post on the linkedin'
    gen_post = state['genrated_post']
    print(f"Post has been successfully made on linkedin.")
    print(f'Post is: \n\n {gen_post[-1].content}')

    return{
        'final_message' :"post has been made successfully congratulation"
    }
