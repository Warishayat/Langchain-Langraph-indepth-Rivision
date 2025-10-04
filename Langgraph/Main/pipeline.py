import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Optional,Literal,Annotated
from typing import TypedDict
import langgraph
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()

conn = sqlite3.connect(database='cahtbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


Model = ChatGoogleGenerativeAI(model='gemini-2.5-flash',api_key=GOOGLE_API_KEY,temperature=.5)

# checkpointer = MemorySaver()

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chatbot(state:ChatState):
    response = Model.invoke(state['messages']).content
    message = [AIMessage(content=response)]
    return{'messages':message}

graph = StateGraph(ChatState)
graph.add_node("chatbot",chatbot)
graph.add_edge(START,'chatbot')
graph.add_edge('chatbot',END)

workflow = graph.compile(checkpointer=checkpointer)

def get_all_threads():
    all_threads = set()
    for cp in checkpointer.list(None):
        all_threads.add(cp.config['configurable']['thread_id'])
    return list(all_threads)