import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Optional,Literal,Annotated
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage,AIMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


Model = ChatGoogleGenerativeAI(model='gemini-2.5-flash',api_key=GOOGLE_API_KEY,temperature=.5)

checkpointer = MemorySaver()

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

initial_state = {'messages':[HumanMessage(content="how to apply german study visa and which universistat are best for llm reserach?")]}

thread_id = '1'
config = {"configurable": {"thread_id": thread_id}}
response=workflow.invoke({'messages':[HumanMessage(content="hello how are you?")]},config=config)
print(workflow.get_state(config=config).values['messages'])
# while True:
#     message = input("User: ")
#     print("User:",message)

#     if message.strip().lower() in ['exit','quit','bye']:
#         break
    
#     response = workflow.invoke({'messages':[HumanMessage(content=message)]},config=config)
#     print("AI :",response['messages'][-1].content)