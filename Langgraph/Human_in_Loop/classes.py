from pydantic import BaseModel
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict



class MessageState(TypedDict):
    query: str
    genrated_post: Annotated[list[AIMessage], add_messages]
    human_feedback: Annotated[list[str], add_messages]