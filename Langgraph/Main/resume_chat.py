import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from pipeline import workflow
import uuid
import warnings

warnings.filterwarnings('ignore')
load_dotenv()


def get_thread_id():
    return str(uuid.uuid4())


# *********************** Session State ***************************
if "conversations" not in st.session_state:
    st.session_state["conversations"] = {} 

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = get_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []


# ****************** functions ***********************************

def new_chat_window():
    thread_id = get_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["conversations"][thread_id] = []
    add_thread(thread_id)
    st.rerun()


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    if thread_id not in st.session_state["conversations"]:
        st.session_state["conversations"][thread_id] = []


def get_conversation(thread_id):
    state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    values = state.values
    return values.get("messages", [])


# *********************** Message History **************************
current_thread = st.session_state["thread_id"]
message_history = st.session_state["conversations"].get(current_thread, [])

for msg in message_history:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])


# *********************** Chatbot ************************************
add_thread(st.session_state["thread_id"])
st.sidebar.title("Langgraph - Chatbot")

if st.sidebar.button("New Chat"):
    new_chat_window()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        # Load conversation from workflow state (if exists)
        message = get_conversation(thread_id)

        msg_list = []
        for m in message:
            if isinstance(m, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            msg_list.append({"role": role, "content": m.content})

        st.session_state["conversations"][thread_id] = msg_list
        st.rerun()  # refresh UI with correct history


# *********************** Input & Streaming **************************
user_input = st.chat_input("Type here")
Config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

if user_input:
    current_thread = st.session_state["thread_id"]

    st.session_state["conversations"][current_thread].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, meta_data in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=Config,
                stream_mode="messages",
            )
        )

    st.session_state["conversations"][current_thread].append(
        {"role": "assistant", "content": ai_message}
    )
