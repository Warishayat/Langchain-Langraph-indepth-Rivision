import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
from pipeline import workflow
import warnings
warnings.filterwarnings('ignore')
load_dotenv()


#streaming in chatbot.
Config = {'configurable' : {'thread_id':'1'}}
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

st.subheader("Chatbot with streming + langgraph")

for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

user_input = st.chat_input("Type here")
if user_input:

    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk,meta_data in workflow.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config  = Config,
                stream_mode = 'messages'
            )
    )
    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
        
   