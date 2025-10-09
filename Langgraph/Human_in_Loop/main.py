from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from classes import MessageState
from nodes import genrated_post,linkedin_publication
from tools import google_search,city_weather
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langgraph.types import Command,interrupt


load_dotenv()


groq_Api_key = os.getenv("groq_api_key")
Model=ChatGroq(model='openai/gpt-oss-20b',api_key=groq_Api_key)


graph = StateGraph(MessageState)
#human in loop
def review_node(state: MessageState):
    print('Human Node wating for the human response')
    print("Genrated Post:\n",state['genrated_post'][-1].content)
    human_feedback = interrupt({
        'Genrated_post': f"Genrated_Post: {state['genrated_post'][-1].content}",
        'feedback': "Provide feedback or type 'done' to finish"
    })
    print('Human feedback is:',human_feedback)

    if human_feedback.lower() == 'done':
        return Command(
            update={'human_feedback' :  state['human_feedback'] + ['finalized']}
            ,goto='linkedin_publication'
        )

    return Command(
        update = {'human_feedback':state['human_feedback'] + [human_feedback]}
        ,goto='genrated_post'
    )


#lets define nodes
graph.add_node('genrated_post',genrated_post)
graph.add_node('review_node',review_node)
graph.add_node('linkedin_publication',linkedin_publication)

#start making edges
graph.set_entry_point('genrated_post')
graph.add_edge(START,'genrated_post')
graph.add_edge('genrated_post','review_node')
graph.set_finish_point('linkedin_publication')

memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)

config = { 
    "configurable": {"thread_id": "010001"}
}

linkedin_topic = input("Enter your LinkedIn topic: ")
initial_state = {
    "query": linkedin_topic, 
    "genrated_post": [], 
    "human_feedback": []
}

for chunk in workflow.stream(initial_state,config=config):
    for node_id, value in chunk.items():
        print("Intruption happen:", node_id)
        if node_id == '__interrupt__':
            while True:
                user_feedback = input("Provide feedback (or type 'done' when finished): ")
                workflow.invoke(Command(resume=user_feedback),config=config)
                if user_feedback.lower() == 'done':
                    break
