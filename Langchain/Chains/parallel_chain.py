import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_groq import ChatGroq
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Problem Statment
# Lets dicuss the pralell chain
# Example 2
# user will pass the detaik text
# paralell kam krega idr 
# we genrate two things from it 1: note 2: quiz
# then combine and wil show to user

model_01 = ChatGroq(model="openai/gpt-oss-20b",temperature=0.7,api_key=GROQ_API_KEY)
model_02 = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)



template_note = PromptTemplate(
    template="Genrate summarize the simple and concise the simple notes and tips and trick from the following text\n {text}",
    input_variables=["text"]
)

template_quiz = PromptTemplate(
    template="""
    Genrate 5 simple question answer from the following text.
    {text}
    """,
    input_variables=["text"]
)

merge_response = PromptTemplate(
    template = """
    You are a helpfull assistant merge the provided notes ->{notes} and quiz -> {quiz} into a single document.
    """,
    input_variables=["notes","quiz"]
)

parser = StrOutputParser()

#RunnableParallel k through hum paralell chain execute krte haen
parallel_chain = RunnableParallel({
    'notes' : template_note | model_01 | parser,
    'quiz' : template_quiz | model_02 | parser
})

merge_chain = merge_response | model_01 | parser
chain = parallel_chain | merge_chain
text = """
    Machine learning (ML) is a branch of artificial intelligence (AI) that focuses on the development of algorithms capable of learning patterns from data and improving performance over time without being explicitly programmed. Unlike traditional computer programs, which follow a rigid set of instructions crafted by human developers, machine learning systems adapt by analyzing examples and drawing generalizations that can later be applied to new, unseen situations. This ability to learn and generalize makes machine learning one of the most powerful technologies of the modern era.
    The concept is not entirely new. As early as the 1950s, pioneers such as Alan Turing and Arthur Samuel envisioned machines that could exhibit forms of intelligence by "learning" from experience. What has changed in recent decades is the unprecedented growth of digital data, affordable computing power, and sophisticated algorithms, all of which have accelerated the progress of machine learning and expanded its practical applications. Today, ML underpins many aspects of daily life, often without people being consciously aware of it. From spam filters in email systems to recommendation engines on streaming platforms, the technology is pervasive.
    Machine learning is both a scientific discipline and an engineering practice. Scientifically, it raises questions about the nature of learning, representation of knowledge, and the boundaries of human versus machine cognition. Practically, it drives innovations across industries, enabling businesses, governments, and individuals to make more informed decisions. Healthcare professionals use ML to predict patient outcomes; financial institutions deploy it to detect fraudulent transactions; self-driving cars rely on it to interpret sensory data and navigate complex environments.
    Despite its rapid growth and widespread use, machine learning also poses challenges and risks. Issues of fairness, interpretability, privacy, and accountability remain central to its responsible deployment. As machine learning systems increasingly influence critical decisions in society, understanding both their capabilities and limitations has become essential.
    This text explores the foundations, methods, applications, and future of machine learning. Beginning with its origins, it will outline fundamental concepts, classify the main types of learning, describe common algorithms, and examine real-world applications. It will also highlight pressing challenges and ethical considerations, before concluding with a reflection on the potential trajectory of machine learning in the coming decades.
"""

res=chain.invoke({"text":text})
print(res)
chain.get_graph().print_ascii()