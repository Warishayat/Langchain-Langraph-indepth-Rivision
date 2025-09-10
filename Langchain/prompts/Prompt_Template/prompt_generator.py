from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate(
        template="""
        You are a helpful {topic_input} assistant.
        Your task is to summarize this {paper_name}.
        The answer should be clear, concise, and relevant.
        Use bullet points to show the response and
        don't give a longer answer than 5 lines.
        Include mathematical equations, and if you don’t
        have enough information, simply say "I don't have
        enough information."
        """,
        input_variables=["topic_input", "paper_name"]  
    )

prompt.save("prompt.json")