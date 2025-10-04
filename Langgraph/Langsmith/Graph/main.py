import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Annotated
from typing import TypedDict
import operator

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
Langsmith_api_key = os.getenv("LangSmith_api_key")

# LangSmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agent in Langgraph"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
if Langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = Langsmith_api_key

# Model
Model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    api_key=GOOGLE_API_KEY, 
    temperature=0
)

# Test call
print(Model.invoke("which khan is most famous?"))

# --- Structured output schema ---
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", gt=0, lt=10)

structured_model = Model.with_structured_output(EvaluationSchema)

# --- State schema for LangGraph ---
class CSS_state(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# --- Traceable functions (nodes) ---
from langsmith import traceable

@traceable(name="Evaluate Language")
def evaluate_language(state: CSS_state):
    prompt = f"Evaluate the language quality of the following essay (0-10):\n\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {"language_feedback": result.feedback, "individual_scores": [result.score]}

@traceable(name="Evaluate Analysis")
def evaluate_analysis(state: CSS_state):
    prompt = f"Evaluate the depth of analysis of the following essay (0-10):\n\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {"analysis_feedback": result.feedback, "individual_scores": [result.score]}

@traceable(name="Evaluate Clarity of Thought")
def evaluate_thought(state: CSS_state):
    prompt = f"Evaluate the clarity of thought in the following essay (0-10):\n\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {"clarity_feedback": result.feedback, "individual_scores": [result.score]}

@traceable(name="Final Evaluation")
def final_evaluate(state: CSS_state):
    # Create overall summary and compute average
    prompt = (
        f"Based on the following feedback, create a summary:\n\n"
        f"Language feedback: {state['language_feedback']}\n"
        f"Analysis feedback: {state['analysis_feedback']}\n"
        f"Clarity feedback: {state['clarity_feedback']}\n"
    )
    overall_summary = Model.invoke(prompt).content
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    return {"overall_feedback": overall_summary, "avg_score": avg_score}

# --- LangGraph setup ---
from langgraph.graph import StateGraph, START, END

graph = StateGraph(CSS_state)

# Add nodes
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluate", final_evaluate)

# Define edges
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

graph.add_edge("evaluate_language", "final_evaluate")
graph.add_edge("evaluate_analysis", "final_evaluate")
graph.add_edge("evaluate_thought", "final_evaluate")

graph.add_edge("final_evaluate", END)

# Compile workflow
workflow = graph.compile()

# --- Wrapper to run whole workflow traceably ---
@traceable(name="Essay Evaluation Workflow")
def run_workflow(essay: str):
    initial_state = {"essay": essay}
    return workflow.invoke(initial_state)

# --- Example run ---
if __name__ == "__main__":
    essay = """
    Quantum physics, also known as quantum mechanics, is the branch of physics that studies
    the behavior of matter and energy at the smallest scales, such as atoms and subatomic particles.
    Unlike classical physics, which follows predictable laws of motion, quantum physics reveals
    a world governed by probabilities and uncertainties. One of its central principles is the
    wave-particle duality, which shows that particles like electrons and photons can behave both
    as waves and particles depending on how they are observed. Another key concept, Heisenberg’s
    uncertainty principle, states that certain properties, such as position and momentum, cannot
    both be measured with complete precision at the same time.
    
    Despite its abstract and sometimes puzzling concepts, quantum physics has had profound practical
    impacts on technology and modern life. The development of semiconductors, which are the foundation
    of computers and smartphones, relies on quantum mechanics. Other technologies, such as lasers,
    magnetic resonance imaging (MRI), and even the emerging field of quantum computing, stem directly
    from quantum principles.
    """
    result = run_workflow(essay)
    print("\n--- Final Evaluation Result ---")
    print(result)
