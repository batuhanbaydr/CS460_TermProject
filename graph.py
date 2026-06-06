from langgraph.graph import StateGraph, START, END
from state import PromptState
from classifier import classify_prompt
from runner import run_prompt_on_models
from evaluator import evaluate_prompt_quality
from optimizer import optimize_prompt
from risk_detector import detect_prompt_risks


def classify_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that classifies the original user prompt.
    """

    original_prompt = state["original_prompt"]

    classification = classify_prompt(original_prompt)

    state["classification_result"] = classification
    state["prompt_type"] = classification.get("primary_type", "general")

    return state

def detect_risks_node(state: PromptState) -> PromptState:
    """
    LangGraph node that detects prompt risks and likely failure modes.
    """

    original_prompt = state["original_prompt"]
    classification_result = state["classification_result"]

    risk_report = detect_prompt_risks(original_prompt, classification_result)

    state["risk_report"] = risk_report

    return state


def run_original_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that sends the original prompt to all selected test models.
    """

    original_prompt = state["original_prompt"]
    model_outputs = run_prompt_on_models(original_prompt)

    state["model_outputs_before"] = model_outputs

    return state


def evaluate_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that evaluates the original prompt and model outputs.
    """

    original_prompt = state["original_prompt"]
    model_outputs = state["model_outputs_before"]

    classification_result = state["classification_result"]
    risk_report = state["risk_report"]

    evaluation = evaluate_prompt_quality(
        original_prompt,
        model_outputs,
        classification_result,
        risk_report,
    )

    state["evaluation_before"] = evaluation

    return state


def decide_next_step(state: PromptState) -> str:
    """
    Decides whether the prompt needs optimization.
    """

    evaluation = state["evaluation_before"]

    if evaluation.get("needs_improvement", True):
        return "optimize_prompt"

    return "end"


def optimize_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that improves the original prompt based on evaluation feedback.
    """

    original_prompt = state["original_prompt"]
    evaluation = state["evaluation_before"]

    classification_result = state["classification_result"]
    risk_report = state["risk_report"]

    improved_prompt = optimize_prompt(
        original_prompt,
        evaluation,
        classification_result,
        risk_report,
    )

    state["improved_prompt"] = improved_prompt

    return state


def run_improved_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that sends the improved prompt to all selected test models.
    """

    improved_prompt = state["improved_prompt"]
    model_outputs = run_prompt_on_models(improved_prompt)

    state["model_outputs_after"] = model_outputs

    return state


def evaluate_improved_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that evaluates the improved prompt and improved model outputs.
    """

    improved_prompt = state["improved_prompt"]
    model_outputs = state["model_outputs_after"]

    classification_result = state["classification_result"]
    risk_report = state["risk_report"]

    evaluation = evaluate_prompt_quality(
        improved_prompt,
        model_outputs,
        classification_result,
        risk_report,
    )

    state["evaluation_after"] = evaluation

    return state


def build_graph():
    """
    Builds the LangGraph workflow.

    Workflow:
    START
      -> classify_prompt
      -> detect_risks
      -> run_original_prompt
      -> evaluate_prompt
      -> conditional decision:
            if needs_improvement=True:
                optimize_prompt
                -> run_improved_prompt
                -> evaluate_improved_prompt
                -> END
            else:
                END
    """

    graph = StateGraph(PromptState)

    graph.add_node("classify_prompt", classify_prompt_node)
    graph.add_node("run_original_prompt", run_original_prompt_node)
    graph.add_node("evaluate_prompt", evaluate_prompt_node)
    graph.add_node("optimize_prompt", optimize_prompt_node)
    graph.add_node("run_improved_prompt", run_improved_prompt_node)
    graph.add_node("evaluate_improved_prompt", evaluate_improved_prompt_node)
    graph.add_node("detect_risks", detect_risks_node)

    graph.add_edge(START, "classify_prompt")
    graph.add_edge("classify_prompt", "detect_risks")
    graph.add_edge("detect_risks", "run_original_prompt")
    graph.add_edge("run_original_prompt", "evaluate_prompt")

    graph.add_conditional_edges(
        "evaluate_prompt",
        decide_next_step,
        {
            "optimize_prompt": "optimize_prompt",
            "end": END,
        },
    )

    graph.add_edge("optimize_prompt", "run_improved_prompt")
    graph.add_edge("run_improved_prompt", "evaluate_improved_prompt")
    graph.add_edge("evaluate_improved_prompt", END)

    return graph.compile()