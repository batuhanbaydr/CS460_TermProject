from langgraph.graph import StateGraph, START, END
from state import PromptState
from runner import run_prompt_on_models
from evaluator import evaluate_prompt_quality
from optimizer import optimize_prompt


def run_original_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that sends the original prompt to all selected models.
    """

    original_prompt = state["original_prompt"]

    model_outputs = run_prompt_on_models(original_prompt)

    state["model_outputs_before"] = model_outputs

    return state


def evaluate_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that evaluates the original prompt
    based on the outputs from different models.
    """

    original_prompt = state["original_prompt"]
    model_outputs = state["model_outputs_before"]

    evaluation = evaluate_prompt_quality(original_prompt, model_outputs)

    state["evaluation_before"] = evaluation

    return state


def optimize_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that improves the original prompt based on evaluation feedback.
    """

    original_prompt = state["original_prompt"]
    evaluation = state["evaluation_before"]

    improved_prompt = optimize_prompt(original_prompt, evaluation)

    state["improved_prompt"] = improved_prompt

    return state


def run_improved_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that sends the improved prompt to all selected models.
    """

    improved_prompt = state["improved_prompt"]

    model_outputs = run_prompt_on_models(improved_prompt)

    state["model_outputs_after"] = model_outputs

    return state


def evaluate_improved_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that evaluates the improved prompt
    based on the outputs from different models.
    """

    improved_prompt = state["improved_prompt"]
    model_outputs = state["model_outputs_after"]

    evaluation = evaluate_prompt_quality(improved_prompt, model_outputs)

    state["evaluation_after"] = evaluation

    return state


def build_graph():
    """
    Builds the current LangGraph workflow.

    Current workflow:
    START
      -> run_original_prompt
      -> evaluate_prompt
      -> optimize_prompt
      -> run_improved_prompt
      -> evaluate_improved_prompt
      -> END
    """

    graph = StateGraph(PromptState)

    graph.add_node("run_original_prompt", run_original_prompt_node)
    graph.add_node("evaluate_prompt", evaluate_prompt_node)
    graph.add_node("optimize_prompt", optimize_prompt_node)
    graph.add_node("run_improved_prompt", run_improved_prompt_node)
    graph.add_node("evaluate_improved_prompt", evaluate_improved_prompt_node)

    graph.add_edge(START, "run_original_prompt")
    graph.add_edge("run_original_prompt", "evaluate_prompt")
    graph.add_edge("evaluate_prompt", "optimize_prompt")
    graph.add_edge("optimize_prompt", "run_improved_prompt")
    graph.add_edge("run_improved_prompt", "evaluate_improved_prompt")
    graph.add_edge("evaluate_improved_prompt", END)

    return graph.compile()