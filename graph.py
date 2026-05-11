from langgraph.graph import StateGraph, START, END
from state import PromptState
from runner import run_prompt_on_models
from evaluator import evaluate_prompt_quality


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


def build_graph():
    """
    Builds the current LangGraph workflow.

    Current workflow:
    START -> run_original_prompt -> evaluate_prompt -> END
    """

    graph = StateGraph(PromptState)

    graph.add_node("run_original_prompt", run_original_prompt_node)
    graph.add_node("evaluate_prompt", evaluate_prompt_node)

    graph.add_edge(START, "run_original_prompt")
    graph.add_edge("run_original_prompt", "evaluate_prompt")
    graph.add_edge("evaluate_prompt", END)

    return graph.compile()