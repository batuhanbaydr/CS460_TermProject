from langgraph.graph import StateGraph, START, END
from state import PromptState
from runner import run_prompt_on_models


def run_original_prompt_node(state: PromptState) -> PromptState:
    """
    LangGraph node that sends the original prompt to all selected models.
    """

    original_prompt = state["original_prompt"]

    model_outputs = run_prompt_on_models(original_prompt)

    state["model_outputs_before"] = model_outputs

    return state


def build_graph():
    """
    Builds the first simple LangGraph workflow.

    Current workflow:
    START -> run_original_prompt -> END
    """

    graph = StateGraph(PromptState)

    graph.add_node("run_original_prompt", run_original_prompt_node)

    graph.add_edge(START, "run_original_prompt")
    graph.add_edge("run_original_prompt", END)

    return graph.compile()