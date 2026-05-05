from typing import TypedDict, Dict, Any


class PromptState(TypedDict):
    """
    Shared state object for the LangGraph workflow.

    Each node in the graph will read from and write to this state.
    """

    original_prompt: str
    task_type: str

    model_outputs_before: Dict[str, Dict[str, Any]]

    evaluation_before: Dict[str, Any]

    improved_prompt: str

    model_outputs_after: Dict[str, Dict[str, Any]]

    evaluation_after: Dict[str, Any]

    final_report: str