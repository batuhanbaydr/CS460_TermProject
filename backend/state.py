from typing import TypedDict, Dict, Any


class PromptState(TypedDict):
    """
    Shared state object for the LangGraph workflow.

    Each node in the graph reads from and writes to this state.
    """

    # Original user input
    original_prompt: str
    task_type: str

    # Prompt classification
    classification_result: Dict[str, Any]
    prompt_type: str

    # Prompt risk analysis
    risk_report: Dict[str, Any]

    # Model outputs before optimization
    model_outputs_before: Dict[str, Dict[str, Any]]

    # Evaluation before optimization
    evaluation_before: Dict[str, Any]

    # Optimized prompt
    improved_prompt: str

    # Model outputs after optimization
    model_outputs_after: Dict[str, Dict[str, Any]]

    # Evaluation after optimization
    evaluation_after: Dict[str, Any]

    # Final summary/report
    final_report: str