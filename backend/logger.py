import json
import os
from datetime import datetime


RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_results.jsonl")


def get_consistency_score(evaluation: dict):
    """
    Safely extracts cross-model consistency score.
    """

    if not evaluation:
        return None

    cross_eval = evaluation.get("cross_model_evaluation", {})
    return cross_eval.get("consistency_score")


def build_log_record(final_state: dict) -> dict:
    """
    Builds a compact log record from the final LangGraph state.
    """

    before_eval = final_state.get("evaluation_before", {})
    after_eval = final_state.get("evaluation_after", {})

    before_score = before_eval.get("overall_score")
    after_score = after_eval.get("overall_score")

    before_consistency = get_consistency_score(before_eval)
    after_consistency = get_consistency_score(after_eval)

    score_change = None
    if isinstance(before_score, (int, float)) and isinstance(after_score, (int, float)):
        score_change = round(after_score - before_score, 2)

    consistency_change = None
    if isinstance(before_consistency, (int, float)) and isinstance(after_consistency, (int, float)):
        consistency_change = round(after_consistency - before_consistency, 2)

    optimization_performed = bool(final_state.get("improved_prompt"))

    if optimization_performed:
        final_decision = (
            "improved_prompt_accepted"
            if after_eval.get("needs_improvement") is False
            else "improved_prompt_needs_refinement"
        )
    else:
        final_decision = "original_prompt_accepted"

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),

        "original_prompt": final_state.get("original_prompt", ""),
        "prompt_type": final_state.get("prompt_type", ""),
        "classification_result": final_state.get("classification_result", {}),
        "risk_report": final_state.get("risk_report", {}),

        "model_outputs_before": final_state.get("model_outputs_before", {}),
        "evaluation_before": before_eval,

        "optimization_performed": optimization_performed,
        "improved_prompt": final_state.get("improved_prompt", ""),
        "model_outputs_after": final_state.get("model_outputs_after", {}),
        "evaluation_after": after_eval,

        "original_score": before_score,
        "improved_score": after_score,
        "score_change": score_change,

        "original_consistency_score": before_consistency,
        "improved_consistency_score": after_consistency,
        "consistency_change": consistency_change,

        "final_decision": final_decision,
    }


def save_run(final_state: dict) -> str:
    """
    Saves one run as a JSONL record.
    Returns the path of the saved results file.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    record = build_log_record(final_state)

    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return RESULTS_FILE