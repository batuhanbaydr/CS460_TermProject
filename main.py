from graph import build_graph


def print_results(title: str, results: dict):
    print("\n==============================")
    print(title)
    print("==============================")

    for model_name, result in results.items():
        print(f"\n--- {model_name} ---")

        if result["success"]:
            print(result["output"])
        else:
            print(f"ERROR: {result['error']}")


def print_evaluation(title: str, evaluation: dict):
    print("\n==============================")
    print(title)
    print("==============================")

    if "error" in evaluation:
        print("Evaluator error:")
        print(evaluation["error"])
        print(evaluation.get("raw_response", ""))
        return

    print(f"\nOverall Score: {evaluation['overall_score']} / 5")
    print(f"Needs Improvement: {evaluation['needs_improvement']}")

    prompt_eval = evaluation["prompt_evaluation"]

    print("\nPrompt-Level Evaluation:")
    for key, value in prompt_eval.items():
        if key not in ["main_issues", "suggestions"]:
            print(f"- {key}: {value}")

    if prompt_eval.get("main_issues"):
        print("\nPrompt Issues:")
        for issue in prompt_eval["main_issues"]:
            print(f"- {issue}")

    if prompt_eval.get("suggestions"):
        print("\nPrompt Suggestions:")
        for suggestion in prompt_eval["suggestions"]:
            print(f"- {suggestion}")

    print("\nPer-Model Evaluations:")
    for model_name, model_eval in evaluation["model_evaluations"].items():
        print(f"\n--- {model_name} ---")
        for key, value in model_eval.items():
            print(f"- {key}: {value}")

    cross_eval = evaluation["cross_model_evaluation"]

    print("\nCross-Model Evaluation:")
    for key, value in cross_eval.items():
        print(f"- {key}: {value}")

def print_comparison_summary(final_state: dict):
    """
    Prints a before/after comparison summary if optimization happened.
    """

    before_eval = final_state.get("evaluation_before", {})
    after_eval = final_state.get("evaluation_after", {})

    print("\n==============================")
    print("BEFORE / AFTER COMPARISON")
    print("==============================")

    before_score = before_eval.get("overall_score", "N/A")
    before_needs_improvement = before_eval.get("needs_improvement", "N/A")

    print(f"\nOriginal Prompt Score: {before_score} / 5")
    print(f"Original Needs Improvement: {before_needs_improvement}")

    if not after_eval:
        print("\nNo optimization was performed.")
        print("Final Decision: Original prompt accepted.")
        return

    after_score = after_eval.get("overall_score", "N/A")
    after_needs_improvement = after_eval.get("needs_improvement", "N/A")

    print(f"Improved Prompt Score: {after_score} / 5")
    print(f"Improved Needs Improvement: {after_needs_improvement}")

    if isinstance(before_score, (int, float)) and isinstance(after_score, (int, float)):
        score_change = round(after_score - before_score, 2)
        print(f"Score Change: {score_change:+} points")

    before_cross = before_eval.get("cross_model_evaluation", {})
    after_cross = after_eval.get("cross_model_evaluation", {})

    before_consistency = before_cross.get("consistency_score", "N/A")
    after_consistency = after_cross.get("consistency_score", "N/A")

    print(f"\nCross-Model Consistency Before: {before_consistency} / 5")
    print(f"Cross-Model Consistency After: {after_consistency} / 5")

    if isinstance(before_consistency, (int, float)) and isinstance(after_consistency, (int, float)):
        consistency_change = round(after_consistency - before_consistency, 2)
        print(f"Consistency Change: {consistency_change:+} points")

    if after_needs_improvement is False:
        print("\nFinal Decision: Improved prompt accepted.")
    else:
        print("\nFinal Decision: Improved prompt still needs refinement.")


if __name__ == "__main__":
    print("PromptRefiner is starting...")

    user_prompt = input("Enter a prompt to test: ")

    app = build_graph()

    initial_state = {
        "original_prompt": user_prompt,
        "task_type": "general",
        "model_outputs_before": {},
        "evaluation_before": {},
        "improved_prompt": "",
        "model_outputs_after": {},
        "evaluation_after": {},
        "final_report": "",
    }

    final_state = app.invoke(initial_state)

    print("\n==============================")
    print("ORIGINAL PROMPT")
    print("==============================")
    print(final_state["original_prompt"])

    print_results(
        "MODEL OUTPUTS BEFORE IMPROVEMENT",
        final_state["model_outputs_before"]
    )

    print_evaluation(
        "PROMPT EVALUATION BEFORE IMPROVEMENT",
        final_state["evaluation_before"]
    )

    if final_state.get("improved_prompt"):
        print("\n==============================")
        print("IMPROVED PROMPT")
        print("==============================")
        print(final_state["improved_prompt"])

        print_results(
            "MODEL OUTPUTS AFTER IMPROVEMENT",
            final_state["model_outputs_after"]
        )

        print_evaluation(
            "PROMPT EVALUATION AFTER IMPROVEMENT",
            final_state["evaluation_after"]
        )
    else:
        print("\n==============================")
        print("NO OPTIMIZATION NEEDED")
        print("==============================")
        print("The original prompt passed the evaluation, so the optimizer was skipped.")
    
    print_comparison_summary(final_state)