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