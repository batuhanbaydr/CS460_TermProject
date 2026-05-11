from graph import build_graph


def print_results(results: dict):
    print("\n==============================")
    print("MODEL OUTPUTS")
    print("==============================")

    for model_name, result in results.items():
        print(f"\n--- {model_name} ---")

        if result["success"]:
            print(result["output"])
        else:
            print(f"ERROR: {result['error']}")


def print_evaluation(evaluation: dict):
    print("\n==============================")
    print("PROMPT EVALUATION")
    print("==============================")

    print(f"\nOverall Score: {evaluation['overall_score']} / 5")
    print(f"Needs Improvement: {evaluation['needs_improvement']}")

    print("\nScores:")
    for criterion, score in evaluation["scores"].items():
        print(f"- {criterion}: {score}/5")

    if evaluation["problems"]:
        print("\nProblems:")
        for problem in evaluation["problems"]:
            print(f"- {problem}")

    if evaluation["suggestions"]:
        print("\nSuggestions:")
        for suggestion in evaluation["suggestions"]:
            print(f"- {suggestion}")


if __name__ == "__main__":
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

    print_results(final_state["model_outputs_before"])
    print_evaluation(final_state["evaluation_before"])