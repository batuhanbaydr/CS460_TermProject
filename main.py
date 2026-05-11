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