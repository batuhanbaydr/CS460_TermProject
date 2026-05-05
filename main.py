from runner import run_prompt_on_models


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

    outputs = run_prompt_on_models(user_prompt)

    print_results(outputs)