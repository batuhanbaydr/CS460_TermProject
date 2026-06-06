from models import get_test_models


def run_prompt_on_models(user_prompt: str) -> dict:
    """
    Sends the same prompt to all configured test models and returns their outputs.
    """
    models = get_test_models()
    results = {}

    for model_name, model in models.items():
        print(f"\nRunning prompt on {model_name}...\n")

        try:
            response = model.invoke(user_prompt)

            results[model_name] = {
                "success": True,
                "output": response.content,
                "error": None,
            }

        except Exception as e:
            results[model_name] = {
                "success": False,
                "output": None,
                "error": str(e),
            }

    return results