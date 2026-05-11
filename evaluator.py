def evaluate_prompt_quality(original_prompt: str, model_outputs: dict) -> dict:
    """
    Evaluates the prompt using simple rule-based checks.

    This is the first version of the evaluator.
    Later, we can improve it with an LLM-as-judge evaluator.
    """

    scores = {
        "clarity": 0,
        "specificity": 0,
        "output_format": 0,
        "constraint_following": 0,
        "cross_model_consistency": 0,
    }

    problems = []
    suggestions = []

    prompt_lower = original_prompt.lower()

    # 1. Clarity check
    if len(original_prompt.split()) >= 5:
        scores["clarity"] = 4
    else:
        scores["clarity"] = 2
        problems.append("The prompt is very short and may be unclear.")
        suggestions.append("Make the task more explicit.")

    # 2. Specificity check
    specific_words = [
        "summarize",
        "classify",
        "rewrite",
        "explain",
        "compare",
        "analyze",
        "generate",
        "list",
    ]

    if any(word in prompt_lower for word in specific_words):
        scores["specificity"] = 4
    else:
        scores["specificity"] = 2
        problems.append("The prompt does not clearly specify the task type.")
        suggestions.append("Add a clear action verb such as summarize, classify, rewrite, or explain.")

    # 3. Output format check
    format_words = [
        "bullet",
        "json",
        "table",
        "paragraph",
        "list",
        "numbered",
        "format",
    ]

    if any(word in prompt_lower for word in format_words):
        scores["output_format"] = 5
    else:
        scores["output_format"] = 2
        problems.append("The prompt does not specify an output format.")
        suggestions.append("Specify the desired output format, such as bullet points, JSON, or a table.")

    # 4. Constraint check
    constraint_words = [
        "under",
        "within",
        "exactly",
        "at least",
        "at most",
        "only",
        "do not",
        "avoid",
        "must",
    ]

    if any(word in prompt_lower for word in constraint_words):
        scores["constraint_following"] = 5
    else:
        scores["constraint_following"] = 3
        suggestions.append("Add constraints if needed, such as length, tone, audience, or style.")

    # 5. Cross-model consistency check
    successful_outputs = [
        result["output"]
        for result in model_outputs.values()
        if result.get("success") and result.get("output")
    ]

    if len(successful_outputs) < 2:
        scores["cross_model_consistency"] = 1
        problems.append("Not enough successful model outputs to compare consistency.")
    else:
        lengths = [len(output.split()) for output in successful_outputs]
        min_length = min(lengths)
        max_length = max(lengths)

        if min_length == 0:
            scores["cross_model_consistency"] = 1
        else:
            length_ratio = max_length / min_length

            if length_ratio <= 1.5:
                scores["cross_model_consistency"] = 5
            elif length_ratio <= 2.5:
                scores["cross_model_consistency"] = 3
                problems.append("The model outputs have somewhat different lengths.")
                suggestions.append("Make the expected answer length more explicit.")
            else:
                scores["cross_model_consistency"] = 2
                problems.append("The model outputs are very different in length.")
                suggestions.append("Add clearer structure and length requirements.")

    overall_score = round(sum(scores.values()) / len(scores), 2)

    needs_improvement = overall_score < 4

    return {
        "scores": scores,
        "overall_score": overall_score,
        "needs_improvement": needs_improvement,
        "problems": problems,
        "suggestions": suggestions,
    }