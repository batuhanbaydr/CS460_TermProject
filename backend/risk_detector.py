import json
from models import get_judge_model


def extract_json(text: str) -> dict:
    """
    Attempts to extract JSON from an LLM response.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {
        "error": "Failed to parse risk detector JSON.",
        "raw_response": text,
    }


def detect_prompt_risks(original_prompt: str, classification_result: dict) -> dict:
    """
    Uses Claude Sonnet 4.5 to detect prompt risks and likely failure modes.
    """

    judge_model = get_judge_model()

    classification_text = json.dumps(classification_result, indent=2)

    risk_instruction = f"""
You are a prompt risk analysis assistant.

Your task is to analyze a user prompt and identify risks that may cause poor or inconsistent LLM outputs.

User prompt:
{original_prompt}

Prompt classification:
{classification_text}

Evaluate the following risk categories:

1. missing_input
- Does the prompt require text, code, data, an article, an image, or context that is not provided?

2. ambiguous_reference
- Does the prompt use unclear references such as "this", "it", "that", "above", "below", "the text", or "the code" without providing the object?

3. unclear_task
- Is the task vague or underspecified?

4. missing_output_format
- Does the prompt fail to specify how the answer should be structured?

5. missing_constraints
- Does the prompt lack useful constraints such as length, tone, audience, scope, style, or allowed labels?

6. hallucination_risk
- Is the model likely to invent missing content, facts, examples, source text, or assumptions?

7. cross_model_inconsistency_risk
- Is the prompt likely to make different LLMs interpret the task differently?

8. format_failure_risk
- Is there a risk that models will not return the desired format?

For each risk, assign:
- level: "low", "medium", or "high"
- reason: short explanation

Also return:
- overall_risk_level: "low", "medium", or "high"
- main_failure_modes: list of the most important likely failure modes
- recommendations: list of concrete prompt improvements

Be strict. If the prompt is vague or incomplete, risks should be high.

Return only valid JSON.
Do not include markdown.

Use exactly this JSON structure:

{{
  "risks": {{
    "missing_input": {{"level": "", "reason": ""}},
    "ambiguous_reference": {{"level": "", "reason": ""}},
    "unclear_task": {{"level": "", "reason": ""}},
    "missing_output_format": {{"level": "", "reason": ""}},
    "missing_constraints": {{"level": "", "reason": ""}},
    "hallucination_risk": {{"level": "", "reason": ""}},
    "cross_model_inconsistency_risk": {{"level": "", "reason": ""}},
    "format_failure_risk": {{"level": "", "reason": ""}}
  }},
  "overall_risk_level": "",
  "main_failure_modes": [],
  "recommendations": []
}}
"""

    response = judge_model.invoke(risk_instruction)

    return extract_json(response.content)