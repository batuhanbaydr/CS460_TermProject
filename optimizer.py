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
        "diagnosis": "Failed to parse optimizer JSON.",
        "repair_strategy": ["Could not parse structured optimizer output."],
        "improved_prompt": text.strip(),
    }


def optimize_prompt(original_prompt: str, evaluation: dict) -> str:
    """
    Uses the judge model to create a reusable improved prompt template.
    """

    optimizer_model = get_judge_model()

    evaluation_text = json.dumps(evaluation, indent=2)

    optimizer_instruction = f"""
You are a prompt engineering optimizer.

Your job is to rewrite the user's weak prompt into a reusable prompt template that works consistently across different LLMs.

Original user prompt:
{original_prompt}

Evaluation results:
{evaluation_text}

Important goal:
Return an improved prompt TEMPLATE, not a direct response to the user.

The improved prompt should:
- Clearly state the task.
- Include an input placeholder when source content is needed, such as [PASTE TEXT HERE], [PASTE CODE HERE], or [PASTE ARTICLE HERE].
- Include missing-input behavior.
- Prevent hallucination and invented content.
- Include output format instructions when useful.
- Include constraints such as tone, length, audience, or preservation of meaning when useful.
- Be model-independent and understandable by different LLMs.

Critical rules:
- Do not simply say "Please provide the text."
- Do not create example content.
- Do not assume missing content.
- Do not invent facts.
- The improved prompt must be something the user can copy, fill in, and reuse.
- If the original prompt is vague, infer the most likely task but include safeguards for missing input.
- If the task requires source content, include a clear placeholder.
- If the placeholder is empty, instruct the model to ask for the missing input instead of completing the task.

Return only valid JSON.
Do not include markdown.

Use exactly this JSON structure:

{{
  "diagnosis": "",
  "repair_strategy": [],
  "improved_prompt": ""
}}
"""

    response = optimizer_model.invoke(optimizer_instruction)
    parsed = extract_json(response.content)

    return parsed.get("improved_prompt", response.content.strip())