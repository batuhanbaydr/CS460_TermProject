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
        "primary_type": "general",
        "likely_task": "unknown",
        "requires_input": False,
        "is_incomplete": True,
        "confidence": 0,
        "reason": "Failed to parse classifier JSON.",
        "raw_response": text,
    }


def classify_prompt(original_prompt: str) -> dict:
    """
    Classifies the user's prompt into a prompt/task type.
    Uses Claude Sonnet 4.5 as the classifier.
    """

    judge_model = get_judge_model()

    classifier_instruction = f"""
You are a prompt classification assistant.

Your task is to classify the user's prompt into a task type.

User prompt:
{original_prompt}

Possible primary_type values:
- rewriting
- summarization
- classification
- explanation
- code_help
- creative_generation
- question_answering
- missing_input
- general

Classification rules:
- If the prompt refers to missing content using words like "this", "it", "above", "below", "the text", "the code", or "the article" but no actual content is provided, classify it as missing_input.
- If the prompt asks to improve, rewrite, polish, fix, or make something better, the likely_task is rewriting.
- If the prompt asks to summarize something, the likely_task is summarization.
- If the prompt asks to explain code, debug code, or improve code, use code_help.
- If the prompt asks for labels, categories, sentiment, or decisions, use classification.
- If the prompt asks to generate a story, idea, image prompt, name, slogan, or creative output, use creative_generation.
- If the prompt asks a factual or conceptual question, use question_answering.
- If the prompt is too vague to identify clearly, use general or missing_input.

Return only valid JSON.
Do not include markdown.

Use exactly this JSON structure:

{{
  "primary_type": "",
  "likely_task": "",
  "requires_input": true,
  "is_incomplete": true,
  "confidence": 0,
  "reason": ""
}}
"""

    response = judge_model.invoke(classifier_instruction)

    return extract_json(response.content)