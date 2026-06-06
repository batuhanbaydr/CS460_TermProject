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
        "error": "Failed to parse evaluator JSON.",
        "raw_response": text,
    }


def evaluate_prompt_quality(
    original_prompt: str,
    model_outputs: dict,
    classification_result: dict | None = None,
    risk_report: dict | None = None,
) -> dict:
    """
    Uses Claude Sonnet 4.5 as an LLM judge to evaluate:
    1. the prompt,
    2. each model's output,
    3. cross-model consistency.

    It also understands prompt templates with placeholders.
    """

    judge_model = get_judge_model()

    formatted_outputs = json.dumps(model_outputs, indent=2)
    classification_text = json.dumps(classification_result or {}, indent=2)
    risk_text = json.dumps(risk_report or {}, indent=2)

    evaluator_instruction = f"""
You are a strict LLM prompt evaluation judge.

You are evaluating a user's prompt and the outputs produced by two different LLMs.

User prompt:
{original_prompt}

Prompt classification:
{classification_text}

Prompt risk report:
{risk_text}

Model outputs:
{formatted_outputs}

Important distinction:
A prompt may be either:
1. A direct user prompt, or
2. An improved reusable prompt template.

A reusable prompt template may contain placeholders such as:
- [PASTE TEXT HERE]
- [PASTE CODE HERE]
- [PASTE ARTICLE HERE]
- [PASTE PROBLEM DESCRIPTION HERE]
- [DESCRIBE CONTEXT HERE]
- [SPECIFY LABELS]
- [SPECIFY REQUIREMENTS]

Do NOT automatically punish a reusable prompt template just because placeholders are not filled in.

For a reusable prompt template, evaluate whether:
- the placeholders are clear,
- the task is clear,
- missing-input behavior is explicit,
- the model is told not to invent missing content,
- output format is clear,
- constraints are useful,
- the test models handled the empty placeholders safely.

If a prompt template has empty placeholders and the model correctly asks for missing input instead of completing the task, this is GOOD behavior.

If a model sees an empty placeholder and invents content anyway, this is BAD behavior.

Evaluate the prompt and outputs using the following logic.

1. Prompt-level evaluation:
- task_clarity: Is the task clear?
- input_completeness: For direct prompts, does the prompt include required input? For prompt templates, are placeholders and missing-input rules clear?
- specificity: Are details, audience, scope, or goal clear?
- output_format: Does the prompt specify the desired output format?
- constraint_quality: Does the prompt include useful constraints such as length, tone, style, or rules?
- hallucination_safety: Does the prompt prevent the model from inventing missing content?
- model_independence: Is the prompt likely to work similarly across different LLMs?

2. Per-model output evaluation:
For each model output, evaluate:
- relevance: Did the answer respond to the actual prompt?
- instruction_following: Did it follow the prompt?
- missing_input_handling: If required input or placeholder content was missing, did the model ask for it instead of inventing content?
- hallucination_safety: Did the model avoid inventing unsupported content?
- format_compliance: Did it follow any requested format?
- helpfulness: Was the answer useful and appropriate?

3. Cross-model evaluation:
Evaluate whether the two models interpreted the prompt similarly.
Watch for cases where one model asks for missing input while another invents content.

Scoring rules:
- Score every criterion from 1 to 5.
- 1 means very poor.
- 5 means excellent.
- Be strict, but do not be unfair to valid templates.
- For direct prompts, missing required source text/data/code should reduce input_completeness.
- For reusable prompt templates, clear placeholders should count positively.
- For reusable prompt templates, empty placeholders should NOT reduce input_completeness if the prompt tells the model how to handle missing input.
- If both models correctly ask for missing placeholder content, missing_input_handling should be high.
- If one model invents content while the other asks for missing input, cross-model consistency should be low.
- If the improved prompt is a template and both models safely ask for missing input, needs_improvement should usually be false unless there are other major issues.
- If the prompt improves overall safety but reduces cross-model consistency, mark that clearly.
- Return only valid JSON.
- Do not include markdown.

Use exactly this JSON structure:

{{
  "prompt_evaluation": {{
    "task_clarity": 0,
    "input_completeness": 0,
    "specificity": 0,
    "output_format": 0,
    "constraint_quality": 0,
    "hallucination_safety": 0,
    "model_independence": 0,
    "overall_score": 0,
    "main_issues": [],
    "suggestions": []
  }},
  "model_evaluations": {{
    "amazon_nova_lite": {{
      "relevance": 0,
      "instruction_following": 0,
      "missing_input_handling": 0,
      "hallucination_safety": 0,
      "format_compliance": 0,
      "helpfulness": 0,
      "overall_score": 0,
      "main_issue": ""
    }},
    "meta_llama_3_8b_instruct": {{
      "relevance": 0,
      "instruction_following": 0,
      "missing_input_handling": 0,
      "hallucination_safety": 0,
      "format_compliance": 0,
      "helpfulness": 0,
      "overall_score": 0,
      "main_issue": ""
    }}
  }},
  "cross_model_evaluation": {{
    "consistency_score": 0,
    "same_intent": false,
    "same_output_type": false,
    "reason": ""
  }},
  "overall_score": 0,
  "needs_improvement": true
}}
"""

    response = judge_model.invoke(evaluator_instruction)
    evaluation = extract_json(response.content)

    return evaluation