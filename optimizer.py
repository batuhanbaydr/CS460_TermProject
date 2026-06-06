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


def optimize_prompt(
    original_prompt: str,
    evaluation: dict,
    classification_result: dict | None = None,
    risk_report: dict | None = None,
) -> str:
    """
    Uses the judge model to create a reusable improved prompt template.
    Uses classification, risk report, and evaluation results.
    """

    optimizer_model = get_judge_model()

    evaluation_text = json.dumps(evaluation or {}, indent=2)
    classification_text = json.dumps(classification_result or {}, indent=2)
    risk_text = json.dumps(risk_report or {}, indent=2)

    optimizer_instruction = f"""
You are a prompt engineering optimizer.

Your job is to rewrite the user's weak prompt into a reusable prompt template that works consistently across different LLMs.

Original user prompt:
{original_prompt}

Prompt classification:
{classification_text}

Prompt risk report:
{risk_text}

Evaluation results:
{evaluation_text}

Important goal:
Return an improved prompt TEMPLATE, not a direct response to the user.

The improved prompt should:
- Clearly state the task.
- Match the likely task type from the classification result.
- Directly address the main risks from the risk report.
- Include an input placeholder when source content is needed, such as [PASTE TEXT HERE], [PASTE CODE HERE], [PASTE ARTICLE HERE], [PASTE PROBLEM DESCRIPTION HERE], or similar.
- Include missing-input behavior.
- Prevent hallucination and invented content.
- Include output format instructions when useful.
- Include constraints such as tone, length, audience, scope, preservation of meaning, allowed labels, or decision criteria when useful.
- Be model-independent and understandable by different LLMs.

Risk-aware repair rules:
- If missing_input risk is medium or high, add a clear input placeholder and a strict missing-input rule.
- If ambiguous_reference risk is medium or high, replace words like "this", "it", or "that" with explicit placeholders.
- If hallucination_risk is medium or high, add a rule that the model must not invent facts, examples, source content, code, data, or context.
- If cross_model_inconsistency_risk is medium or high, make the task structure and output format more explicit.
- If missing_output_format risk is medium or high, add a clear output format.
- If missing_constraints risk is medium or high, add useful constraints such as length, tone, target audience, scope, or allowed categories.
- If format_failure_risk is medium or high, add exact formatting instructions and validation rules.

Task-specific repair rules:
- For rewriting/improvement tasks, include content placeholder, improvement criteria, target audience, tone, preservation of meaning, and no-invention rule.
- For summarization tasks, include source text placeholder if needed, length/format requirement, and no-outside-information rule.
- For classification tasks, include allowed labels, decision criteria, and structured output format.
- For code_help tasks, include code placeholder, error/context placeholder, explanation steps, and no-assumption rule.
- For question_answering tasks, include context if needed, answer scope, tone, and uncertainty handling.
- For creative_generation tasks, preserve the original creative request and all provided details. Add missing constraints such as number of outputs, style, target audience, naming rules, or output format only when useful. Do not replace existing details with placeholders.

Critical rules:
- Do not simply say "Please provide the text."
- Do not create example content.
- Do not assume missing content.
- Do not invent facts.
- The improved prompt must be something the user can copy, fill in, and reuse.
- If the original prompt is vague, infer the most likely task but include safeguards for missing input.
- If the task requires source content, include a clear placeholder.
- If the placeholder is empty, instruct the model to ask for the missing input instead of completing the task.
- Use priority rules when needed, so different LLMs handle missing input consistently.
- Preserve all concrete details already provided in the original prompt.
- Do not replace provided details with placeholders.
- Only add placeholders for information that is missing.
- If the original prompt already includes enough information to complete the task, improve it by adding structure, constraints, and output format, not by turning it into a blank template.
- For creative generation tasks with enough context, keep the provided topic, quantity, product type, audience, or style details from the original prompt.

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