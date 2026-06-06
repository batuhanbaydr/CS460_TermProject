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
    Uses the judge model to create an improved prompt.
    Uses classification, risk report, and evaluation results.

    Important behavior:
    - If the original prompt is incomplete, it creates a reusable template with placeholders.
    - If the original prompt is already concrete, it preserves the provided details and only adds structure/constraints.
    """

    optimizer_model = get_judge_model()

    evaluation_text = json.dumps(evaluation or {}, indent=2)
    classification_text = json.dumps(classification_result or {}, indent=2)
    risk_text = json.dumps(risk_report or {}, indent=2)

    optimizer_instruction = f"""
You are a careful prompt engineering optimizer.

Your job is to improve the user's prompt so it works more consistently across different LLMs.

Original user prompt:
{original_prompt}

Prompt classification:
{classification_text}

Prompt risk report:
{risk_text}

Evaluation results:
{evaluation_text}

Core objective:
Return an improved prompt that preserves the user's intent and improves reliability across models.

Very important distinction:
- If the original prompt is incomplete or missing required input, return a reusable prompt TEMPLATE with placeholders.
- If the original prompt already contains enough concrete information to complete the task, return a refined version of the same prompt.
- Do NOT turn a complete prompt into a blank generic template.

Preservation priority:
Before rewriting, decide whether the original prompt already contains concrete task information.

Concrete task information may include:
- topic
- quantity
- product type
- source text
- labels
- target audience
- constraints
- desired output
- tone/style
- domain/context
- specific object to work on
- specific question to answer

If the original prompt contains concrete task information, you MUST preserve it.

Examples:
- Original: "Generate 5 brand name ideas for an eco-friendly water bottle company."
  Improved prompt MUST keep:
  - 5
  - brand name ideas
  - eco-friendly
  - water bottle company
  It may add structure, naming constraints, and output format.
  It must NOT turn this into:
  "Product type: [SPECIFY PRODUCT TYPE]"

- Original: "Classify this review as positive or negative: I loved the product."
  Improved prompt MUST keep:
  - review text: "I loved the product"
  - labels: positive or negative
  It may add JSON format or decision criteria.
  It must NOT replace the review with [PASTE REVIEW HERE].

- Original: "Make this better."
  This is incomplete.
  Placeholders are appropriate because no content is provided.

- Original: "Propose a solution."
  This is incomplete.
  A problem-description placeholder is appropriate because no problem is provided.

The improved prompt should:
- Clearly state the task.
- Match the likely task type from the classification result.
- Directly address the main risks from the risk report.
- Preserve all useful details from the original prompt.
- Add placeholders only for information that is actually missing.
- Include missing-input behavior when needed.
- Prevent hallucination and invented content.
- Include output format instructions when useful.
- Include constraints such as tone, length, audience, scope, allowed labels, preservation of meaning, or decision criteria when useful.
- Be understandable by different LLMs.

Risk-aware repair rules:
- If missing_input risk is medium or high:
  - Add a clear input placeholder only for the missing information.
  - Add a strict missing-input rule.
- If ambiguous_reference risk is medium or high:
  - Replace unclear words like "this", "it", "that", "above", or "below" with explicit placeholders or concrete references.
- If hallucination_risk is medium or high:
  - Add a rule that the model must not invent facts, examples, source content, code, data, context, or assumptions.
- If cross_model_inconsistency_risk is medium or high:
  - Make the task structure and output format more explicit.
  - Add priority rules so models behave similarly.
- If missing_output_format risk is medium or high:
  - Add a clear output format.
- If missing_constraints risk is medium or high:
  - Add useful constraints such as length, tone, audience, scope, style, or allowed categories.
- If format_failure_risk is medium or high:
  - Add exact formatting instructions and validation rules.

Task-specific repair rules:

1. Rewriting / improvement tasks:
- If source content is missing, include a content placeholder.
- Ask for improvement criteria, tone, audience, and length only if they are missing.
- Preserve original meaning.
- Add a no-invention rule.
- Output can include improved version and key changes.

2. Summarization tasks:
- If source text/article/document is missing, include a source placeholder.
- If the user asks to summarize general knowledge without source text, preserve the topic and do not add a source placeholder unnecessarily.
- Add length and format constraints.
- Add a no-outside-information rule only when summarizing provided source content.

3. Classification tasks:
- Preserve provided labels and input text if present.
- If labels are missing, add a placeholder for allowed labels.
- If text to classify is missing, add a placeholder for the input.
- Add decision criteria and structured output format.

4. Code help tasks:
- If code is missing, include a code placeholder.
- If error message or expected behavior is missing, add optional placeholders.
- Add step-by-step explanation and no-assumption rules.
- Do not invent code context.

5. Question answering tasks:
- If the question is clear and self-contained, preserve it.
- Add answer scope, tone, and uncertainty handling if helpful.
- If context is required but missing, ask for context.

6. Creative generation tasks:
- Preserve the original creative request and all provided details.
- Keep quantity, topic, product type, audience, theme, style, and constraints from the original prompt.
- Add missing constraints only when useful.
- Do not replace existing details with placeholders.
- If the task is already complete, improve it by adding output format and quality criteria, not by making it generic.

7. Problem-solving tasks:
- If no problem is provided, add a problem-description placeholder.
- If a problem is provided, preserve the problem and add structure such as solution options, feasibility, impact, resources, and recommendation.
- Do not invent a problem.

Priority rules for missing input:
If required input is missing, the improved prompt should tell the model to stop and ask for the missing input before completing the task.

Use wording like:
"First check whether the required input is provided. If it is missing, respond only with: [clarification message]. Do not continue to the task."

This is important because some LLMs continue generating even after noticing missing input.

Rules for complete prompts:
If the original prompt already contains enough information to answer, do NOT add required placeholders that make the prompt incomplete.

For complete prompts:
- Keep the original task.
- Keep the original details.
- Add clarity, format, constraints, and quality criteria.
- Do not ask the user for information that was already provided.
- Do not make the prompt worse by over-generalizing it.

Critical rules:
- Never make a complete prompt less complete.
- Never remove concrete details from the original prompt.
- Never generalize a specific prompt into a blank form.
- Do not replace provided details with placeholders.
- Only add placeholders for genuinely missing information.
- Do not create example content.
- Do not invent facts.
- Do not assume missing content.
- The improved prompt must be something the user can copy and use.
- If optimization would reduce cross-model consistency or remove useful details, keep the original details and only add minimal structure.
- If the original prompt is already good, improve only lightly.
- If the original prompt is seriously incomplete, create a reusable prompt template with placeholders and missing-input rules.

Bad optimization examples:
- Bad: Turning "Generate 5 brand name ideas for an eco-friendly water bottle company" into a blank form asking for product type.
- Bad: Turning "Classify this review as positive or negative: I loved the product" into a blank review template.
- Bad: Adding placeholders for information already present in the original prompt.
- Bad: Generating sample content instead of an improved prompt.

Good optimization examples:
- Original: "Generate 5 brand name ideas for an eco-friendly water bottle company."
  Improved: "Generate exactly 5 brand name ideas for an eco-friendly water bottle company. Return a numbered list. Each name should be 1-3 words, memorable, easy to pronounce, and suitable for a sustainability-focused consumer brand. For each name, include one short explanation. Do not include more than 5 names."

- Original: "Make this better."
  Improved: "Improve the content below while preserving its original meaning. CONTENT TO IMPROVE: [PASTE CONTENT HERE]. If no content is provided, respond only with: 'Please provide the content you would like me to improve.' Do not invent content."

- Original: "Propose a solution."
  Improved: "Propose a solution for the problem below. PROBLEM DESCRIPTION: [PASTE PROBLEM DESCRIPTION HERE]. If no problem is provided, ask for the problem statement, context, desired outcome, and constraints before proposing a solution. Do not invent a problem."

Return only valid JSON.
Do not include markdown.
Do not include commentary outside the JSON.

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