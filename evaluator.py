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


def evaluate_prompt_quality(original_prompt: str, model_outputs: dict) -> dict:
    """
    Uses Claude Sonnet 4.5 as an LLM judge to evaluate:
    1. the original prompt,
    2. each model's output,
    3. cross-model consistency.
    """

    judge_model = get_judge_model()

    formatted_outputs = json.dumps(model_outputs, indent=2)

    evaluator_instruction = f"""
You are a strict LLM prompt evaluation judge.

You are evaluating a user's prompt and the outputs produced by two different LLMs.

User prompt:
{original_prompt}

Model outputs:
{formatted_outputs}

Evaluate the prompt and outputs using the following logic.

1. Prompt-level evaluation:
- task_clarity: Is the user asking for a clear task?
- input_completeness: Does the prompt include all necessary input? For example, "Make this better" is incomplete because no text is provided.
- specificity: Are details, audience, scope, or goal clear?
- output_format: Does the prompt specify the desired output format?
- constraint_quality: Does the prompt include useful constraints, such as length, tone, style, or rules?
- hallucination_safety: Does the prompt prevent the model from inventing missing content?
- model_independence: Is the prompt likely to work similarly across different LLMs?

2. Per-model output evaluation:
For each model output, evaluate:
- relevance: Did the answer respond to the user's actual prompt?
- instruction_following: Did it follow the prompt?
- missing_input_handling: If required input was missing, did the model ask for it instead of inventing content?
- hallucination_safety: Did the model avoid inventing unsupported content?
- format_compliance: Did it follow any requested format?
- helpfulness: Was the answer useful and appropriate?

3. Cross-model evaluation:
Evaluate whether the two models interpreted the prompt similarly.
Watch for cases where one model asks for missing input while another invents content.

Important:
- Score every criterion from 1 to 5.
- 1 means very poor.
- 5 means excellent.
- Be strict.
- If the prompt requires missing text, data, code, an article, an image, or context, input_completeness should be low.
- However, if the prompt is an improved prompt template and includes a clear placeholder such as [PASTE TEXT HERE], [PASTE CODE HERE], [PASTE ARTICLE HERE], [INSERT TEXT], or similar, do not punish it for missing input.
- In that case, evaluate whether the placeholder and missing-input behavior are clear.
- If the model correctly asks for missing input instead of inventing content, missing_input_handling and hallucination_safety should be high.
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