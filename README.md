# Using LLMs for Prompt Engineering

This project is a Python-based multi-LLM prompt evaluation and improvement tool. The project’s goal is to test whether a user-written prompt works consistently across different LLMs and, if not, generate a clearer and more model-independent version.

This project is developed for the topic “Using LLMs for Prompt Engineering.”

## Current Status

We implemented the core backend pipeline. The current system can:

* take a user prompt from the terminal,
* send the same prompt to two different LLMs,
* evaluate the prompt and model outputs using a stronger judge model,
* detect vague, incomplete, or inconsistent prompts,
* generate an improved prompt template when needed,
* retest the improved prompt,
* show a before/after comparison.

## Models Used

We use AWS Bedrock models.

Test models:

* Amazon Nova Lite
* Meta Llama 3 8B Instruct

Judge / optimizer model:

* Claude Sonnet 4.5

Nova Lite and Llama are used to test whether the same prompt works across different model families. Claude Sonnet 4.5 is used separately for evaluation and optimization to reduce bias.

## Workflow

The workflow is implemented with LangGraph:

User prompt

↓

Run prompt on Nova Lite and Llama 3 8B

↓

Evaluate prompt and both outputs with Claude Sonnet 4.5

↓

If the prompt is good, stop

↓

If the prompt needs improvement, generate an improved prompt, retest it, and compare before/after results.

## Evaluation System

The evaluator checks three levels:

1. Prompt-level evaluation: clarity, completeness, specificity, output format, constraints, hallucination safety, and model independence.
2. Per-model evaluation: relevance, instruction following, missing-input handling, hallucination safety, format compliance, and helpfulness.
3. Cross-model evaluation: whether both models understood the prompt similarly and produced consistent outputs.
