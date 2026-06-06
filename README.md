# Using LLMs for Prompt Engineering

This project is a multi-LLM prompt evaluation and improvement tool. The goal is to test whether a user-written prompt works consistently across different LLMs and, if needed, generate a clearer and more model-independent version.

The project is developed for the topic “Using LLMs for Prompt Engineering.”

## Current Status

We implemented both the backend pipeline and a React demo interface.

The current system can:

* take a user prompt from the frontend or terminal,
* classify the prompt type,
* detect prompt risks,
* send the same prompt to two different LLMs,
* evaluate the prompt and model outputs using a stronger judge model,
* identify vague, incomplete, risky, or inconsistent prompts,
* generate an improved prompt when needed,
* retest the improved prompt,
* compare the original and improved prompt,
* save experiment results.

## Models Used

We use AWS Bedrock models.

Test models:

* Amazon Nova Lite
* Meta Llama 3 8B Instruct

Judge / optimizer model:

* Claude Sonnet 4.5

Nova Lite and Llama are used to test whether the same prompt works across different model families. Claude Sonnet 4.5 is used separately for classification, risk analysis, evaluation, and optimization.

## Workflow

The workflow is implemented with LangGraph.

User prompt

↓

Classify prompt

↓

Detect risks

↓

Run prompt on Nova Lite and Llama 3 8B

↓

Evaluate prompt and both model outputs with Claude Sonnet 4.5

↓

If the prompt is good, stop

↓

If the prompt needs improvement, generate an improved prompt

↓

Run the improved prompt on both models

↓

Evaluate the improved outputs

↓

Compare original vs improved prompt

## Evaluation System

The evaluator checks three levels:

1. Prompt-level evaluation: clarity, completeness, specificity, output format, constraints, hallucination safety, and model independence.
2. Per-model evaluation: relevance, instruction following, missing-input handling, hallucination safety, format compliance, and helpfulness.
3. Cross-model evaluation: whether both models understood the prompt similarly and produced consistent outputs.

## Risk Detection

The risk detector checks whether the prompt may fail because of:

* missing input,
* ambiguous references such as “this” or “it,”
* unclear task definition,
* missing output format,
* missing constraints,
* hallucination risk,
* cross-model inconsistency risk,
* format failure risk.

The risk report is used by the evaluator and optimizer to make better decisions.

## Optimizer

The optimizer improves weak prompts by adding:

* clearer task instructions,
* missing input placeholders when needed,
* output format requirements,
* constraints such as tone, length, audience, or scope,
* rules to prevent hallucination,
* model-independent structure.

The optimizer also preserves concrete details from the original prompt. For example, if the original prompt already says “Generate 5 brand name ideas for an eco-friendly water bottle company,” the optimizer keeps those details and only adds structure and constraints.

## Frontend

The project includes a React frontend demo.

The frontend shows:

* prompt input panel,
* prompt classification,
* risk report,
* model outputs before improvement,
* evaluation before improvement,
* improved prompt,
* model outputs after improvement,
* evaluation after improvement,
* before/after comparison.

## Project Structure

CS460_TermProject/

backend/

* api.py
* main.py
* graph.py
* state.py
* models.py
* runner.py
* classifier.py
* risk_detector.py
* evaluator.py
* optimizer.py
* logger.py
* requirements.txt

frontend/

* src/
* package.json
* vite.config.ts

README.md

.gitignore

## How to Run

Backend:

cd backend

uvicorn api:app --reload --port 8000

Frontend:

cd frontend

npm install

npm run dev

Then open:

http://localhost:5173

## Example Test Prompts

Good prompt:

Summarize the benefits of exercise in exactly 10 bullet points.

Weak prompt:

Make this better.

Weak prompt:

Propose a solution.

Creative prompt:

Generate 5 brand name ideas for an eco-friendly water bottle company.

## Current Limitations

* The system depends on AWS Bedrock access.
* LLM outputs may vary between runs.
* The frontend is a demo interface, not a production application.
* The evaluator is also LLM-based, so its judgments may not always be perfectly consistent.
* More benchmark prompts are needed for systematic testing.
