import { useState } from "react";
import "./App.css";

type RiskLevel = "low" | "medium" | "high";

type ClassificationResult = {
  primary_type: string;
  likely_task: string;
  requires_input: boolean;
  is_incomplete: boolean;
  confidence: number;
  reason: string;
};

type RiskReport = {
  overall_risk_level: RiskLevel;
  risks: Record<
    string,
    {
      level: RiskLevel;
      reason: string;
    }
  >;
  main_failure_modes: string[];
  recommendations: string[];
};

type ModelOutput = {
  success: boolean;
  output: string | null;
  error: string | null;
};

type PromptEvaluation = {
  task_clarity: number;
  input_completeness: number;
  specificity: number;
  output_format: number;
  constraint_quality: number;
  hallucination_safety: number;
  model_independence: number;
  overall_score: number;
  main_issues: string[];
  suggestions: string[];
};

type ModelEvaluation = {
  relevance: number;
  instruction_following: number;
  missing_input_handling: number;
  hallucination_safety: number;
  format_compliance: number;
  helpfulness: number;
  overall_score: number;
  main_issue: string;
};

type CrossModelEvaluation = {
  consistency_score: number;
  same_intent: boolean;
  same_output_type: boolean;
  reason: string;
};

type Evaluation = {
  prompt_evaluation: PromptEvaluation;
  model_evaluations: Record<string, ModelEvaluation>;
  cross_model_evaluation: CrossModelEvaluation;
  overall_score: number;
  needs_improvement: boolean;
};

type FinalState = {
  original_prompt: string;
  prompt_type: string;
  classification_result: ClassificationResult;
  risk_report: RiskReport;
  model_outputs_before: Record<string, ModelOutput>;
  evaluation_before: Evaluation;
  improved_prompt: string;
  model_outputs_after: Record<string, ModelOutput>;
  evaluation_after: Evaluation;
};

const mockResult: FinalState = {
  original_prompt: "Make this better",
  prompt_type: "missing_input",
  classification_result: {
    primary_type: "missing_input",
    likely_task: "rewriting",
    requires_input: true,
    is_incomplete: true,
    confidence: 95,
    reason:
      "The prompt uses 'this' to refer to content that should be improved, but no actual content is provided. Once content is provided, the task would be rewriting or improvement.",
  },
  risk_report: {
    overall_risk_level: "high",
    risks: {
      missing_input: {
        level: "high",
        reason:
          "The prompt references missing content, but no text, code, image, or other input is provided.",
      },
      ambiguous_reference: {
        level: "high",
        reason:
          "The word 'this' is unclear because there is no previous object or content to refer to.",
      },
      unclear_task: {
        level: "high",
        reason:
          "'Better' is subjective and does not specify whether the user wants clarity, grammar, tone, structure, or another kind of improvement.",
      },
      missing_output_format: {
        level: "medium",
        reason:
          "The prompt does not say whether the output should include only the improved version or also an explanation of changes.",
      },
      hallucination_risk: {
        level: "high",
        reason:
          "Without content, a model may invent sample content or assume a task that the user did not provide.",
      },
    },
    main_failure_modes: [
      "A model may invent content to improve.",
      "A model may give generic advice instead of completing the task.",
      "Different models may interpret the prompt in different ways.",
    ],
    recommendations: [
      "Provide the actual content that needs improvement.",
      "Specify what aspect should be improved, such as clarity, tone, grammar, or conciseness.",
      "Add the desired output format.",
      "Tell the model not to invent missing content.",
    ],
  },
  model_outputs_before: {
    amazon_nova_lite: {
      success: true,
      output:
        "To improve the image, you could enhance colors, increase sharpness, reduce noise, and improve lighting. If you provide the original image, I can give more specific suggestions.",
      error: null,
    },
    meta_llama_3_8b_instruct: {
      success: true,
      output:
        "I'm happy to help, but I don't see any text to improve. Could you please provide the text you'd like me to make better?",
      error: null,
    },
  },
  evaluation_before: {
    overall_score: 1,
    needs_improvement: true,
    prompt_evaluation: {
      task_clarity: 1,
      input_completeness: 1,
      specificity: 1,
      output_format: 1,
      constraint_quality: 1,
      hallucination_safety: 1,
      model_independence: 1,
      overall_score: 1,
      main_issues: [
        "No content is provided.",
        "The word 'this' is ambiguous.",
        "The prompt does not define what 'better' means.",
        "One model invented an image-improvement task.",
      ],
      suggestions: [
        "Add a content placeholder.",
        "Specify the improvement focus.",
        "Add a missing-input rule.",
      ],
    },
    model_evaluations: {
      amazon_nova_lite: {
        relevance: 1,
        instruction_following: 1,
        missing_input_handling: 1,
        hallucination_safety: 1,
        format_compliance: 2,
        helpfulness: 1,
        overall_score: 1,
        main_issue:
          "The model hallucinated an image-improvement scenario even though no image was provided.",
      },
      meta_llama_3_8b_instruct: {
        relevance: 5,
        instruction_following: 5,
        missing_input_handling: 5,
        hallucination_safety: 5,
        format_compliance: 5,
        helpfulness: 5,
        overall_score: 5,
        main_issue:
          "None. The model correctly asked for missing input instead of inventing content.",
      },
    },
    cross_model_evaluation: {
      consistency_score: 1,
      same_intent: false,
      same_output_type: false,
      reason:
        "The models behaved very differently. Nova invented an image task, while Llama asked for missing text.",
    },
  },
  improved_prompt: `Improve the content provided below while preserving its original meaning.

CONTENT TO IMPROVE:
[PASTE YOUR CONTENT HERE]

IMPROVEMENT FOCUS:
[e.g., clarity, grammar, conciseness, tone, professionalism]

INSTRUCTIONS:
- First check whether content is provided.
- If no content is provided, respond only with: "Please provide the content you would like me to improve."
- Do not invent or assume missing content.
- Preserve the original meaning.
- Return the improved version and a short list of key changes.`,
  model_outputs_after: {
    amazon_nova_lite: {
      success: true,
      output: "Please provide the content you would like me to improve.",
      error: null,
    },
    meta_llama_3_8b_instruct: {
      success: true,
      output: "Please provide the content you would like me to improve.",
      error: null,
    },
  },
  evaluation_after: {
    overall_score: 5,
    needs_improvement: false,
    prompt_evaluation: {
      task_clarity: 5,
      input_completeness: 5,
      specificity: 5,
      output_format: 5,
      constraint_quality: 5,
      hallucination_safety: 5,
      model_independence: 5,
      overall_score: 5,
      main_issues: [],
      suggestions: ["Could add examples of valid improvement focus values."],
    },
    model_evaluations: {
      amazon_nova_lite: {
        relevance: 5,
        instruction_following: 5,
        missing_input_handling: 5,
        hallucination_safety: 5,
        format_compliance: 5,
        helpfulness: 5,
        overall_score: 5,
        main_issue: "None.",
      },
      meta_llama_3_8b_instruct: {
        relevance: 5,
        instruction_following: 5,
        missing_input_handling: 5,
        hallucination_safety: 5,
        format_compliance: 5,
        helpfulness: 5,
        overall_score: 5,
        main_issue: "None.",
      },
    },
    cross_model_evaluation: {
      consistency_score: 5,
      same_intent: true,
      same_output_type: true,
      reason:
        "Both models handled the missing placeholder safely and asked for the missing content.",
    },
  },
};

function formatLabel(value: string) {
  return value.replaceAll("_", " ");
}

function getRiskClass(level: string) {
  return `risk-${level.toLowerCase()}`;
}

function ScoreBadge({ score }: { score: number }) {
  return <span className="score-badge">{score}/5</span>;
}

function RiskBadge({ level }: { level: string }) {
  return <span className={`risk-badge ${getRiskClass(level)}`}>{level}</span>;
}

function OutputCard({
  title,
  output,
}: {
  title: string;
  output: ModelOutput;
}) {
  return (
    <div className="output-card">
      <div className="output-header">
        <h4>{title}</h4>
        <span className={output.success ? "success-pill" : "error-pill"}>
          {output.success ? "Success" : "Error"}
        </span>
      </div>
      <pre>{output.success ? output.output : output.error}</pre>
    </div>
  );
}

function EvaluationCard({
  title,
  evaluation,
}: {
  title: string;
  evaluation: Evaluation;
}) {
  const promptScores = evaluation.prompt_evaluation;

  return (
    <section className="result-card">
      <div className="result-card-header">
        <div>
          <p className="card-kicker">Evaluation</p>
          <h3>{title}</h3>
        </div>
        <div className="score-cluster">
          <ScoreBadge score={evaluation.overall_score} />
          <span
            className={
              evaluation.needs_improvement ? "needs-badge bad" : "needs-badge good"
            }
          >
            {evaluation.needs_improvement ? "Needs improvement" : "Passed"}
          </span>
        </div>
      </div>

      <div className="score-grid">
        {Object.entries(promptScores)
          .filter(
            ([key]) =>
              !["main_issues", "suggestions", "overall_score"].includes(key)
          )
          .map(([key, value]) => (
            <div className="score-item" key={key}>
              <span>{formatLabel(key)}</span>
              <strong>{value as number}/5</strong>
            </div>
          ))}
      </div>

      {promptScores.main_issues.length > 0 && (
        <div className="list-block">
          <h4>Main Issues</h4>
          <ul>
            {promptScores.main_issues.map((issue) => (
              <li key={issue}>{issue}</li>
            ))}
          </ul>
        </div>
      )}

      {promptScores.suggestions.length > 0 && (
        <div className="list-block">
          <h4>Suggestions</h4>
          <ul>
            {promptScores.suggestions.map((suggestion) => (
              <li key={suggestion}>{suggestion}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="model-eval-grid">
        {Object.entries(evaluation.model_evaluations).map(
          ([modelName, modelEval]) => (
            <div className="model-eval-card" key={modelName}>
              <h4>{formatLabel(modelName)}</h4>
              <div className="mini-score-row">
                <span>Overall</span>
                <strong>{modelEval.overall_score}/5</strong>
              </div>
              <p>{modelEval.main_issue || "No major issue detected."}</p>
            </div>
          )
        )}
      </div>

      <div className="cross-card">
        <div>
          <span className="label">Cross-model consistency</span>
          <strong>{evaluation.cross_model_evaluation.consistency_score}/5</strong>
        </div>
        <p>{evaluation.cross_model_evaluation.reason}</p>
      </div>
    </section>
  );
}

function App() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState<FinalState | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    if (!prompt.trim()) {
      alert("Please enter a prompt first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/analyze-prompt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const beforeScore = result?.evaluation_before.overall_score ?? 0;
  const afterScore = result?.evaluation_after.overall_score ?? null;
  const beforeConsistency =
    result?.evaluation_before.cross_model_evaluation.consistency_score ?? 0;
  const afterConsistency =
    result?.evaluation_after.cross_model_evaluation.consistency_score ?? null;

  return (
    <main className="app">
      <div className="shell">
        <section className="hero">
          <p className="eyebrow">BetterPrompts App by Batuhan Baydar & Bahar Akbaş</p>
          <h1>Evaluate and improve prompts across multiple LLMs.</h1>
          <p className="subtitle">
            Test whether a prompt behaves consistently across different models.
            The system classifies the prompt, detects risks, evaluates outputs,
            and improves the prompt when needed.
          </p>
        </section>

        <section className="layout">
          <aside className="info-panel">
            <h3>Pipeline</h3>
            <p>
              The backend runs a multi-step LangGraph workflow using two test
              models and a separate judge/optimizer model.
            </p>

            <div className="pipeline-list">
              <div className="pipeline-item">1. Classify prompt</div>
              <div className="pipeline-item">2. Detect risks</div>
              <div className="pipeline-item">3. Run test models</div>
              <div className="pipeline-item">4. Evaluate outputs</div>
              <div className="pipeline-item">5. Optimize if needed</div>
            </div>
          </aside>

          <section className="prompt-card">
            <div className="card-header">
              <div>
                <h2>Prompt Input</h2>
                <p>Write the prompt you want to test.</p>
              </div>
              <span className="status-badge">Ready</span>
            </div>

            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Example: Propose a solution."
              rows={8}
            />

            <div className="actions">
              <p className="hint">{prompt.length} characters</p>
              <button onClick={handleRun} disabled={loading}>
                {loading ? "Running..." : "Run"}
              </button>
            </div>
          </section>
        </section>
        {error && <p className="error-message">{error}</p>}

        {result && (
          <section className="results-section">
            <div className="section-heading">
              <p className="eyebrow small">Analysis Result</p>
              <h2>BetterPrompts Report</h2>
              <p>
                Below is the full pipeline result, including classification,
                risks, model outputs, evaluations, optimization, and final
                comparison.
              </p>
            </div>

            <section className="result-card">
              <div className="result-card-header">
                <div>
                  <p className="card-kicker">Step 1</p>
                  <h3>Prompt Classification</h3>
                </div>
                <RiskBadge
                  level={
                    result.classification_result.is_incomplete ? "high" : "low"
                  }
                />
              </div>

              <div className="classification-grid">
                <div className="mini-card">
                  <span className="label">Prompt Type</span>
                  <strong>{formatLabel(result.classification_result.primary_type)}</strong>
                </div>
                <div className="mini-card">
                  <span className="label">Likely Task</span>
                  <strong>{formatLabel(result.classification_result.likely_task)}</strong>
                </div>
                <div className="mini-card">
                  <span className="label">Incomplete</span>
                  <strong>
                    {result.classification_result.is_incomplete ? "Yes" : "No"}
                  </strong>
                </div>
                <div className="mini-card">
                  <span className="label">Confidence</span>
                  <strong>{result.classification_result.confidence}%</strong>
                </div>
              </div>

              <div className="reason-box">
                <span className="label">Reason</span>
                <p>{result.classification_result.reason}</p>
              </div>
            </section>

            <section className="result-card">
              <div className="result-card-header">
                <div>
                  <p className="card-kicker">Step 2</p>
                  <h3>Risk Report</h3>
                </div>
                <RiskBadge level={result.risk_report.overall_risk_level} />
              </div>

              <div className="risk-grid">
                {Object.entries(result.risk_report.risks).map(
                  ([riskName, riskData]) => (
                    <div className="risk-item" key={riskName}>
                      <div className="risk-item-header">
                        <strong>{formatLabel(riskName)}</strong>
                        <RiskBadge level={riskData.level} />
                      </div>
                      <p>{riskData.reason}</p>
                    </div>
                  )
                )}
              </div>

              <div className="two-column">
                <div className="list-block">
                  <h4>Main Failure Modes</h4>
                  <ul>
                    {result.risk_report.main_failure_modes.map((mode) => (
                      <li key={mode}>{mode}</li>
                    ))}
                  </ul>
                </div>

                <div className="list-block">
                  <h4>Recommendations</h4>
                  <ul>
                    {result.risk_report.recommendations.map((rec) => (
                      <li key={rec}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </section>

            <section className="result-card">
              <div className="result-card-header">
                <div>
                  <p className="card-kicker">Step 3</p>
                  <h3>Model Outputs Before Improvement</h3>
                </div>
              </div>

              <div className="outputs-grid">
                <OutputCard
                  title="Amazon Nova Lite"
                  output={result.model_outputs_before.amazon_nova_lite}
                />
                <OutputCard
                  title="Meta Llama 3 8B Instruct"
                  output={result.model_outputs_before.meta_llama_3_8b_instruct}
                />
              </div>
            </section>

            <EvaluationCard
              title="Evaluation Before Improvement"
              evaluation={result.evaluation_before}
            />

            {result.improved_prompt && (
              <>
                <section className="result-card">
                  <div className="result-card-header">
                    <div>
                      <p className="card-kicker">Step 5</p>
                      <h3>Improved Prompt</h3>
                    </div>
                    <span className="success-pill">Optimized</span>
                  </div>

                  <pre className="prompt-template">{result.improved_prompt}</pre>
                </section>

                <section className="result-card">
                  <div className="result-card-header">
                    <div>
                      <p className="card-kicker">Step 6</p>
                      <h3>Model Outputs After Improvement</h3>
                    </div>
                  </div>

                  <div className="outputs-grid">
                    <OutputCard
                      title="Amazon Nova Lite"
                      output={result.model_outputs_after.amazon_nova_lite}
                    />
                    <OutputCard
                      title="Meta Llama 3 8B Instruct"
                      output={
                        result.model_outputs_after.meta_llama_3_8b_instruct
                      }
                    />
                  </div>
                </section>

                <EvaluationCard
                  title="Evaluation After Improvement"
                  evaluation={result.evaluation_after}
                />
              </>
            )}

            <section className="comparison-card">
              <div>
                <p className="card-kicker">Final Step</p>
                <h3>Before / After Comparison</h3>
              </div>

              <div className="comparison-grid">
                <div>
                  <span>Original Score</span>
                  <strong>{beforeScore}/5</strong>
                </div>
                <div>
                  <span>Improved Score</span>
                  <strong>{afterScore !== null ? `${afterScore}/5` : "N/A"}</strong>
                </div>
                <div>
                  <span>Score Change</span>
                  <strong>
                    {afterScore !== null
                      ? `${(afterScore - beforeScore).toFixed(1)}`
                      : "N/A"}
                  </strong>
                </div>
                <div>
                  <span>Consistency Before</span>
                  <strong>{beforeConsistency}/5</strong>
                </div>
                <div>
                  <span>Consistency After</span>
                  <strong>
                    {afterConsistency !== null ? `${afterConsistency}/5` : "N/A"}
                  </strong>
                </div>
                <div>
                  <span>Final Decision</span>
                  <strong className="decision">Improved prompt accepted</strong>
                </div>
              </div>
            </section>
          </section>
        )}
      </div>
    </main>
  );
}

export default App;