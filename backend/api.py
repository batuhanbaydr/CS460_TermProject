from summary import build_experiment_summary
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import build_graph
from logger import save_run


app = FastAPI(title="PromptRefiner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzePromptRequest(BaseModel):
    prompt: str


@app.get("/")
def root():
    return {"message": "PromptRefiner API is running"}

@app.get("/experiment-summary")
def get_experiment_summary():
    return build_experiment_summary()

@app.post("/analyze-prompt")
def analyze_prompt(request: AnalyzePromptRequest):
    graph = build_graph()

    initial_state = {
        "original_prompt": request.prompt,
        "task_type": "general",
        "classification_result": {},
        "prompt_type": "",
        "risk_report": {},
        "model_outputs_before": {},
        "evaluation_before": {},
        "improved_prompt": "",
        "model_outputs_after": {},
        "evaluation_after": {},
        "final_report": "",
    }

    final_state = graph.invoke(initial_state)
    save_run(final_state)

    return final_state