import argparse
import json
import os
import time
from typing import Any, Dict, List

from graph import build_graph
from logger import save_run


BENCHMARK_FILE = os.path.join("data", "benchmark_prompts.json")


def load_benchmark_prompts(path: str = BENCHMARK_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        prompts = json.load(file)

    if not isinstance(prompts, list):
        raise ValueError("Benchmark file must contain a list of prompt objects.")

    return prompts


def build_initial_state(prompt_text: str, benchmark_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "original_prompt": prompt_text,
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

        # Benchmark metadata
        "benchmark_id": benchmark_item.get("id", ""),
        "benchmark_category": benchmark_item.get("category", ""),
        "benchmark_difficulty": benchmark_item.get("difficulty", ""),
        "expected_weakness": benchmark_item.get("expected_weakness", ""),
        "evaluation_focus": benchmark_item.get("evaluation_focus", []),
        "ideal_prompt_components": benchmark_item.get("ideal_prompt_components", []),
    }


def run_benchmark(limit: int | None = None, delay_seconds: float = 0.0) -> None:
    prompts = load_benchmark_prompts()

    if limit is not None:
        prompts = prompts[:limit]

    graph = build_graph()

    print("Starting benchmark run")
    print(f"Total prompts to run: {len(prompts)}")
    print("----------------------------------------")

    successful_runs = 0
    failed_runs = 0

    for index, item in enumerate(prompts, start=1):
        prompt_id = item.get("id", f"P{index:03d}")
        category = item.get("category", "unknown")
        prompt_text = item.get("prompt", "")

        print(f"\n[{index}/{len(prompts)}] Running {prompt_id} ({category})")
        print(f"Prompt: {prompt_text}")

        if not prompt_text.strip():
            print("Skipped: empty prompt")
            failed_runs += 1
            continue

        try:
            initial_state = build_initial_state(prompt_text, item)
            final_state = graph.invoke(initial_state)

            # Preserve benchmark metadata after graph execution.
            final_state["benchmark_id"] = item.get("id", "")
            final_state["benchmark_category"] = item.get("category", "")
            final_state["benchmark_difficulty"] = item.get("difficulty", "")
            final_state["expected_weakness"] = item.get("expected_weakness", "")
            final_state["evaluation_focus"] = item.get("evaluation_focus", [])
            final_state["ideal_prompt_components"] = item.get("ideal_prompt_components", [])

            save_run(final_state)

            before_score = final_state.get("evaluation_before", {}).get("overall_score")
            after_score = final_state.get("evaluation_after", {}).get("overall_score")
            improved_prompt = final_state.get("improved_prompt", "")

            print(f"Original score: {before_score}")
            print(f"Improved score: {after_score}")
            print(f"Optimization performed: {bool(improved_prompt)}")

            successful_runs += 1

        except Exception as error:
            print(f"Failed: {error}")
            failed_runs += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print("\n----------------------------------------")
    print("Benchmark finished")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PromptRefiner on benchmark prompts.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of benchmark prompts to run.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay between runs in seconds.",
    )

    args = parser.parse_args()
    run_benchmark(limit=args.limit, delay_seconds=args.delay)