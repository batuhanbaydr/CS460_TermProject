import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


RESULTS_FILE = os.path.join("results", "experiment_results.jsonl")


def load_jsonl(path: str = RESULTS_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def get_score(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        possible_keys = [
            "overall_score",
            "score",
            "prompt_score",
            "quality_score",
            "final_score",
        ]

        for key in possible_keys:
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])

    return None


def average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def build_experiment_summary(path: str = RESULTS_FILE) -> Dict[str, Any]:
    records = load_jsonl(path)

    original_scores = []
    improved_scores = []
    score_gains = []

    category_stats = defaultdict(lambda: {
        "count": 0,
        "original_scores": [],
        "improved_scores": [],
        "score_gains": [],
    })

    difficulty_stats = defaultdict(lambda: {
        "count": 0,
        "original_scores": [],
        "improved_scores": [],
        "score_gains": [],
    })

    optimized_count = 0

    for record in records:
        before_score = get_score(record.get("evaluation_before"))
        after_score = get_score(record.get("evaluation_after"))

        category = record.get("benchmark_category", "unknown") or "unknown"
        difficulty = record.get("benchmark_difficulty", "unknown") or "unknown"

        improved_prompt = record.get("improved_prompt", "")
        if improved_prompt:
            optimized_count += 1

        category_stats[category]["count"] += 1
        difficulty_stats[difficulty]["count"] += 1

        if before_score is not None:
            original_scores.append(before_score)
            category_stats[category]["original_scores"].append(before_score)
            difficulty_stats[difficulty]["original_scores"].append(before_score)

        if after_score is not None:
            improved_scores.append(after_score)
            category_stats[category]["improved_scores"].append(after_score)
            difficulty_stats[difficulty]["improved_scores"].append(after_score)

        if before_score is not None and after_score is not None:
            gain = after_score - before_score
            score_gains.append(gain)
            category_stats[category]["score_gains"].append(gain)
            difficulty_stats[difficulty]["score_gains"].append(gain)

    category_breakdown = {}
    for category, stats in category_stats.items():
        category_breakdown[category] = {
            "count": stats["count"],
            "average_original_score": average(stats["original_scores"]),
            "average_improved_score": average(stats["improved_scores"]),
            "average_score_gain": average(stats["score_gains"]),
        }

    difficulty_breakdown = {}
    for difficulty, stats in difficulty_stats.items():
        difficulty_breakdown[difficulty] = {
            "count": stats["count"],
            "average_original_score": average(stats["original_scores"]),
            "average_improved_score": average(stats["improved_scores"]),
            "average_score_gain": average(stats["score_gains"]),
        }

    return {
        "total_runs": len(records),
        "optimized_runs": optimized_count,
        "average_original_score": average(original_scores),
        "average_improved_score": average(improved_scores),
        "average_score_gain": average(score_gains),
        "category_breakdown": category_breakdown,
        "difficulty_breakdown": difficulty_breakdown,
    }


if __name__ == "__main__":
    summary = build_experiment_summary()
    print(json.dumps(summary, indent=2))