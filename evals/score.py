import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower()).strip()


def score_task(task: Dict[str, Any], output: str) -> float:
    normalized_output = normalize(output)
    expected = [normalize(x) for x in task["expected"]]

    if task["mode"] == "exact":
        return 1.0 if normalized_output in expected else 0.0

    if task["mode"] == "contains":
        return 1.0 if any(exp in normalized_output for exp in expected) else 0.0

    raise ValueError(f"Unsupported mode: {task['mode']}")


def score_results(tasks_path: Path, outputs_path: Path) -> Dict[str, Any]:
    tasks = json.loads(tasks_path.read_text())
    outputs = json.loads(outputs_path.read_text())

    if len(tasks["tasks"]) != len(outputs["results"]):
        raise ValueError("Number of tasks and results do not match")

    total_weight = sum(task["weight"] for task in tasks["tasks"])
    earned_weight = 0.0
    per_task = []

    for task, result in zip(tasks["tasks"], outputs["results"]):
        score = score_task(task, result["output"])
        earned_weight += score * task["weight"]
        per_task.append(
            {
                "id": task["id"],
                "prompt": task["prompt"],
                "expected": task["expected"],
                "output": result["output"],
                "score": score,
                "weight": task["weight"],
            }
        )

    percentage = (earned_weight / total_weight * 100.0) if total_weight else 0.0
    return {"percentage": round(percentage, 2), "per_task": per_task}


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python evals/score.py <tasks.json> <results.json>")
        sys.exit(1)

    tasks_path = Path(sys.argv[1])
    outputs_path = Path(sys.argv[2])
    result = score_results(tasks_path, outputs_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
