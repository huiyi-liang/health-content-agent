"""Upload one completed run and execute basic deterministic LangSmith checks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from langsmith import Client

from config import load_project_environment


CITATION_PATTERN = re.compile(r"\[(R\d+)\]")
DEFAULT_DATASET = "health-content-agent-eval"


def exported_run_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return the saved run that this lightweight experiment evaluates."""

    return inputs["run_record"]


def workflow_completed(outputs: dict[str, Any]) -> bool:
    """Check whether automation ended with a Reviewer PASS."""

    return outputs.get("final_status") == "completed"


def retry_limit_respected(outputs: dict[str, Any]) -> bool:
    """Check the deterministic three-correction-cycle safety limit."""

    retry_count = outputs.get("retry_count")
    return isinstance(retry_count, int) and 0 <= retry_count <= 3


def citation_ids_exist(outputs: dict[str, Any]) -> bool:
    """Check that final-draft R# citations refer to exported research sources."""

    draft = outputs.get("final_draft")
    if not isinstance(draft, dict):
        return False

    source_ids = {
        source.get("source_id")
        for source in outputs.get("deep_research_sources", [])
        if isinstance(source, dict)
    }
    article_text = "\n".join(
        str(section.get("content", ""))
        for section in draft.get("sections", [])
        if isinstance(section, dict)
    )
    cited_ids = set(CITATION_PATTERN.findall(article_text))
    return bool(cited_ids) and cited_ids.issubset(source_ids)


def expected_history_exists(outputs: dict[str, Any]) -> bool:
    """Check core history shape without claiming medical or editorial accuracy."""

    return (
        len(outputs.get("discovery_queries", [])) == 3
        and len(outputs.get("article_ideas", [])) == 3
        and outputs.get("selected_idea") is not None
        and len(outputs.get("deep_research_queries", [])) >= 3
        and bool(outputs.get("draft_history"))
        and bool(outputs.get("reviewer_history"))
    )


def load_run_record(path: Path) -> dict[str, Any]:
    """Read and minimally validate one exported workflow JSON file."""

    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not record.get("run_id"):
        raise ValueError("The run export must be a JSON object with a run_id.")
    return record


def run_evaluation(
    run_file: Path,
    *,
    dataset_name: str = DEFAULT_DATASET,
) -> None:
    """Upload one run example and record deterministic scores in LangSmith."""

    load_project_environment()
    record = load_run_record(run_file)
    client = Client()

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Completed health-content-agent runs used for structural baseline "
                "evaluation. This is not yet a medical-accuracy golden dataset."
            ),
        )

    example = client.create_example(
        dataset_id=dataset.id,
        inputs={"run_record": record},
        metadata={"run_id": record["run_id"], "topic": record.get("topic")},
    )
    client.evaluate(
        exported_run_target,
        data=[example],
        evaluators=[
            workflow_completed,
            retry_limit_respected,
            citation_ids_exist,
            expected_history_exists,
        ],
        experiment_prefix="health-content-agent-eval",
        max_concurrency=1,
        metadata={"evaluation_type": "deterministic_structural_baseline"},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one exported workflow run in LangSmith."
    )
    parser.add_argument("run_file", type=Path)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_evaluation(args.run_file, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
