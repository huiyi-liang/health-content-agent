"""End-of-run JSON export built from LangGraph checkpoint history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import BaseModel


SCHEMA_VERSION = "1.0"
DEFAULT_RUNS_DIRECTORY = Path(__file__).resolve().parent / "data" / "runs"
TERMINAL_STATUSES = {"completed", "human_review_required", "failed"}
REVIEW_DESTINATIONS = {
    "increment_retry",
    "completed",
    "human_review_required",
}


class RunExportError(RuntimeError):
    """Raised when checkpoint history cannot produce a valid run export."""


def build_run_export(snapshots: Iterable[Any]) -> dict[str, Any]:
    """Build one JSON-safe evaluation record from checkpoint snapshots."""

    ordered_snapshots = sorted(
        snapshots,
        key=lambda snapshot: snapshot.metadata.get("step", -1),
    )
    if not ordered_snapshots:
        raise RunExportError("No checkpoint history exists for this run.")

    final_state = ordered_snapshots[-1].values
    final_status = final_state.get("final_status")
    if final_status not in TERMINAL_STATUSES:
        raise RunExportError("The workflow has not reached a terminal status.")

    draft_history: list[dict[str, Any]] = []
    reviewer_history: list[dict[str, Any]] = []
    workflow_errors: list[dict[str, Any]] = []
    previous_error: dict[str, Any] | None = None

    for snapshot in ordered_snapshots:
        state = snapshot.values
        next_nodes = set(snapshot.next)

        # A Writer checkpoint points next to Reviewer. This captures the
        # initial draft and every later full draft produced by either retry path.
        if next_nodes == {"reviewer"} and state.get("draft") is not None:
            draft = _json_safe(state["draft"])
            draft_history.append(
                {
                    "draft_sequence_number": len(draft_history) + 1,
                    "draft_type": "initial" if not draft_history else "revision",
                    "title": draft["title"],
                    "sections": draft["sections"],
                }
            )

        # These destinations occur immediately after a valid Reviewer result.
        if next_nodes.intersection(REVIEW_DESTINATIONS):
            review_status = state.get("review_status")
            if review_status in {"PASS", "FAIL"}:
                reviewer_history.append(
                    {
                        "review_sequence_number": len(reviewer_history) + 1,
                        "review_status": review_status,
                        "failure_type": state.get("failure_type"),
                        "flagged_sections": list(
                            state.get("flagged_sections", [])
                        ),
                        "review_feedback": state.get("review_feedback"),
                        "retry_count": state.get("retry_count", 0),
                    }
                )

        error = state.get("workflow_error")
        serialized_error = _json_safe(error) if error is not None else None
        if serialized_error is not None and serialized_error != previous_error:
            workflow_errors.append(serialized_error)
        previous_error = serialized_error

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": final_state["run_id"],
        "topic": final_state["topic"],
        "discovery_queries": list(final_state.get("discovery_queries", [])),
        "discovery_sources": _json_safe(
            final_state.get("discovery_sources", [])
        ),
        "article_ideas": _json_safe(final_state.get("article_ideas", [])),
        "selected_idea": _json_safe(final_state.get("selected_idea")),
        "deep_research_queries": list(
            final_state.get("deep_research_queries", [])
        ),
        "deep_research_sources": _json_safe(
            final_state.get("deep_research_sources", [])
        ),
        "draft_history": draft_history,
        "reviewer_history": reviewer_history,
        "retry_count": final_state.get("retry_count", 0),
        "workflow_errors": workflow_errors,
        "final_draft": _json_safe(final_state.get("draft")),
        "final_status": final_status,
    }


def export_run_history(
    workflow: Any,
    config: Mapping[str, Any],
    *,
    output_directory: Path = DEFAULT_RUNS_DIRECTORY,
) -> Path:
    """Write checkpoint history to data/runs/<run_id>.json and return its path."""

    export = build_run_export(workflow.get_state_history(config))
    run_id = export["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise RunExportError("A non-empty run_id is required for export.")
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise RunExportError("run_id cannot contain path separators.")

    output_path = output_directory / f"{run_id}.json"
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(export, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError) as exc:
        raise RunExportError(f"Could not write run export: {output_path}") from exc
    return output_path


def _json_safe(value: Any) -> Any:
    """Convert workflow Pydantic values and containers to JSON-safe values."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
