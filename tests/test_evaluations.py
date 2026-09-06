"""Offline tests for the deterministic LangSmith evaluation functions."""

import json

from evaluations import (
    citation_ids_exist,
    expected_history_exists,
    load_run_record,
    retry_limit_respected,
    workflow_completed,
)


def _run_record() -> dict:
    return {
        "run_id": "run-1",
        "topic": "Type 2 diabetes",
        "discovery_queries": ["d1", "d2", "d3"],
        "article_ideas": [{"idea_id": "A1"}] * 3,
        "selected_idea": {"idea_id": "A1"},
        "deep_research_queries": ["r1", "r2", "r3"],
        "deep_research_sources": [{"source_id": "R1"}],
        "draft_history": [{"draft_type": "initial"}],
        "reviewer_history": [{"review_status": "PASS"}],
        "retry_count": 1,
        "final_draft": {
            "title": "Article",
            "sections": [{"section_id": "intro", "content": "Claim. [R1]"}],
        },
        "final_status": "completed",
    }


def test_structural_evaluators_accept_valid_export() -> None:
    record = _run_record()

    assert workflow_completed(record)
    assert retry_limit_respected(record)
    assert citation_ids_exist(record)
    assert expected_history_exists(record)


def test_citation_evaluator_rejects_unknown_source_id() -> None:
    record = _run_record()
    record["final_draft"]["sections"][0]["content"] = "Claim. [R99]"

    assert not citation_ids_exist(record)


def test_load_run_record_reads_valid_json(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_run_record()), encoding="utf-8")

    assert load_run_record(path)["run_id"] == "run-1"
