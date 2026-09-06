"""Offline tests for complete end-of-run checkpoint-history export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.types import Command

from graph import build_workflow
from persistence import export_run_history
from state import ArticleDraft, ArticleIdea, ArticleSection, GraphState, Source


IDEAS = [
    ArticleIdea(
        idea_id=f"A{number}",
        title=f"Idea {number}",
        article_angle=f"Angle {number}",
        reason=f"Reason {number}",
    )
    for number in range(1, 4)
]


def _source(source_id: str, query: str, highlight: str) -> Source:
    return Source(
        source_id=source_id,
        query=query,
        title=f"Source {source_id}",
        url=f"https://example.com/{source_id.lower()}",
        publication_date=None,
        source_type="web",
        highlights=[highlight],
    )


def _draft(version: int) -> ArticleDraft:
    return ArticleDraft(
        title="Test article",
        sections=[
            ArticleSection(
                section_id="introduction",
                heading=None,
                content=f"Introduction version {version}. [R1]",
            ),
            ArticleSection(
                section_id="practical_steps",
                heading="Practical steps",
                content=f"Practical section version {version}. [R1]",
            ),
        ],
    )


def test_export_preserves_ordered_drafts_reviews_and_nested_models(
    tmp_path: Path,
) -> None:
    """Exercise only mocked nodes; no Nebius or You.com client is called."""

    revision_number = 0
    review_number = 0

    def discovery(state: GraphState) -> dict[str, Any]:
        return {
            "discovery_queries": ["d1", "d2", "d3"],
            "discovery_sources": [_source("D1", "d1", "Discovery evidence")],
            "workflow_error": None,
        }

    def opportunity(state: GraphState) -> dict[str, Any]:
        return {"article_ideas": IDEAS, "workflow_error": None}

    def research(state: GraphState) -> dict[str, Any]:
        return {
            "deep_research_queries": ["r1", "r2", "r3"],
            "deep_research_sources": [
                _source("R1", "r1", "Full research Highlight")
            ],
            "workflow_error": None,
        }

    def writer(state: GraphState) -> dict[str, Any]:
        return {"draft": _draft(0), "workflow_error": None}

    def writer_revision(state: GraphState) -> dict[str, Any]:
        nonlocal revision_number
        revision_number += 1
        return {"draft": _draft(revision_number), "workflow_error": None}

    def reviewer(state: GraphState) -> dict[str, Any]:
        nonlocal review_number
        review_number += 1
        if review_number <= 2:
            return {
                "review_status": "FAIL",
                "failure_type": "style_or_readability",
                "flagged_sections": ["practical_steps"],
                "review_feedback": f"Revision feedback {review_number}",
                "workflow_error": None,
            }
        return {
            "review_status": "PASS",
            "failure_type": None,
            "flagged_sections": [],
            "review_feedback": None,
            "workflow_error": None,
        }

    workflow = build_workflow(
        discovery=discovery,
        opportunity=opportunity,
        deep_research=research,
        writer=writer,
        writer_revision=writer_revision,
        reviewer=reviewer,
    )
    run_id = "persistence-test"
    config = {"configurable": {"thread_id": run_id}}
    workflow.invoke(
        {"run_id": run_id, "topic": "Test topic", "retry_count": 0},
        config,
    )
    workflow.invoke(Command(resume="A1"), config)

    output_path = export_run_history(
        workflow,
        config,
        output_directory=tmp_path,
    )
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == tmp_path / "persistence-test.json"
    assert exported["schema_version"] == "1.0"
    assert [entry["draft_type"] for entry in exported["draft_history"]] == [
        "initial",
        "revision",
        "revision",
    ]
    assert [
        entry["sections"][0]["content"]
        for entry in exported["draft_history"]
    ] == [
        "Introduction version 0. [R1]",
        "Introduction version 1. [R1]",
        "Introduction version 2. [R1]",
    ]
    assert [
        entry["review_status"] for entry in exported["reviewer_history"]
    ] == ["FAIL", "FAIL", "PASS"]
    assert [
        entry["retry_count"] for entry in exported["reviewer_history"]
    ] == [0, 1, 2]
    assert exported["discovery_sources"][0]["highlights"] == [
        "Discovery evidence"
    ]
    assert exported["deep_research_sources"][0]["highlights"] == [
        "Full research Highlight"
    ]
    assert exported["final_draft"]["sections"][1]["section_id"] == (
        "practical_steps"
    )
    assert exported["final_status"] == "completed"
