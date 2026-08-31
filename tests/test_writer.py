"""Focused unit tests for the standalone Writer Agent node."""

from copy import deepcopy
from typing import Any

import pytest

from nodes.writer import (
    WriterDraftOutput,
    format_writer_evidence,
    writer_node,
)
from state import ArticleDraft, ArticleIdea, ArticleSection, Source


SELECTED_IDEA = ArticleIdea(
    idea_id="A2",
    title="Questions to Ask Before Changing a Diabetes Medication",
    article_angle=(
        "Explain the tradeoffs patients may need to discuss when cost, side "
        "effects, or treatment burden make a medication change relevant."
    ),
    reason="Medication changes can affect a patient's day-to-day care plan.",
)


def _deep_research_sources() -> list[Source]:
    """Create normalized evidence fixtures without calling You.com."""

    return [
        Source(
            source_id="R1",
            query="diabetes medication changes clinical guidance",
            title="Medication Change Guidance",
            url="https://nih.gov/example-guidance",
            publication_date=None,
            source_type="web",
            highlights=[
                "Medication decisions should account for benefits and risks.",
                "Monitoring needs may differ after a treatment change.",
            ],
        ),
        Source(
            source_id="R2",
            query="diabetes shared decision making medication cost",
            title="Shared Decision-Making Considerations",
            url="https://example.org/shared-decisions",
            publication_date="2026-08-20T10:00:00",
            source_type="news",
            highlights=["Cost can be part of a medication discussion."],
        ),
    ]


VALID_DRAFT = {
    "title": "Questions to Ask Before Changing a Diabetes Medication",
    "sections": [
        {
            "section_id": "introduction",
            "heading": None,
            "content": (
                "Changing medication can involve questions about benefits, "
                "risks, and monitoring. [R1]"
            ),
        },
        {
            "section_id": "questions_for_your_care_team",
            "heading": "Questions for Your Care Team",
            "content": (
                "Ask how the change may affect monitoring and whether cost "
                "should be part of the decision. [R1][R2]"
            ),
        },
    ],
}


class FakeStructuredModel:
    """Return a controlled structured draft instead of calling Nebius."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        assert messages[0][0] == "system"
        assert "The Deep Research sources are the only source" in messages[0][1]
        assert "Return only the structured article output" in messages[0][1]
        assert "editorial direction, not as medical evidence" in messages[0][1]
        assert "Do not create new numerical estimates" in messages[0][1]
        assert "approximately 350 to 500 words" in messages[0][1]
        assert "Return exactly 4 sections" in messages[0][1]
        assert "at most two medical factual sentences" in messages[0][1]
        assert "no more than one quantitative medical outcome" in messages[0][1]
        assert "working title, not a factual claim" in messages[0][1]
        assert "Place each citation immediately" in messages[0][1]
        assert "# Consumer Health Article Style Guide" in messages[0][1]

        writer_input = messages[1][1]
        assert "Idea ID: A2" in writer_input
        assert f"Title: {SELECTED_IDEA.title}" in writer_input
        assert f"Article angle: {SELECTED_IDEA.article_angle}" in writer_input
        assert "# Consumer Health Article Style Guide" not in writer_input
        assert "[R1]" in writer_input
        assert "Title: Medication Change Guidance" in writer_input
        assert "URL: https://nih.gov/example-guidance" in writer_input
        assert "Date: Not available" in writer_input
        assert "- Medication decisions should account for benefits and risks." in writer_input
        assert "- Monitoring needs may differ after a treatment change." in writer_input
        return self.result


class FakeLLM:
    """Mimic only the structured-output methods used by the Writer."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def with_structured_output(
        self,
        schema: type[WriterDraftOutput],
        *,
        method: str,
        strict: bool,
    ) -> FakeStructuredModel:
        assert schema is WriterDraftOutput
        assert method == "json_schema"
        assert strict is True
        return FakeStructuredModel(self.result)


def test_valid_writer_output_creates_structured_draft_and_preserves_sources() -> None:
    sources = _deep_research_sources()
    original_sources = deepcopy(sources)

    update = writer_node(
        {
            "run_id": "run-601",
            "topic": "Type 2 diabetes",
            "selected_idea": SELECTED_IDEA,
            "deep_research_sources": sources,
        },
        llm=FakeLLM(VALID_DRAFT),
    )

    draft = update["draft"]
    assert isinstance(draft, ArticleDraft)
    assert all(isinstance(section, ArticleSection) for section in draft.sections)
    assert draft.sections[0].heading is None
    assert [section.section_id for section in draft.sections] == [
        "introduction",
        "questions_for_your_care_team",
    ]
    assert len({section.section_id for section in draft.sections}) == len(
        draft.sections
    )
    assert "[R1][R2]" in draft.sections[1].content
    assert sources == original_sources
    assert update["workflow_error"] is None


def test_nonexistent_citation_fails_clearly() -> None:
    invalid_draft = deepcopy(VALID_DRAFT)
    invalid_draft["sections"][1]["content"] += " Unsupported citation. [R99]"

    update = writer_node(
        {
            "run_id": "run-602",
            "topic": "Type 2 diabetes",
            "selected_idea": SELECTED_IDEA,
            "deep_research_sources": _deep_research_sources(),
        },
        llm=FakeLLM(invalid_draft),
    )

    assert "draft" not in update
    assert update["workflow_error"].node == "writer"
    assert update["workflow_error"].type == "draft_validation_error"
    assert "R99" in update["workflow_error"].message


def test_missing_selected_idea_fails_before_calling_the_llm() -> None:
    update = writer_node(
        {
            "run_id": "run-603",
            "topic": "Type 2 diabetes",
            "deep_research_sources": _deep_research_sources(),
        }
    )

    assert "draft" not in update
    assert update["workflow_error"].type == "missing_selected_idea"


def test_empty_deep_research_evidence_fails_before_calling_the_llm() -> None:
    update = writer_node(
        {
            "run_id": "run-604",
            "topic": "Type 2 diabetes",
            "selected_idea": SELECTED_IDEA,
            "deep_research_sources": [],
        }
    )

    assert "draft" not in update
    assert update["workflow_error"].type == "missing_deep_research_evidence"


@pytest.mark.parametrize(
    "section_ids",
    [
        ["", "questions_for_your_care_team"],
        ["introduction", "introduction"],
        ["introduction", "Questions-For-Care-Team"],
    ],
    ids=["blank", "duplicate", "not-snake-case"],
)
def test_invalid_section_ids_fail_clearly(section_ids: list[str]) -> None:
    invalid_draft = deepcopy(VALID_DRAFT)
    for section, section_id in zip(invalid_draft["sections"], section_ids):
        section["section_id"] = section_id

    update = writer_node(
        {
            "run_id": "run-605",
            "topic": "Type 2 diabetes",
            "selected_idea": SELECTED_IDEA,
            "deep_research_sources": _deep_research_sources(),
        },
        llm=FakeLLM(invalid_draft),
    )

    assert "draft" not in update
    assert update["workflow_error"].type == "draft_validation_error"


def test_evidence_formatting_contains_source_ids_and_complete_highlights() -> None:
    formatted = format_writer_evidence(_deep_research_sources())

    assert "[R1]" in formatted
    assert "[R2]" in formatted
    assert "Medication decisions should account for benefits and risks." in formatted
    assert "Monitoring needs may differ after a treatment change." in formatted
    assert "Cost can be part of a medication discussion." in formatted
