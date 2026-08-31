"""Focused unit tests for the standalone Opportunity Agent node."""

from copy import deepcopy
from typing import Any

import pytest

from nodes.opportunity import OpportunityIdeaPlan, opportunity_agent_node
from state import ArticleIdea, Source


VALID_IDEAS = [
    {
        "title": "Why Diabetes Burnout Can Disrupt Daily Care",
        "article_angle": (
            "Explain diabetes burnout, how it can affect routines, and practical "
            "ways patients can discuss support with their care team."
        ),
        "reason": (
            "Daily management can become emotionally exhausting and interfere "
            "with tasks patients need to repeat every day."
        ),
    },
    {
        "title": "Questions to Ask Before Changing a Diabetes Medication",
        "article_angle": (
            "Cover the tradeoffs patients may want to discuss when cost, side "
            "effects, or treatment burden make a medication change relevant."
        ),
        "reason": (
            "Patients may face choices that affect both glucose management and "
            "their ability to follow a treatment plan."
        ),
    },
    {
        "title": "How Poor Sleep Can Complicate Type 2 Diabetes Management",
        "article_angle": (
            "Explore the two-way relationship between sleep problems and daily "
            "diabetes management, including what patients can track and discuss."
        ),
        "reason": (
            "Sleep difficulties can shape how patients feel and manage their "
            "condition from one day to the next."
        ),
    },
]


def _discovery_sources() -> list[Source]:
    """Create small normalized evidence fixtures without calling You.com."""

    return [
        Source(
            source_id="D1",
            query="type 2 diabetes daily management patient experiences",
            title="Living With Daily Diabetes Care",
            url="https://example.org/daily-care",
            publication_date=None,
            source_type="web",
            highlights=["Exact web Highlight.", "A second exact Highlight."],
        ),
        Source(
            source_id="D2",
            query="recent type 2 diabetes developments patient impact",
            title="A Recent Diabetes Development",
            url="https://news.example.org/development",
            publication_date="2026-08-20T10:00:00",
            source_type="news",
            highlights=["Exact news Highlight."],
        ),
    ]


class FakeStructuredModel:
    """Return a controlled structured response instead of calling Nebius."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        assert messages[0][0] == "system"
        assert "establish unverified medical facts" in messages[0][1]
        assert "Avoid causal headline words" in messages[0][1]
        assert messages[1][0] == "human"
        assert "Health topic: Type 2 diabetes" in messages[1][1]
        assert "[D1]" in messages[1][1]
        assert "Query: type 2 diabetes daily management patient experiences" in messages[1][1]
        assert "Title: Living With Daily Diabetes Care" in messages[1][1]
        assert "URL: https://example.org/daily-care" in messages[1][1]
        assert "Type: web" in messages[1][1]
        assert "Date: Not available" in messages[1][1]
        assert "- Exact web Highlight." in messages[1][1]
        assert "- A second exact Highlight." in messages[1][1]
        return self.result


class FakeLLM:
    """Mimic only the LangChain methods used by the Opportunity Agent."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def with_structured_output(
        self,
        schema: type[OpportunityIdeaPlan],
        *,
        method: str,
        strict: bool,
    ) -> FakeStructuredModel:
        assert schema is OpportunityIdeaPlan
        assert method == "json_schema"
        assert strict is True
        return FakeStructuredModel(self.result)


def test_valid_response_creates_three_article_ideas_without_changing_sources() -> None:
    sources = _discovery_sources()
    original_sources = deepcopy(sources)

    update = opportunity_agent_node(
        {
            "run_id": "run-004",
            "topic": "Type 2 diabetes",
            "discovery_sources": sources,
        },
        llm=FakeLLM({"ideas": VALID_IDEAS}),
    )

    ideas = update["article_ideas"]
    assert len(ideas) == 3
    assert all(isinstance(idea, ArticleIdea) for idea in ideas)
    assert [idea.idea_id for idea in ideas] == ["A1", "A2", "A3"]
    assert [idea.title for idea in ideas] == [idea["title"] for idea in VALID_IDEAS]
    assert [idea.article_angle for idea in ideas] == [
        idea["article_angle"] for idea in VALID_IDEAS
    ]
    assert [idea.reason for idea in ideas] == [idea["reason"] for idea in VALID_IDEAS]
    assert sources == original_sources
    assert update["workflow_error"] is None


def test_empty_discovery_evidence_fails_before_calling_the_llm() -> None:
    update = opportunity_agent_node(
        {
            "run_id": "run-005",
            "topic": "Type 2 diabetes",
            "discovery_sources": [],
        }
    )

    assert "article_ideas" not in update
    assert update["workflow_error"].node == "opportunity_agent"
    assert update["workflow_error"].type == "missing_discovery_evidence"


@pytest.mark.parametrize(
    "ideas",
    [
        VALID_IDEAS[:2],
        [*VALID_IDEAS, VALID_IDEAS[0]],
        [{**VALID_IDEAS[0], "title": "   "}, *VALID_IDEAS[1:]],
        [{**VALID_IDEAS[0], "article_angle": "   "}, *VALID_IDEAS[1:]],
        [{**VALID_IDEAS[0], "reason": "   "}, *VALID_IDEAS[1:]],
    ],
    ids=[
        "fewer-than-three",
        "more-than-three",
        "blank-title",
        "blank-article-angle",
        "blank-reason",
    ],
)
def test_invalid_structured_idea_output_fails_clearly(
    ideas: list[dict[str, str]],
) -> None:
    update = opportunity_agent_node(
        {
            "run_id": "run-006",
            "topic": "Type 2 diabetes",
            "discovery_sources": _discovery_sources(),
        },
        llm=FakeLLM({"ideas": ideas}),
    )

    assert "article_ideas" not in update
    assert update["workflow_error"].node == "opportunity_agent"
    assert update["workflow_error"].type == "idea_generation_error"
