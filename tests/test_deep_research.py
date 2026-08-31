"""Focused unit tests for human selection and standalone Deep Research."""

from copy import deepcopy
from datetime import date
from typing import Any, Sequence

import pytest
from pydantic import ValidationError

from config import PREFERRED_MEDICAL_DOMAINS
from nodes.opportunity import human_selection_node, select_article_idea
from nodes.research import (
    DeepResearchQueryPlan,
    deep_research_node,
    plan_deep_research_queries,
)
from state import ArticleIdea


SELECTED_IDEA = ArticleIdea(
    idea_id="A2",
    title="Questions to Ask Before Changing a Diabetes Medication",
    article_angle=(
        "Explain the tradeoffs patients may need to discuss when cost, side "
        "effects, or treatment burden make a medication change relevant."
    ),
    reason=(
        "Medication changes can affect both glucose management and a patient's "
        "ability to follow the treatment plan."
    ),
)

ARTICLE_IDEAS = [
    ArticleIdea(
        idea_id="A1",
        title="Understanding Diabetes Burnout",
        article_angle="Explain diabetes burnout and practical support options.",
        reason="Daily management can become emotionally exhausting.",
    ),
    SELECTED_IDEA,
    ArticleIdea(
        idea_id="A3",
        title="Sleep and Type 2 Diabetes Management",
        article_angle="Explore how sleep problems can affect daily management.",
        reason="Sleep difficulties can shape day-to-day self-care.",
    ),
]

DEEP_RESEARCH_QUERIES = [
    "type 2 diabetes medication change clinical guidance side effects",
    "type 2 diabetes shared decision making medication cost adherence",
    "switching type 2 diabetes medicines safety monitoring patient guidance",
]


class FakeStructuredModel:
    """Return controlled planner output without contacting Nebius."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        assert messages[0][0] == "system"
        assert "Never guess a calendar year" in messages[0][1]
        assert messages[1][0] == "human"
        assert f"Current date: {date.today().isoformat()}" in messages[1][1]
        assert "Idea ID: A2" in messages[1][1]
        assert f"Title: {SELECTED_IDEA.title}" in messages[1][1]
        assert f"Article angle: {SELECTED_IDEA.article_angle}" in messages[1][1]
        assert f"Reason: {SELECTED_IDEA.reason}" in messages[1][1]
        return self.result


class FakeLLM:
    """Mimic the structured-output methods used by the query planner."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def with_structured_output(
        self,
        schema: type[DeepResearchQueryPlan],
        *,
        method: str,
        strict: bool,
    ) -> FakeStructuredModel:
        assert schema is DeepResearchQueryPlan
        assert method == "json_schema"
        assert strict is True
        return FakeStructuredModel(self.result)


def test_valid_human_selection_returns_complete_idea_without_modifying_choices() -> None:
    original_ideas = deepcopy(ARTICLE_IDEAS)

    selected = select_article_idea(ARTICLE_IDEAS, "a2")
    update = human_selection_node(
        {
            "run_id": "run-501",
            "topic": "Type 2 diabetes",
            "article_ideas": ARTICLE_IDEAS,
        },
        selected_idea_id="A2",
    )

    assert selected is SELECTED_IDEA
    assert update["selected_idea"] is SELECTED_IDEA
    assert update["selected_idea"].model_dump() == SELECTED_IDEA.model_dump()
    assert ARTICLE_IDEAS == original_ideas
    assert update["workflow_error"] is None


def test_invalid_human_selection_fails_clearly() -> None:
    update = human_selection_node(
        {
            "run_id": "run-502",
            "topic": "Type 2 diabetes",
            "article_ideas": ARTICLE_IDEAS,
        },
        selected_idea_id="A9",
    )

    assert "selected_idea" not in update
    assert update["workflow_error"].node == "human_selection"
    assert update["workflow_error"].type == "invalid_selection"


def test_deep_research_planner_accepts_exactly_three_queries() -> None:
    plan = plan_deep_research_queries(
        SELECTED_IDEA,
        llm=FakeLLM({"queries": DEEP_RESEARCH_QUERIES}),
    )

    assert plan.queries == DEEP_RESEARCH_QUERIES


@pytest.mark.parametrize(
    "queries",
    [
        DEEP_RESEARCH_QUERIES[:2],
        [*DEEP_RESEARCH_QUERIES, "a fourth query"],
        [DEEP_RESEARCH_QUERIES[0], "   ", DEEP_RESEARCH_QUERIES[2]],
    ],
    ids=["fewer-than-three", "more-than-three", "blank-query"],
)
def test_deep_research_planner_rejects_invalid_queries(
    queries: list[str],
) -> None:
    with pytest.raises(ValidationError):
        plan_deep_research_queries(
            SELECTED_IDEA,
            llm=FakeLLM({"queries": queries}),
        )


def test_missing_selected_idea_fails_before_planning_or_searching() -> None:
    update = deep_research_node(
        {
            "run_id": "run-503",
            "topic": "Type 2 diabetes",
        }
    )

    assert "deep_research_queries" not in update
    assert "deep_research_sources" not in update
    assert update["workflow_error"].node == "deep_research"
    assert update["workflow_error"].type == "missing_selected_idea"


def test_deep_research_combines_three_searches_and_preserves_evidence() -> None:
    duplicate_url = "https://nih.gov/shared-medication-guidance"
    responses = {
        DEEP_RESEARCH_QUERIES[0]: {
            "results": {
                "web": [
                    {
                        "title": "Medication Guidance",
                        "url": duplicate_url,
                        "contents": {
                            "highlights": [
                                "First medical Highlight EXACT.",
                                "Second medical Highlight EXACT.",
                            ]
                        },
                    }
                ],
                "news": [
                    {
                        "title": "Medication Safety Update",
                        "url": "https://fda.gov/example-safety-update",
                        "page_age": "2026-08-20T10:00:00",
                        "contents": {"highlights": ["News Highlight EXACT."]},
                    }
                ],
            }
        },
        DEEP_RESEARCH_QUERIES[1]: {
            "results": {
                "web": [
                    {
                        "title": "Shared Decision Making",
                        "url": duplicate_url,
                        "contents": {
                            "highlights": ["Different-query Highlight EXACT."]
                        },
                    }
                ]
            }
        },
        DEEP_RESEARCH_QUERIES[2]: {
            "results": {
                "news": [
                    {
                        "title": "Monitoring After a Medication Switch",
                        "url": "https://example.org/monitoring",
                        "contents": {"highlights": ["Monitoring Highlight EXACT."]},
                    }
                ]
            }
        },
    }
    requested_queries: list[str] = []

    def fake_planner(selected_idea: ArticleIdea) -> DeepResearchQueryPlan:
        assert selected_idea == SELECTED_IDEA
        return DeepResearchQueryPlan(queries=DEEP_RESEARCH_QUERIES)

    def fake_search_request(
        query: str,
        *,
        result_count: int,
        boost_domains: Sequence[str],
    ) -> dict[str, Any]:
        assert result_count == 4
        assert tuple(boost_domains) == PREFERRED_MEDICAL_DOMAINS
        requested_queries.append(query)
        return responses[query]

    update = deep_research_node(
        {
            "run_id": "run-504",
            "topic": "Type 2 diabetes",
            "selected_idea": SELECTED_IDEA,
        },
        query_planner=fake_planner,
        search_request=fake_search_request,
    )

    sources = update["deep_research_sources"]
    assert requested_queries == DEEP_RESEARCH_QUERIES
    assert update["deep_research_queries"] == DEEP_RESEARCH_QUERIES
    assert [source.source_id for source in sources] == ["R1", "R2", "R3", "R4"]
    assert [source.source_type for source in sources] == [
        "web",
        "news",
        "web",
        "news",
    ]
    assert [source.url for source in sources].count(duplicate_url) == 2
    assert [source.query for source in sources] == [
        DEEP_RESEARCH_QUERIES[0],
        DEEP_RESEARCH_QUERIES[0],
        DEEP_RESEARCH_QUERIES[1],
        DEEP_RESEARCH_QUERIES[2],
    ]
    assert [source.highlights for source in sources] == [
        ["First medical Highlight EXACT.", "Second medical Highlight EXACT."],
        ["News Highlight EXACT."],
        ["Different-query Highlight EXACT."],
        ["Monitoring Highlight EXACT."],
    ]
    assert update["workflow_error"] is None


def test_no_deep_research_evidence_returns_workflow_error_after_three_searches() -> None:
    request_count = 0

    def fake_planner(selected_idea: ArticleIdea) -> DeepResearchQueryPlan:
        return DeepResearchQueryPlan(queries=DEEP_RESEARCH_QUERIES)

    def empty_search_request(
        query: str,
        *,
        result_count: int,
        boost_domains: Sequence[str],
    ) -> dict[str, Any]:
        nonlocal request_count
        request_count += 1
        return {"results": {"web": [], "news": []}}

    update = deep_research_node(
        {
            "run_id": "run-505",
            "topic": "Type 2 diabetes",
            "selected_idea": SELECTED_IDEA,
        },
        query_planner=fake_planner,
        search_request=empty_search_request,
    )

    assert request_count == 3
    assert update["deep_research_queries"] == DEEP_RESEARCH_QUERIES
    assert update["deep_research_sources"] == []
    assert update["workflow_error"].node == "deep_research"
    assert update["workflow_error"].type == "no_usable_evidence"
