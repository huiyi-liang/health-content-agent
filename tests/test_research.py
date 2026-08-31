"""Focused unit tests for the standalone Discovery Research node."""

from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from nodes.discovery import (
    DiscoveryQueryPlan,
    _evidence_preview_for_display,
    discovery_research_node,
    plan_discovery_queries,
)


PLANNED_QUERIES = [
    "type 2 diabetes daily management challenges patient experiences",
    "type 2 diabetes decisions questions patients ask their care team",
    "recent type 2 diabetes developments with practical patient impact",
]


class FakeStructuredModel:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        assert messages[0][0] == "system"
        assert "Never guess a calendar year" in messages[0][1]
        assert messages[1] == (
            "human",
            f"Current date: {date.today().isoformat()}\nHealth topic: Type 2 diabetes",
        )
        return self.result


class FakeLLM:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def with_structured_output(
        self,
        schema: type[DiscoveryQueryPlan],
        *,
        method: str,
        strict: bool,
    ) -> FakeStructuredModel:
        assert schema is DiscoveryQueryPlan
        assert method == "json_schema"
        assert strict is True
        return FakeStructuredModel(self.result)


def test_query_planner_accepts_exactly_three_valid_queries() -> None:
    plan = plan_discovery_queries(
        "Type 2 diabetes",
        llm=FakeLLM({"queries": PLANNED_QUERIES}),
    )

    assert plan.queries == PLANNED_QUERIES


@pytest.mark.parametrize(
    "queries",
    [
        PLANNED_QUERIES[:2],
        [*PLANNED_QUERIES, "a fourth query"],
        [PLANNED_QUERIES[0], "   ", PLANNED_QUERIES[2]],
    ],
    ids=["fewer-than-three", "more-than-three", "blank-query"],
)
def test_query_planner_rejects_invalid_query_lists(queries: list[str]) -> None:
    with pytest.raises(ValidationError):
        plan_discovery_queries(
            "Type 2 diabetes",
            llm=FakeLLM({"queries": queries}),
        )


def test_discovery_combines_three_searches_and_preserves_evidence() -> None:
    long_highlight = "A" * 350
    duplicate_url = "https://example.org/shared-article"
    responses = {
        PLANNED_QUERIES[0]: {
            "results": {
                "web": [
                    {
                        "title": "Daily Management",
                        "url": duplicate_url,
                        "contents": {"highlights": [long_highlight]},
                    }
                ],
                "news": [
                    {
                        "title": "Recent Development",
                        "url": "https://news.example.org/development",
                        "page_age": "2026-08-20T10:00:00",
                        "contents": {"highlights": ["News Highlight EXACT."]},
                    }
                ],
            }
        },
        PLANNED_QUERIES[1]: {
            "results": {
                "web": [
                    {
                        "title": "Questions for Care Teams",
                        "url": duplicate_url,
                        "contents": {"highlights": ["Different Query Highlight."]},
                    }
                ]
            }
        },
        PLANNED_QUERIES[2]: {
            "results": {
                "news": [
                    {
                        "title": "Another Development",
                        "url": "https://news.example.org/another-development",
                        "page_age": "2026-08-25T09:00:00",
                        "contents": {"highlights": ["Third Query Highlight."]},
                    }
                ]
            }
        },
    }
    requested_queries: list[str] = []

    def fake_planner(topic: str) -> DiscoveryQueryPlan:
        assert topic == "Type 2 diabetes"
        return DiscoveryQueryPlan(queries=PLANNED_QUERIES)

    def fake_search_request(query: str, *, result_count: int) -> dict[str, Any]:
        assert result_count == 4
        requested_queries.append(query)
        return responses[query]

    update = discovery_research_node(
        {"run_id": "run-001", "topic": "Type 2 diabetes"},
        query_planner=fake_planner,
        search_request=fake_search_request,
    )

    sources = update["discovery_sources"]
    assert update["discovery_queries"] == PLANNED_QUERIES
    assert requested_queries == PLANNED_QUERIES
    assert len(sources) == 4
    assert [source.source_id for source in sources] == ["D1", "D2", "D3", "D4"]
    assert [source.source_type for source in sources] == [
        "web",
        "news",
        "web",
        "news",
    ]
    assert [source.url for source in sources].count(duplicate_url) == 2
    assert [source.query for source in sources] == [
        PLANNED_QUERIES[0],
        PLANNED_QUERIES[0],
        PLANNED_QUERIES[1],
        PLANNED_QUERIES[2],
    ]
    assert [source.highlights for source in sources] == [
        [long_highlight],
        ["News Highlight EXACT."],
        ["Different Query Highlight."],
        ["Third Query Highlight."],
    ]
    assert update["workflow_error"] is None


def test_discovery_returns_workflow_error_when_all_three_searches_are_empty() -> None:
    request_count = 0

    def fake_planner(topic: str) -> DiscoveryQueryPlan:
        return DiscoveryQueryPlan(queries=PLANNED_QUERIES)

    def empty_search_request(query: str, *, result_count: int) -> dict[str, Any]:
        nonlocal request_count
        request_count += 1
        return {"results": {"web": [], "news": []}}

    update = discovery_research_node(
        {"run_id": "run-002", "topic": "Type 2 diabetes"},
        query_planner=fake_planner,
        search_request=empty_search_request,
    )

    assert request_count == 3
    assert update["discovery_sources"] == []
    assert update["workflow_error"].node == "discovery_research"
    assert update["workflow_error"].type == "no_usable_evidence"


def test_combined_evidence_preview_is_300_characters_and_display_only() -> None:
    stored_highlights = ["A" * 200, "B" * 200]

    displayed_preview = _evidence_preview_for_display(stored_highlights)

    assert stored_highlights == ["A" * 200, "B" * 200]
    assert displayed_preview == f"{'A' * 200} {'B' * 99}... [truncated]"
