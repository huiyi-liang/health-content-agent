"""Standalone Discovery Research node and live smoke test."""

from __future__ import annotations

import argparse
from datetime import date
from typing import Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from llm import create_nebius_llm
from prompts import (
    DISCOVERY_QUERY_PLANNER_SYSTEM_PROMPT,
    discovery_query_planner_user_prompt,
)
from state import GraphState, Source, WorkflowError
from tools.search import (
    YouSearchError,
    normalize_all_search_results,
    request_you_search,
)


DISPLAY_HIGHLIGHT_LIMIT = 300

NonBlankQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class DiscoveryQueryPlan(BaseModel):
    """The validated structured output expected from the Discovery planner."""

    model_config = ConfigDict(extra="forbid")

    queries: list[NonBlankQuery] = Field(min_length=3, max_length=3)


QueryPlanner = Callable[[str], DiscoveryQueryPlan]
SearchRequest = Callable[..., dict[str, Any]]


def plan_discovery_queries(topic: str, *, llm: Any | None = None) -> DiscoveryQueryPlan:
    """Use one structured LLM call to plan exactly three Discovery queries."""

    if not topic.strip():
        raise ValueError("Discovery topic cannot be empty.")

    model = llm or create_nebius_llm()
    structured_model = model.with_structured_output(
        DiscoveryQueryPlan,
        method="json_schema",
        strict=True,
    )
    result = structured_model.invoke(
        [
            ("system", DISCOVERY_QUERY_PLANNER_SYSTEM_PROMPT),
            (
                "human",
                discovery_query_planner_user_prompt(
                    topic,
                    current_date=date.today().isoformat(),
                ),
            ),
        ]
    )
    return DiscoveryQueryPlan.model_validate(result)


def discovery_research_node(
    state: GraphState,
    *,
    query_planner: QueryPlanner | None = None,
    search_request: SearchRequest = request_you_search,
    result_count: int = 4,
) -> dict[str, Any]:
    """Plan three queries and collect normalized web and news evidence."""

    topic = state["topic"]
    planner = query_planner or plan_discovery_queries

    try:
        plan = DiscoveryQueryPlan.model_validate(planner(topic))
    except Exception as exc:
        return _error_update("query_planning_error", str(exc))

    discovery_sources: list[Source] = []
    for query in plan.queries:
        try:
            # One request may return both web and news sections.
            response = search_request(query, result_count=result_count)
            query_sources = normalize_all_search_results(
                response,
                query=query,
                id_prefix="D",
                start_number=len(discovery_sources) + 1,
            )
        except YouSearchError as exc:
            return {
                "discovery_queries": plan.queries,
                "discovery_sources": discovery_sources,
                **_error_update("search_error", str(exc)),
            }

        # Preserve every query-result pair, including repeated URLs.
        discovery_sources.extend(query_sources)

    if not discovery_sources:
        return {
            "discovery_queries": plan.queries,
            "discovery_sources": [],
            **_error_update(
                "no_usable_evidence",
                "Discovery completed three searches but found no usable evidence.",
            ),
        }

    return {
        "discovery_queries": plan.queries,
        "discovery_sources": discovery_sources,
        "workflow_error": None,
    }


def _error_update(error_type: str, message: str) -> dict[str, WorkflowError]:
    """Create an error update identifying the Discovery Research node."""

    return {
        "workflow_error": WorkflowError(
            node="discovery_research",
            type=error_type,
            message=message,
        )
    }


def _evidence_preview_for_display(highlights: list[str]) -> str:
    """Create one 300-character terminal preview from all Source Highlights."""

    combined_highlights = " ".join(highlights)
    if len(combined_highlights) <= DISPLAY_HIGHLIGHT_LIMIT:
        return combined_highlights
    return f"{combined_highlights[:DISPLAY_HIGHLIGHT_LIMIT].rstrip()}... [truncated]"


def _print_discovery(topic: str, update: dict[str, Any]) -> None:
    """Print Discovery results for the manual live smoke test."""

    print("Topic")
    print(topic)
    print()
    print("LLM-generated Discovery queries")
    for number, query in enumerate(update.get("discovery_queries", []), start=1):
        print(f"{number}. {query}")

    print()
    print("Discovery evidence")
    for source in update.get("discovery_sources", []):
        print(f"Source ID: {source.source_id}")
        print(f"Query: {source.query}")
        print(f"Type: {source.source_type}")
        print(f"Title: {source.title}")
        print(f"URL: {source.url}")
        print(f"Date: {source.publication_date or 'Not available'}")
        print("Evidence preview (first 300 characters total):")
        preview = _evidence_preview_for_display(source.highlights)
        print(f"- {preview or 'No Highlights returned'}")
        print()


def _parse_args() -> argparse.Namespace:
    """Read optional Discovery smoke-test settings from the terminal."""

    parser = argparse.ArgumentParser(description="Run Discovery Research live.")
    parser.add_argument("--topic", default="Type 2 diabetes")
    parser.add_argument("--count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run the standalone live Discovery smoke test."""

    args = _parse_args()
    state: GraphState = {"run_id": "discovery-smoke-test", "topic": args.topic}
    update = discovery_research_node(state, result_count=args.count)

    workflow_error = update.get("workflow_error")
    if workflow_error is not None:
        raise SystemExit(
            f"Discovery failed in {workflow_error.node}: {workflow_error.message}"
        )

    _print_discovery(args.topic, update)


if __name__ == "__main__":
    main()
