"""Standalone Deep Research node for a human-selected article idea."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from config import PREFERRED_MEDICAL_DOMAINS
from llm import create_nebius_llm
from prompts import (
    DEEP_RESEARCH_QUERY_PLANNER_SYSTEM_PROMPT,
    DEEP_RESEARCH_RETRY_SYSTEM_PROMPT,
    deep_research_query_planner_user_prompt,
    deep_research_retry_user_prompt,
)
from state import ArticleIdea, GraphState, Source, WorkflowError
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


class DeepResearchQueryPlan(BaseModel):
    """The validated structured output expected from the Deep Research planner."""

    model_config = ConfigDict(extra="forbid")

    queries: list[NonBlankQuery] = Field(min_length=3, max_length=3)


DeepResearchQueryPlanner = Callable[[ArticleIdea], DeepResearchQueryPlan]
DeepResearchRetryPlanner = Callable[
    [ArticleIdea, str, list[str], list[Source]],
    DeepResearchQueryPlan,
]
SearchRequest = Callable[..., dict[str, Any]]


def plan_deep_research_queries(
    selected_idea: ArticleIdea,
    *,
    llm: Any | None = None,
) -> DeepResearchQueryPlan:
    """Use one structured LLM call to plan three targeted research queries."""

    model = llm or create_nebius_llm()
    structured_model = model.with_structured_output(
        DeepResearchQueryPlan,
        method="json_schema",
        strict=True,
    )
    result = structured_model.invoke(
        [
            ("system", DEEP_RESEARCH_QUERY_PLANNER_SYSTEM_PROMPT),
            (
                "human",
                deep_research_query_planner_user_prompt(
                    current_date=date.today().isoformat(),
                    idea_id=selected_idea.idea_id,
                    title=selected_idea.title,
                    article_angle=selected_idea.article_angle,
                    reason=selected_idea.reason,
                ),
            ),
        ]
    )
    return DeepResearchQueryPlan.model_validate(result)


def format_existing_research(deep_research_sources: list[Source]) -> str:
    """Format prior evidence for retry planning without rewriting Highlights."""

    formatted_sources: list[str] = []
    for source in deep_research_sources:
        highlights = "\n".join(f"- {highlight}" for highlight in source.highlights)
        formatted_sources.append(
            f"""[{source.source_id}]
Query: {source.query}
Title: {source.title}
Type: {source.source_type}
Date: {source.publication_date or "Not available"}
Highlights:
{highlights}"""
        )
    return "\n\n".join(formatted_sources)


def plan_deep_research_retry_queries(
    selected_idea: ArticleIdea,
    review_feedback: str,
    existing_queries: list[str],
    existing_sources: list[Source],
    *,
    llm: Any | None = None,
) -> DeepResearchQueryPlan:
    """Plan three new searches aimed at the Reviewer's evidence gap."""

    if not review_feedback.strip():
        raise ValueError("Deep Research retry requires Reviewer feedback.")

    model = llm or create_nebius_llm()
    structured_model = model.with_structured_output(
        DeepResearchQueryPlan,
        method="json_schema",
        strict=True,
    )
    result = structured_model.invoke(
        [
            ("system", DEEP_RESEARCH_RETRY_SYSTEM_PROMPT),
            (
                "human",
                deep_research_retry_user_prompt(
                    current_date=date.today().isoformat(),
                    idea_id=selected_idea.idea_id,
                    title=selected_idea.title,
                    article_angle=selected_idea.article_angle,
                    reason=selected_idea.reason,
                    review_feedback=review_feedback,
                    existing_queries="\n".join(
                        f"- {query}" for query in existing_queries
                    ),
                    existing_evidence=format_existing_research(existing_sources),
                ),
            ),
        ]
    )
    return DeepResearchQueryPlan.model_validate(result)


def deep_research_node(
    state: GraphState,
    *,
    query_planner: DeepResearchQueryPlanner | None = None,
    retry_query_planner: DeepResearchRetryPlanner | None = None,
    search_request: SearchRequest = request_you_search,
    result_count: int = 4,
) -> dict[str, Any]:
    """Plan three searches and collect normalized evidence for one selected idea."""

    selected_idea = state.get("selected_idea")
    if selected_idea is None:
        return _error_update(
            "missing_selected_idea",
            "Deep Research requires a complete selected ArticleIdea.",
        )

    is_retry = state.get("failure_type") == "insufficient_research"
    existing_queries = list(state.get("deep_research_queries", [])) if is_retry else []
    existing_sources = list(state.get("deep_research_sources", [])) if is_retry else []

    try:
        validated_idea = ArticleIdea.model_validate(selected_idea)
        if is_retry:
            review_feedback = state.get("review_feedback")
            if review_feedback is None or not review_feedback.strip():
                raise ValueError("Deep Research retry requires Reviewer feedback.")
            retry_planner = retry_query_planner or plan_deep_research_retry_queries
            plan = DeepResearchQueryPlan.model_validate(
                retry_planner(
                    validated_idea,
                    review_feedback,
                    existing_queries,
                    existing_sources,
                )
            )
            previous_queries = {query.strip().casefold() for query in existing_queries}
            repeated_queries = [
                query
                for query in plan.queries
                if query.strip().casefold() in previous_queries
            ]
            normalized_new_queries = [
                query.strip().casefold() for query in plan.queries
            ]
            if repeated_queries or len(set(normalized_new_queries)) != 3:
                raise ValueError(
                    "Deep Research retry queries must be new and not repeat "
                    "one another or the existing query history."
                )
        else:
            planner = query_planner or plan_deep_research_queries
            plan = DeepResearchQueryPlan.model_validate(planner(validated_idea))
    except Exception as exc:
        return _error_update("query_planning_error", str(exc))

    new_sources: list[Source] = []
    combined_queries = [*existing_queries, *plan.queries]
    for query in plan.queries:
        try:
            # These domains are preferences, not an allowlist.
            response = search_request(
                query,
                result_count=result_count,
                boost_domains=PREFERRED_MEDICAL_DOMAINS,
            )
            query_sources = normalize_all_search_results(
                response,
                query=query,
                id_prefix="R",
                start_number=len(existing_sources) + len(new_sources) + 1,
            )
        except YouSearchError as exc:
            return {
                "deep_research_queries": combined_queries,
                "deep_research_sources": [*existing_sources, *new_sources],
                **_error_update("search_error", str(exc)),
            }

        # Preserve every query-result pair, including repeated URLs.
        new_sources.extend(query_sources)

    if not new_sources:
        return {
            "deep_research_queries": combined_queries,
            "deep_research_sources": existing_sources,
            **_error_update(
                "no_usable_evidence",
                "Deep Research completed three searches but found no usable evidence.",
            ),
        }

    return {
        "deep_research_queries": combined_queries,
        "deep_research_sources": [*existing_sources, *new_sources],
        "workflow_error": None,
    }


def _error_update(error_type: str, message: str) -> dict[str, WorkflowError]:
    """Create an error update identifying the Deep Research node."""

    return {
        "workflow_error": WorkflowError(
            node="deep_research",
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


def _print_deep_research(state: GraphState) -> None:
    """Print selected idea, queries, and display-only evidence previews."""

    selected_idea = state["selected_idea"]
    if selected_idea is None:
        return

    print()
    print("Selected idea")
    print(f"Title: {selected_idea.title}")
    print(f"Article angle: {selected_idea.article_angle}")
    print(f"Reason: {selected_idea.reason}")
    print()
    print("Deep Research queries")
    for number, query in enumerate(state.get("deep_research_queries", []), start=1):
        print(f"{number}. {query}")

    print()
    print("Deep Research evidence")
    for source in state.get("deep_research_sources", []):
        print(f"Source ID: {source.source_id}")
        print(f"Query: {source.query}")
        print(f"Type: {source.source_type}")
        print(f"Title: {source.title}")
        print(f"URL: {source.url}")
        print(f"Date: {source.publication_date or 'Not available'}")
        print("Highlight preview (first 300 characters total):")
        preview = _evidence_preview_for_display(source.highlights)
        print(f"- {preview or 'No Highlights returned'}")
        print()
