"""Opportunity Agent, UI-independent selection, and phased smoke tests."""

from __future__ import annotations

import argparse
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from llm import create_nebius_llm
from nodes.discovery import discovery_research_node
from nodes.research import (
    _print_deep_research,
    deep_research_node,
)
from nodes.reviewer import _print_reviewer_result, reviewer_node
from nodes.writer import _print_writer_result, writer_node
from prompts import OPPORTUNITY_AGENT_SYSTEM_PROMPT, opportunity_agent_user_prompt
from state import ArticleIdea, GraphState, Source, WorkflowError


# This type rejects missing, empty, and whitespace-only idea fields.
NonBlankText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class OpportunityIdeaCandidate(BaseModel):
    """One idea returned by the LLM before Python assigns its stable ID."""

    model_config = ConfigDict(extra="forbid")

    title: NonBlankText
    article_angle: NonBlankText
    reason: NonBlankText


class OpportunityIdeaPlan(BaseModel):
    """The validated structured output expected from the Opportunity LLM."""

    model_config = ConfigDict(extra="forbid")

    # Matching limits require exactly three final ideas.
    ideas: list[OpportunityIdeaCandidate] = Field(min_length=3, max_length=3)


class ArticleIdeaSelectionError(ValueError):
    """Raised when a human selection does not match exactly one available idea."""


def format_discovery_evidence(discovery_sources: list[Source]) -> str:
    """Format normalized Sources as readable evidence without changing them."""

    formatted_sources: list[str] = []
    for source in discovery_sources:
        highlights = "\n".join(f"- {highlight}" for highlight in source.highlights)
        if not highlights:
            highlights = "- No Highlights returned"

        formatted_sources.append(
            f"""[{source.source_id}]
Query: {source.query}
Title: {source.title}
URL: {source.url}
Type: {source.source_type}
Date: {source.publication_date or "Not available"}
Highlights:
{highlights}"""
        )

    return "\n\n".join(formatted_sources)


def generate_article_ideas(
    topic: str,
    discovery_sources: list[Source],
    *,
    llm: Any | None = None,
) -> list[ArticleIdea]:
    """Use one structured LLM call and assign deterministic article-idea IDs."""

    if not topic.strip():
        raise ValueError("Opportunity topic cannot be empty.")
    if not discovery_sources:
        raise ValueError("Opportunity Agent requires Discovery evidence.")

    # Tests supply a fake model; live execution uses the shared Nebius model.
    model = llm or create_nebius_llm()
    structured_model = model.with_structured_output(
        OpportunityIdeaPlan,
        method="json_schema",
        strict=True,
    )
    result = structured_model.invoke(
        [
            ("system", OPPORTUNITY_AGENT_SYSTEM_PROMPT),
            (
                "human",
                opportunity_agent_user_prompt(
                    topic,
                    format_discovery_evidence(discovery_sources),
                ),
            ),
        ]
    )

    # Validate at our application boundary even when the provider claims its
    # response follows the requested JSON Schema.
    plan = OpportunityIdeaPlan.model_validate(result)

    # The LLM supplies editorial content only. Python always owns stable IDs.
    return [
        ArticleIdea(
            idea_id=f"A{number}",
            title=candidate.title,
            article_angle=candidate.article_angle,
            reason=candidate.reason,
        )
        for number, candidate in enumerate(plan.ideas, start=1)
    ]


def opportunity_agent_node(
    state: GraphState,
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Read Discovery evidence and return only the article-ideas state update."""

    discovery_sources = state.get("discovery_sources", [])
    if not discovery_sources:
        return _error_update(
            "missing_discovery_evidence",
            "Opportunity Agent requires at least one Discovery Source.",
        )

    try:
        article_ideas = generate_article_ideas(
            state["topic"],
            discovery_sources,
            llm=llm,
        )
    except Exception as exc:
        return _error_update("idea_generation_error", str(exc))

    return {
        "article_ideas": article_ideas,
        "workflow_error": None,
    }


def select_article_idea(
    article_ideas: list[ArticleIdea],
    selected_idea_id: str,
) -> ArticleIdea:
    """Return the complete selected idea without depending on a user interface."""

    normalized_id = selected_idea_id.strip().upper()
    matching_ideas = [
        idea for idea in article_ideas if idea.idea_id == normalized_id
    ]
    if len(matching_ideas) != 1:
        available_ids = ", ".join(idea.idea_id for idea in article_ideas)
        raise ArticleIdeaSelectionError(
            f"Select exactly one available idea ID: {available_ids}."
        )
    return matching_ideas[0]


def human_selection_node(
    state: GraphState,
    *,
    selected_idea_id: str,
) -> dict[str, Any]:
    """Validate a UI-provided choice and write the complete idea to state."""

    try:
        selected_idea = select_article_idea(
            state.get("article_ideas", []),
            selected_idea_id,
        )
    except ArticleIdeaSelectionError as exc:
        return _selection_error_update(str(exc))

    return {
        "selected_idea": selected_idea,
        "workflow_error": None,
    }


def _error_update(error_type: str, message: str) -> dict[str, WorkflowError]:
    """Create a Phase 4 update using the existing workflow error contract."""

    return {
        "workflow_error": WorkflowError(
            node="opportunity_agent",
            type=error_type,
            message=message,
        )
    }


def _selection_error_update(message: str) -> dict[str, WorkflowError]:
    """Create an error update for an invalid human selection."""

    return {
        "workflow_error": WorkflowError(
            node="human_selection",
            type="invalid_selection",
            message=message,
        )
    }


def _print_opportunity_result(topic: str, state: GraphState) -> None:
    """Print a compact live smoke-test result without repeating all evidence."""

    print("Topic")
    print(topic)
    print()
    print("Discovery queries")
    for number, query in enumerate(state.get("discovery_queries", []), start=1):
        print(f"{number}. {query}")

    print()
    print(f"Discovery sources retrieved: {len(state.get('discovery_sources', []))}")
    print()
    print("Recommended article ideas")
    for idea in state.get("article_ideas", []):
        print()
        print(idea.idea_id)
        print(f"Title: {idea.title}")
        print(f"Article angle: {idea.article_angle}")
        print(f"Why patients may care: {idea.reason}")


def _parse_args() -> argparse.Namespace:
    """Read optional smoke-test settings from the terminal command."""

    parser = argparse.ArgumentParser(description="Run Opportunity Agent live.")
    parser.add_argument("--topic", default="Type 2 diabetes")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--deep-research",
        action="store_true",
        help="Continue through manual selection and live Deep Research.",
    )
    parser.add_argument(
        "--writer",
        action="store_true",
        help="Continue through manual selection, Deep Research, and Writer.",
    )
    parser.add_argument(
        "--reviewer",
        action="store_true",
        help="Continue through Writer and Reviewer, then display the next route.",
    )
    return parser.parse_args()


def main() -> None:
    """Run Opportunity live, optionally continuing through Deep Research."""

    args = _parse_args()
    state: GraphState = {
        "run_id": "opportunity-smoke-test",
        "topic": args.topic,
        "retry_count": 0,
    }

    discovery_update = discovery_research_node(state, result_count=args.count)
    discovery_error = discovery_update.get("workflow_error")
    if discovery_error is not None:
        raise SystemExit(
            f"Discovery failed in {discovery_error.node}: {discovery_error.message}"
        )
    state.update(discovery_update)

    opportunity_update = opportunity_agent_node(state)
    opportunity_error = opportunity_update.get("workflow_error")
    if opportunity_error is not None:
        raise SystemExit(
            "Opportunity failed in "
            f"{opportunity_error.node}: {opportunity_error.message}"
        )
    state.update(opportunity_update)

    _print_opportunity_result(args.topic, state)

    # input() belongs only to this manual terminal path. Reusable selection
    # logic above receives an ID and has no dependency on a terminal UI.
    if not (args.deep_research or args.writer or args.reviewer):
        return

    selected_idea_id = input("\nSelect one idea ID (A1, A2, or A3): ")
    selection_update = human_selection_node(
        state,
        selected_idea_id=selected_idea_id,
    )
    selection_error = selection_update.get("workflow_error")
    if selection_error is not None:
        raise SystemExit(
            f"Selection failed in {selection_error.node}: {selection_error.message}"
        )
    state.update(selection_update)

    deep_research_update = deep_research_node(state, result_count=args.count)
    deep_research_error = deep_research_update.get("workflow_error")
    if deep_research_error is not None:
        raise SystemExit(
            "Deep Research failed in "
            f"{deep_research_error.node}: {deep_research_error.message}"
        )
    state.update(deep_research_update)

    if not (args.writer or args.reviewer):
        _print_deep_research(state)
        return

    writer_update = writer_node(state)
    writer_error = writer_update.get("workflow_error")
    if writer_error is not None:
        raise SystemExit(
            f"Writer failed in {writer_error.node}: {writer_error.message}"
        )
    state.update(writer_update)

    _print_writer_result(state)

    if not args.reviewer:
        return

    reviewer_update = reviewer_node(state)
    reviewer_error = reviewer_update.get("workflow_error")
    if reviewer_error is not None:
        raise SystemExit(
            f"Reviewer failed in {reviewer_error.node}: {reviewer_error.message}"
        )
    state.update(reviewer_update)

    # The route is displayed for inspection only. Phase 7 does not execute it.
    _print_reviewer_result(state)


if __name__ == "__main__":
    # This runs only for `python -m nodes.opportunity`, not when it is imported.
    main()
