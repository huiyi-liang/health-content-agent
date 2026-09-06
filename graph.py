"""LangGraph orchestration and terminal harness for the complete Phase 8 flow."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from nodes.discovery import discovery_research_node
from nodes.opportunity import human_selection_node, opportunity_agent_node
from nodes.research import deep_research_node
from nodes.reviewer import reviewer_node, route_review_result
from nodes.writer import _print_writer_result, writer_node, writer_revision_node
from persistence import export_run_history
from state import (
    ArticleDraft,
    ArticleIdea,
    ArticleSection,
    GraphState,
    Source,
    WorkflowError,
)


WorkflowNode = Callable[[GraphState], dict[str, Any]]
StepRoute = Literal["continue", "failed"]
ReviewGraphRoute = Literal[
    "completed",
    "begin_retry",
    "human_review_required",
    "failed",
]
RetryTarget = Literal["deep_research", "writer_revision"]


def create_in_memory_checkpointer() -> InMemorySaver:
    """Create a temporary checkpointer that trusts our state-model classes."""

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            Source,
            ArticleIdea,
            ArticleSection,
            ArticleDraft,
            WorkflowError,
        ]
    )
    return InMemorySaver(serde=serializer)


def human_selection_interrupt_node(state: GraphState) -> dict[str, Any]:
    """Pause the graph and validate the human's resumed article-idea ID."""

    article_ideas = state.get("article_ideas", [])
    selected_idea_id = interrupt(
        {
            "question": "Select one article idea by idea_id.",
            "article_ideas": [
                idea.model_dump(mode="json") for idea in article_ideas
            ],
        }
    )
    if not isinstance(selected_idea_id, str):
        return {
            "workflow_error": WorkflowError(
                node="human_selection",
                type="invalid_selection",
                message="The resumed selection must be an article idea ID string.",
            )
        }
    return human_selection_node(state, selected_idea_id=selected_idea_id)


def increment_retry_count_node(state: GraphState) -> dict[str, int]:
    """Count one newly authorized automated correction cycle."""

    return {"retry_count": state.get("retry_count", 0) + 1}


def completed_node(state: GraphState) -> dict[str, str]:
    """Mark a Reviewer PASS as successful automated completion."""

    return {"final_status": "completed"}


def human_review_required_node(state: GraphState) -> dict[str, str]:
    """Stop automation after three completed correction cycles."""

    return {"final_status": "human_review_required"}


def failed_node(state: GraphState) -> dict[str, str]:
    """Mark an unrecoverable node or tool error without discarding state."""

    return {"final_status": "failed"}


def route_after_workflow_node(state: GraphState) -> StepRoute:
    """Stop after a system/tool error; otherwise continue along the graph."""

    return "failed" if state.get("workflow_error") is not None else "continue"


def route_after_review(state: GraphState) -> ReviewGraphRoute:
    """Choose completion, retry, escalation, or failure deterministically."""

    if state.get("workflow_error") is not None:
        return "failed"
    if state.get("review_status") == "PASS":
        return "completed"
    if state.get("retry_count", 0) >= 3:
        return "human_review_required"

    # The Phase 7 helper validates the failure-to-destination mapping. At this
    # point either retry destination means one correction cycle may begin.
    route_review_result(state)
    return "begin_retry"


def route_retry_target(state: GraphState) -> RetryTarget:
    """Choose research expansion versus targeted Writer revision."""

    failure_type = state.get("failure_type")
    if failure_type == "insufficient_research":
        return "deep_research"
    if failure_type in {
        "unsupported_claim",
        "style_or_readability",
        "patient_value",
    }:
        return "writer_revision"
    raise ValueError("Cannot choose a retry target without a valid failure type.")


def build_workflow(
    *,
    discovery: WorkflowNode = discovery_research_node,
    opportunity: WorkflowNode = opportunity_agent_node,
    deep_research: WorkflowNode = deep_research_node,
    writer: WorkflowNode = writer_node,
    writer_revision: WorkflowNode = writer_revision_node,
    reviewer: WorkflowNode = reviewer_node,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the workflow while allowing mocked nodes in integration tests."""

    builder = StateGraph(GraphState)
    builder.add_node("discovery", discovery)
    builder.add_node("opportunity", opportunity)
    builder.add_node("human_selection", human_selection_interrupt_node)
    builder.add_node("deep_research", deep_research)
    builder.add_node("writer", writer)
    builder.add_node("writer_revision", writer_revision)
    builder.add_node("reviewer", reviewer)
    builder.add_node("increment_retry", increment_retry_count_node)
    builder.add_node("completed", completed_node)
    builder.add_node("human_review_required", human_review_required_node)
    builder.add_node("failed", failed_node)

    builder.add_edge(START, "discovery")
    builder.add_conditional_edges(
        "discovery",
        route_after_workflow_node,
        {"continue": "opportunity", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "opportunity",
        route_after_workflow_node,
        {"continue": "human_selection", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "human_selection",
        route_after_workflow_node,
        {"continue": "deep_research", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "deep_research",
        route_after_workflow_node,
        {"continue": "writer", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "writer",
        route_after_workflow_node,
        {"continue": "reviewer", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "writer_revision",
        route_after_workflow_node,
        {"continue": "reviewer", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "completed": "completed",
            "begin_retry": "increment_retry",
            "human_review_required": "human_review_required",
            "failed": "failed",
        },
    )
    builder.add_conditional_edges(
        "increment_retry",
        route_retry_target,
        {
            "deep_research": "deep_research",
            "writer_revision": "writer_revision",
        },
    )
    builder.add_edge("completed", END)
    builder.add_edge("human_review_required", END)
    builder.add_edge("failed", END)

    # A checkpointer is required for interrupt/resume. InMemorySaver is
    # intentionally temporary and is appropriate for tests and this CLI demo.
    return builder.compile(
        checkpointer=checkpointer or create_in_memory_checkpointer()
    )


def _display_stream_event(
    node_name: str,
    update: dict[str, Any],
    state: GraphState,
) -> None:
    """Print compact stage-specific progress without dumping full evidence."""

    labels = {
        "discovery": "Discovery complete",
        "opportunity": "Opportunity Agent complete",
        "human_selection": "Human selection saved",
        "deep_research": "Deep Research complete",
        "writer": "Writer draft complete",
        "writer_revision": "Writer revision complete",
        "reviewer": "Reviewer complete",
        "increment_retry": "Automated retry started",
        "completed": "Workflow completed",
        "human_review_required": "Automation stopped for human review",
        "failed": "Workflow failed",
    }
    print(f"\nStage: {labels.get(node_name, node_name)}")

    if node_name == "opportunity":
        for idea in update.get("article_ideas", []):
            print(f"{idea.idea_id}: {idea.title}")
            print(f"  Angle: {idea.article_angle}")
            print(f"  Reason: {idea.reason}")
    elif node_name == "human_selection":
        idea = update.get("selected_idea")
        if idea is not None:
            print(f"Selected: {idea.idea_id} — {idea.title}")
    elif node_name == "deep_research":
        print(f"Total queries: {len(update.get('deep_research_queries', []))}")
        print(f"Total sources: {len(update.get('deep_research_sources', []))}")
    elif node_name == "writer_revision":
        print(f"Revised section IDs: {state.get('flagged_sections', [])}")
        print(f"Correction cycle: {state.get('retry_count', 0)}")
    elif node_name == "reviewer":
        workflow_error = state.get("workflow_error")
        if workflow_error is not None:
            print(
                "Reviewer system error: "
                f"{workflow_error.type} / {workflow_error.message}"
            )
            print("Route: Workflow failed")
            return

        print(f"Reviewer: {update.get('review_status')}")
        print(f"Failure: {update.get('failure_type')}")
        print(f"Flagged sections: {update.get('flagged_sections', [])}")
        print(f"Feedback: {update.get('review_feedback')}")
        print(f"Retry count: {state.get('retry_count', 0)}")
        graph_route = route_after_review(state)
        route_labels = {
            "completed": "Final completion",
            "begin_retry": (
                "Deep Research retry"
                if state.get("failure_type") == "insufficient_research"
                else "Writer revision"
            ),
            "human_review_required": "Human review required",
            "failed": "Workflow failed",
        }
        print(f"Route: {route_labels[graph_route]}")
    elif node_name == "increment_retry":
        print(f"Retry count: {update.get('retry_count')}")
    elif node_name in {"completed", "human_review_required", "failed"}:
        print(f"Final status: {update.get('final_status')}")
        if node_name == "human_review_required":
            print(
                "This is an editorial safety escalation, not a system or API failure."
            )


def _stream_until_pause_or_end(
    app: Any,
    graph_input: GraphState | Command,
    config: dict[str, Any],
) -> tuple[Any, ...]:
    """Stream visible node progress and return any human interrupts."""

    interrupts: tuple[Any, ...] = ()
    # get_state() can lag one streamed event because LangGraph commits the new
    # checkpoint after yielding that event. Keep a local display snapshot and
    # merge each node update immediately so terminal output sees current data.
    current_state: GraphState = dict(app.get_state(config).values)
    if isinstance(graph_input, dict):
        current_state.update(graph_input)

    for event in app.stream(graph_input, config, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "__interrupt__":
                interrupts = tuple(update)
            elif isinstance(update, dict):
                current_state.update(update)
                _display_stream_event(node_name, update, current_state)
    return interrupts


def _parse_args() -> argparse.Namespace:
    """Read terminal settings for the live end-to-end graph smoke test."""

    parser = argparse.ArgumentParser(description="Run the Phase 8 LangGraph flow.")
    parser.add_argument("--topic", default="Type 2 diabetes")
    return parser.parse_args()


def main() -> None:
    """Run the real graph, pause for selection, resume, and show final state."""

    args = _parse_args()
    run_id = f"graph-smoke-{uuid4()}"
    config = {"configurable": {"thread_id": run_id}}
    app = build_workflow()

    print(f"Topic: {args.topic}")
    interrupts = _stream_until_pause_or_end(
        app,
        {
            "run_id": run_id,
            "topic": args.topic,
            "retry_count": 0,
        },
        config,
    )
    if not interrupts:
        final_state = app.get_state(config).values
        print(f"Final status: {final_state.get('final_status')}")
        if final_state.get("final_status") in {
            "completed",
            "human_review_required",
            "failed",
        }:
            export_path = export_run_history(app, config)
            print(f"Exported run: {export_path}")
        return

    selection = input("\nSelect one idea ID (A1, A2, or A3): ")
    _stream_until_pause_or_end(
        app,
        Command(resume=selection),
        config,
    )

    final_state: GraphState = app.get_state(config).values
    print()
    print(f"Retry count: {final_state.get('retry_count', 0)}")
    print(f"Final status: {final_state.get('final_status')}")
    if final_state.get("workflow_error") is not None:
        error = final_state["workflow_error"]
        print(f"Workflow error: {error.node} / {error.type} / {error.message}")
    if final_state.get("draft") is not None:
        _print_writer_result(final_state)
    if final_state.get("final_status") in {
        "completed",
        "human_review_required",
        "failed",
    }:
        export_path = export_run_history(app, config)
        print(f"Exported run: {export_path}")


if __name__ == "__main__":
    main()
