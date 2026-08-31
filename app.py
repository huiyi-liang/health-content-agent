"""Thin Streamlit interface around the existing LangGraph workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from graph import build_workflow
from state import ArticleDraft, ArticleIdea, GraphState, Source, WorkflowError


@st.cache_resource
def get_workflow() -> Any:
    """Keep one compiled graph/checkpointer alive across Streamlit reruns."""

    return build_workflow()


def graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build the LangGraph configuration that identifies one workflow run."""

    return {"configurable": {"thread_id": thread_id}}


def article_display_blocks(
    draft: ArticleDraft,
) -> list[tuple[str | None, str]]:
    """Return reader-facing headings and content without exposing section IDs."""

    return [(section.heading, section.content) for section in draft.sections]


def source_display_entries(
    sources: list[Source],
) -> list[tuple[str, str, str]]:
    """Return deterministic source ID, title, and URL display values."""

    return [(source.source_id, source.title, source.url) for source in sources]


def flagged_section_labels(
    draft: ArticleDraft,
    flagged_section_ids: list[str],
) -> list[str]:
    """Translate internal section IDs into reader-facing labels for the UI."""

    labels_by_id = {
        section.section_id: section.heading or "Introduction"
        for section in draft.sections
    }
    return [
        labels_by_id.get(section_id, "Article section")
        for section_id in flagged_section_ids
    ]


def format_workflow_error(error: WorkflowError | dict[str, Any]) -> str:
    """Create a short UI message without a traceback or raw exception dump."""

    validated_error = WorkflowError.model_validate(error)
    stage = validated_error.node.replace("_", " ").title()
    message = " ".join(validated_error.message.split())
    if len(message) > 400:
        message = f"{message[:397]}..."
    return f"The workflow stopped during {stage}: {message}"


def _progress_message(node_name: str, state: GraphState) -> str:
    """Translate graph-node events into simple product-level progress text."""

    if node_name == "discovery":
        return "Topic research complete. Identifying article opportunities..."
    if node_name == "opportunity":
        return "Article opportunities ready. Waiting for your selection..."
    if node_name == "human_selection":
        return "Selection saved. Researching the selected article..."
    if node_name == "deep_research":
        return "Research complete. Writing the article..."
    if node_name in {"writer", "writer_revision"}:
        return "Draft ready. Reviewing the article..."
    if node_name == "reviewer":
        if state.get("review_status") == "PASS":
            return "Automated review passed. Finishing the article..."
        return "Automated review found an issue. Preparing a correction..."
    if node_name == "increment_retry":
        if state.get("failure_type") == "insufficient_research":
            return "Gathering additional evidence..."
        return "Revising the flagged article sections..."
    if node_name == "completed":
        return "Workflow completed."
    if node_name == "human_review_required":
        return "Automated corrections stopped. Human review is required."
    if node_name == "failed":
        return "The workflow stopped because a system or tool step failed."
    return "Workflow is continuing..."


def run_graph_with_progress(
    workflow: Any,
    graph_input: GraphState | Command,
    config: dict[str, Any],
    *,
    initial_label: str,
) -> bool:
    """Run or resume LangGraph while showing compact progress to the user."""

    paused_for_selection = False
    current_state: GraphState = dict(workflow.get_state(config).values)
    if isinstance(graph_input, dict):
        current_state.update(graph_input)

    with st.status(initial_label, expanded=True) as status:
        for event in workflow.stream(graph_input, config, stream_mode="updates"):
            for node_name, update in event.items():
                if node_name == "__interrupt__":
                    paused_for_selection = True
                    status.write("Three article ideas are ready for your review.")
                    continue

                if isinstance(update, dict):
                    current_state.update(update)
                status.write(_progress_message(node_name, current_state))

        final_status = current_state.get("final_status")
        if paused_for_selection:
            status.update(label="Choose an article idea", state="complete")
        elif final_status == "failed":
            status.update(label="Workflow stopped", state="error")
        else:
            status.update(label="Workflow finished", state="complete")

    return paused_for_selection


def render_article(draft: ArticleDraft) -> None:
    """Render a structured ArticleDraft as a readable Streamlit article."""

    st.title(draft.title)
    for heading, content in article_display_blocks(draft):
        if heading is not None:
            st.header(heading)
        for paragraph in content.split("\n\n"):
            if paragraph.strip():
                st.markdown(paragraph.strip())


def render_sources(sources: list[Source]) -> None:
    """Render stored source records; no LLM-generated bibliography is used."""

    st.header("Sources")
    for source_id, title, url in source_display_entries(sources):
        st.markdown(f"**[{source_id}] {title}**")
        st.markdown(f"[Open source]({url})")


def render_article_ideas(
    article_ideas: list[ArticleIdea],
    thread_id: str,
) -> str | None:
    """Display all ideas and return a confirmed radio selection when present."""

    st.header("Choose an article idea")
    for idea in article_ideas:
        st.subheader(f"{idea.idea_id}: {idea.title}")
        st.markdown("**Article angle**")
        st.write(idea.article_angle)
        st.markdown("**Why patients may care**")
        st.write(idea.reason)

    idea_by_id = {idea.idea_id: idea for idea in article_ideas}
    with st.form(f"article_selection_form_{thread_id}"):
        selected_id = st.radio(
            "Select one idea",
            options=list(idea_by_id),
            index=None,
            key=f"article_selection_{thread_id}",
            format_func=lambda idea_id: (
                f"{idea_id}: {idea_by_id[idea_id].title}"
            ),
        )
        confirmed = st.form_submit_button("Continue with selected idea")

    if confirmed and selected_id is None:
        st.warning("Select one article idea before continuing.")
        return None
    return selected_id if confirmed else None


def render_finished_state(state: GraphState) -> None:
    """Display completed, escalated, or failed workflow state appropriately."""

    final_status = state.get("final_status")
    workflow_error = state.get("workflow_error")

    if final_status == "failed":
        st.header("Workflow could not finish")
        if workflow_error is not None:
            st.error(format_workflow_error(workflow_error))
        else:
            st.error("The workflow stopped before it could finish.")
        return

    draft_value = state.get("draft")
    if draft_value is None:
        st.error("The workflow finished without an article draft to display.")
        return

    draft = ArticleDraft.model_validate(draft_value)
    sources = [
        Source.model_validate(source)
        for source in state.get("deep_research_sources", [])
    ]

    if final_status == "completed":
        st.success("Automated evidence and editorial review passed.")
    elif final_status == "human_review_required":
        st.warning(
            "Automated review could not resolve the remaining issue. "
            "Human editorial review is required."
        )
        st.markdown(f"**Primary failure:** {state.get('failure_type') or 'Unknown'}")
        st.markdown(f"**Retry count:** {state.get('retry_count', 0)}")
        labels = flagged_section_labels(
            draft,
            list(state.get("flagged_sections", [])),
        )
        st.markdown(
            "**Flagged sections:** " + (", ".join(labels) if labels else "None")
        )
        st.markdown("**Reviewer feedback**")
        st.write(state.get("review_feedback") or "No feedback was provided.")

    st.header("Final article")
    render_article(draft)
    render_sources(sources)


def main() -> None:
    """Start or reconnect to one LangGraph run through a thin Streamlit UI."""

    st.set_page_config(page_title="Health Content Agent", layout="centered")
    st.title("Health Content Agent")
    st.write(
        "Research patient-focused article ideas, select one, and generate an "
        "evidence-grounded draft."
    )

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    workflow = get_workflow()

    with st.form("topic_form"):
        topic = st.text_input(
            "Health topic",
            placeholder="Type 2 diabetes",
        )
        start_requested = st.form_submit_button("Research article ideas")

    if start_requested:
        normalized_topic = topic.strip()
        if not normalized_topic:
            st.warning("Enter a health topic before starting.")
        else:
            thread_id = f"streamlit-{uuid4()}"
            st.session_state.thread_id = thread_id
            try:
                run_graph_with_progress(
                    workflow,
                    {
                        "run_id": thread_id,
                        "topic": normalized_topic,
                        "retry_count": 0,
                    },
                    graph_config(thread_id),
                    initial_label="Researching the topic...",
                )
            except Exception:
                st.error(
                    "The workflow encountered an unexpected error. "
                    "Please start a new run."
                )

    thread_id = st.session_state.thread_id
    if thread_id is None:
        return

    config = graph_config(thread_id)
    state: GraphState = dict(workflow.get_state(config).values)
    if not state:
        st.warning("This in-memory workflow session expired. Start a new run.")
        return

    final_status = state.get("final_status")
    if final_status in {"completed", "human_review_required", "failed"}:
        render_finished_state(state)
        return

    article_ideas = [
        ArticleIdea.model_validate(idea)
        for idea in state.get("article_ideas", [])
    ]
    if not article_ideas:
        return

    selected_id = render_article_ideas(article_ideas, thread_id)
    if selected_id is None:
        return

    # Resume the same checkpoint. The graph validates this ID against its own
    # ArticleIdea list and writes the complete selected object to selected_idea.
    try:
        run_graph_with_progress(
            workflow,
            Command(resume=selected_id),
            config,
            initial_label="Researching the selected article...",
        )
    except Exception:
        st.error(
            "The workflow encountered an unexpected error while continuing. "
            "Please start a new run."
        )
        return

    updated_state: GraphState = dict(workflow.get_state(config).values)
    if updated_state.get("final_status") in {
        "completed",
        "human_review_required",
        "failed",
    }:
        render_finished_state(updated_state)


if __name__ == "__main__":
    main()
