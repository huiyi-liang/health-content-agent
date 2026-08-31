"""Focused retry, revision, and LangGraph integration tests for Phase 8."""

from copy import deepcopy
from typing import Any, Sequence

from langgraph.types import Command

from config import PREFERRED_MEDICAL_DOMAINS
from graph import (
    _stream_until_pause_or_end,
    build_workflow,
    create_in_memory_checkpointer,
)
from nodes.research import (
    DeepResearchQueryPlan,
    deep_research_node,
    plan_deep_research_retry_queries,
)
from nodes.writer import WriterRevisionOutput, writer_revision_node
from prompts import writer_revision_user_prompt
from state import (
    ArticleDraft,
    ArticleIdea,
    ArticleSection,
    GraphState,
    Source,
    WorkflowError,
)


SELECTED_IDEA = ArticleIdea(
    idea_id="A1",
    title="Questions to Ask Before Changing a Diabetes Medication",
    article_angle="Explain practical questions to discuss before a medication change.",
    reason="Patients may need help preparing for treatment decisions.",
)

ORIGINAL_QUERIES = ["original query one", "original query two", "original query three"]
RETRY_QUERIES = ["new gap query one", "new gap query two", "new gap query three"]
REVIEW_FEEDBACK = "Find evidence about monitoring after a medication change."


def _source(source_id: str = "R1", query: str = "original query one") -> Source:
    """Create one normalized research Source fixture."""

    return Source(
        source_id=source_id,
        query=query,
        title=f"Evidence {source_id}",
        url=f"https://example.org/{source_id.lower()}",
        publication_date=None,
        source_type="web",
        highlights=[f"Complete Highlight for {source_id}."],
    )


def _draft(content_suffix: str = "") -> ArticleDraft:
    """Create a three-section draft so preservation is easy to verify."""

    return ArticleDraft(
        title="Questions to Ask Before Changing a Diabetes Medication",
        sections=[
            ArticleSection(
                section_id="introduction",
                heading=None,
                content="Introduction stays unchanged. [R1]",
            ),
            ArticleSection(
                section_id="monitoring_questions",
                heading="Monitoring Questions",
                content=f"Ask what monitoring may change. [R1]{content_suffix}",
            ),
            ArticleSection(
                section_id="closing",
                heading="Preparing for Your Visit",
                content="Closing stays unchanged. [R1]",
            ),
        ],
    )


class RetryFakeStructuredModel:
    """Verify retry context and return controlled new queries."""

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        assert "targeted Deep Research retry" in messages[0][1]
        planner_input = messages[1][1]
        assert REVIEW_FEEDBACK in planner_input
        assert "original query one" in planner_input
        assert "[R1]" in planner_input
        assert "Complete Highlight for R1." in planner_input
        return {"queries": RETRY_QUERIES}


class RetryFakeLLM:
    """Mimic structured output for the Deep Research retry planner."""

    def with_structured_output(
        self,
        schema: type[DeepResearchQueryPlan],
        *,
        method: str,
        strict: bool,
    ) -> RetryFakeStructuredModel:
        assert schema is DeepResearchQueryPlan
        assert method == "json_schema"
        assert strict is True
        return RetryFakeStructuredModel()


def test_deep_research_retry_prompt_includes_feedback_and_prior_research() -> None:
    plan = plan_deep_research_retry_queries(
        SELECTED_IDEA,
        REVIEW_FEEDBACK,
        ORIGINAL_QUERIES,
        [_source()],
        llm=RetryFakeLLM(),
    )

    assert plan.queries == RETRY_QUERIES


def test_deep_research_retry_appends_queries_sources_and_continues_ids() -> None:
    existing_sources = [_source("R1"), _source("R2", "original query two")]
    original_sources = deepcopy(existing_sources)
    requested_queries: list[str] = []

    def fake_retry_planner(
        selected_idea: ArticleIdea,
        feedback: str,
        existing_queries: list[str],
        existing_evidence: list[Source],
    ) -> DeepResearchQueryPlan:
        assert selected_idea == SELECTED_IDEA
        assert feedback == REVIEW_FEEDBACK
        assert existing_queries == ORIGINAL_QUERIES
        assert existing_evidence == existing_sources
        return DeepResearchQueryPlan(queries=RETRY_QUERIES)

    def fake_search_request(
        query: str,
        *,
        result_count: int,
        boost_domains: Sequence[str],
    ) -> dict[str, Any]:
        assert result_count == 4
        assert tuple(boost_domains) == PREFERRED_MEDICAL_DOMAINS
        requested_queries.append(query)
        return {
            "results": {
                "web": [
                    {
                        "title": f"Result for {query}",
                        "url": "https://example.org/repeated-url",
                        "contents": {"highlights": [f"Highlight for {query}"]},
                    }
                ]
            }
        }

    update = deep_research_node(
        {
            "run_id": "run-801",
            "topic": "Type 2 diabetes",
            "selected_idea": SELECTED_IDEA,
            "deep_research_queries": ORIGINAL_QUERIES,
            "deep_research_sources": existing_sources,
            "review_status": "FAIL",
            "failure_type": "insufficient_research",
            "review_feedback": REVIEW_FEEDBACK,
        },
        retry_query_planner=fake_retry_planner,
        search_request=fake_search_request,
    )

    assert requested_queries == RETRY_QUERIES
    assert update["deep_research_queries"] == [*ORIGINAL_QUERIES, *RETRY_QUERIES]
    assert [source.source_id for source in update["deep_research_sources"]] == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ]
    assert update["deep_research_sources"][:2] == original_sources
    assert existing_sources == original_sources


class RevisionFakeStructuredModel:
    """Verify revision context and return one controlled revised section."""

    def __init__(self, citation: str = "[R1]") -> None:
        self.citation = citation

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        assert "Revise only the sections" in messages[0][1]
        assert "unsupported or overstated claim" in messages[0][1]
        assert "Evidence rules override" in messages[0][1]
        assert "# Consumer Health Article Style Guide" in messages[0][1]
        revision_input = messages[1][1]
        assert "Primary failure type: unsupported_claim" in revision_input
        assert "Automated correction cycle: 1" in revision_input
        assert "Use deletion-first repair" in revision_input
        assert "- monitoring_questions" in revision_input
        assert REVIEW_FEEDBACK in revision_input
        assert "[section_id: introduction]" in revision_input
        assert "Complete Highlight for R1." in revision_input
        return {
            "sections": [
                {
                    "section_id": "monitoring_questions",
                    "heading": "Monitoring to Discuss",
                    "content": f"Ask which monitoring steps apply. {self.citation}",
                }
            ]
        }


class RevisionFakeLLM:
    """Mimic structured output for targeted Writer revision."""

    def __init__(self, citation: str = "[R1]") -> None:
        self.citation = citation

    def with_structured_output(
        self,
        schema: type[WriterRevisionOutput],
        *,
        method: str,
        strict: bool,
    ) -> RevisionFakeStructuredModel:
        assert schema is WriterRevisionOutput
        assert method == "json_schema"
        assert strict is True
        return RevisionFakeStructuredModel(self.citation)


def _revision_state() -> GraphState:
    """Create state containing all targeted Writer revision inputs."""

    return {
        "run_id": "run-802",
        "topic": "Type 2 diabetes",
        "selected_idea": SELECTED_IDEA,
        "deep_research_sources": [_source()],
        "draft": _draft(),
        "review_status": "FAIL",
        "failure_type": "unsupported_claim",
        "flagged_sections": ["monitoring_questions"],
        "review_feedback": REVIEW_FEEDBACK,
        "retry_count": 1,
    }


def test_writer_revision_changes_only_flagged_section_and_preserves_ids() -> None:
    state = _revision_state()
    original_draft = deepcopy(state["draft"])
    update = writer_revision_node(state, llm=RevisionFakeLLM())

    revised = update["draft"]
    assert revised.title == original_draft.title
    assert [section.section_id for section in revised.sections] == [
        section.section_id for section in original_draft.sections
    ]
    assert revised.sections[0] == original_draft.sections[0]
    assert revised.sections[2] == original_draft.sections[2]
    assert revised.sections[1].content == "Ask which monitoring steps apply. [R1]"
    assert state["draft"] == original_draft


def test_writer_revision_rejects_nonexistent_citation() -> None:
    update = writer_revision_node(_revision_state(), llm=RevisionFakeLLM("[R99]"))

    assert "draft" not in update
    assert update["workflow_error"].type == "revision_validation_error"
    assert "R99" in update["workflow_error"].message


def test_repeated_unsupported_claim_uses_mandatory_deletion_strategy() -> None:
    prompt = writer_revision_user_prompt(
        idea_id=SELECTED_IDEA.idea_id,
        title=SELECTED_IDEA.title,
        article_angle=SELECTED_IDEA.article_angle,
        reason=SELECTED_IDEA.reason,
        failure_type="unsupported_claim",
        retry_count=2,
        flagged_sections=["monitoring_questions"],
        review_feedback=REVIEW_FEEDBACK,
        current_draft="Current draft",
        research_evidence="[R1] Evidence",
    )

    assert "Automated correction cycle: 2" in prompt
    assert "repeated unsupported-claim correction" in prompt
    assert "Rebuild each flagged section conservatively from scratch" in prompt
    assert "at most two short medical factual sentences" in prompt


def _mock_discovery(state: GraphState) -> dict[str, Any]:
    return {
        "discovery_queries": ["q1", "q2", "q3"],
        "discovery_sources": [_source("D1")],
        "workflow_error": None,
    }


def _mock_opportunity(state: GraphState) -> dict[str, Any]:
    return {
        "article_ideas": [
            SELECTED_IDEA,
            ArticleIdea(
                idea_id="A2",
                title="Idea two",
                article_angle="A second angle.",
                reason="A second reason.",
            ),
            ArticleIdea(
                idea_id="A3",
                title="Idea three",
                article_angle="A third angle.",
                reason="A third reason.",
            ),
        ],
        "workflow_error": None,
    }


def _mock_writer(state: GraphState) -> dict[str, Any]:
    return {"draft": _draft(), "workflow_error": None}


def _pass_review(state: GraphState) -> dict[str, Any]:
    return {
        "review_status": "PASS",
        "failure_type": None,
        "flagged_sections": [],
        "review_feedback": None,
        "workflow_error": None,
    }


def _run_graph(app: Any, run_id: str = "graph-test") -> GraphState:
    """Invoke through the real interrupt and resume with the human's A1 choice."""

    config = {"configurable": {"thread_id": run_id}}
    paused = app.invoke(
        {"run_id": run_id, "topic": "Type 2 diabetes", "retry_count": 0},
        config,
    )
    assert "__interrupt__" in paused
    interrupt_payload = paused["__interrupt__"][0].value
    assert len(interrupt_payload["article_ideas"]) == 3

    app.invoke(Command(resume="A1"), config)
    return app.get_state(config).values


def test_langgraph_happy_path_pauses_for_human_then_completes() -> None:
    def initial_research(state: GraphState) -> dict[str, Any]:
        assert state["selected_idea"] == SELECTED_IDEA
        return {
            "deep_research_queries": ["r1", "r2", "r3"],
            "deep_research_sources": [_source()],
            "workflow_error": None,
        }

    app = build_workflow(
        discovery=_mock_discovery,
        opportunity=_mock_opportunity,
        deep_research=initial_research,
        writer=_mock_writer,
        reviewer=_pass_review,
    )
    final_state = _run_graph(app, "happy-path")

    assert final_state["selected_idea"] == SELECTED_IDEA
    assert final_state["final_status"] == "completed"
    assert final_state["retry_count"] == 0


def test_terminal_stream_displays_current_reviewer_update(
    capsys: Any,
) -> None:
    def initial_research(state: GraphState) -> dict[str, Any]:
        return {
            "deep_research_queries": ["r1", "r2", "r3"],
            "deep_research_sources": [_source()],
            "workflow_error": None,
        }

    app = build_workflow(
        discovery=_mock_discovery,
        opportunity=_mock_opportunity,
        deep_research=initial_research,
        writer=_mock_writer,
        reviewer=_pass_review,
    )
    config = {"configurable": {"thread_id": "terminal-stream"}}
    interrupts = _stream_until_pause_or_end(
        app,
        {
            "run_id": "terminal-stream",
            "topic": "Type 2 diabetes",
            "retry_count": 0,
        },
        config,
    )
    assert interrupts

    _stream_until_pause_or_end(app, Command(resume="A1"), config)
    output = capsys.readouterr().out

    assert "Reviewer: PASS" in output
    assert "Route: Final completion" in output
    assert "Final status: completed" in output


def test_checkpointer_restores_state_models_without_unregistered_warning(
    caplog: Any,
) -> None:
    checkpointer = create_in_memory_checkpointer()
    encoded = checkpointer.serde.dumps_typed(_source())
    restored = checkpointer.serde.loads_typed(encoded)

    assert restored == _source()
    assert "Deserializing unregistered type" not in caplog.text


def test_langgraph_writer_retry_revises_then_completes() -> None:
    review_calls = 0
    revision_calls = 0

    def research(state: GraphState) -> dict[str, Any]:
        return {
            "deep_research_queries": ["r1", "r2", "r3"],
            "deep_research_sources": [_source()],
            "workflow_error": None,
        }

    def reviewer(state: GraphState) -> dict[str, Any]:
        nonlocal review_calls
        review_calls += 1
        if review_calls == 1:
            return {
                "review_status": "FAIL",
                "failure_type": "unsupported_claim",
                "flagged_sections": ["monitoring_questions"],
                "review_feedback": "Remove the unsupported claim.",
                "workflow_error": None,
            }
        return _pass_review(state)

    def revision(state: GraphState) -> dict[str, Any]:
        nonlocal revision_calls
        revision_calls += 1
        return {"draft": _draft(" Revised."), "workflow_error": None}

    app = build_workflow(
        discovery=_mock_discovery,
        opportunity=_mock_opportunity,
        deep_research=research,
        writer=_mock_writer,
        writer_revision=revision,
        reviewer=reviewer,
    )
    final_state = _run_graph(app, "writer-retry")

    assert final_state["final_status"] == "completed"
    assert final_state["retry_count"] == 1
    assert revision_calls == 1
    assert review_calls == 2


def test_langgraph_deep_research_retry_appends_then_completes() -> None:
    research_calls = 0
    writer_calls = 0
    review_calls = 0

    def research(state: GraphState) -> dict[str, Any]:
        nonlocal research_calls
        research_calls += 1
        if research_calls == 1:
            return {
                "deep_research_queries": ["r1", "r2", "r3"],
                "deep_research_sources": [_source("R1")],
                "workflow_error": None,
            }
        assert state["failure_type"] == "insufficient_research"
        return {
            "deep_research_queries": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "deep_research_sources": [_source("R1"), _source("R2", "r4")],
            "workflow_error": None,
        }

    def writer(state: GraphState) -> dict[str, Any]:
        nonlocal writer_calls
        writer_calls += 1
        return {"draft": _draft(), "workflow_error": None}

    def reviewer(state: GraphState) -> dict[str, Any]:
        nonlocal review_calls
        review_calls += 1
        if review_calls == 1:
            return {
                "review_status": "FAIL",
                "failure_type": "insufficient_research",
                "flagged_sections": ["monitoring_questions"],
                "review_feedback": REVIEW_FEEDBACK,
                "workflow_error": None,
            }
        return _pass_review(state)

    app = build_workflow(
        discovery=_mock_discovery,
        opportunity=_mock_opportunity,
        deep_research=research,
        writer=writer,
        reviewer=reviewer,
    )
    final_state = _run_graph(app, "research-retry")

    assert final_state["final_status"] == "completed"
    assert final_state["retry_count"] == 1
    assert len(final_state["deep_research_queries"]) == 6
    assert [source.source_id for source in final_state["deep_research_sources"]] == [
        "R1",
        "R2",
    ]
    assert research_calls == 2
    assert writer_calls == 2


def test_langgraph_stops_before_fourth_automated_retry() -> None:
    revision_calls = 0
    review_calls = 0

    def research(state: GraphState) -> dict[str, Any]:
        return {
            "deep_research_queries": ["r1", "r2", "r3"],
            "deep_research_sources": [_source()],
            "workflow_error": None,
        }

    def always_fail_review(state: GraphState) -> dict[str, Any]:
        nonlocal review_calls
        review_calls += 1
        return {
            "review_status": "FAIL",
            "failure_type": "style_or_readability",
            "flagged_sections": ["monitoring_questions"],
            "review_feedback": "Make this section easier to read.",
            "workflow_error": None,
        }

    def revision(state: GraphState) -> dict[str, Any]:
        nonlocal revision_calls
        revision_calls += 1
        return {"draft": _draft(f" Revision {revision_calls}."), "workflow_error": None}

    app = build_workflow(
        discovery=_mock_discovery,
        opportunity=_mock_opportunity,
        deep_research=research,
        writer=_mock_writer,
        writer_revision=revision,
        reviewer=always_fail_review,
    )
    final_state = _run_graph(app, "retry-limit")

    assert final_state["final_status"] == "human_review_required"
    assert final_state["retry_count"] == 3
    assert revision_calls == 3
    assert review_calls == 4
    assert "Revision 3." in final_state["draft"].sections[1].content


def test_langgraph_marks_unrecoverable_node_error_as_failed() -> None:
    def failed_discovery(state: GraphState) -> dict[str, Any]:
        return {
            "workflow_error": WorkflowError(
                node="discovery_research",
                type="search_error",
                message="Mocked You.com failure.",
            )
        }

    app = build_workflow(discovery=failed_discovery)
    config = {"configurable": {"thread_id": "failed-path"}}
    result = app.invoke(
        {
            "run_id": "failed-path",
            "topic": "Type 2 diabetes",
            "retry_count": 0,
        },
        config,
    )

    assert result["final_status"] == "failed"
    assert result["workflow_error"].type == "search_error"
    assert "__interrupt__" not in result
