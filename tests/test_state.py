"""Small tests for the Phase 1 data and state contracts."""

from state import ArticleDraft, ArticleIdea, ArticleSection, GraphState, Source


def test_valid_source_can_be_created() -> None:
    source = Source(
        source_id="D1",
        query="type 2 diabetes sleep patient questions",
        title="Sleep and Type 2 Diabetes",
        url="https://example.com/sleep-and-diabetes",
        publication_date=None,
        source_type="web",
        highlights=["Sleep can be part of a patient's management discussion."],
    )

    assert source.source_id == "D1"
    assert source.source_type == "web"


def test_valid_article_idea_can_be_created() -> None:
    idea = ArticleIdea(
        idea_id="I1",
        title="How Sleep Habits Fit Into Type 2 Diabetes Management",
        article_angle="Questions patients can discuss with their care team.",
        reason="It connects a daily experience with an actionable care need.",
    )

    assert idea.idea_id == "I1"
    assert idea.article_angle.startswith("Questions")


def test_article_draft_can_contain_structured_sections() -> None:
    draft = ArticleDraft(
        title="Sleep and Type 2 Diabetes",
        sections=[
            ArticleSection(
                section_id="introduction",
                heading=None,
                content="Sleep and diabetes management can affect daily life.",
            ),
            ArticleSection(
                section_id="questions-for-care-team",
                heading="Questions to Ask Your Care Team",
                content="Prepare questions before your next appointment.",
            ),
        ],
    )

    assert len(draft.sections) == 2
    assert draft.sections[1].section_id == "questions-for-care-team"


def test_initial_graph_state_needs_only_run_id_and_topic() -> None:
    state: GraphState = {
        "run_id": "run-001",
        "topic": "Type 2 diabetes and sleep",
    }

    assert state["run_id"] == "run-001"
    assert state["topic"] == "Type 2 diabetes and sleep"


def test_initial_review_fields_can_be_unset() -> None:
    state: GraphState = {
        "run_id": "run-002",
        "topic": "Type 2 diabetes",
        "review_status": None,
        "failure_type": None,
        "flagged_sections": [],
        "review_feedback": None,
    }

    assert state["review_status"] is None
    assert state["failure_type"] is None
    assert state["flagged_sections"] == []
    assert state["review_feedback"] is None
