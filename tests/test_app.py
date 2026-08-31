"""Small tests for deterministic Streamlit presentation helpers."""

from app import (
    article_display_blocks,
    flagged_section_labels,
    format_workflow_error,
    source_display_entries,
)
from state import ArticleDraft, ArticleSection, Source, WorkflowError


def _draft() -> ArticleDraft:
    return ArticleDraft(
        title="A readable article",
        sections=[
            ArticleSection(
                section_id="introduction",
                heading=None,
                content="Introduction with a citation [R1].",
            ),
            ArticleSection(
                section_id="questions_for_your_care_team",
                heading="Questions for Your Care Team",
                content="A reader-facing section.",
            ),
        ],
    )


def test_article_blocks_preserve_content_without_exposing_section_ids() -> None:
    blocks = article_display_blocks(_draft())

    assert blocks == [
        (None, "Introduction with a citation [R1]."),
        ("Questions for Your Care Team", "A reader-facing section."),
    ]
    assert "introduction" not in str(blocks)
    assert "questions_for_your_care_team" not in str(blocks)


def test_source_entries_use_stored_source_values() -> None:
    source = Source(
        source_id="R1",
        query="care questions",
        title="Stored Source Title",
        url="https://example.org/source",
        publication_date=None,
        source_type="web",
        highlights=["Stored evidence."],
    )

    assert source_display_entries([source]) == [
        ("R1", "Stored Source Title", "https://example.org/source")
    ]


def test_flagged_section_ids_become_reader_facing_labels() -> None:
    assert flagged_section_labels(
        _draft(),
        ["introduction", "questions_for_your_care_team"],
    ) == ["Introduction", "Questions for Your Care Team"]


def test_workflow_error_is_short_and_has_no_traceback() -> None:
    message = format_workflow_error(
        WorkflowError(
            node="deep_research",
            type="search_error",
            message="You.com request failed.",
        )
    )

    assert message == (
        "The workflow stopped during Deep Research: You.com request failed."
    )
    assert "Traceback" not in message
