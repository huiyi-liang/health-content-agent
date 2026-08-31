"""Standalone evidence-grounded Reviewer and deterministic route helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from llm import create_nebius_llm
from prompts import reviewer_system_prompt, reviewer_user_prompt
from state import (
    ArticleDraft,
    FailureType,
    GraphState,
    ReviewStatus,
    Source,
    WorkflowError,
)


STYLE_GUIDE_PATH = Path(__file__).resolve().parents[1] / "config" / "article_style.md"
ReviewRoute = Literal["human_review", "deep_research", "rewrite"]
REWRITE_FAILURES: set[FailureType] = {
    "unsupported_claim",
    "style_or_readability",
    "patient_value",
}


class ReviewerValidationError(ValueError):
    """Raised when a structured review violates deterministic review rules."""


class ReviewerOutput(BaseModel):
    """Strict structured output expected from the Reviewer LLM."""

    model_config = ConfigDict(extra="forbid")

    review_status: ReviewStatus
    failure_type: FailureType | None
    flagged_sections: list[str]
    review_feedback: str | None


def load_editorial_style_guide() -> str:
    """Load the same reusable editorial guide used by the Writer."""

    style_guide = STYLE_GUIDE_PATH.read_text(encoding="utf-8").strip()
    if not style_guide:
        raise ValueError("The editorial style guide is empty.")
    return style_guide


def format_reviewer_draft(draft: ArticleDraft) -> str:
    """Make stable section IDs and reader-facing headings easy to distinguish."""

    formatted_sections: list[str] = [f"Article title: {draft.title}"]
    for section in draft.sections:
        formatted_sections.append(
            f"""[section_id: {section.section_id}]
Heading: {section.heading or "None"}
Content:
{section.content}"""
        )
    return "\n\n".join(formatted_sections)


def format_reviewer_evidence(deep_research_sources: list[Source]) -> str:
    """Format complete stored evidence without rewriting its Highlights."""

    formatted_sources: list[str] = []
    for source in deep_research_sources:
        highlights = "\n".join(f"- {highlight}" for highlight in source.highlights)
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


def generate_review(
    draft: ArticleDraft,
    deep_research_sources: list[Source],
    style_guide: str,
    *,
    llm: Any | None = None,
) -> ReviewerOutput:
    """Use the shared Nebius model for one structured Reviewer judgment."""

    model = llm or create_nebius_llm()
    structured_model = model.with_structured_output(
        ReviewerOutput,
        method="json_schema",
        strict=True,
    )
    result = structured_model.invoke(
        [
            ("system", reviewer_system_prompt(style_guide)),
            (
                "human",
                reviewer_user_prompt(
                    draft=format_reviewer_draft(draft),
                    research_evidence=format_reviewer_evidence(
                        deep_research_sources
                    ),
                    allowed_section_ids=[
                        section.section_id for section in draft.sections
                    ],
                ),
            ),
        ]
    )
    return ReviewerOutput.model_validate(result)


def normalize_flagged_section_ids(
    review: ReviewerOutput,
    draft: ArticleDraft,
) -> ReviewerOutput:
    """Remove a known label only when it reveals an exact valid section ID."""

    valid_section_ids = {section.section_id for section in draft.sections}
    normalized_ids: list[str] = []

    for returned_value in review.flagged_sections:
        stripped_value = returned_value.strip()
        normalized_value = stripped_value

        label, separator, possible_id = stripped_value.partition(":")
        if separator and label.strip().casefold() == "section_id":
            candidate = possible_id.strip()
            if candidate in valid_section_ids:
                normalized_value = candidate

        normalized_ids.append(normalized_value)

    return review.model_copy(update={"flagged_sections": normalized_ids})


def validate_review(review: ReviewerOutput, draft: ArticleDraft) -> None:
    """Check status consistency and ensure every flag is a real section ID."""

    valid_section_ids = {section.section_id for section in draft.sections}
    unknown_section_ids = set(review.flagged_sections) - valid_section_ids
    if unknown_section_ids:
        unknown = ", ".join(sorted(unknown_section_ids))
        raise ReviewerValidationError(
            f"Reviewer returned unknown section IDs: {unknown}."
        )

    if review.review_status == "PASS":
        if (
            review.failure_type is not None
            or review.flagged_sections
            or review.review_feedback is not None
        ):
            raise ReviewerValidationError(
                "PASS requires failure_type=None, flagged_sections=[], and "
                "review_feedback=None."
            )
        return

    if review.failure_type is None:
        raise ReviewerValidationError("FAIL requires one failure_type.")
    if review.failure_type in REWRITE_FAILURES and not review.flagged_sections:
        raise ReviewerValidationError(
            "Writer-routed failures require at least one flagged section ID."
        )
    if review.review_feedback is None or not review.review_feedback.strip():
        raise ReviewerValidationError(
            "FAIL requires specific, actionable review_feedback."
        )


def reviewer_node(
    state: GraphState,
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Review the draft and return only the four editorial review fields."""

    draft = state.get("draft")
    if draft is None:
        return _error_update("missing_draft", "Reviewer requires an ArticleDraft.")

    deep_research_sources = state.get("deep_research_sources", [])
    if not deep_research_sources:
        return _error_update(
            "missing_deep_research_evidence",
            "Reviewer requires at least one Deep Research Source.",
        )

    try:
        validated_draft = ArticleDraft.model_validate(draft)
        style_guide = load_editorial_style_guide()
    except Exception as exc:
        return _error_update("reviewer_input_error", str(exc))

    try:
        review = generate_review(
            validated_draft,
            deep_research_sources,
            style_guide,
            llm=llm,
        )
    except Exception as exc:
        return _error_update("review_generation_error", str(exc))

    try:
        review = normalize_flagged_section_ids(review, validated_draft)
        validate_review(review, validated_draft)
    except ReviewerValidationError as exc:
        return _error_update("review_validation_error", str(exc))

    return {
        "review_status": review.review_status,
        "failure_type": review.failure_type,
        "flagged_sections": list(review.flagged_sections),
        "review_feedback": review.review_feedback,
        "workflow_error": None,
    }


def route_review_result(state: GraphState) -> ReviewRoute:
    """Map review state to a route without asking an LLM what happens next."""

    retry_count = state.get("retry_count", 0)
    if not isinstance(retry_count, int) or retry_count < 0:
        raise ValueError("retry_count must be a non-negative integer.")
    if retry_count >= 3:
        return "human_review"

    review_status = state.get("review_status")
    failure_type = state.get("failure_type")
    if review_status == "PASS":
        return "human_review"
    if review_status == "FAIL" and failure_type == "insufficient_research":
        return "deep_research"
    if review_status == "FAIL" and failure_type in REWRITE_FAILURES:
        return "rewrite"

    raise ValueError(
        "Cannot route an incomplete or inconsistent review result."
    )


def _error_update(error_type: str, message: str) -> dict[str, WorkflowError]:
    """Create an error update identifying the Reviewer node."""

    return {
        "workflow_error": WorkflowError(
            node="reviewer",
            type=error_type,
            message=message,
        )
    }


def _print_reviewer_result(state: GraphState) -> None:
    """Display the judgment and route without executing that route."""

    print()
    print("Reviewer result")
    print(f"Review status: {state.get('review_status')}")
    print(f"Primary failure: {state.get('failure_type')}")
    print(f"Flagged sections: {state.get('flagged_sections', [])}")
    print(f"Feedback: {state.get('review_feedback')}")
    print(f"Deterministic next route: {route_review_result(state)}")
    print(f"Retry count: {state.get('retry_count', 0)}")
