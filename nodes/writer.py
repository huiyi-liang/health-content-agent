"""Standalone evidence-grounded Writer Agent node for Phase 6."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm import create_nebius_llm
from prompts import (
    writer_revision_system_prompt,
    writer_revision_user_prompt,
    writer_system_prompt,
    writer_user_prompt,
)
from state import (
    ArticleDraft,
    ArticleIdea,
    ArticleSection,
    GraphState,
    Source,
    WorkflowError,
)


STYLE_GUIDE_PATH = Path(__file__).resolve().parents[1] / "config" / "webmd_style.md"
SECTION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
CITATION_PATTERN = re.compile(r"\[(R\d+)\]")


class WriterValidationError(ValueError):
    """Raised when a structured draft violates deterministic Writer rules."""


class WriterSectionOutput(BaseModel):
    """Strict LLM transport shape with an explicitly nullable heading."""

    model_config = ConfigDict(extra="forbid")

    section_id: str
    heading: str | None
    content: str


class WriterDraftOutput(BaseModel):
    """Strict structured-output shape converted into the shared ArticleDraft."""

    model_config = ConfigDict(extra="forbid")

    title: str
    sections: list[WriterSectionOutput]


class WriterRevisionOutput(BaseModel):
    """Structured transport containing only sections requested for revision."""

    model_config = ConfigDict(extra="forbid")

    sections: list[WriterSectionOutput] = Field(min_length=1)


def load_editorial_style_guide() -> str:
    """Load the reusable editorial guide from the project's config directory."""

    style_guide = STYLE_GUIDE_PATH.read_text(encoding="utf-8").strip()
    if not style_guide:
        raise ValueError("The editorial style guide is empty.")
    return style_guide


def format_writer_evidence(deep_research_sources: list[Source]) -> str:
    """Format all stored Deep Research evidence without rewriting Highlights."""

    formatted_sources: list[str] = []
    for source in deep_research_sources:
        highlights = "\n".join(f"- {highlight}" for highlight in source.highlights)
        formatted_sources.append(
            f"""[{source.source_id}]
Title: {source.title}
URL: {source.url}
Date: {source.publication_date or "Not available"}
Highlights:
{highlights}"""
        )

    return "\n\n".join(formatted_sources)


def format_current_draft(draft: ArticleDraft) -> str:
    """Expose stable section IDs and complete current content for revision."""

    parts: list[str] = [f"Article title: {draft.title}"]
    for section in draft.sections:
        parts.append(
            f"""[section_id: {section.section_id}]
Heading: {section.heading or "None"}
Content:
{section.content}"""
        )
    return "\n\n".join(parts)


def generate_article_draft(
    selected_idea: ArticleIdea,
    deep_research_sources: list[Source],
    style_guide: str,
    *,
    llm: Any | None = None,
) -> ArticleDraft:
    """Use one structured LLM call to create an ArticleDraft."""

    model = llm or create_nebius_llm()
    structured_model = model.with_structured_output(
        WriterDraftOutput,
        method="json_schema",
        strict=True,
    )
    result = structured_model.invoke(
        [
            ("system", writer_system_prompt(style_guide)),
            (
                "human",
                writer_user_prompt(
                    idea_id=selected_idea.idea_id,
                    title=selected_idea.title,
                    article_angle=selected_idea.article_angle,
                    reason=selected_idea.reason,
                    research_evidence=format_writer_evidence(
                        deep_research_sources
                    ),
                ),
            ),
        ]
    )
    structured_result = WriterDraftOutput.model_validate(result)
    return ArticleDraft.model_validate(structured_result.model_dump())


def generate_article_revision(
    selected_idea: ArticleIdea,
    current_draft: ArticleDraft,
    failure_type: str,
    retry_count: int,
    flagged_sections: list[str],
    review_feedback: str,
    deep_research_sources: list[Source],
    style_guide: str,
    *,
    llm: Any | None = None,
) -> WriterRevisionOutput:
    """Use one structured LLM call to revise only flagged sections."""

    model = llm or create_nebius_llm()
    structured_model = model.with_structured_output(
        WriterRevisionOutput,
        method="json_schema",
        strict=True,
    )
    result = structured_model.invoke(
        [
            ("system", writer_revision_system_prompt(style_guide)),
            (
                "human",
                writer_revision_user_prompt(
                    idea_id=selected_idea.idea_id,
                    title=selected_idea.title,
                    article_angle=selected_idea.article_angle,
                    reason=selected_idea.reason,
                    failure_type=failure_type,
                    retry_count=retry_count,
                    flagged_sections=flagged_sections,
                    review_feedback=review_feedback,
                    current_draft=format_current_draft(current_draft),
                    research_evidence=format_writer_evidence(
                        deep_research_sources
                    ),
                ),
            ),
        ]
    )
    return WriterRevisionOutput.model_validate(result)


def validate_draft_structure(draft: ArticleDraft) -> None:
    """Validate stable section IDs and other basic structured-draft rules."""

    if not draft.title.strip():
        raise WriterValidationError("Draft title cannot be blank.")
    if not draft.sections:
        raise WriterValidationError("Draft must contain at least one section.")

    seen_section_ids: set[str] = set()
    for section in draft.sections:
        if not SECTION_ID_PATTERN.fullmatch(section.section_id):
            raise WriterValidationError(
                "Section IDs must be non-empty snake_case identifiers."
            )
        if section.section_id in seen_section_ids:
            raise WriterValidationError(
                f"Duplicate section ID: {section.section_id}."
            )
        if section.heading is not None and not section.heading.strip():
            raise WriterValidationError(
                f"Section heading cannot be blank: {section.section_id}."
            )
        if not section.content.strip():
            raise WriterValidationError(
                f"Section content cannot be blank: {section.section_id}."
            )
        seen_section_ids.add(section.section_id)


def validate_draft_citations(
    draft: ArticleDraft,
    deep_research_sources: list[Source],
) -> None:
    """Ensure every [R#] used by the draft exists in stored research Sources."""

    available_source_ids = {source.source_id for source in deep_research_sources}
    draft_parts = [draft.title]
    for section in draft.sections:
        if section.heading is not None:
            draft_parts.append(section.heading)
        draft_parts.append(section.content)

    used_source_ids = set(CITATION_PATTERN.findall("\n".join(draft_parts)))
    nonexistent_ids = used_source_ids - available_source_ids
    if nonexistent_ids:
        invalid_ids = ", ".join(
            sorted(nonexistent_ids, key=lambda source_id: int(source_id[1:]))
        )
        raise WriterValidationError(
            f"Draft cites nonexistent research source IDs: {invalid_ids}."
        )


def writer_node(
    state: GraphState,
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Create and validate a draft using selected idea and stored research."""

    selected_idea = state.get("selected_idea")
    if selected_idea is None:
        return _error_update(
            "missing_selected_idea",
            "Writer requires a complete selected ArticleIdea.",
        )

    deep_research_sources = state.get("deep_research_sources", [])
    if not deep_research_sources:
        return _error_update(
            "missing_deep_research_evidence",
            "Writer requires at least one Deep Research Source.",
        )

    try:
        validated_idea = ArticleIdea.model_validate(selected_idea)
        style_guide = load_editorial_style_guide()
    except Exception as exc:
        return _error_update("writer_input_error", str(exc))

    try:
        draft = generate_article_draft(
            validated_idea,
            deep_research_sources,
            style_guide,
            llm=llm,
        )
    except Exception as exc:
        return _error_update("draft_generation_error", str(exc))

    try:
        validate_draft_structure(draft)
        validate_draft_citations(draft, deep_research_sources)
    except WriterValidationError as exc:
        return _error_update("draft_validation_error", str(exc))

    return {
        "draft": draft,
        "workflow_error": None,
    }


def writer_revision_node(
    state: GraphState,
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Revise flagged sections and deterministically preserve all others."""

    selected_idea = state.get("selected_idea")
    current_draft = state.get("draft")
    deep_research_sources = state.get("deep_research_sources", [])
    flagged_sections = list(state.get("flagged_sections", []))
    review_feedback = state.get("review_feedback")
    failure_type = state.get("failure_type")
    retry_count = state.get("retry_count", 0)

    if selected_idea is None:
        return _error_update(
            "missing_selected_idea",
            "Writer revision requires a complete selected ArticleIdea.",
        )
    if current_draft is None:
        return _error_update(
            "missing_draft",
            "Writer revision requires the current ArticleDraft.",
        )
    if not deep_research_sources:
        return _error_update(
            "missing_deep_research_evidence",
            "Writer revision requires at least one Deep Research Source.",
        )
    if not flagged_sections:
        return _error_update(
            "missing_flagged_sections",
            "Writer revision requires at least one flagged section ID.",
        )
    if review_feedback is None or not review_feedback.strip():
        return _error_update(
            "missing_review_feedback",
            "Writer revision requires Reviewer feedback.",
        )
    if failure_type not in {
        "unsupported_claim",
        "style_or_readability",
        "patient_value",
    }:
        return _error_update(
            "invalid_revision_failure_type",
            "Writer revision requires a failure type routed to Writer.",
        )
    if not isinstance(retry_count, int) or retry_count < 1:
        return _error_update(
            "invalid_retry_count",
            "Writer revision requires a positive automated retry count.",
        )

    try:
        validated_idea = ArticleIdea.model_validate(selected_idea)
        validated_draft = ArticleDraft.model_validate(current_draft)
        style_guide = load_editorial_style_guide()

        draft_section_ids = {
            section.section_id for section in validated_draft.sections
        }
        unknown_flags = set(flagged_sections) - draft_section_ids
        if unknown_flags:
            unknown = ", ".join(sorted(unknown_flags))
            raise WriterValidationError(
                f"Writer revision received unknown section IDs: {unknown}."
            )
        if len(set(flagged_sections)) != len(flagged_sections):
            raise WriterValidationError("Flagged section IDs must be unique.")
    except Exception as exc:
        return _error_update("revision_input_error", str(exc))

    try:
        revision = generate_article_revision(
            validated_idea,
            validated_draft,
            failure_type,
            retry_count,
            flagged_sections,
            review_feedback,
            deep_research_sources,
            style_guide,
            llm=llm,
        )
    except Exception as exc:
        return _error_update("revision_generation_error", str(exc))

    try:
        returned_ids = [section.section_id for section in revision.sections]
        if len(set(returned_ids)) != len(returned_ids):
            raise WriterValidationError(
                "Writer revision returned duplicate section IDs."
            )
        if set(returned_ids) != set(flagged_sections):
            raise WriterValidationError(
                "Writer revision must return exactly the flagged section IDs."
            )

        replacements = {
            section.section_id: ArticleSection.model_validate(
                section.model_dump()
            )
            for section in revision.sections
        }
        revised_draft = ArticleDraft(
            title=validated_draft.title,
            sections=[
                replacements.get(section.section_id, section)
                for section in validated_draft.sections
            ],
        )
        validate_draft_structure(revised_draft)
        validate_draft_citations(revised_draft, deep_research_sources)
    except WriterValidationError as exc:
        return _error_update("revision_validation_error", str(exc))

    return {
        "draft": revised_draft,
        "workflow_error": None,
    }


def _error_update(error_type: str, message: str) -> dict[str, WorkflowError]:
    """Create an error update identifying the Writer node."""

    return {
        "workflow_error": WorkflowError(
            node="writer",
            type=error_type,
            message=message,
        )
    }


def _print_writer_result(state: GraphState) -> None:
    """Render the structured draft and a deterministic source list."""

    selected_idea = state["selected_idea"]
    draft = state["draft"]
    if selected_idea is None or draft is None:
        return

    sources = state.get("deep_research_sources", [])

    print()
    print("Selected idea")
    print(f"Title: {selected_idea.title}")
    print(f"Article angle: {selected_idea.article_angle}")
    print()
    print("Research summary")
    print(f"Deep Research sources: {len(sources)}")
    print(f"Available source IDs: {', '.join(source.source_id for source in sources)}")
    print()
    print("Draft article")
    print()
    print(f"# {draft.title}")
    for section in draft.sections:
        print()
        if section.heading is not None:
            print(f"## {section.heading}")
            print()
        print(section.content)

    print()
    print("## Sources")
    print()
    for source in sources:
        print(f"[{source.source_id}] {source.title} — {source.url}")
