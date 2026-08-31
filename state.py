"""Data contracts shared by the health-content workflow."""

from typing import Literal, NotRequired, Required, TypedDict

from pydantic import BaseModel, ConfigDict


SourceType = Literal["web", "news"]
ReviewStatus = Literal["PASS", "FAIL"]
FailureType = Literal[
    "insufficient_research",
    "unsupported_claim",
    "style_or_readability",
    "patient_value",
]


class ContractModel(BaseModel):
    """Base settings shared by the workflow's validated data models."""

    model_config = ConfigDict(extra="forbid")


class Source(ContractModel):
    """One normalized search result used as research evidence."""

    source_id: str
    query: str
    title: str
    url: str
    publication_date: str | None = None
    source_type: SourceType
    highlights: list[str]


class ArticleIdea(ContractModel):
    """One patient-oriented article opportunity."""

    idea_id: str
    title: str
    article_angle: str
    reason: str


class ArticleSection(ContractModel):
    """A stable article section with a reader-facing heading and content."""

    section_id: str
    heading: str | None = None
    content: str


class ArticleDraft(ContractModel):
    """A structured article draft."""

    title: str
    sections: list[ArticleSection]


class WorkflowError(ContractModel):
    """A system or tool failure, separate from an editorial review failure."""

    node: str
    type: str
    message: str


class GraphState(TypedDict, total=False):
    """Shared information that moves through the future LangGraph workflow."""

    run_id: Required[str]
    topic: Required[str]
    discovery_queries: NotRequired[list[str]]
    discovery_sources: NotRequired[list[Source]]
    article_ideas: NotRequired[list[ArticleIdea]]
    selected_idea: NotRequired[ArticleIdea | None]
    deep_research_queries: NotRequired[list[str]]
    deep_research_sources: NotRequired[list[Source]]
    draft: NotRequired[ArticleDraft | None]
    review_status: NotRequired[ReviewStatus | None]
    failure_type: NotRequired[FailureType | None]
    flagged_sections: NotRequired[list[str]]
    review_feedback: NotRequired[str | None]
    retry_count: NotRequired[int]
    workflow_error: NotRequired[WorkflowError | None]
    final_status: NotRequired[str | None]
