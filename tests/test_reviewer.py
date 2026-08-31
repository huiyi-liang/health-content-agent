"""Focused unit tests for the Reviewer and deterministic routing helper."""

from copy import deepcopy
from typing import Any

import pytest

from nodes.reviewer import ReviewerOutput, reviewer_node, route_review_result
from state import ArticleDraft, ArticleSection, FailureType, GraphState, Source


def _draft() -> ArticleDraft:
    """Create a structured draft with stable section IDs."""

    return ArticleDraft(
        title="Questions to Ask Before Changing a Diabetes Medication",
        sections=[
            ArticleSection(
                section_id="introduction",
                heading=None,
                content="Medication changes can involve benefits and risks. [R1]",
            ),
            ArticleSection(
                section_id="questions_for_your_care_team",
                heading="Questions for Your Care Team",
                content="Ask how monitoring may change. [R1]",
            ),
        ],
    )


def _sources() -> list[Source]:
    """Create stored evidence fixtures without calling You.com."""

    return [
        Source(
            source_id="R1",
            query="diabetes medication changes clinical guidance",
            title="Medication Change Guidance",
            url="https://nih.gov/example-guidance",
            publication_date=None,
            source_type="web",
            highlights=[
                "Medication decisions should account for benefits and risks.",
                "Monitoring needs may differ after a treatment change.",
            ],
        )
    ]


class FakeStructuredModel:
    """Return controlled Reviewer output instead of calling Nebius."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        system_prompt = messages[0][1]
        assert messages[0][0] == "system"
        assert "1. `insufficient_research`" in system_prompt
        assert "4. `patient_value`" in system_prompt
        assert "plain-language paraphrase" in system_prompt
        assert "only when the mismatch is material" in system_prompt
        assert "every section affected" in system_prompt
        assert "Do not return a recommended action or route" in system_prompt
        assert "# Consumer Health Article Style Guide" in system_prompt

        reviewer_input = messages[1][1]
        assert "Allowed `flagged_sections` values:" in reviewer_input
        assert "- introduction" in reviewer_input
        assert "- questions_for_your_care_team" in reviewer_input
        assert 'Correct: ["introduction"]' in reviewer_input
        assert 'Incorrect: ["section_id: introduction"]' in reviewer_input
        assert "[section_id: introduction]" in reviewer_input
        assert "Heading: Questions for Your Care Team" in reviewer_input
        assert "[R1]" in reviewer_input
        assert "Title: Medication Change Guidance" in reviewer_input
        assert "Medication decisions should account for benefits and risks." in reviewer_input
        assert "# Consumer Health Article Style Guide" not in reviewer_input
        return self.result


class FakeLLM:
    """Mimic only the structured-output methods used by the Reviewer."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def with_structured_output(
        self,
        schema: type[ReviewerOutput],
        *,
        method: str,
        strict: bool,
    ) -> FakeStructuredModel:
        assert schema is ReviewerOutput
        assert method == "json_schema"
        assert strict is True
        return FakeStructuredModel(self.result)


def _state() -> GraphState:
    """Create the complete state needed by the Reviewer."""

    return {
        "run_id": "run-701",
        "topic": "Type 2 diabetes",
        "draft": _draft(),
        "deep_research_sources": _sources(),
        "retry_count": 0,
    }


def test_valid_pass_has_exact_empty_fields_and_preserves_inputs() -> None:
    state = _state()
    original_draft = deepcopy(state["draft"])
    original_sources = deepcopy(state["deep_research_sources"])

    update = reviewer_node(
        state,
        llm=FakeLLM(
            {
                "review_status": "PASS",
                "failure_type": None,
                "flagged_sections": [],
                "review_feedback": None,
            }
        ),
    )

    assert update == {
        "review_status": "PASS",
        "failure_type": None,
        "flagged_sections": [],
        "review_feedback": None,
        "workflow_error": None,
    }
    assert state["draft"] == original_draft
    assert state["deep_research_sources"] == original_sources


@pytest.mark.parametrize(
    "failure_type",
    [
        "insufficient_research",
        "unsupported_claim",
        "style_or_readability",
        "patient_value",
    ],
)
def test_each_allowed_failure_type_is_accepted(failure_type: FailureType) -> None:
    update = reviewer_node(
        _state(),
        llm=FakeLLM(
            {
                "review_status": "FAIL",
                "failure_type": failure_type,
                "flagged_sections": ["questions_for_your_care_team"],
                "review_feedback": "Revise this section using the supplied evidence.",
            }
        ),
    )

    assert update["review_status"] == "FAIL"
    assert update["failure_type"] == failure_type
    assert update["flagged_sections"] == ["questions_for_your_care_team"]
    assert update["workflow_error"] is None


def test_unknown_section_id_fails_clearly() -> None:
    update = reviewer_node(
        _state(),
        llm=FakeLLM(
            {
                "review_status": "FAIL",
                "failure_type": "unsupported_claim",
                "flagged_sections": ["section_that_does_not_exist"],
                "review_feedback": "Remove the unsupported statement.",
            }
        ),
    )

    assert "review_status" not in update
    assert update["workflow_error"].type == "review_validation_error"
    assert "section_that_does_not_exist" in update["workflow_error"].message


def test_known_section_id_wrapper_is_normalized_before_validation() -> None:
    update = reviewer_node(
        _state(),
        llm=FakeLLM(
            {
                "review_status": "FAIL",
                "failure_type": "unsupported_claim",
                "flagged_sections": [
                    "section_id: questions_for_your_care_team"
                ],
                "review_feedback": "Remove the unsupported statement.",
            }
        ),
    )

    assert update["flagged_sections"] == ["questions_for_your_care_team"]
    assert update["workflow_error"] is None


def test_unknown_wrapped_section_id_still_fails_validation() -> None:
    update = reviewer_node(
        _state(),
        llm=FakeLLM(
            {
                "review_status": "FAIL",
                "failure_type": "unsupported_claim",
                "flagged_sections": ["section_id: imaginary_section"],
                "review_feedback": "Remove the unsupported statement.",
            }
        ),
    )

    assert "review_status" not in update
    assert update["workflow_error"].type == "review_validation_error"
    assert "imaginary_section" in update["workflow_error"].message


@pytest.mark.parametrize(
    "invalid_result",
    [
        {
            "review_status": "PASS",
            "failure_type": "unsupported_claim",
            "flagged_sections": [],
            "review_feedback": None,
        },
        {
            "review_status": "PASS",
            "failure_type": None,
            "flagged_sections": [],
            "review_feedback": "Looks good",
        },
        {
            "review_status": "PASS",
            "failure_type": None,
            "flagged_sections": ["introduction"],
            "review_feedback": None,
        },
        {
            "review_status": "FAIL",
            "failure_type": None,
            "flagged_sections": ["introduction"],
            "review_feedback": "Clarify the introduction.",
        },
        {
            "review_status": "FAIL",
            "failure_type": "style_or_readability",
            "flagged_sections": ["introduction"],
            "review_feedback": "   ",
        },
        {
            "review_status": "FAIL",
            "failure_type": "unsupported_claim",
            "flagged_sections": [],
            "review_feedback": "Remove the unsupported statement.",
        },
    ],
    ids=[
        "pass-with-failure",
        "pass-with-feedback",
        "pass-with-flags",
        "fail-without-type",
        "fail-with-blank-feedback",
        "rewrite-failure-without-flags",
    ],
)
def test_invalid_status_field_combinations_fail(invalid_result: dict[str, Any]) -> None:
    update = reviewer_node(_state(), llm=FakeLLM(invalid_result))

    assert "review_status" not in update
    assert update["workflow_error"].type == "review_validation_error"


def test_router_maps_pass_to_human_review() -> None:
    assert (
        route_review_result(
            {
                "run_id": "run-702",
                "topic": "Type 2 diabetes",
                "review_status": "PASS",
                "failure_type": None,
                "retry_count": 0,
            }
        )
        == "human_review"
    )


def test_router_maps_insufficient_research_to_deep_research() -> None:
    assert (
        route_review_result(
            {
                "run_id": "run-703",
                "topic": "Type 2 diabetes",
                "review_status": "FAIL",
                "failure_type": "insufficient_research",
                "retry_count": 1,
            }
        )
        == "deep_research"
    )


@pytest.mark.parametrize(
    "failure_type",
    ["unsupported_claim", "style_or_readability", "patient_value"],
)
def test_router_maps_other_failures_to_rewrite(failure_type: FailureType) -> None:
    state: GraphState = {
        "run_id": "run-704",
        "topic": "Type 2 diabetes",
        "review_status": "FAIL",
        "failure_type": failure_type,
        "retry_count": 2,
    }

    assert route_review_result(state) == "rewrite"


def test_retry_limit_routes_to_human_review() -> None:
    state: GraphState = {
        "run_id": "run-705",
        "topic": "Type 2 diabetes",
        "review_status": "FAIL",
        "failure_type": "insufficient_research",
        "retry_count": 3,
    }

    assert route_review_result(state) == "human_review"


def test_missing_reviewer_inputs_fail_before_calling_llm() -> None:
    no_draft = reviewer_node(
        {
            "run_id": "run-706",
            "topic": "Type 2 diabetes",
            "deep_research_sources": _sources(),
        }
    )
    no_evidence = reviewer_node(
        {
            "run_id": "run-707",
            "topic": "Type 2 diabetes",
            "draft": _draft(),
            "deep_research_sources": [],
        }
    )

    assert no_draft["workflow_error"].type == "missing_draft"
    assert no_evidence["workflow_error"].type == "missing_deep_research_evidence"
