"""Focused unit tests for You.com response normalization."""

import json

import httpx
import pytest

from tools.search import (
    YOU_SEARCH_URL,
    YouSearchEmptyResultsError,
    YouSearchResponseError,
    _safe_search_trace_inputs,
    normalize_all_search_results,
    normalize_search_results,
    request_you_search,
)


SAMPLE_RESPONSE = {
    "results": {
        "web": [
            {
                "title": "Diabetes and Daily Life",
                "url": "https://example.org/diabetes-daily-life",
                "contents": {
                    "highlights": [
                        "First Highlight with original capitalization.",
                        "Second Highlight remains unchanged.",
                    ]
                },
            }
        ],
        "news": [
            {
                "title": "New Diabetes Treatment Study Published",
                "url": "https://news.example.org/diabetes-study",
                "page_age": "2026-08-20T09:30:00",
                "contents": {
                    "highlights": ["The study reported its primary findings."]
                },
            }
        ],
    },
    "metadata": {"query": "type 2 diabetes"},
}


def test_langsmith_search_trace_inputs_remove_credentials_and_client() -> None:
    safe_inputs = _safe_search_trace_inputs(
        {
            "query": "type 2 diabetes",
            "result_count": 4,
            "api_key": "secret-test-value",
            "client": object(),
        }
    )

    assert safe_inputs == {
        "query": "type 2 diabetes",
        "result_count": 4,
    }


def test_web_results_normalize_and_preserve_highlights() -> None:
    sources = normalize_search_results(
        SAMPLE_RESPONSE,
        query="type 2 diabetes",
        source_type="web",
        id_prefix="D",
    )

    assert sources[0].source_id == "D1"
    assert sources[0].source_type == "web"
    assert sources[0].publication_date is None
    assert sources[0].highlights == [
        "First Highlight with original capitalization.",
        "Second Highlight remains unchanged.",
    ]


def test_news_results_normalize_with_publication_date() -> None:
    sources = normalize_search_results(
        SAMPLE_RESPONSE,
        query="type 2 diabetes treatment developments",
        source_type="news",
        id_prefix="D",
    )

    assert sources[0].source_type == "news"
    assert sources[0].publication_date == "2026-08-20T09:30:00"
    assert sources[0].title == "New Diabetes Treatment Study Published"


def test_custom_prefix_and_start_number_are_deterministic() -> None:
    response = {
        "results": {
            "web": [SAMPLE_RESPONSE["results"]["web"][0]] * 3,
        }
    }

    sources = normalize_search_results(
        response,
        query="patient questions",
        source_type="web",
        id_prefix="R",
        start_number=4,
    )

    assert [source.source_id for source in sources] == ["R4", "R5", "R6"]


@pytest.mark.parametrize(
    "response",
    [
        {"results": {"web": []}},
        {"results": {}},
    ],
)
def test_empty_result_section_raises_clear_error(response: dict) -> None:
    with pytest.raises(YouSearchEmptyResultsError):
        normalize_search_results(
            response,
            query="type 2 diabetes",
            source_type="web",
            id_prefix="D",
        )


def test_malformed_highlights_raise_clear_error() -> None:
    response = {
        "results": {
            "web": [
                {
                    "title": "Example",
                    "url": "https://example.org",
                    "contents": {"highlights": "not-a-list"},
                }
            ]
        }
    }

    with pytest.raises(YouSearchResponseError, match="list of strings"):
        normalize_search_results(
            response,
            query="type 2 diabetes",
            source_type="web",
            id_prefix="D",
        )


def test_combined_normalization_skips_only_results_without_highlights() -> None:
    response = {
        "results": {
            "web": [
                {
                    "title": "Extraction Failed",
                    "url": "https://example.org/no-highlights",
                },
                SAMPLE_RESPONSE["results"]["web"][0],
            ],
            "news": [
                {
                    "title": "Empty Extraction",
                    "url": "https://news.example.org/empty-highlights",
                    "contents": {"highlights": []},
                },
                SAMPLE_RESPONSE["results"]["news"][0],
            ],
        }
    }

    sources = normalize_all_search_results(
        response,
        query="type 2 diabetes",
        id_prefix="D",
    )

    assert [source.source_id for source in sources] == ["D1", "D2"]
    assert [source.title for source in sources] == [
        "Diabetes and Daily Life",
        "New Diabetes Treatment Study Published",
    ]
    assert [source.highlights for source in sources] == [
        [
            "First Highlight with original capitalization.",
            "Second Highlight remains unchanged.",
        ],
        ["The study reported its primary findings."],
    ]


def test_combined_normalization_returns_empty_when_no_highlights_are_usable() -> None:
    response = {
        "results": {
            "web": [
                {
                    "title": "No Contents",
                    "url": "https://example.org/no-contents",
                }
            ],
            "news": [
                {
                    "title": "No Highlight Field",
                    "url": "https://news.example.org/no-highlight-field",
                    "contents": {},
                }
            ],
        }
    }

    sources = normalize_all_search_results(
        response,
        query="type 2 diabetes",
        id_prefix="D",
    )

    assert sources == []


def test_request_uses_post_auth_highlights_and_domain_boosts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.method == "POST"
        assert str(request.url) == YOU_SEARCH_URL
        assert request.headers["X-API-Key"] == "test-key"
        assert body == {
            "query": "type 2 diabetes",
            "count": 2,
            "extraction": {"extraction_mode": "highlights"},
            "boost_domains": ["nih.gov", "cdc.gov"],
        }
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = request_you_search(
            "type 2 diabetes",
            result_count=2,
            boost_domains=["nih.gov", "cdc.gov"],
            api_key="test-key",
            client=client,
        )

    assert response == SAMPLE_RESPONSE
