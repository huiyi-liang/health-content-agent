"""You.com Search API boundary and deterministic Source normalization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import httpx
from dotenv import load_dotenv

from state import Source


YOU_SEARCH_URL = "https://ydc-index.io/v1/search"
SearchResultType = Literal["web", "news"]


# These errors let future LangGraph nodes distinguish configuration, network,
# response-shape, and empty-result failures without inspecting error text.
class YouSearchError(RuntimeError):
    """Base error raised by the You.com search-tool boundary."""


class YouSearchConfigurationError(YouSearchError):
    """Raised when local search configuration is missing or invalid."""


class YouSearchAPIError(YouSearchError):
    """Raised when You.com cannot complete the HTTP request."""


class YouSearchResponseError(YouSearchError):
    """Raised when You.com returns an unexpected response shape."""


class YouSearchEmptyResultsError(YouSearchError):
    """Raised when the requested web or news section has no results."""


def request_you_search(
    query: str,
    *,
    result_count: int = 5,
    boost_domains: Sequence[str] | None = None,
    freshness: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Send one Highlights-enabled search request to You.com.

    You.com decides whether a response contains web results, news results, or
    both. Choosing which section to use happens later during normalization.
    """

    if not query.strip():
        raise YouSearchConfigurationError("Search query cannot be empty.")
    if not 1 <= result_count <= 100:
        raise YouSearchConfigurationError("result_count must be between 1 and 100.")

    resolved_api_key = api_key or _load_api_key()
    # This is the exact provider-facing request. Highlights are requested here,
    # before any application normalization happens.
    request_body: dict[str, Any] = {
        "query": query,
        "count": result_count,
        "extraction": {"extraction_mode": "highlights"},
    }
    if boost_domains:
        request_body["boost_domains"] = list(boost_domains)
    if freshness:
        request_body["freshness"] = freshness

    headers = {
        "X-API-Key": resolved_api_key,
        "Content-Type": "application/json",
    }

    try:
        # Tests can inject a mocked client. Live calls omit it and use the
        # short-lived client created in the else branch.
        if client is not None:
            response = client.post(YOU_SEARCH_URL, headers=headers, json=request_body)
        else:
            with httpx.Client(timeout=30.0) as owned_client:
                response = owned_client.post(
                    YOU_SEARCH_URL,
                    headers=headers,
                    json=request_body,
                )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise YouSearchAPIError(
            f"You.com returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise YouSearchAPIError(f"You.com request failed: {exc}") from exc
    except ValueError as exc:
        raise YouSearchResponseError("You.com returned invalid JSON.") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise YouSearchResponseError(
            "You.com response must contain a 'results' object."
        )

    # Return the raw You.com dictionary. A separate function translates it
    # into our application's Source model.
    return payload


def normalize_search_results(
    response: Mapping[str, Any],
    *,
    query: str,
    source_type: SearchResultType,
    id_prefix: str,
    start_number: int = 1,
) -> list[Source]:
    """Convert one You.com result section into application Source objects."""

    if not id_prefix:
        raise YouSearchConfigurationError("id_prefix cannot be empty.")
    if start_number < 1:
        raise YouSearchConfigurationError("start_number must be at least 1.")

    results = response.get("results")
    if not isinstance(results, Mapping):
        raise YouSearchResponseError(
            "You.com response must contain a 'results' object."
        )

    # source_type selects results.web or results.news from the same API response.
    provider_results = results.get(source_type)
    if provider_results is None:
        raise YouSearchEmptyResultsError(
            f"You.com returned no '{source_type}' result section for this query."
        )
    if not isinstance(provider_results, list):
        raise YouSearchResponseError(
            f"You.com 'results.{source_type}' must be a list."
        )
    if not provider_results:
        raise YouSearchEmptyResultsError(
            f"You.com returned an empty '{source_type}' result set."
        )

    normalized_sources: list[Source] = []
    # enumerate supplies predictable numbers; the LLM never creates Source IDs.
    for index, result in enumerate(provider_results, start=start_number):
        normalized_sources.append(
            _normalize_one_result(
                result,
                query=query,
                source_type=source_type,
                source_id=f"{id_prefix}{index}",
            )
        )

    return normalized_sources


def normalize_all_search_results(
    response: Mapping[str, Any],
    *,
    query: str,
    id_prefix: str,
    start_number: int = 1,
) -> list[Source]:
    """Normalize every available web and news result from one API response."""

    results = response.get("results")
    if not isinstance(results, Mapping):
        raise YouSearchResponseError(
            "You.com response must contain a 'results' object."
        )

    normalized_sources: list[Source] = []
    # One You.com request may contain both sections. Normalize web first and
    # then news, continuing the same ID sequence across both sections.
    for source_type in ("web", "news"):
        provider_results = results.get(source_type)
        if provider_results is None or provider_results == []:
            continue

        if not isinstance(provider_results, list):
            raise YouSearchResponseError(
                f"You.com 'results.{source_type}' must be a list."
            )

        # Highlight extraction can fail for an individual page even when the
        # overall search request succeeds. Such a result cannot serve as our
        # evidence, so skip it while preserving every usable result.
        usable_results = [
            result
            for result in provider_results
            if _has_retrieved_highlights(result)
        ]
        if not usable_results:
            continue

        filtered_response = {"results": {source_type: usable_results}}

        section_sources = normalize_search_results(
            filtered_response,
            query=query,
            source_type=source_type,
            id_prefix=id_prefix,
            start_number=start_number + len(normalized_sources),
        )
        normalized_sources.extend(section_sources)

    return normalized_sources


def _has_retrieved_highlights(result: Any) -> bool:
    """Return False only when a result has no usable extracted Highlights."""

    # Leave other malformed shapes for _normalize_one_result() to report
    # clearly instead of silently hiding provider response problems.
    if not isinstance(result, Mapping):
        return True

    contents = result.get("contents")
    if contents is None:
        return False
    if not isinstance(contents, Mapping):
        return True

    highlights = contents.get("highlights")
    if highlights is None:
        return False
    if isinstance(highlights, list) and not highlights:
        return False

    return True


def search_you(
    query: str,
    *,
    source_type: SearchResultType = "web",
    result_count: int = 5,
    id_prefix: str = "D",
    start_number: int = 1,
    boost_domains: Sequence[str] | None = None,
    freshness: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> list[Source]:
    """Search You.com and return only normalized application Source objects."""

    # This convenience function composes the two main tool responsibilities:
    # make one request, then normalize one requested result section.
    response = request_you_search(
        query,
        result_count=result_count,
        boost_domains=boost_domains,
        freshness=freshness,
        api_key=api_key,
        client=client,
    )
    return normalize_search_results(
        response,
        query=query,
        source_type=source_type,
        id_prefix=id_prefix,
        start_number=start_number,
    )


def _load_api_key() -> str:
    # search.py is inside tools/, so parents[1] points to the project root.
    project_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(project_env, override=False)
    api_key = os.getenv("YOU_API_KEY")
    if not api_key:
        raise YouSearchConfigurationError(
            "YOU_API_KEY is missing. Add it to the local .env file."
        )
    return api_key


def _normalize_one_result(
    result: Any,
    *,
    query: str,
    source_type: SearchResultType,
    source_id: str,
) -> Source:
    if not isinstance(result, Mapping):
        raise YouSearchResponseError("Each You.com result must be an object.")

    title = result.get("title")
    url = result.get("url")
    publication_date = result.get("page_age")
    contents = result.get("contents")

    if not isinstance(title, str) or not title:
        raise YouSearchResponseError("Each You.com result must have a title.")
    if not isinstance(url, str) or not url:
        raise YouSearchResponseError("Each You.com result must have a URL.")
    if publication_date is not None and not isinstance(publication_date, str):
        raise YouSearchResponseError("You.com 'page_age' must be a string or null.")
    if not isinstance(contents, Mapping):
        raise YouSearchResponseError(
            "Each You.com result must contain Highlights in 'contents'."
        )

    highlights = contents.get("highlights")
    if not isinstance(highlights, list) or not all(
        isinstance(highlight, str) for highlight in highlights
    ):
        raise YouSearchResponseError(
            "You.com 'contents.highlights' must be a list of strings."
        )

    # Only fields in our Source contract move downstream. Provider-specific
    # extras such as thumbnails and descriptions are deliberately left out.
    return Source(
        source_id=source_id,
        query=query,
        title=title,
        url=url,
        publication_date=publication_date,
        source_type=source_type,
        highlights=list(highlights),
    )


def _print_sources(query: str, sources: Sequence[Source]) -> None:
    # Printing is only for the manual Phase 2 smoke test; it does not alter data.
    print(f"Query sent: {query}")
    print()
    for source in sources:
        print(f"Source ID: {source.source_id}")
        print(f"Type: {source.source_type}")
        print(f"Title: {source.title}")
        print(f"URL: {source.url}")
        print(f"Date: {source.publication_date or 'Not available'}")
        print(f"Query: {source.query}")
        print("Highlights:")
        if source.highlights:
            for highlight in source.highlights:
                print(f"- {highlight}")
        else:
            print("- No Highlights returned")
        print()


def _parse_args() -> argparse.Namespace:
    # argparse converts terminal flags such as "--type news" into Python values.
    parser = argparse.ArgumentParser(description="Run a live You.com smoke test.")
    parser.add_argument("--type", choices=("web", "news"), default="web")
    parser.add_argument("--query")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--prefix", default="D")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--boost-domain", action="append", dest="boost_domains")
    parser.add_argument("--freshness")
    return parser.parse_args()


def main() -> None:
    # main() is the manual live-search entry point, not an automated unit test.
    args = _parse_args()
    default_queries = {
        "web": "type 2 diabetes patient concerns",
        "news": "type 2 diabetes treatment developments",
    }
    query = args.query or default_queries[args.type]
    try:
        sources = search_you(
            query,
            source_type=args.type,
            result_count=args.count,
            id_prefix=args.prefix,
            start_number=args.start,
            boost_domains=args.boost_domains,
            freshness=args.freshness,
        )
    except YouSearchError as exc:
        raise SystemExit(f"Search failed: {exc}") from exc
    _print_sources(query, sources)


if __name__ == "__main__":
    # This runs only for `python -m tools.search`, not when another module imports it.
    main()
