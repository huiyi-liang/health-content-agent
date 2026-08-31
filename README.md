# Health Content Agent

A beginner-oriented Week 3 project that will turn a health topic into researched article ideas and, after human selection, a cited article draft.

Phase 3 adds a standalone Discovery Research node. It uses one Nebius LLM call
to plan exactly three queries, makes exactly three You.com requests, and
normalizes all available web and news evidence into `Source` objects. It does
not yet run a complete LangGraph or Streamlit workflow.

Phase 4 adds a standalone Opportunity Agent. It reads the topic and normalized
Discovery evidence, uses one structured Nebius LLM call, and returns exactly
three `ArticleIdea` objects with deterministic IDs. It does not yet select an
idea or continue to Deep Research.

Phase 5 adds a terminal-only human checkpoint and standalone Deep Research.
After the user selects a complete `ArticleIdea`, one Nebius LLM call plans
exactly three targeted queries. Three You.com requests then produce normalized
`Source` evidence with sequential `R#` IDs. The workflow stops after research.

Phase 6 adds a standalone Writer Agent. It reads the complete selected idea,
Deep Research evidence, and the editorial style guide, then returns a
structured `ArticleDraft`. Python validates section IDs and verifies that every
inline `[R#]` citation refers to a stored research Source.

Phase 7 adds a standalone Reviewer Agent and a deterministic routing helper.
The Reviewer identifies one primary editorial failure using the supplied draft,
research evidence, and style guide. Python validates its structured result and
maps that result to a future route without executing a workflow loop.

Phase 8 connects the tested nodes with LangGraph. It pauses for human article
selection, resumes the same in-memory workflow, expands research after an
`insufficient_research` review, revises only flagged sections for other review
failures, and stops before a fourth automated correction cycle.

Shared Nebius model initialization lives in `llm.py`. Future LLM-based nodes
should import `create_nebius_llm()` rather than repeat provider configuration.

## Setup

Add your local provider configuration to the `.env` file:

```text
YOU_API_KEY=<your key>
NEBIUS_API_KEY=<your Nebius key>
```

The `.env` file is ignored by Git. Never put a real key in `.env.example`.

## Run the unit tests

From this project directory, run:

```bash
uv sync --dev
uv run pytest
```

The unit tests use mocks and sample responses; they call neither Nebius nor
You.com.

## Run Discovery Research live

```bash
uv run python -m nodes.discovery
```

The default topic is `Type 2 diabetes`. Supply another topic with:

```bash
uv run python -m nodes.discovery --topic "Type 2 diabetes and sleep"
```

## Run the Opportunity Agent live

This smoke test runs the existing live Discovery pipeline first and then sends
its evidence to the Opportunity Agent:

```bash
uv run python -m nodes.opportunity
```

Supply another topic with:

```bash
uv run python -m nodes.opportunity --topic "Type 2 diabetes and sleep"
```

## Run the Phase 5 smoke test

Add `--deep-research` to continue from Opportunity ideas to manual selection
and live Deep Research:

```bash
uv run python -m nodes.opportunity --deep-research
```

The terminal will display `A1`, `A2`, and `A3` and wait for one idea ID. It
then plans and runs exactly three Deep Research searches and stops.

## Run the Phase 6 Writer smoke test

Add `--writer` to run through manual selection, Deep Research, and Writer:

```bash
uv run python -m nodes.opportunity --writer
```

The displayed bibliography is generated from stored Sources rather than by
the Writer LLM. The workflow stops after displaying the structured draft.

## Run the Phase 7 Reviewer smoke test

Add `--reviewer` to run through manual selection, Deep Research, Writer, and
Reviewer:

```bash
uv run python -m nodes.opportunity --reviewer
```

The terminal displays the review result and deterministic next route but does
not execute that route.

## Run the Phase 8 LangGraph smoke test

Run the complete graph and select an article idea when it pauses:

```bash
uv run python -m graph
```

Use a different topic with:

```bash
uv run python -m graph --topic "Type 2 diabetes and sleep"
```

The graph uses an in-memory checkpointer for terminal interrupt/resume. Its
checkpoints disappear when the process exits; this is not final run storage.

## Run live smoke tests

Web results:

```bash
uv run python -m tools.search --type web
```

News results:

```bash
uv run python -m tools.search --type news --freshness month
```

You.com uses one search endpoint for both kinds of results. It automatically
adds a `results.news` section when it detects news intent. Therefore, a news
query can occasionally return no news section; the smoke test reports that
case clearly instead of relabeling web results as news.

Optional domain boosts can be repeated:

```bash
uv run python -m tools.search --type web --boost-domain nih.gov --boost-domain cdc.gov
```
