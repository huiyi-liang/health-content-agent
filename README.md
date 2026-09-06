# Health Content Agent

A multi-agent application that turns one health topic into
patient-focused article ideas and then into an evidence-grounded article draft.

The project uses Python, LangGraph, Nebius, the You.com Search API, and
Streamlit. It is designed as a learning project, so responsibilities are kept
small and explicit instead of hidden behind many abstraction layers.

## Problem statement

A mature consumer-health publisher usually already has foundational articles
such as “What is type 2 diabetes?” and broad summaries of symptoms, causes, and
treatments. Publishing more versions of those articles may not give readers
much new value.

This application starts with a free-form topic such as:

```text
Type 2 diabetes
Type 2 diabetes and sleep
```

It searches for more specific opportunities involving:

- questions patients are asking;
- challenges people face while managing a condition;
- decisions that require practical guidance;
- patient experiences and concerns; and
- meaningful recent developments with patient implications.

After a human chooses an article idea, the application gathers more focused
medical evidence, writes a cited article, and reviews it against the supplied
evidence and editorial style guide.

The application does **not** medically approve an article, diagnose a reader,
publish content automatically, or replace clinical review.

## Workflow diagram

```mermaid
flowchart TD
    START([User enters a health topic]) --> D[Discovery Research]
    D --> O[Opportunity Agent]
    O --> H{{Human selects A1, A2, or A3}}
    H --> DR[Deep Research]
    DR --> W[Writer Agent]
    W --> RV[Reviewer Agent]

    RV -->|PASS| C([Completed])
    RV -->|Insufficient research| RC[Increment retry count]
    RV -->|Unsupported claim| RC
    RV -->|Style or readability| RC
    RV -->|Patient value| RC

    RC -->|Evidence gap| DRR[Deep Research retry]
    DRR -->|Append queries and sources| W

    RC -->|Draft problem| WR[Writer revises flagged sections]
    WR --> RV

    RV -->|A fourth correction would be needed| HR([Human review required])

    D -. System or tool error .-> F([Failed])
    O -. System error .-> F
    H -. Invalid selection .-> F
    DR -. Search or model error .-> F
    W -. Model or validation error .-> F
    RV -. Model or validation error .-> F
```

LangGraph acts like the workflow conductor: it decides which already-defined
component runs next, carries shared state between components, pauses for the
human decision, and enforces the retry limit.

## What each node does

### Quick reference

| Node | Main question | Main state output |
| --- | --- | --- |
| Discovery Research | What might be worth writing about? | `discovery_queries`, `discovery_sources` |
| Opportunity Agent | Which three ideas offer the strongest patient value? | `article_ideas` |
| Human Selection | Which idea did the user choose? | `selected_idea` |
| Deep Research | What evidence is needed for that specific idea? | `deep_research_queries`, `deep_research_sources` |
| Writer Agent | How can the evidence become a useful article? | `draft` |
| Reviewer Agent | What is the highest-priority problem, if any? | Review fields and feedback |
| Deterministic routing | Which node should run next? | Route, retry count, or final status |

### 1. Discovery Research

**File:** `nodes/discovery.py`

#### Purpose

Answer: **What might be worth writing about for someone dealing with this
health topic?**

#### Reads from state

- `topic`

#### What happens

- The Nebius LLM plans exactly three complementary search queries.
- Each query makes one You.com request.
- Available web and news results are both consumed.
- You.com Highlights are preserved rather than rewritten by the LLM.
- Python normalizes results into `Source` objects with IDs such as `D1`, `D2`,
  and `D3`.

#### Writes to state

- `discovery_queries`
- `discovery_sources`

#### Boundary

Discovery finds possible directions. It does not choose the final idea or
prove that enough evidence exists to write an article.

### 2. Opportunity Agent

**File:** `nodes/opportunity.py`

#### Purpose

Answer: **Which three article ideas are most likely to provide useful,
specific value to patients?**

#### Reads from state

- `topic`
- `discovery_sources`

#### What happens

- One LLM call considers at least six candidates internally.
- It selects the strongest three using patient relevance, specificity, and
  usefulness or actionability.
- Python assigns the stable IDs `A1`, `A2`, and `A3`.

#### Writes to state

- `article_ideas`: exactly three complete `ArticleIdea` objects

Each idea contains:

- `idea_id`
- `title`
- `article_angle`
- `reason`

#### Boundary

The Opportunity Agent recommends ideas. It does not select one for the user,
perform Deep Research, or judge whether final evidence will be sufficient.

### 3. Human Selection

**Files:** `graph.py`, `nodes/opportunity.py`, and `app.py`

#### Purpose

Pause automation so the human—not the LLM—chooses which idea moves forward.

#### Reads from state

- `article_ideas`

#### What happens

- LangGraph interrupts after the Opportunity Agent.
- Streamlit displays the title, angle, and reason for all three ideas.
- The user selects one stable idea ID.
- The selected ID resumes the same LangGraph thread.
- Python validates the ID and retrieves the complete matching `ArticleIdea`.

#### Writes to state

- `selected_idea`: the complete chosen `ArticleIdea`, not only its title

#### Boundary

Discovery and Opportunity do not run again after resume. The interface only
collects the choice; selection validation remains UI-independent Python code.

### 4. Deep Research

**File:** `nodes/research.py`

#### Purpose

Answer: **What evidence is needed to write this specific selected article?**

#### Reads from state

For the first research pass:

- `selected_idea`

For a Reviewer-triggered retry, it also reads:

- `review_feedback`
- existing `deep_research_queries`
- existing `deep_research_sources`

#### What happens

- The LLM reads the full selected idea and plans exactly three targeted
  queries.
- Each query makes one You.com request.
- Medical domains are preferred through domain boosting, not used as a strict
  allowlist.
- Web and news results become normalized `Source` objects with IDs such as
  `R1`, `R2`, and `R3`.
- Highlights remain unchanged.

#### Writes to state

- `deep_research_queries`
- `deep_research_sources`

On retry, new queries and Sources are appended. Existing evidence is preserved,
and R-numbering continues with the next available number.

#### Boundary

The LLM decides what to search. You.com retrieves the evidence. Deterministic
Python normalizes and stores it without an intermediate LLM summary.

### 5. Writer Agent

**File:** `nodes/writer.py`

#### Purpose

Turn the selected idea and supplied research evidence into a readable,
consumer-health article.

#### Reads from state and configuration

- `selected_idea`
- `deep_research_sources`
- `config/article_style.md`

For a revision, it also reads:

- the current `draft`
- `flagged_sections`
- `review_feedback`

#### What happens

- The LLM writes a structured article using only supplied evidence for medical
  factual claims.
- Inline citations use stored IDs such as `[R1]` or `[R1][R3]`.
- Python verifies that every citation ID exists in `deep_research_sources`.
- The MVP uses plain-language findings instead of detailed study statistics
  such as odds ratios, confidence intervals, P-values, or regression
  coefficients.
- On revision, only Reviewer-flagged sections may change.

#### Writes to state

- `draft`: one structured `ArticleDraft`

Each draft contains:

- an article `title`
- a list of `ArticleSection` objects
- a stable `section_id`, optional reader-facing `heading`, and `content` for
  each section

#### Boundary

The Writer cannot search for new evidence, invent R# IDs, or decide whether its
own article passes review. Unflagged sections remain unchanged during revision.

### 6. Reviewer Agent

**File:** `nodes/reviewer.py`

#### Purpose

Answer: **What is the single highest-priority problem with this draft, if one
exists?**

#### Reads from state and configuration

- `draft`
- `deep_research_sources`
- `config/article_style.md`

#### What happens

- The Reviewer compares the draft with the actual supplied evidence.
- It does not treat its general medical knowledge as evidence.
- It chooses one primary failure using this priority:

  1. `insufficient_research`
  2. `unsupported_claim`
  3. `style_or_readability`
  4. `patient_value`

- For a failure, it identifies affected sections using stable `section_id`
  values rather than reader-facing headings.

#### Writes to state

- `review_status`
- `failure_type`
- `flagged_sections`
- `review_feedback`

#### Boundary

The Reviewer determines **what is wrong**. It does not choose which graph node
runs next and does not rewrite the article itself.

### 7. Deterministic routing

**File:** `graph.py`

#### Purpose

Turn the structured review result into a fixed, predictable workflow route.

#### Reads from state

- `review_status`
- `failure_type`
- `retry_count`
- `workflow_error`

#### Routing rules

- PASS → `completed`
- `insufficient_research` → increment the retry count and run Deep Research
- `unsupported_claim` → increment the retry count and revise the Writer draft
- `style_or_readability` → increment the retry count and revise the Writer
  draft
- `patient_value` → increment the retry count and revise the Writer draft
- another correction needed after three loops → `human_review_required`
- unrecoverable system or tool error → `failed`

#### Writes or controls

- the next graph edge
- updated `retry_count` when a correction begins
- `final_status` when automation stops

#### Boundary

This prevents an LLM from inventing graph transitions or creating an unlimited
loop. The Reviewer determines **what is wrong**; deterministic routing decides
**what happens next**.

## Supporting components

| File | Responsibility |
| --- | --- |
| `state.py` | Shared `Source`, `ArticleIdea`, `ArticleDraft`, review, error, and `GraphState` contracts |
| `tools/search.py` | You.com HTTP requests, web/news normalization, Highlights preservation, and deterministic source IDs |
| `prompts.py` | Task instructions for the LLM-based components |
| `llm.py` | One shared Nebius model configuration used by every LLM node |
| `config/article_style.md` | Reusable synthetic consumer-health editorial guidance |
| `graph.py` | LangGraph sequencing, interrupt/resume, routing, and retry enforcement |
| `persistence.py` | End-of-run checkpoint-history reconstruction and JSON export |
| `app.py` | Thin Streamlit input, selection, progress, and result presentation layer |

## LLM work versus deterministic Python work

| LLM judgment | Deterministic Python |
| --- | --- |
| Plan search queries | Call You.com and validate HTTP responses |
| Recommend article ideas | Assign D#, R#, and A# IDs |
| Write article sections | Normalize results into `Source` objects |
| Review article quality | Validate structured models and citation IDs |
| Identify the primary failure | Select graph routes and enforce retry count |

You can think of the LLM as the team member handling judgment and language,
while Python is the operations checklist that performs exact, repeatable steps.

## State ownership

LangGraph state is the source of truth for live workflow data, including the
topic, evidence, ideas, selected idea, draft, review result, retry count, error,
and final status.

Streamlit session state stores only the active LangGraph thread ID, plus normal
widget values. The thread ID lets the UI reconnect to the correct in-memory
checkpoint after Streamlit reruns its script.

The current checkpointer is intentionally in memory, so restarting the
application clears active unfinished runs. When a run reaches a terminal status,
its evaluation history is exported to `data/runs/<run_id>.json` before that
in-memory history is lost.

## Setup

This project uses `uv`. From the project root, install the locked dependencies:

```bash
uv sync --dev
```

Add your real keys only to the local `.env` file:

```text
YOU_API_KEY=<your You.com key>
NEBIUS_API_KEY=<your Nebius key>
```

The `.env` file is ignored by Git. Do not put real credentials in
`.env.example` or source code.

## Run the Streamlit application

```bash
uv run streamlit run app.py
```

Then:

1. enter a health topic;
2. click **Research article ideas**;
3. review the three ideas;
4. select one idea;
5. click **Continue with selected idea**; and
6. wait for the research, writing, review, and possible correction cycle.

A completed run displays the structured article and a deterministic clickable
source list. Its Discovery evidence, draft versions, Reviewer decisions, and
final state are also saved under `data/runs/`. A run that reaches the correction
limit displays the latest draft and Reviewer feedback for human review.

## Run the tests

```bash
uv run pytest
```

The automated test scripts act like rehearsals: controlled fake LLM and search
responses play the external roles, so each Python component can be checked
repeatedly without waiting for APIs or spending provider credits.

| Test script | What it verifies |
| --- | --- |
| `tests/test_state.py` | The shared Source, idea, draft, section, review, and GraphState contracts |
| `tests/test_search.py` | You.com request construction, web/news normalization, Highlights preservation, and deterministic source IDs |
| `tests/test_research.py` | Discovery query validation, exactly three mocked searches, combined evidence, and errors |
| `tests/test_opportunity.py` | Exactly three structured article ideas, deterministic A# IDs, and evidence preservation |
| `tests/test_deep_research.py` | Human selection, targeted query validation, medical-domain preferences, and R# evidence |
| `tests/test_writer.py` | Structured sections, evidence formatting, source citation IDs, and Writer input failures |
| `tests/test_reviewer.py` | Review result consistency, valid flagged section IDs, and deterministic route decisions |
| `tests/test_phase8.py` | LangGraph interrupt/resume, research retry, Writer revision, retry limit, and failure paths |
| `tests/test_app.py` | Deterministic article, source, flagged-section, and error-display helpers used by Streamlit |

Pytest automatically discovers functions whose names begin with `test_`. Each
test arranges a controlled example, runs one part of the application, and uses
assertions to compare the actual result with the expected result. A test marked
`PASSED` means that behavior matched its contract; it does not mean a live API
call was made.

Run one test script while learning or debugging a component:

```bash
uv run pytest tests/test_writer.py -v
```

Run one specific scenario:

```bash
uv run pytest tests/test_phase8.py::test_langgraph_stops_before_fourth_automated_retry -v
```

The automated tests do not call Nebius or You.com. The standalone smoke tests
below are separate manual checks that use the real services and may consume
credits.

## Terminal workflow test

The complete graph can also be run without Streamlit:

```bash
uv run python -m graph
```

Use another topic with:

```bash
uv run python -m graph --topic "Type 2 diabetes and sleep"
```

## Standalone component smoke tests

```bash
# You.com web search
uv run python -m tools.search --type web

# Discovery only
uv run python -m nodes.discovery

# Discovery and Opportunity
uv run python -m nodes.opportunity

# Continue through Deep Research
uv run python -m nodes.opportunity --deep-research

# Continue through Writer
uv run python -m nodes.opportunity --writer

# Continue through Reviewer
uv run python -m nodes.opportunity --reviewer
```

These live smoke tests use your API credentials and may consume provider
credits.

## Current MVP boundaries

The application intentionally does not include:

- automatic publishing;
- clinical approval or full medical fact-checking;
- existing-content coverage checks;
- SEO or search-demand scoring;
- user accounts or saved workflow history;
- a database;
- persistence for unfinished/in-progress runs; or
- autonomous search loops.

These boundaries keep the project understandable and make each component easy
to inspect and test independently.
