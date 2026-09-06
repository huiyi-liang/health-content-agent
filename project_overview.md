# Health Content Agent: Project Overview and Learnings

## Project overview

The Health Content Agent turns one free-form health topic into a patient-focused, evidence-grounded article draft. It is intended for a mature consumer-health publisher that needs more specific coverage of patient questions, management challenges, decisions, and meaningful recent developments.

The application uses Python, LangGraph, Nebius, You.com Search, and Streamlit. Like a conductor, LangGraph runs the components and carries state. LLMs handle language and judgment; Python handles IDs, normalization, validation, routing, citations, and retry limits.

The flow is **Discovery → Opportunity → human selection → Deep Research → Writer → Reviewer**. Discovery plans three searches; Opportunity returns three ideas; the user selects one; Deep Research gathers article-specific evidence; Writer produces a cited structured draft; and Reviewer identifies one primary problem. Python then completes, retries research or writing, or escalates after three corrections.

The MVP does not diagnose, clinically approve, publish, or retain runs across restarts.

## Datasets and evidence used

The project uses no fixed training dataset or patient records. Each run builds evidence from You.com titles, URLs, optional dates, and query-specific Highlights. **The external tool call is `request_you_search()` in `tools/search.py`; its `httpx.Client.post(...)` call sends the query to You.com.** One response may contain both `results.web` and `results.news`; Python preserves Highlights and creates `Source` objects with `D#` discovery or `R#` research IDs. Duplicate URLs remain because queries may retrieve different Highlights. The synthetic `config/article_style.md` supplies editorial guidance. Tests use mocked fixtures; smoke tests use live services.

## Prompts used during vibe coding

Vibe-coding prompts were phase-scoped so each added one testable layer. The production prompt patterns were:

- **Discovery:** current date, never guess a year, three complementary patient-oriented queries, and no article ideas.
- **Opportunity:** consider six candidates, return three using patient relevance, specificity, and usefulness; do not score SEO or evidence sufficiency.
- **Deep Research:** full selected idea, three nonredundant queries, credible-domain preferences, and gap-filling retries.
- **Writer:** selected angle, editorial guide, evidence-only medical facts, valid `[R#]` citations, and flagged-section-only revision.
- **Reviewer:** compare with supplied evidence, return one prioritized failure, and use the dynamic valid `section_id` list.
- **Orchestration:** let the Reviewer decide what is wrong, let deterministic Python decide where to route, and stop before a fourth automated correction loop.

## Reviewer iterations and trade-offs

The Reviewer required the most iteration because valid structured output did not guarantee useful judgment. Early runs returned headings or invented values instead of stable `section_id`s. Passing the draft's valid IDs to the Reviewer and validating them in Python fixed the interface, but not whether its evidence judgment was correct.

Repeated `unsupported_claim` failures came next. The Writer retained claims or statistics that Highlights did not fully support, and revisions did not always remove them. We tightened both prompts, preserved unflagged sections, and re-ran citation-ID checks after revision.

For a reliable MVP happy path, we shortened articles and removed odds ratios, confidence intervals, P-values, and similar statistics. This made grounding easier but produced less detailed articles: the system favors a narrow supported draft over a richer unsupported one.

One primary failure keeps routing understandable, but multiple problems may consume several cycles. The three-retry limit prevents runaway loops and hands the latest work to a human.

## Learnings, observations, and evaluation concerns

Structured output validates an answer's **shape**, not its truth. It is like checking that every box on a form is filled: the entries can still be wrong. Deterministic checks can verify allowed statuses, real section IDs, and existing `[R#]` citations, but they cannot prove that a cited Highlight actually supports the nearby claim.

Evaluation therefore cannot rely only on “the graph reached PASS.” A lenient Reviewer could pass a weak draft, while an overly strict Reviewer could cause unnecessary retries. The Writer and Reviewer also use the same model in this MVP, which is simple and inexpensive but may reproduce similar blind spots.

Testing should combine unit tests for PASS and each failure, graph tests for routes/retries/revisions, and repeated live evaluations across broad and specific topics. A human editor should score claim support, patient value, readability, route choice, and usability. The 8-of-10 target still needs a documented evaluation set and repeated runs.

The broader lesson is to separate judgment from control. LLMs plan, write, and review; Python enforces IDs, state, routes, and limits. Human escalation is a safety outcome, not automatically a failure.

## Future evaluation plan and golden dataset

A **golden dataset** is a fixed collection of examples with expert-reviewed expected results. It is the workflow's answer key, but it should not require the LLM to reproduce one exact sentence. For creative outputs such as queries, ideas, and articles, the “gold” should be a scoring rubric, required properties, acceptable alternatives, and known unacceptable behavior.

An initial dataset could contain 20–30 representative topics balanced across:

- broad and specific conditions;
- common management questions and patient experiences;
- timely topics where the current-date rule matters;
- topics with strong evidence, limited evidence, and conflicting evidence; and
- easy happy paths plus cases expected to require research retry, Writer revision, or human escalation.

Because live search results change, each example should save a snapshot of the normalized You.com evidence used for evaluation. Otherwise, a test could change because the web changed rather than because the agent changed. The dataset should be versioned and divided into a development set used while adjusting prompts and a held-out set used only for final evaluation.

### What to evaluate at each step

| Component | Golden input and expected annotation | Main evaluation question |
| --- | --- | --- |
| **Discovery planner** | Topic plus acceptable research dimensions and prohibited generic/duplicate queries | Are there exactly three relevant, complementary queries with no invented year? |
| **Search tool** | Frozen web/news API payloads and expected `Source` objects | Are Highlights, source type, query provenance, dates, and deterministic IDs preserved correctly? |
| **Opportunity Agent** | Frozen Discovery evidence plus editor-rated strong and weak ideas | Are the three ideas specific, patient-relevant, useful, and supported as opportunities by the evidence? |
| **Deep Research** | Selected idea plus an expert checklist of evidence the article needs | Do the searches cover the important evidence gaps and retrieve useful, credible sources? |
| **Writer** | Frozen evidence plus a claim-to-source support map | What percentage of medical claims are supported, correctly cited, readable, and within the selected angle? |
| **Reviewer** | Intentionally clean and flawed drafts with expert labels for failure type and section IDs | Does it distinguish insufficient evidence from an unsupported Writer claim, flag the right sections, and PASS only clean drafts? |
| **Full graph** | Topic, fixed evidence fixtures, expected routes, and a usability rubric | Does the workflow complete or escalate correctly without exceeding the retry limit? |

The Reviewer dataset is especially important. It should contain matched examples where only one detail changes: a supported claim becomes unsupported, evidence is removed to create `insufficient_research`, readability is degraded, or practical patient value is removed. These controlled pairs reveal whether the Reviewer understands the failure definitions instead of simply preferring FAIL or reacting to writing style.

### Suggested measures

- **Deterministic contracts:** 100% valid schemas, real source and section IDs, valid citations, correct routes, and correct retry counts.
- **Writer grounding:** supported medical claims divided by all medical claims, plus the rate of citations attached to the correct source.
- **Reviewer accuracy:** PASS/FAIL accuracy, precision and recall for each failure type, correct-section rate, and deterministic route accuracy.
- **Retrieval quality:** percentage of expert-required evidence needs covered by the retrieved Highlights, not merely the number of sources returned.
- **End-to-end value:** usable-draft rate, correct-escalation rate, average runtime, API cost, and number of correction cycles.

At least two reviewers—ideally one consumer-health editor and one medically knowledgeable reviewer—should label the difficult examples and resolve disagreements. Since LLM output can vary even with low temperature, important cases should run several times. This evaluation would show whether a prompt change genuinely improves the system rather than fixing only the latest example.

# Part 2: The Agent Framework

## The Primer: Your one-liner

*My agent helps consumer-health editors turn a health topic into an evidence-grounded article draft in a Streamlit app, replacing hours of searching, source tracking, drafting, and review; it plans, searches, proposes, writes, and reviews using one external retrieval tool, hands off for idea selection, final review, or after three unresolved corrections, and succeeds when an editor reaches a usable draft or clear escalation in under 15 minutes on at least 8 of 10 test topics.*

The “under 15 minutes” and “8 of 10 topics” values are evaluation targets, not validated production results.

## The Framework

| Field | Answer |
| --- | --- |
| **Agent goal** | Turn a free-form health topic into a human-selected, researched, cited consumer-health article draft with automated editorial review. |
| **Where do people use it?** | In a Streamlit web application. A terminal harness is also available for development and smoke testing. |
| **What steps does it take, in order?** | 1) Discover opportunities, 2) propose three ideas, 3) pause for human selection, 4) research the choice, 5) draft, and 6) review and route through no more than three correction cycles. |
| **What can it actually do?** | You.com Search is the read-only external tool. Nebius plans and writes; Python normalizes, validates, assigns IDs, and routes; Streamlit collects input but does not publish. |
| **What does it need to remember?** | LangGraph state and an in-memory checkpointer hold the active run. Streamlit stores the thread ID and necessary UI values; there is no cross-session memory. |
| **What should it never do?** | Never diagnose, invent evidence or citations, expose secrets, automatically publish, call a draft medically approved, or loop indefinitely. |
| **Human-in-the-loop** | A person selects the article idea and performs final editorial review. The system also hands off when three automated corrections cannot resolve the primary issue. |
| **What happens when something breaks?** | Tool or model failures create a clear `WorkflowError` and stop the run. Editorial failures route to more research or targeted rewriting, while the retry guard prevents a fourth automated correction. |
| **How do you know it worked?** | Target: produce a usable draft or transparent human-review handoff in under 15 minutes for at least 8 of 10 representative topics. |
