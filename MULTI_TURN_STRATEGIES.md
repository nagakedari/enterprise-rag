# Multi-turn Conversation Strategies

This document covers how `/chat` moved from single question-and-answer to
multi-turn conversations. It describes the options for memory, context
storage and follow-up handling, what this repo implements and why, how it
was tested, and what's still open. Retrieval and agent strategies are in
[RAG_STRATEGIES.md](RAG_STRATEGIES.md).

> Code: `src/conversation/` (models, store, memory, contextualizer, service),
> wired into `src/api/main.py`. Config: `ConversationConfig` in `src/config.py`.

---

## 1. Architecture

The agent (`src/agentic/`) stays **single-turn and unaware of history**. A
conversation layer in front of it turns each new message into something the
agent can answer on its own:

```
POST /chat {query, conversation_id?}
   │
   ├─ load conversation (or create one)            ConversationStore (SQLite)
   │
   ├─ first turn? ──► answer as-is                  no extra LLM call
   │
   └─ contextualize(message, history)               CONVERSATION_MODEL (gpt-4o)
         ├─ chit_chat              ──► direct reply                 no retrieval
         ├─ about_previous_answer  ──► rework the last answer       no retrieval
         └─ new_question / follow_up
                 ──► agent(standalone question)                     AGENT_MODEL
   │
   └─ store turn → update entity state → fold old turns into summary
```

Keeping the agent stateless was deliberate:
- **Retrieval and evaluation are unchanged.** The agent, its prompt and the
  evaluation pipeline see the same kind of input as before: one standalone
  question.
- **Grounding stays clean.** Earlier answers never enter the agent's
  context, so they can't be mistaken for filing evidence. Every new fact
  comes from a fresh search.
- **Each step can be tested on its own.** The rewrite (message → standalone
  question) is a small, cheap step with its own test set.

---

## 2. Turn kinds

The contextualizer classifies every message after the first:

| Kind | Example | Handling | Retrieval |
|---|---|---|---|
| `new_question` | "How did Amazon's AWS revenue change over time?" | Sent to the agent **unchanged**. Rewriting a self-contained question can only change what was asked. | yes |
| `follow_up` | "what about Intel?", "and the quarter before that?", "why did it drop?" | Rewritten as a standalone question, then sent to the agent | yes |
| `about_previous_answer` | "make that a one-line summary", "as a table" | The light model reworks the stored answer, keeping its figures and citations | **no** |
| `chit_chat` | "thanks!" | Direct reply | **no** |

Greetings on the *first* turn are still handled by the agent's planner
(`requires_filings=false`, see RAG_STRATEGIES.md), so they also skip
retrieval.

---

## 3. Memory strategies

### 3.1 Options

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| **Full buffer** | Send the whole conversation every turn | Simple; nothing lost | Cost and latency grow every turn; eventually hits the context limit; old irrelevant turns distract the model |
| **Sliding window** | Last N turns verbatim | Bounded size; exact recent references | Forgets anything older than N |
| **Token-budget window** | As many recent turns as fit in a token budget | Adapts to turn length | Needs token counting; still forgets |
| **Running summary** | An LLM folds old turns into a short summary | Keeps the gist of long conversations at bounded size | Extra LLM call; lossy, and figures can blur |
| **Summary + window (hybrid)** | Summary of old turns, verbatim recent turns | Best of both: precise recent, compact old | Two mechanisms to maintain |
| **Entity / slot state** | Track structured facts (active company, period, metric) | Deterministic, cheap; resolves "and the quarter before?" | Only covers the fields modelled |
| **Vector memory** | Embed past turns; retrieve relevant ones per message | Scales to very long histories; recalls old turns by topic | Extra index; retrieval can miss; overkill for short sessions |
| **Long-term user memory** | Per-user profile or preferences across conversations | Personalisation ("I usually ask about NVIDIA") | Needs user identity, consent, retention policy |
| **Source / citation memory** | Keep what each turn retrieved | Enables "show me the source" and reuse without re-searching | Source numbering across turns must stay stable |

### 3.2 Implemented: summary + window + entity state

`src/conversation/memory.py`:
1. **Entity state.** The companies and filing labels from the agent's plan
   each turn, most recent first, up to 6 each. No LLM call. Shown to the
   contextualizer as `ACTIVE CONTEXT`.
2. **Sliding window.** The last `CONVERSATION_WINDOW_TURNS` (4) turns
   verbatim, with each answer truncated to `CONVERSATION_MAX_ANSWER_CHARS`
   (1500). The resolved standalone question is shown as well, so the model
   can see how earlier follow-ups were read.
3. **Running summary.** When a turn leaves the window it's folded into a
   summary of under 120 words by the light model: one call per evicted turn,
   and none at all for conversations of 4 turns or fewer. If it fails, the
   old summary is kept and the fold is retried on a later turn.

**Why this mix:** analyst conversations are mostly short, with references to
the last one or two turns ("what about…", "and before that?"). The window
handles those exactly, the entity state makes company and period carry-over
cheap and reliable, and the summary keeps longer sessions bounded. Vector
memory and long-term user memory aren't needed at this scale (see §8).

**Source memory, partly:** each turn stores a compact list of its sources
(`source_file`, pages), returned by `GET /conversations/{id}`. Retrieved
text isn't reused across turns: follow-ups search again, which keeps
grounding simple at the cost of some repeated retrieval.

---

## 4. Follow-up resolution (query contextualization)

### 4.1 Options

| Strategy | How it works | Trade-off |
|---|---|---|
| **Condense-question rewrite** | An LLM rewrites the follow-up as a standalone question, then normal single-turn RAG runs | One extra call; retrieval and agent unchanged; easy to test |
| **History into the agent/planner** | Pass the conversation to the planner and answer step | No rewrite call, but couples the agent to history, inflates every agent turn, and earlier answers can leak into "evidence" |
| **Rewrite + multi-query** | The rewrite produces several phrasings | Better recall, more searches |
| **Entity-filter carry-over only** | Carry company/period filters, keep the user's text | Cheap, but "why did it drop?" stays unanswerable |

### 4.2 Implemented: condense-question rewrite + routing

`src/conversation/contextualizer.py`, a forced `contextualize` function
call that returns `{kind, standalone_question, direct_reply}`. The rules
were all added because of test failures, and all are general:
- **Ask exactly what the user asked.** Keep the intent words: compare, why,
  how did it change, trend.
- **Carry over only what the message leaves implicit.** For "what about
  Intel?", keep the metric and period, replace the company.
- **Never compute dates.** Relative periods are written relative to a period
  already named ("the quarter immediately before the quarter ended July 1,
  2023"). The agent's filing catalog resolves them, because filing labels
  and fiscal calendars differ by company.
- **New questions pass through unchanged** (a code guard, not just the
  prompt).

**Test, contextualizer only** (fixed two-turn Apple history, 7–8 messages):

| Version | Kinds correct | Failures |
|---|---|---|
| gpt-4o-mini, first prompt | 6/7 | Rewrote a new trend question into "latest quarter"; dropped "compare"; computed "quarter ended July 1, 2022" (Apple's year-earlier quarter ended June 25, 2022) |
| + new-question guard, stricter prompt (mini) | 12/14 | "what about Intel?" labelled new_question by **both** models: the prompt's "new subject" rule was ambiguous. Mini still computed dates |
| + "leaves something implicit" definition, **gpt-4o** | **16/16** | — |

**Bug found in use (2026-11-06): explicit quarter labels rewritten as
relative periods.** Intel conversation:
- "net profit … Q3 2023"
- → "how about q4 of the same year?" became "the quarter immediately
  *after* the quarter ended Sep 30, 2023"
- → "can find about q1?" became "the quarter immediately *before* the
  quarter ended Sep 30, 2023", which was answered with **Q2** ($1,473M).

Cause: the "describe relative periods relative to a named period" rule
(added to stop computed dates) was applied to labels the user named. The
"reuse a period wording from the conversation" rule pulled in the answer's
date in place of the user's label. The old wording also computed dates
("q1" → "quarter ended March 31, 2023"; Intel's Q1 actually ended April 1).

Fix:
- The prompt separates **explicit** periods (a label, fiscal year or date:
  kept as written, with only a missing year filled in) from **relative**
  ones (only for words like before, after, previous, same quarter last
  year).
- A code guard: if the message names a quarter label (`q1`–`q4`) that the
  rewrite drops, the rewriter is retried once with a correction note.

Results (14 cases including the Intel conversation): 9/14 before, 14/14
after, in two runs. The test's strict check flags "same quarter in 2022",
which is correct, as label arithmetic on filing labels. Live replay: "q1" →
"Q1 2023" → 2023 Q1 INTC filing → "quarter ended April 1, 2023, net loss of
$2,768 million".

**Open issue, agent side:** "Q4 2023" is not in the corpus. The planner
silently searched the latest filing (Q3) and answered with the Q3 figure as
if it were Q4. The rule to report periods that aren't covered exists in the
prompt but isn't enforced in code.

So the rewrite uses **gpt-4o** (`CONVERSATION_MODEL`), at about $0.005 per
follow-up turn. Reworking the previous answer and the summary use
**gpt-4o-mini** (`CONVERSATION_LIGHT_MODEL`): those tasks only restate text
that's already there.

---

## 5. Context storage strategies

### 5.1 Options

| Backend | Good for | Limits |
|---|---|---|
| **In-memory dict** | Tests, demos | Lost on restart; one process only |
| **SQLite file** | Single-server apps, local dev; zero infrastructure; persistent | One writer at a time; not shared across servers |
| **Redis** | Many API instances; TTL built in; fast | Extra service; persistence needs configuring |
| **Postgres** | Production with analytics and audit; shared; transactional | Extra service and migrations (this repo already runs one for Airflow) |
| **Document DB** (Mongo, DynamoDB) | Very large scale, flexible schema | Operational overhead |
| **Client-side history** | Stateless servers: the client sends the history each turn | Client can tamper with or bloat history; server can't summarise |

### 5.2 Implemented: SQLite (default) + in-memory

`src/conversation/store.py`, behind a four-method `ConversationStore`
interface (`get`, `save`, `delete`, `purge_older_than`). A Redis or Postgres
store only needs those four methods.

- **Schema:** `conversations(id, created_at, updated_at, summary,
  summarized_turns, state JSON)` and `turns(conversation_id, idx, data
  JSON)`. Turns are append-only, so only new rows are written per turn.
- **Location:** `CONVERSATION_DB_PATH`, default `./storage/conversations.db`
  (git-ignored).
- **Concurrency:** a connection per call (safe under FastAPI's thread pool),
  WAL mode for concurrent reads.
- **Retention:** conversations not updated for `CONVERSATION_TTL_DAYS` (30)
  are purged at server start. `DELETE /conversations/{id}` removes one
  immediately.
- **When to move off SQLite:** more than one API instance, or a need for
  shared history across services. Redis suits session-only memory; Postgres
  suits keeping history for audit.

---

## 6. What each model call sees

| Call | Input | Never sees |
|---|---|---|
| Contextualizer (gpt-4o) | summary + last 4 turns (answers truncated) + entity state + new message | retrieved filing text |
| Agent planner and answer (gpt-4o) | the standalone question + filing catalog | conversation history |
| Rework previous answer (gpt-4o-mini) | previous question + previous answer + request | filings, older turns |
| Summary (gpt-4o-mini) | current summary + evicted turns | filings |

**Cost per turn on top of single-turn:** turn 1, $0. Later turns add one
gpt-4o rewrite (about $0.005) plus a gpt-4o-mini summary call once the
conversation passes 4 turns. Reworks and chit-chat **replace** the agent
run, so they make the turn cheaper than before.

---

## 7. API

```bash
# New conversation: conversation_id is returned
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"query": "What was Apple'\''s iPhone revenue for the latest quarter?"}'

# Follow-up: send the id back
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"query": "what about Mac?", "conversation_id": "<id>"}'

curl -s localhost:8000/conversations/<id>          # turns, summary, entity state
curl -s -X DELETE localhost:8000/conversations/<id>
```

New `ChatResponse` fields: `conversation_id`, `turn_index`, `turn_kind`,
`standalone_query` (the question actually answered). A request without
`conversation_id` behaves as before, plus a new conversation is recorded.
An unknown id returns 404.

| Env var | Default | Meaning |
|---|---|---|
| `CONVERSATION_STORE` | `sqlite` | `sqlite` or `memory` |
| `CONVERSATION_DB_PATH` | `./storage/conversations.db` | SQLite file |
| `CONVERSATION_MODEL` | `gpt-4o` | Follow-up rewriting and turn classification |
| `CONVERSATION_LIGHT_MODEL` | `gpt-4o-mini` | Reworking the last answer; running summary |
| `CONVERSATION_WINDOW_TURNS` | `4` | Turns shown verbatim |
| `CONVERSATION_MAX_ANSWER_CHARS` | `1500` | Truncation of past answers in history |
| `CONVERSATION_MAX_TOKENS` | `1024` | Output cap for reworked answers |
| `CONVERSATION_TTL_DAYS` | `30` | Purge conversations idle this long |

---

## 8. Evaluating multi-turn

The golden Q&A set is single-turn, so multi-turn needs its own checks,
cheapest first:
1. **Rewrite accuracy** (no retrieval, cents): fixed histories, then follow-up
   messages with the expected kind and a reference standalone question.
   Checks routing, entity carry-over and intent preservation. This is §4.2's
   test and should become a small regression set.
2. **Scripted dialogues** (agent cost per turn): 3–5-turn conversations built
   from gold questions (turn 1 = a gold question, later turns = follow-ups
   whose standalone form is another gold question). Score each turn's answer
   with the existing metrics against that gold answer.
3. **Conversation-level checks:** consistency (no contradiction of an earlier
   answer), no-retrieval routing (reworks and chit-chat must not trigger
   searches), and cost and latency per turn.

---

## 9. Status and limitations

**Tested:**
- Mocked end-to-end through `/chat`: turn routing, the agent receiving the
  standalone question, reworks and chit-chat skipping the agent, entity
  state, summary folding at window 2, 404s, delete.
- SQLite round-trip.
- Live contextualizer: 16/16.

**Not yet tested:** a live multi-turn conversation through retrieval (Docker
and Weaviate were down at the time).

**Known limitations / next steps:**
- **Concurrent turns on the same conversation:** the second write hits the
  turn-index primary key and fails (HTTP 502). Clients should send turns in
  order; a per-conversation lock or retry would fix it.
- **No user identity:** anyone with an id can read or delete that
  conversation. Add auth and an owner column before multi-user use.
- **Reworking only sees the last answer.** "Combine your last two answers"
  isn't supported.
- **The summary is lossy.** It's kept short and figures are copied exactly,
  but detail from turns older than 4 back is reduced to the summary.
- **No streaming,** and the evaluation UI has no chat view yet.
- **Retrieved text isn't reused across turns.** Every follow-up searches
  again (simple and well grounded, but repeats work).
