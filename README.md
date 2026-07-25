# Enterprise AI Support Platform

A production-grade AI support agent built across 12 sessions using LangGraph, Google Gemini, and FastAPI. Each session adds one architectural capability on top of the last, resulting in a fully operational multi-agent system with human-in-the-loop approval, persistent state, and time-travel debugging.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. Set environment variables
cp .env.example .env          # then fill in GOOGLE_API_KEY
# Optional: GITHUB_TOKEN, GITHUB_REPO for real GitHub issue creation

# 4. Start the server
python api.py                 # → http://localhost:8000

# 5. Or run the CLI test harness
python support_agent.py
```

---

## Project Files

| File | Purpose |
|---|---|
| `support_agent.py` | Core agent: graph, nodes, tools, state schema, all helper functions |
| `api.py` | FastAPI server exposing all agent capabilities as REST endpoints |
| `index.html` | Single-file frontend served statically by the API |
| `requirements.txt` | All Python dependencies |
| `support.db` | SQLite checkpoint store (auto-created; gitignored) |

---

## Architecture Overview

```
User Ticket
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  TRIAGE SUBGRAPH                                        │
│  ingress_node → [PII mask + injection filter]          │
│  classify_node → technical | billing | fraud | general  │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   SUPERVISOR (hub)         DISPATCHER (fan-out)
   hub-and-spoke LLM        parallel Send API
   routing                  ┌──────┬──────┐
          │                 │      │      │
          ▼              tech  billing  fraud
   tech_support             │      │      │
   fraud_handler            └──────┴──────┘
   general_handler                 │
          │                   synthesizer
          └──────────────────────────┐
                                     ▼
                             github_tool_node
                             [HITL interrupt]
                                     │
                                     ▼
                              final_response
```

---

## Session-by-Session Breakdown

### Session 1 — The Blueprint
**Capability:** Graph skeleton and static routing.

Built the first LangGraph `StateGraph` with a typed `SupportState` schema. Added `classify_node` (LLM-based ticket categoriser returning one of four categories) and conditional routing via `route_by_category`. No tools, no persistence — just the skeleton that every later session builds on.

Key additions: `StateGraph`, `StateSchema`, `classify_node`, `route_by_category`, `general_handler`, `billing_handler`, `technical_handler` stubs.

---

### Session 2 — Tool Binding
**Capability:** CRM lookup and knowledge base search tools.

Introduced `get_customer_details` (mock CRM lookup) and `search_knowledge_base` (keyword-matched KB articles). Built `agent_node` as a single-pass ReAct step using `llm.bind_tools()`. Added `tool_node` (LangGraph `ToolNode`) and `route_after_agent` to loop between agent and tools. Added `respond_node` to extract the final AIMessage.

Key additions: `@tool` decorators, `ToolNode`, `agent_node`, `respond_node`, `route_after_agent`.

---

### Session 3 — The ReAct Architecture
**Capability:** Full ReAct loop with circuit breaker and duplicate detection.

Replaced the single-pass agent with a proper ReAct loop: `agent_node` now increments `iteration_count` on every call and hard-stops at `MAX_ITERATIONS=5`. Added fingerprint-based duplicate tool call detection — if the agent calls the same tool with the same arguments twice, it escalates rather than looping forever. Added `check_fraud_signals` as a third tool and `fraud_handler`.

Key additions: `MAX_ITERATIONS`, `iteration_count`, `get_tool_fingerprint`, `build_escalation_response`, `trim_context`, `AGENT_SYSTEM_PROMPT`.

---

### Session 4 — Persistence and Threading
**Capability:** SQLite checkpoint store; conversation threads survive restarts.

Wired `SqliteSaver` as the graph checkpointer. Every `graph.invoke()` now takes a `config` dict with `thread_id`. All state is written to `support.db` after every node. Added `run_ticket()` and `stream_ticket()` as the canonical entry points — both generate or accept a `thread_id` and handle first-turn vs. follow-up turn logic. Added `get_conversation_history()` to read the checkpoint ledger.

Key additions: `SqliteSaver`, `thread_id`, `run_ticket`, `stream_ticket`, `get_conversation_history`, `get_active_threads`.

---

### Session 5 — Context Management
**Capability:** Summarisation node; bounded context window.

Added `summarization_node`: when `messages` exceeds `SUMMARY_THRESHOLD=8`, the node compresses history into `system_summary` and trims the message list to the last 4 entries using `RemoveMessage`. Replaced the default `add_messages` reducer with a custom `deduplicate_messages` reducer that handles both `RemoveMessage` deletions and duplicate prevention. Added `route_after_classify` to check the threshold before routing to `agent_node`.

Key additions: `summarization_node`, `SUMMARIZATION_PROMPT`, `SUMMARY_THRESHOLD`, `deduplicate_messages`, `system_summary` field, `RemoveMessage`.

---

### Session 6 — Guardrails and Bounding
**Capability:** PII detection and injection pattern filtering at ingress and egress.

Added `ingress_node` as the new graph entry point. Uses Presidio `AnalyzerEngine` to detect and anonymise PII (credit cards, SSNs, phone numbers, emails, etc.) before the ticket reaches the LLM. Runs regex against 14 injection patterns — detected injection sets `is_safe=False` and routes to `blocked_response_node` (zero LLM tokens). Added `egress_node` to scan `final_response` for PII leakage and uncertainty markers before delivery.

Key additions: `ingress_node`, `egress_node`, `blocked_response_node`, `route_after_ingress`, `PII_ENTITIES`, `INJECTION_PATTERNS`, `UNCERTAINTY_MARKERS`, `pii_detected`, `injection_detected`, `is_safe`, `sanitized_input`.

---

### Session 7 — Multi-Agent Topologies
**Capability:** Independent triage and tech-support subgraphs; shared state reducers.

Refactored the flat graph into a master + subgraph architecture. `build_triage_subgraph()` compiles ingress → classify independently. `build_tech_support_subgraph()` compiles the summarisation → agent → tool → respond → egress loop independently. The master graph composes them as nodes. Replaced `SupportState` with `SharedState` — a unified schema that all subgraphs share. Switched `tool_results` and `internal_notes` to `operator.add` reducers so parallel writers never silently overwrite each other.

Key additions: `SharedState`, `build_triage_subgraph`, `build_tech_support_subgraph`, `operator.add` reducers, `route_after_triage`.

---

### Session 8 — The Supervisor Orchestrator
**Capability:** LLM-powered hub-and-spoke routing via structured output.

Added `supervisor_node`: an LLM call using `with_structured_output(SupervisorDecision)` that reads full state and returns one of five routing decisions (`triage`, `tech_support`, `fraud_handler`, `general_handler`, `FINISH`). Every worker returns to `supervisor_node` after completing — no worker routes directly to END. The supervisor applies a `MAX_DELEGATIONS=5` safety limit before calling the LLM. Added `delegation_count` and `next_worker` to state.

Key additions: `supervisor_node`, `supervisor_router`, `SupervisorDecision`, `SUPERVISOR_PROMPT`, `MAX_DELEGATIONS`, `delegation_count`, `next_worker`.

---

### Session 9 — Shared Scratchpads and Consensus
**Capability:** Three parallel specialist agents running simultaneously via LangGraph's Send API.

Added `dispatcher_node` which returns `list[Send]` to fan out to `tech_analysis_agent`, `billing_analysis_agent`, and `fraud_analysis_agent` in a single superstep. Each agent has a scoped LLM (bound to one tool only), runs one tool call, and writes a structured finding dict to `internal_notes`. After the superstep, `synthesizer_node` reads all findings, filters `severity='none'`, sorts by severity, and calls the LLM once to produce a unified customer response.

Key additions: `dispatcher_node`, `tech_analysis_agent`, `billing_analysis_agent`, `fraud_analysis_agent`, `synthesizer_node`, `Send`, `SYNTHESIZER_PROMPT`, `extract_finding_from_response`.

---

### Session 10 — Write Access
**Capability:** GitHub issue creation with idempotency; write tool behind severity gate.

Added `create_github_issue` — a write tool that POSTs to the GitHub API (or runs in mock mode). Protected by `generate_idempotency_key()`: a SHA-256 hash of `thread_id + title + body` ensures the same logical issue is never created twice, even on retry. `tech_analysis_agent` populates `github_draft` in state when severity is `high` or `critical`. `github_tool_node` reads the draft and executes the write. Wired `github_tool_node` between `tech_analysis_agent` and `synthesizer` in the master graph.

Key additions: `create_github_issue`, `generate_idempotency_key`, `github_tool_node`, `github_draft`, `github_issue_url`.

---

### Session 11 — Interruptions and Breakpoints
**Capability:** Human-in-the-loop approval gate; no write happens without explicit human sign-off.

Replaced the automatic write in `github_tool_node` with `interrupt()` — the graph pauses mid-execution and returns control to the caller with `snapshot.next = ['github_tool_node']`. Three approval paths added: `approve_github_issue()` resumes with `Command(resume={'approved': True})`; `deny_github_issue()` resumes with `approved=False`; `edit_and_approve_github_issue()` calls `update_state()` then resumes with an edited draft. Added `get_pending_approvals()` to query all suspended threads.

FastAPI endpoints: `GET /api/pending-approvals`, `POST /api/approve/{thread_id}`, `POST /api/deny/{thread_id}`, `POST /api/edit-approve/{thread_id}`.

UI: Human Approval Gate panel with Approve / Deny / Edit & Approve actions; 30-second polling sidebar showing the live approval queue.

Key additions: `interrupt()`, `Command(resume=...)`, `get_pending_approvals`, `approve_github_issue`, `deny_github_issue`, `edit_and_approve_github_issue`.

---

### Session 12 — Time Travel and State Forensics
**Capability:** Full checkpoint ledger inspection, anomaly detection, state correction, and time-travel re-invocation.

The checkpoint ledger that has been recording every node execution since Session 4 is now used actively.

**`state_forensics(thread_id)`** reads the full checkpoint history via `graph.get_state_history()` and returns a structured report: total steps, a timeline entry per checkpoint (source, category, message count, PII flags, GitHub URL presence), detected anomaly flags (circuit breaker proximity, possible misclassification, PII not sanitised, injection not blocked, empty final response), and human intervention markers (checkpoints where `source='update'` — written by explicit `update_state()` calls).

**`find_bad_checkpoint(thread_id, field, bad_value)`** iterates history chronologically and returns the `parent_config` of the first checkpoint where `state[field]` contains `bad_value` — the state immediately before the wrong value was written. Used as the time-travel target.

**`apply_correction(thread_id, field, new_value)`** calls `graph.update_state()` to inject a corrected value into the checkpoint ledger. The correction is immediately readable via `graph.get_state()` and visible in subsequent forensics runs as a human intervention.

**`time_travel(thread_id, target_step)`** loads the checkpoint at `target_step` from the history and calls `graph.invoke(None, config=target.config)`. This creates a new execution branch from that point — the original branch is preserved in the ledger unchanged.

FastAPI endpoints: `GET /api/forensics/{thread_id}`, `POST /api/time-travel`, `POST /api/correct`.

UI: Forensics Timeline Panel with per-step inspection, flag highlighting (red for anomalies, amber for human interventions), and a ⏱ Travel button on every checkpoint row. Time Travel Modal with optional correction fields (field + new value) and a Re-run button that calls `/api/correct` then `/api/time-travel` in sequence. Completion banner displayed after all 12 session checks pass.

Key additions: `state_forensics`, `find_bad_checkpoint`, `apply_correction`, `time_travel`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/run` | Submit a ticket; returns full result including interrupt status |
| `POST` | `/api/stream` | Submit a ticket; streams node-by-node SSE events |
| `POST` | `/api/verify` | Run the Session 12 verification suite (5 checks) |
| `GET` | `/api/threads` | List all thread IDs in the checkpoint store |
| `GET` | `/api/history/{thread_id}` | Checkpoint history for a thread |
| `GET` | `/api/pending-approvals` | Threads currently paused at the HITL gate |
| `POST` | `/api/approve/{thread_id}` | Resume with approval; executes the GitHub write |
| `POST` | `/api/deny/{thread_id}` | Resume with denial; skips the write |
| `POST` | `/api/edit-approve/{thread_id}` | Inject edited draft then resume with approval |
| `GET` | `/api/forensics/{thread_id}` | Full forensics report for a thread |
| `POST` | `/api/time-travel` | Re-invoke graph from a specific checkpoint step |
| `POST` | `/api/correct` | Inject a state correction via `update_state()` |
| `GET` | `/health` | System status and configuration |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key — get one at aistudio.google.com |
| `GITHUB_TOKEN` | No | GitHub personal access token for real issue creation |
| `GITHUB_REPO` | No | Target repo in `owner/repo` format (e.g. `acme/support`) |

Without `GITHUB_TOKEN` and `GITHUB_REPO`, `create_github_issue` runs in mock mode and returns a fake URL — all other functionality is unaffected.

---

## Conclusion

This project demonstrates that a production-grade AI agent is not a single LLM call — it is a system. Over 12 sessions, the platform evolved from a three-node routing stub into a full multi-agent architecture with:

- **Reliability**: circuit breaker, duplicate detection, idempotent writes
- **Security**: PII anonymisation at ingress, injection filtering, egress scanning
- **Scalability**: parallel specialist agents running in a single superstep
- **Observability**: every node execution recorded in a queryable checkpoint ledger
- **Control**: human approval gate that pauses graph execution before any write
- **Debuggability**: time-travel re-invocation and state correction without data loss

The same SQLite file that started as a simple persistence layer in Session 4 ends as a full forensics database in Session 12 — queryable, correctable, and branchable. Every design decision along the way was driven by a real production concern, not abstraction for its own sake.
