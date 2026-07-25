"""
Enterprise AI Support Platform — FastAPI Backend
Session 12 of 12 — Time Travel & State Forensics
"""

import os
import json
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from support_agent import (
    run_ticket, stream_ticket,
    run_session_verification,
    get_conversation_history,
    get_active_threads,
    get_pending_approvals,
    approve_github_issue,
    deny_github_issue,
    edit_and_approve_github_issue,
    state_forensics,
    find_bad_checkpoint,
    apply_correction,
    time_travel,
    SharedState,
    TOOLS, MAX_ITERATIONS,
    SUMMARY_THRESHOLD,
    INJECTION_PATTERNS,
    MAX_DELEGATIONS,
    generate_idempotency_key,
    graph,
)

app = FastAPI(title="Enterprise AI Support Platform", version="8.0.0")


# ── Subgraph helpers (Session 7) ─────────────────────────────────

def infer_subgraph(node_name: str) -> str:
    triage_nodes = {
        'ingress_node', 'classify_node', 'blocked_response_node'
    }
    tech_nodes = {
        'summarization_check', 'summarization_node', 'agent_node',
        'tool_node', 'respond_node', 'egress_node'
    }
    parallel_nodes = {
        'dispatcher', 'tech_analysis_agent', 'billing_analysis_agent',
        'fraud_analysis_agent', 'synthesizer',
    }
    if node_name == 'supervisor':
        return 'supervisor'
    if node_name in triage_nodes:
        return 'triage'
    if node_name in tech_nodes:
        return 'tech_support'
    if node_name in parallel_nodes:
        return 'parallel'
    if node_name == 'github_tool_node':
        return 'write'
    return 'master'


def build_subgraph_trace(result: dict) -> list:
    """
    Builds a human-readable trace of which subgraphs ran.
    Used by the UI subgraph execution panel.
    """
    trace = []

    if result.get('pii_detected') is not None:
        trace.append({
            'subgraph': 'triage',
            'nodes':    ['ingress_node', 'classify_node'],
            'outcome':  ('blocked' if not result.get('is_safe', True)
                         else 'classified:' + result.get('category', '')),
        })

    if result.get('final_response') and result.get('is_safe', True):
        category = result.get('category', '')
        if category in ('billing', 'technical'):
            trace.append({
                'subgraph': 'tech_support',
                'nodes':    ['agent_node', 'tool_node', 'respond_node', 'egress_node'],
                'outcome':  (f"responded | "
                             f"{len(result.get('tool_calls_log', []))} tool calls"),
            })
        elif category == 'fraud':
            trace.append({
                'subgraph': 'master',
                'nodes':    ['fraud_handler'],
                'outcome':  'fraud analysis complete',
            })
        elif category == 'general':
            trace.append({
                'subgraph': 'master',
                'nodes':    ['general_handler'],
                'outcome':  'general response sent',
            })

    return trace

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    ticket:          str
    thread_id:       Optional[str] = None
    return_existing: bool = False   # True when stream already ran this thread


class StreamRequest(BaseModel):
    ticket:    str
    thread_id: Optional[str] = None


@app.post("/api/run")
def run(req: RunRequest):
    used_thread_id = req.thread_id
    result = run_ticket(req.ticket, thread_id=used_thread_id,
                        return_existing=req.return_existing)

    used_thread_id = result.get('thread_id', used_thread_id or '')

    # Extract tool call log
    tool_calls_log = []
    for msg in result.get('messages', []):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({
                    'tool_name': tc['name'],
                    'args':      tc['args'],
                    'call_id':   tc['id'],
                })

    # Match each call_id to a ToolMessage result
    tool_results_map = {}
    for msg in result.get('messages', []):
        if hasattr(msg, 'tool_call_id'):
            try:
                tool_results_map[msg.tool_call_id] = json.loads(msg.content)
            except Exception:
                tool_results_map[msg.tool_call_id] = msg.content

    for entry in tool_calls_log:
        entry['result'] = tool_results_map.get(entry['call_id'], {})

    iterations_used = result.get('iteration_count', 0)

    # Detect if execution was interrupted (Session 11)
    interrupted = False
    snapshot_next = []
    if used_thread_id:
        try:
            config = {'configurable': {'thread_id': used_thread_id}}
            snap = graph.get_state(config)
            interrupted = 'github_tool_node' in (snap.next or [])
            snapshot_next = list(snap.next or [])
        except Exception:
            pass

    response_payload = {
        "category":              result.get("category", ""),
        "final_response":        result.get("final_response", ""),
        "is_safe":               result.get("is_safe", True),
        "pii_detected":          result.get("pii_detected", False),
        "injection_detected":    result.get("injection_detected", False),
        "sanitized_input":       result.get("sanitized_input", "")[:100],
        "blocked":               not result.get("is_safe", True),
        "iteration_count":       iterations_used,
        "raw_input":             result.get("raw_input", ""),
        "tool_calls_log":        tool_calls_log,
        "circuit_breaker_fired": iterations_used > MAX_ITERATIONS - 1,
        "iterations_used":       iterations_used,
        "max_iterations":        MAX_ITERATIONS,
        "thread_id":             used_thread_id,
        "message_count":         len(result.get("messages", [])),
        "system_summary":        result.get("system_summary", ""),
        "summary_active":        bool(result.get("system_summary", "").strip()),
        "summary_threshold":     SUMMARY_THRESHOLD,
        "internal_notes":        result.get("internal_notes", []),
        "delegation_count":      result.get("delegation_count", 0),
        "next_worker":           result.get("next_worker", ""),
        "max_delegations":       MAX_DELEGATIONS,
        "supervisor_notes":      [
            n for n in result.get("internal_notes", [])
            if isinstance(n, dict) and n.get("agent") == "supervisor"
        ],
        "agent_findings":        [
            n for n in result.get("internal_notes", [])
            if isinstance(n, dict) and
            n.get("agent") in (
                "tech_analysis", "billing_analysis", "fraud_analysis"
            )
        ],
        "parallel_executed":     any(
            n.get("agent") in ("tech_analysis", "billing_analysis", "fraud_analysis")
            for n in result.get("internal_notes", [])
            if isinstance(n, dict)
        ),
        "subgraph_trace":        build_subgraph_trace({
            **result,
            "tool_calls_log": tool_calls_log,
        }),
        "github_draft":          result.get("github_draft", {}),
        "github_issue_url":      result.get("github_issue_url", ""),
        "github_created":        bool(result.get("github_issue_url", "").strip()),
        # Session 11 — interrupt/approval fields
        "interrupted":           interrupted,
        "pending_approval":      interrupted,
        "snapshot_next":         snapshot_next,
    }
    return response_payload


@app.post("/api/stream")
async def stream(req: StreamRequest):
    # Generate the UUID here so the start event exposes it and /api/run can reuse it
    thread_id_used = req.thread_id if req.thread_id else str(uuid.uuid4())

    def generate():
        start_payload = {"type": "start", "thread_id": thread_id_used}
        yield f"data: {json.dumps(start_payload)}\n\n"

        for node_name, snapshot in stream_ticket(req.ticket, thread_id=thread_id_used):
            payload = {
                "node":     node_name,
                "category": snapshot.get("category", ""),
                "response": snapshot.get("final_response", ""),
                "subgraph": infer_subgraph(node_name),
            }

            # Enrich ingress_node events with security data
            if node_name == 'ingress_node':
                payload['pii_detected']       = snapshot.get('pii_detected', False)
                payload['injection_detected']  = snapshot.get('injection_detected', False)
                payload['is_safe']             = snapshot.get('is_safe', True)
                payload['sanitized_input']     = snapshot.get('sanitized_input', '')[:80]

            # Enrich blocked_response_node events
            if node_name == 'blocked_response_node':
                payload['blocked'] = True
                payload['reason']  = (
                    "injection_detected"
                    if snapshot.get('injection_detected')
                    else "unsafe_content"
                )

            # Enrich summarization_node events
            if node_name == 'summarization_node':
                payload['summary_fired'] = True
                payload['summary']       = snapshot.get('system_summary', '')[:120]
                payload['msgs_before']   = len(snapshot.get('messages', []))

            # Enrich agent_node events with tool call info and iteration data
            if node_name == 'agent_node':
                msgs = snapshot.get('messages', [])
                if msgs:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        payload['tool_calls'] = [
                            {'name': tc['name'], 'args': tc['args']}
                            for tc in last_msg.tool_calls
                        ]
                payload['iteration']      = snapshot.get('iteration_count', 0)
                payload['max_iterations'] = MAX_ITERATIONS
                payload['summary_active'] = bool(
                    snapshot.get('system_summary', '').strip()
                )

            # Enrich tool_node events with tool results
            if node_name == 'tool_node':
                msgs = snapshot.get('messages', [])
                tool_results = []
                for msg in msgs:
                    if hasattr(msg, 'tool_call_id'):
                        tool_results.append({
                            'tool_name': getattr(msg, 'name', ''),
                            'content':   msg.content,
                        })
                if tool_results:
                    payload['tool_results'] = tool_results

            # Enrich supervisor node events (Session 8)
            if node_name == 'supervisor':
                payload['supervisor']      = True
                payload['delegation']      = snapshot.get('delegation_count', 0)
                payload['max_delegations'] = MAX_DELEGATIONS
                payload['next_worker']     = snapshot.get('next_worker', '')

            # Enrich parallel agent events (Session 9)
            if node_name in ('tech_analysis_agent', 'billing_analysis_agent', 'fraud_analysis_agent'):
                payload['parallel']    = True
                payload['agent_name']  = node_name
                notes = snapshot.get('internal_notes', [])
                # Find the most recent finding from this agent
                agent_key = node_name.replace('_agent', '')
                agent_findings = [
                    n for n in notes
                    if isinstance(n, dict) and n.get('agent') == agent_key
                ]
                payload['finding'] = agent_findings[-1] if agent_findings else {}

            if node_name == 'synthesizer':
                payload['synthesis_complete'] = True
                payload['findings_merged']    = len([
                    n for n in snapshot.get('internal_notes', [])
                    if isinstance(n, dict) and
                    n.get('agent') in (
                        'tech_analysis', 'billing_analysis', 'fraud_analysis'
                    )
                ])

            # Enrich github_tool_node events (Session 10)
            if node_name == 'github_tool_node':
                payload['github_node']    = True
                payload['github_url']     = snapshot.get('github_issue_url', '')
                payload['github_draft']   = snapshot.get('github_draft', {})
                payload['github_created'] = bool(snapshot.get('github_issue_url', ''))

            yield f"data: {json.dumps(payload)}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/verify")
def verify():
    result = run_session_verification()
    return result


@app.get("/api/threads")
def list_threads():
    """
    Returns all active thread_ids from the SQLite checkpointer.
    Used by the UI thread selector dropdown.
    """
    threads = get_active_threads()
    return {"threads": threads, "count": len(threads)}


@app.get("/api/history/{thread_id}")
def get_history(thread_id: str):
    """
    Returns the full checkpoint history for a thread_id.
    Each entry: step, node, category, iteration,
    message_count, final_response, is_end, checkpoint_id.
    """
    history = get_conversation_history(thread_id)
    return {
        "thread_id": thread_id,
        "count":     len(history),
        "history":   history,
    }


@app.get("/api/pending-approvals")
def list_pending_approvals():
    """
    Returns all threads currently suspended at github_tool_node.
    Polled by the UI approval queue every 30 seconds.
    """
    pending = get_pending_approvals()
    return {
        "count":   len(pending),
        "pending": pending,
    }


@app.post("/api/approve/{thread_id}")
def approve_thread(thread_id: str):
    """
    Resumes a suspended thread with approval.
    Executes github_tool_node with original draft.
    """
    result = approve_github_issue(thread_id)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result


@app.post("/api/deny/{thread_id}")
def deny_thread(thread_id: str):
    """
    Resumes a suspended thread with denial.
    Skips the write. Records denial in internal_notes.
    """
    result = deny_github_issue(thread_id)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result


class EditApproveRequest(BaseModel):
    edited_title:  str
    edited_body:   str
    edited_labels: list = []


@app.post("/api/edit-approve/{thread_id}")
def edit_approve_thread(thread_id: str, req: EditApproveRequest):
    """
    Updates github_draft with human edits then resumes with approval.
    Calls graph.update_state() then Command(resume={'approved': True}).
    """
    result = edit_and_approve_github_issue(
        thread_id=thread_id,
        edited_title=req.edited_title,
        edited_body=req.edited_body,
        edited_labels=req.edited_labels,
    )
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result


@app.get("/api/forensics/{thread_id}")
def get_forensics(thread_id: str):
    """
    Returns the full forensics report for a thread.
    Includes: timeline, flags, human_interventions, recommendation.
    Used by the Forensics Timeline Panel in the UI.
    """
    report = state_forensics(thread_id)
    return report


class TimeTravelRequest(BaseModel):
    thread_id:   str
    target_step: int


@app.post("/api/time-travel")
def run_time_travel(req: TimeTravelRequest):
    """
    Re-invokes the graph from a specific checkpoint step.
    Creates a new branch. Original branch preserved.
    Returns new final_response.
    """
    result = time_travel(req.thread_id, req.target_step)
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result


class CorrectionRequest(BaseModel):
    thread_id:   str
    field:       str
    new_value:   str
    target_step: int = None


@app.post("/api/correct")
def apply_state_correction(req: CorrectionRequest):
    """
    Injects a correction into the checkpoint ledger via update_state().
    If target_step provided: also re-invokes from that step.
    """
    import json as _json

    try:
        parsed_value = _json.loads(req.new_value)
    except (ValueError, TypeError):
        parsed_value = req.new_value

    result = apply_correction(
        thread_id   = req.thread_id,
        field       = req.field,
        new_value   = parsed_value,
        target_step = req.target_step,
    )

    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error'))
    return result


@app.get("/health")
def health():
    return {
        "status":              "ok",
        "session":             12,
        "tools":               len(TOOLS),
        "max_iterations":      MAX_ITERATIONS,
        "max_delegations":     MAX_DELEGATIONS,
        "summary_threshold":   SUMMARY_THRESHOLD,
        "injection_patterns":  len(INJECTION_PATTERNS),
        "parallel_agents":     3,
        "write_access":        True,
        "interrupt_before":    ["github_tool_node"],
        "hitl_active":         True,
        "time_travel":         True,
        "forensics":           True,
        "architecture":        "production_complete",
        "persistence":         "sqlite",
        "sessions_complete":   12,
    }


# Serve frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
