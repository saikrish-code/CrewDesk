import os
import json
import pytest
from backend.app.logging.audit import AuditLogger, print_run_trace

def test_audit_logger_redaction():
    logger = AuditLogger(run_id="test_run_redact")
    secret_text = "My key is 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 and token is tvly-12345."
    redacted = logger.redact(secret_text)
    assert "gsk_" not in redacted
    assert "tvly-" not in redacted
    assert "[REDACTED]" in redacted

def test_audit_logger_jsonl_and_trace():
    run_id = "test_run_audit_1"
    logger = AuditLogger(run_id=run_id, logs_dir="logs")
    
    event = logger.log_event(
        agent_name="PlannerAgent",
        input_data={"topic": "Test Topic", "key": "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"},
        output_data={"status": "ok"}
    )
    
    assert event["run_id"] == run_id
    assert "gsk_" not in event["input"]["key"]
    assert event["input"]["key"] == "[REDACTED]"

    logger.finalize_run(
        status="completed",
        start_time="2026-10-20T12:00:00Z",
        end_time="2026-10-20T12:00:05Z",
        total_duration_seconds=5.0,
        agent_durations={"PlannerAgent": 1.0},
        llm_usage=2,
        search_usage=1,
        source_count=3,
        revision_count=1,
        transition_history=[{"source": "START", "destination": "Planner", "reason": "ok", "revision_count": 0, "budget_status": "2/25", "timestamp": "2026-10-20T12:00:01Z"}],
        plan={"original_topic": "Test"},
        draft={"markdown_content": "# Report"},
        review={"approved": True}
    )

    jsonl_path = os.path.join("logs", f"{run_id}.jsonl")
    trace_path = os.path.join("logs", f"{run_id}_trace.json")
    summary_path = os.path.join("logs", f"{run_id}_summary.json")

    assert os.path.exists(jsonl_path)
    assert os.path.exists(trace_path)
    assert os.path.exists(summary_path)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 1
        parsed_line = json.loads(lines[0])
        assert parsed_line["agent_name"] == "PlannerAgent"

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        assert "summary" in trace_data
        assert trace_data["summary"]["status"] == "completed"
        assert "transition_history" in trace_data

    print_run_trace(run_id)
