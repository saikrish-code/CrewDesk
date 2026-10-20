import os
import json
import uuid
import re
import datetime
from typing import Dict, Any, List, Optional
from backend.app.config import settings

class AuditLogger:
    def __init__(self, run_id: str, logs_dir: str = "logs"):
        self.run_id = run_id
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.logs_dir, f"{run_id}.jsonl")
        self.trace_path = os.path.join(self.logs_dir, f"{run_id}_trace.json")
        self.summary_path = os.path.join(self.logs_dir, f"{run_id}_summary.json")
        self.secret_patterns = [
            re.compile(r"gsk_[a-zA-Z0-9]{20,}", re.IGNORECASE),
            re.compile(r"tvly-[a-zA-Z0-9-_]+", re.IGNORECASE),
            re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
            re.compile(r"api[_-]key\s*[:=]\s*['\"]?[a-zA-Z0-9-_]{10,}", re.IGNORECASE)
        ]

    def redact(self, data: Any) -> Any:
        if isinstance(data, str):
            val = data
            for pattern in self.secret_patterns:
                val = pattern.sub("[REDACTED]", val)
            return val
        elif isinstance(data, dict):
            return {k: self.redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.redact(item) for item in data]
        return data

    def log_event(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event = {
            "run_id": self.run_id,
            "event_id": event_id,
            "agent_name": agent_name,
            "input": self.redact(input_data),
            "output": self.redact(output_data),
            "error": self.redact(error),
            "metadata": self.redact(metadata or {}),
            "timestamp": timestamp
        }

        line = json.dumps(event, default=str) + "\n"
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)

        return event

    def finalize_run(
        self,
        status: str,
        start_time: str,
        end_time: str,
        total_duration_seconds: float,
        agent_durations: Dict[str, float],
        llm_usage: int,
        search_usage: int,
        source_count: int,
        revision_count: int,
        transition_history: List[Dict[str, Any]],
        plan: Optional[Dict[str, Any]] = None,
        draft: Optional[Dict[str, Any]] = None,
        review: Optional[Dict[str, Any]] = None
    ) -> None:
        summary = {
            "run_id": self.run_id,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "total_duration_seconds": total_duration_seconds,
            "agent_durations": agent_durations,
            "llm_call_count": llm_usage,
            "search_request_count": search_usage,
            "source_count": source_count,
            "revision_count": revision_count
        }

        trace = {
            "summary": summary,
            "transition_history": self.redact(transition_history),
            "research_plan": self.redact(plan),
            "final_draft": self.redact(draft),
            "final_review": self.redact(review)
        }

        self._atomic_write(self.summary_path, summary)
        self._atomic_write(self.trace_path, trace)

    def _atomic_write(self, filepath: str, data: Any) -> None:
        temp_path = filepath + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_path, filepath)

def print_run_trace(run_id: str, logs_dir: str = "logs") -> None:
    trace_path = os.path.join(logs_dir, f"{run_id}_trace.json")
    if not os.path.exists(trace_path):
        print(f"Trace not found for run ID: {run_id}")
        return

    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    summary = trace.get("summary", {})
    print(f"=== CrewDesk Run Trace: {run_id} ===")
    print(f"Status: {summary.get('status')}")
    print(f"Duration: {summary.get('total_duration_seconds')}s")
    print(f"LLM Calls: {summary.get('llm_call_count')} | Search Requests: {summary.get('search_request_count')}")
    print(f"Sources: {summary.get('source_count')} | Revisions: {summary.get('revision_count')}")
    print("\n--- Transition History ---")
    for t in trace.get("transition_history", []):
        print(f"[{t.get('timestamp')}] {t.get('source')} -> {t.get('destination')} | Reason: {t.get('reason')} | Budget: {t.get('budget_status')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_run_trace(sys.argv[1])
    else:
        print("Usage: python -m backend.app.logging.audit <run_id>")
