import asyncio
import datetime
import json
import os
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from backend.app.graph.workflow import CrewDeskWorkflow
from backend.app.logging.audit import AuditLogger
from backend.app.logging import logger

router = APIRouter(prefix="/api")

active_runs: Dict[str, Dict[str, Any]] = {}
run_queues: Dict[str, List[asyncio.Queue]] = {}

class RunRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    context: Optional[str] = None
    llm_budget: Optional[int] = 25
    search_budget: Optional[int] = 15

    @field_validator("topic")
    @classmethod
    def validate_topic_not_whitespace(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Topic cannot be blank or whitespace-only")
        if len(v.strip()) < 3:
            raise ValueError("Topic must be at least 3 characters")
        return v.strip()

class RunResponse(BaseModel):
    run_id: str
    status: str
    original_topic: str

async def broadcast_event(run_id: str, event_type: str, data: Dict[str, Any]):
    event_payload = {
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "run_id": run_id,
        "event_type": event_type,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data": data
    }
    if run_id in run_queues:
        for q in run_queues[run_id]:
            await q.put(event_payload)

async def execute_workflow_task(run_id: str, topic: str, context: Optional[str], llm_budget: int, search_budget: int):
    run_queues.setdefault(run_id, [])
    workflow = CrewDeskWorkflow(llm_budget=llm_budget, search_budget=search_budget)
    
    await broadcast_event(run_id, "run_started", {"topic": topic})
    
    loop = asyncio.get_running_loop()
    try:
        await broadcast_event(run_id, "agent_started", {"agent_name": "PlannerAgent"})
        state = await loop.run_in_executor(None, workflow.run, topic, context)
        active_runs[run_id] = state
        
        if state["status"] == "completed":
            await broadcast_event(run_id, "run_completed", {"summary": state})
        elif state["status"] == "budget_exhausted":
            await broadcast_event(run_id, "budget_warning", {"error": state.get("error")})
            await broadcast_event(run_id, "run_failed", {"error": "Budget exhausted"})
        else:
            await broadcast_event(run_id, "run_failed", {"error": state.get("error") or "Run failed"})
            
    except Exception as e:
        logger.error(f"Workflow execution failed for run {run_id}: {e}")
        active_runs[run_id] = {"status": "failed", "error": str(e)}
        await broadcast_event(run_id, "run_failed", {"error": str(e)})

@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(run_req: RunRequest, background_tasks: BackgroundTasks):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    active_runs[run_id] = {"run_id": run_id, "original_topic": run_req.topic, "status": "running"}
    
    background_tasks.add_task(
        execute_workflow_task,
        run_id,
        run_req.topic,
        run_req.context,
        run_req.llm_budget or 25,
        run_req.search_budget or 15
    )
    
    return RunResponse(run_id=run_id, status="running", original_topic=run_req.topic)

@router.get("/runs/{run_id}", response_model=Dict[str, Any])
async def get_run_status(run_id: str):
    if run_id in active_runs:
        return active_runs[run_id]
    
    summary_path = os.path.join("logs", f"{run_id}_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    raise HTTPException(status_code=404, detail="Run not found")

@router.get("/runs/{run_id}/report", response_model=Dict[str, Any])
async def get_run_report(run_id: str):
    state = active_runs.get(run_id)
    if state and state.get("current_draft"):
        return state["current_draft"]

    trace_path = os.path.join("logs", f"{run_id}_trace.json")
    if os.path.exists(trace_path):
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
            if trace.get("final_draft"):
                return trace["final_draft"]

    raise HTTPException(status_code=404, detail="Report not found")

@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request):
    if run_id not in active_runs and not os.path.exists(os.path.join("logs", f"{run_id}_summary.json")):
        raise HTTPException(status_code=404, detail="Run not found")

    queue = asyncio.Queue()
    run_queues.setdefault(run_id, []).append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"id: {event['event_id']}\nevent: {event['event_type']}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f": ping\n\n"
        finally:
            if run_id in run_queues and queue in run_queues[run_id]:
                run_queues[run_id].remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
