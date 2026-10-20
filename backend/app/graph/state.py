from typing import TypedDict, List, Dict, Any, Optional
from backend.app.schemas.plan import ResearchPlan
from backend.app.schemas.research import ResearchReportData
from backend.app.schemas.report import ResearchReport
from backend.app.schemas.critic import CriticReview

class CrewDeskState(TypedDict):
    run_id: str
    original_topic: str
    context: Optional[str]
    research_plan: Optional[Dict[str, Any]]
    research_findings: Optional[Dict[str, Any]]
    current_draft: Optional[Dict[str, Any]]
    critic_review: Optional[Dict[str, Any]]
    revision_count: int
    llm_budget: int
    llm_usage: int
    status: str
    error: Optional[str]
    timestamps: Dict[str, str]
    transition_history: List[Dict[str, Any]]
