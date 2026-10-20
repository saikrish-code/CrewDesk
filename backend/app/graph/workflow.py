import datetime
import uuid
from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from backend.app.graph.state import CrewDeskState
from backend.app.agents.planner import PlannerAgent
from backend.app.agents.researcher import ResearcherAgent
from backend.app.agents.writer import WriterAgent
from backend.app.agents.critic import CriticAgent
from backend.app.schemas.plan import ResearchPlan
from backend.app.schemas.research import ResearchReportData
from backend.app.schemas.report import ResearchReport
from backend.app.schemas.critic import CriticReview
from backend.app.logging import logger

class CrewDeskWorkflow:
    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        writer: Optional[WriterAgent] = None,
        critic: Optional[CriticAgent] = None,
        llm_budget: int = 50
    ):
        self.planner = planner or PlannerAgent()
        self.researcher = researcher or ResearcherAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self.default_budget = llm_budget
        self.graph = self._build_graph()

    def _record_transition(self, state: CrewDeskState, source: str, destination: str, reason: str) -> None:
        transition = {
            "source": source,
            "destination": destination,
            "reason": reason,
            "revision_count": state["revision_count"],
            "budget_status": f"{state['llm_usage']}/{state['llm_budget']}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        state["transition_history"].append(transition)
        logger.info(f"Transition: {source} -> {destination} | Reason: {reason} | Rev: {state['revision_count']} | Budget: {state['llm_usage']}/{state['llm_budget']}")

    def planning_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["llm_usage"] >= state["llm_budget"]:
            state["status"] = "budget_exceeded"
            state["error"] = "LLM call budget exhausted during planning"
            return state

        try:
            plan = self.planner.create_plan(state["original_topic"], state.get("context"))
            state["research_plan"] = plan.model_dump()
            state["llm_usage"] += 1
            self._record_transition(state, "START", "Planner", "Plan created successfully")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "START", "Error", str(e))
        return state

    def research_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["llm_usage"] >= state["llm_budget"] or state["status"] == "failed":
            state["status"] = "budget_exceeded"
            return state

        try:
            plan_obj = ResearchPlan(**state["research_plan"])
            research_data = self.researcher.conduct_research(plan_obj)
            state["research_findings"] = research_data.model_dump()
            state["llm_usage"] += len(plan_obj.sub_questions)
            self._record_transition(state, "Planner", "Researcher", "Research gathered successfully")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "Planner", "Error", str(e))
        return state

    def writing_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["llm_usage"] >= state["llm_budget"] or state["status"] == "failed":
            state["status"] = "budget_exceeded"
            return state

        try:
            plan_obj = ResearchPlan(**state["research_plan"])
            research_data = ResearchReportData(**state["research_findings"])
            critic_feedback = None
            if state.get("critic_review"):
                issues_text = "; ".join([i["corrective_action"] for i in state["critic_review"].get("issues", [])])
                critic_feedback = f"{state['critic_review'].get('summary', '')} Action items: {issues_text}"

            state["revision_count"] += 1
            report = self.writer.write_report(plan_obj, research_data, critic_feedback=critic_feedback, revision_number=state["revision_count"])
            state["current_draft"] = report.model_dump()
            state["llm_usage"] += 1
            source_node = "Critic" if state["revision_count"] > 1 else "Researcher"
            self._record_transition(state, source_node, "Writer", f"Draft written for revision {state['revision_count']}")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "Writer", "Error", str(e))
        return state

    def criticism_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["llm_usage"] >= state["llm_budget"] or state["status"] == "failed":
            state["status"] = "budget_exceeded"
            return state

        try:
            plan_obj = ResearchPlan(**state["research_plan"])
            research_data = ResearchReportData(**state["research_findings"])
            report_obj = ResearchReport(**state["current_draft"])
            review = self.critic.evaluate_report(state["original_topic"], plan_obj, research_data, report_obj, revision_number=state["revision_count"])
            state["critic_review"] = review.model_dump()
            state["llm_usage"] += 1
            self._record_transition(state, "Writer", "Critic", f"Review evaluated: approved={review.approved}")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "Critic", "Error", str(e))
        return state

    def finalization_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["status"] != "failed" and state["status"] != "budget_exceeded":
            state["status"] = "completed"
        self._record_transition(state, "Critic", "Finalizer", f"Run completed with status: {state['status']}")
        return state

    def _should_continue(self, state: CrewDeskState) -> str:
        if state["status"] == "failed" or state["status"] == "budget_exceeded":
            return "finalize"
        review = state.get("critic_review")
        if not review:
            return "finalize"
        if review.get("approved", False):
            return "finalize"
        if state["revision_count"] >= 2:
            return "finalize"
        return "rewrite"

    def _build_graph(self) -> Any:
        workflow = StateGraph(CrewDeskState)
        workflow.add_node("planning", self.planning_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("writing", self.writing_node)
        workflow.add_node("criticism", self.criticism_node)
        workflow.add_node("finalization", self.finalization_node)

        workflow.add_edge(START, "planning")
        workflow.add_edge("planning", "research")
        workflow.add_edge("research", "writing")
        workflow.add_edge("writing", "criticism")
        workflow.add_conditional_edges(
            "criticism",
            self._should_continue,
            {
                "rewrite": "writing",
                "finalize": "finalization"
            }
        )
        workflow.add_edge("finalization", END)
        return workflow.compile()

    def run(self, topic: str, context: Optional[str] = None) -> CrewDeskState:
        initial_state: CrewDeskState = {
            "run_id": f"run_{uuid.uuid4().hex[:8]}",
            "original_topic": topic,
            "context": context,
            "research_plan": None,
            "research_findings": None,
            "current_draft": None,
            "critic_review": None,
            "revision_count": 0,
            "llm_budget": self.default_budget,
            "llm_usage": 0,
            "status": "running",
            "error": None,
            "timestamps": {"start": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            "transition_history": []
        }
        final_state = self.graph.invoke(initial_state)
        final_state["timestamps"]["end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return final_state
