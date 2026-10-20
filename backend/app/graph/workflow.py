import datetime
import uuid
from typing import Dict, Any, Optional
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
from backend.app.services.budget import BudgetManager
from backend.app.services.llm import BudgetExceededError
from backend.app.services.tavily import SearchBudgetExceededError
from backend.app.logging import logger

class CrewDeskWorkflow:
    def __init__(
        self,
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        writer: Optional[WriterAgent] = None,
        critic: Optional[CriticAgent] = None,
        llm_budget: int = 25,
        search_budget: int = 15
    ):
        self.budget_manager = BudgetManager(max_llm_calls=llm_budget, max_search_calls=search_budget)
        self.planner = planner or PlannerAgent()
        self.planner.llm_service.budget_manager = self.budget_manager
        
        self.researcher = researcher or ResearcherAgent()
        self.researcher.llm_service.budget_manager = self.budget_manager
        self.researcher.tavily_service.budget_manager = self.budget_manager

        self.writer = writer or WriterAgent()
        self.writer.llm_service.budget_manager = self.budget_manager

        self.critic = critic or CriticAgent()
        self.critic.llm_service.budget_manager = self.budget_manager

        self.default_budget = llm_budget
        self.graph = self._build_graph()

    def _record_transition(self, state: CrewDeskState, source: str, destination: str, reason: str) -> None:
        stats = self.budget_manager.get_stats()
        state["llm_usage"] = stats["llm_usage"]
        transition = {
            "source": source,
            "destination": destination,
            "reason": reason,
            "revision_count": state["revision_count"],
            "budget_status": f"{stats['llm_usage']}/{stats['max_llm_calls']}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        state["transition_history"].append(transition)
        logger.info(f"Transition: {source} -> {destination} | Reason: {reason} | Rev: {state['revision_count']} | Budget: {stats['llm_usage']}/{stats['max_llm_calls']}")

    def planning_node(self, state: CrewDeskState) -> CrewDeskState:
        try:
            plan = self.planner.create_plan(state["original_topic"], state.get("context"))
            state["research_plan"] = plan.model_dump()
            self._record_transition(state, "START", "Planner", "Plan created successfully")
        except (BudgetExceededError, SearchBudgetExceededError) as e:
            state["status"] = "budget_exhausted"
            state["error"] = str(e)
            self._record_transition(state, "START", "Finalizer", "Budget exhausted during planning")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "START", "Error", str(e))
        return state

    def research_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["status"] in ["failed", "budget_exhausted"]:
            return state

        try:
            plan_obj = ResearchPlan(**state["research_plan"])
            research_data = self.researcher.conduct_research(plan_obj)
            state["research_findings"] = research_data.model_dump()
            self._record_transition(state, "Planner", "Researcher", "Research gathered successfully")
        except (BudgetExceededError, SearchBudgetExceededError) as e:
            state["status"] = "budget_exhausted"
            state["error"] = str(e)
            self._record_transition(state, "Planner", "Finalizer", "Budget exhausted during research")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "Planner", "Error", str(e))
        return state

    def writing_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["status"] in ["failed", "budget_exhausted"]:
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
            source_node = "Critic" if state["revision_count"] > 1 else "Researcher"
            self._record_transition(state, source_node, "Writer", f"Draft written for revision {state['revision_count']}")
        except (BudgetExceededError, SearchBudgetExceededError) as e:
            state["status"] = "budget_exhausted"
            state["error"] = str(e)
            if state.get("current_draft"):
                state["current_draft"]["markdown_content"] += "\n\n> [INCOMPLETE: Budget Exhausted]"
            self._record_transition(state, "Writer", "Finalizer", "Budget exhausted during writing")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "Writer", "Error", str(e))
        return state

    def criticism_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["status"] in ["failed", "budget_exhausted"]:
            return state

        try:
            plan_obj = ResearchPlan(**state["research_plan"])
            research_data = ResearchReportData(**state["research_findings"])
            report_obj = ResearchReport(**state["current_draft"])
            review = self.critic.evaluate_report(state["original_topic"], plan_obj, research_data, report_obj, revision_number=state["revision_count"])
            state["critic_review"] = review.model_dump()
            self._record_transition(state, "Writer", "Critic", f"Review evaluated: approved={review.approved}")
        except (BudgetExceededError, SearchBudgetExceededError) as e:
            state["status"] = "budget_exhausted"
            state["error"] = str(e)
            self._record_transition(state, "Critic", "Finalizer", "Budget exhausted during criticism")
        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            self._record_transition(state, "Critic", "Error", str(e))
        return state

    def finalization_node(self, state: CrewDeskState) -> CrewDeskState:
        if state["status"] not in ["failed", "budget_exhausted"]:
            state["status"] = "completed"
        self._record_transition(state, "Critic", "Finalizer", f"Run completed with status: {state['status']}")
        return state

    def _should_continue(self, state: CrewDeskState) -> str:
        if state["status"] in ["failed", "budget_exhausted"]:
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
        stats = self.budget_manager.get_stats()
        initial_state: CrewDeskState = {
            "run_id": f"run_{uuid.uuid4().hex[:8]}",
            "original_topic": topic,
            "context": context,
            "research_plan": None,
            "research_findings": None,
            "current_draft": None,
            "critic_review": None,
            "revision_count": 0,
            "llm_budget": stats["max_llm_calls"],
            "llm_usage": stats["llm_usage"],
            "status": "running",
            "error": None,
            "timestamps": {"start": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            "transition_history": []
        }
        final_state = self.graph.invoke(initial_state)
        final_state["timestamps"]["end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        final_state["llm_usage"] = self.budget_manager.get_stats()["llm_usage"]
        return final_state
