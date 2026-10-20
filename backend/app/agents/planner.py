from typing import Optional, Dict, Any
from backend.app.schemas.plan import ResearchPlan
from backend.app.services.llm import BaseLLMService, GroqLLMService
from backend.app.logging import logger

class PlannerAgent:
    def __init__(self, llm_service: Optional[BaseLLMService] = None):
        self.llm_service = llm_service or GroqLLMService()
        self.run_traces: list[Dict[str, Any]] = []

    def create_plan(self, topic: str, context: Optional[str] = None) -> ResearchPlan:
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be blank")
        if len(topic.strip()) > 500:
            raise ValueError("Topic is excessively long")

        prompt = f"Create a comprehensive research plan for the topic: '{topic}'."
        if context:
            prompt += f"\nAdditional context: {context}"
        prompt += "\nProvide 3 to 4 distinct sub-questions, objectives, priority source types, and a report outline."

        trace_entry = {
            "agent": "PlannerAgent",
            "input": {"topic": topic, "context": context},
            "output": None,
            "success": False,
            "error": None
        }

        try:
            plan: ResearchPlan = self.llm_service.invoke(
                prompt=prompt,
                agent="PlannerAgent",
                purpose="create_research_plan",
                temperature=0.3,
                response_model=ResearchPlan
            )
            trace_entry["output"] = plan.model_dump()
            trace_entry["success"] = True
            self.run_traces.append(trace_entry)
            logger.info(f"PlannerAgent successfully created plan for topic: {topic}")
            return plan
        except Exception as e:
            trace_entry["error"] = str(e)
            self.run_traces.append(trace_entry)
            logger.error(f"PlannerAgent failed to create plan: {e}")
            raise e
