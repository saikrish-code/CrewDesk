from typing import Optional, Dict, Any, List
from backend.app.schemas.plan import ResearchPlan
from backend.app.schemas.research import ResearchReportData
from backend.app.schemas.report import ResearchReport
from backend.app.schemas.critic import CriticReview, QualityScores, CriticIssue
from backend.app.services.llm import BaseLLMService, GroqLLMService
from backend.app.logging import logger

class CriticAgent:
    def __init__(self, llm_service: Optional[BaseLLMService] = None, max_revisions: int = 2):
        self.llm_service = llm_service or GroqLLMService()
        self.max_revisions = max_revisions
        self.run_traces: List[Dict[str, Any]] = []

    def evaluate_report(
        self,
        topic: str,
        plan: ResearchPlan,
        research_data: ResearchReportData,
        report: ResearchReport,
        revision_number: int = 1
    ) -> CriticReview:
        sources_list = []
        for sq_res in research_data.results:
            for s in sq_res.sources:
                sources_list.append(f"URL: {s.url} | Title: {s.title}")

        prompt = (
            f"Review the following research report draft for topic: '{topic}' (Revision #{revision_number}).\n\n"
            f"Planned Sub-Questions:\n" + "\n".join([f"- {sq.question}" for sq in plan.sub_questions]) + "\n\n"
            f"Available Sources:\n" + "\n".join(sources_list) + "\n\n"
            f"Report Content:\n{report.markdown_content}\n\n"
            f"Cited Source IDs in Report: {report.cited_source_ids}\n\n"
            f"Evaluate coverage, factual consistency, citation validity, unsupported claims, clarity, and limitations.\n"
            f"Provide numeric quality scores (0-10), issues with severity (low, medium, high, critical), explanation, and corrective action, "
            f"and whether the report is approved."
        )

        trace_entry = {
            "agent": "CriticAgent",
            "revision_number": revision_number,
            "input_topic": topic,
            "output": None,
            "success": False,
            "error": None
        }

        try:
            review: CriticReview = self.llm_service.invoke(
                prompt=prompt,
                agent="CriticAgent",
                purpose=f"review_report_rev_{revision_number}",
                temperature=0.2,
                response_model=CriticReview
            )

            has_critical_issues = any(i.severity.lower() in ["high", "critical"] for i in review.issues)
            if has_critical_issues:
                review.approved = False

            if revision_number > self.max_revisions:
                review.approved = True
                review.summary += " (Forced approval due to revision limit reached.)"

            trace_entry["output"] = review.model_dump()
            trace_entry["success"] = True
            self.run_traces.append(trace_entry)
            logger.info(f"CriticAgent evaluated revision {revision_number}: approved={review.approved}")
            return review

        except Exception as e:
            error_msg = str(e)
            trace_entry["error"] = error_msg
            self.run_traces.append(trace_entry)
            logger.error(f"CriticAgent evaluation failed: {error_msg}")
            raise e
