from typing import List, Optional, Dict, Any
from backend.app.schemas.plan import ResearchPlan
from backend.app.schemas.research import ResearchReportData
from backend.app.schemas.report import ResearchReport, SectionMetadata
from backend.app.services.llm import BaseLLMService, GroqLLMService
from backend.app.logging import logger

class WriterAgent:
    def __init__(self, llm_service: Optional[BaseLLMService] = None):
        self.llm_service = llm_service or GroqLLMService()
        self.run_traces: List[Dict[str, Any]] = []

    def write_report(
        self,
        plan: ResearchPlan,
        research_data: ResearchReportData,
        critic_feedback: Optional[str] = None,
        revision_number: int = 1
    ) -> ResearchReport:
        sources_map = {}
        source_id_list = []
        source_context_lines = []
        idx = 1
        for sq_res in research_data.results:
            for s in sq_res.sources:
                if s.url not in sources_map:
                    sid = f"S{idx}"
                    sources_map[s.url] = sid
                    source_id_list.append(sid)
                    source_context_lines.append(f"[{sid}] Title: {s.title} | URL: {s.url} | Date: {s.published_date} | Snippet: {s.snippet}")
                    idx += 1

        findings_context = []
        for sq_res in research_data.results:
            findings_context.append(f"Sub-Question: {sq_res.sub_question}")
            if not sq_res.findings:
                findings_context.append("- Findings: Insufficient evidence available.")
            for f in sq_res.findings:
                mapped_sids = [sources_map.get(u, "S1") for u in f.supporting_urls if u in sources_map]
                findings_context.append(f"- Finding: {f.finding_text} (Supported by: {', '.join(mapped_sids) or 'None'}, Verified: {f.verified}, Limitations: {f.limitations})")

        prompt = (
            f"Write a comprehensive, professional Markdown research report on the topic: '{plan.original_topic}'.\n\n"
            f"Report Outline:\n" + "\n".join([f"- {item}" for item in plan.report_outline]) + "\n\n"
            f"Research Findings & Evidence:\n" + "\n".join(findings_context) + "\n\n"
            f"Available Sources Registry:\n" + "\n".join(source_context_lines) + "\n\n"
        )

        if critic_feedback:
            prompt += f"\nRevision Feedback from Critic (Revision #{revision_number}):\n{critic_feedback}\nAddress this feedback directly without unrelated rewrites.\n"

        prompt += (
            f"\nRequirements:\n"
            f"1. Include Title, Executive Summary, sections for each sub-question, limitations, conclusion, and References.\n"
            f"2. Use inline citations like [S1], [S2] matching the available sources registry.\n"
            f"3. Do not invent citations.\n"
            f"4. Return structured output including markdown_content, sections_metadata, and cited_source_ids."
        )

        trace_entry = {
            "agent": "WriterAgent",
            "revision_number": revision_number,
            "critic_feedback": critic_feedback,
            "input_topic": plan.original_topic,
            "output": None,
            "success": False,
            "error": None
        }

        try:
            report: ResearchReport = self.llm_service.invoke(
                prompt=prompt,
                agent="WriterAgent",
                purpose=f"write_report_rev_{revision_number}",
                temperature=0.3,
                response_model=ResearchReport
            )

            for cid in report.cited_source_ids:
                if cid not in source_id_list and source_id_list:
                    raise ValueError(f"Invalid cited source ID: {cid}")

            report.revision_number = revision_number
            report.critic_feedback_addressed = critic_feedback
            trace_entry["output"] = report.model_dump()
            trace_entry["success"] = True
            self.run_traces.append(trace_entry)
            logger.info(f"WriterAgent successfully generated report revision {revision_number}")
            return report

        except Exception as e:
            error_msg = str(e)
            trace_entry["error"] = error_msg
            self.run_traces.append(trace_entry)
            logger.error(f"WriterAgent failed to write report: {error_msg}")
            raise e
