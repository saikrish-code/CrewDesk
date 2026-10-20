import pytest
from backend.app.graph.workflow import CrewDeskWorkflow
from backend.app.agents.planner import PlannerAgent
from backend.app.agents.researcher import ResearcherAgent
from backend.app.agents.writer import WriterAgent
from backend.app.agents.critic import CriticAgent
from backend.app.services.llm import FakeLLMService
from backend.app.services.tavily import FakeTavilySearchService
from backend.app.schemas.plan import ResearchPlan, SubQuestion
from backend.app.schemas.report import ResearchReport, SectionMetadata
from backend.app.schemas.critic import CriticReview, QualityScores

def test_workflow_approval_first_try():
    plan = ResearchPlan(
        original_topic="AI Testing",
        sub_questions=[
            SubQuestion(question="Q1?", objective="O1", priority_sources=["S1"]),
            SubQuestion(question="Q2?", objective="O2", priority_sources=["S2"]),
            SubQuestion(question="Q3?", objective="O3", priority_sources=["S3"])
        ],
        report_outline=["Intro", "Body", "Conclusion"]
    )
    report = ResearchReport(markdown_content="# Report", sections_metadata=[], cited_source_ids=[], revision_number=1)
    review = CriticReview(
        approved=True,
        quality_scores=QualityScores(coverage=9.0, factual_consistency=9.0, citation_validity=9.0, clarity=9.0, completeness=9.0),
        issues=[],
        summary="Approved"
    )

    planner_llm = FakeLLMService(canned_responses=[plan])
    researcher_llm = FakeLLMService(canned_responses=[])
    writer_llm = FakeLLMService(canned_responses=[report])
    critic_llm = FakeLLMService(canned_responses=[review])

    planner = PlannerAgent(llm_service=planner_llm)
    researcher = ResearcherAgent(tavily_service=FakeTavilySearchService(canned_results=[]), llm_service=researcher_llm)
    writer = WriterAgent(llm_service=writer_llm)
    critic = CriticAgent(llm_service=critic_llm)

    workflow = CrewDeskWorkflow(planner=planner, researcher=researcher, writer=writer, critic=critic, llm_budget=20)
    state = workflow.run("AI Testing")

    assert state["status"] == "completed"
    assert state["revision_count"] == 1
    assert len(state["transition_history"]) >= 4

def test_workflow_revision_loop():
    plan = ResearchPlan(
        original_topic="AI Testing",
        sub_questions=[
            SubQuestion(question="Q1?", objective="O1", priority_sources=["S1"]),
            SubQuestion(question="Q2?", objective="O2", priority_sources=["S2"]),
            SubQuestion(question="Q3?", objective="O3", priority_sources=["S3"])
        ],
        report_outline=["Intro", "Body", "Conclusion"]
    )
    report1 = ResearchReport(markdown_content="# Draft 1", sections_metadata=[], cited_source_ids=[], revision_number=1)
    report2 = ResearchReport(markdown_content="# Draft 2", sections_metadata=[], cited_source_ids=[], revision_number=2)
    review_reject = CriticReview(
        approved=False,
        quality_scores=QualityScores(coverage=5.0, factual_consistency=5.0, citation_validity=5.0, clarity=5.0, completeness=5.0),
        issues=[],
        summary="Needs revision"
    )
    review_approve = CriticReview(
        approved=True,
        quality_scores=QualityScores(coverage=9.0, factual_consistency=9.0, citation_validity=9.0, clarity=9.0, completeness=9.0),
        issues=[],
        summary="Approved now"
    )

    planner_llm = FakeLLMService(canned_responses=[plan])
    researcher_llm = FakeLLMService(canned_responses=[])
    writer_llm = FakeLLMService(canned_responses=[report1, report2])
    critic_llm = FakeLLMService(canned_responses=[review_reject, review_approve])

    planner = PlannerAgent(llm_service=planner_llm)
    researcher = ResearcherAgent(tavily_service=FakeTavilySearchService(canned_results=[]), llm_service=researcher_llm)
    writer = WriterAgent(llm_service=writer_llm)
    critic = CriticAgent(llm_service=critic_llm)

    workflow = CrewDeskWorkflow(planner=planner, researcher=researcher, writer=writer, critic=critic, llm_budget=30)
    state = workflow.run("AI Testing")

    assert state["status"] == "completed"
    assert state["revision_count"] == 2
    transitions = [t["destination"] for t in state["transition_history"]]
    assert "Writer" in transitions
    assert transitions.count("Writer") == 2

def test_workflow_budget_exceeded():
    planner = PlannerAgent(llm_service=FakeLLMService(should_fail=True))
    workflow = CrewDeskWorkflow(planner=planner, llm_budget=0)
    state = workflow.run("AI Testing")
    assert state["status"] == "budget_exhausted"
