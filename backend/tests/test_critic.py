import pytest
from backend.app.schemas.plan import ResearchPlan, SubQuestion
from backend.app.schemas.research import ResearchReportData, SubQuestionResearch, ResearchSource
from backend.app.schemas.report import ResearchReport
from backend.app.schemas.critic import CriticReview, QualityScores, CriticIssue
from backend.app.agents.critic import CriticAgent
from backend.app.services.llm import FakeLLMService

@pytest.fixture
def critic_setup():
    plan = ResearchPlan(
        original_topic="Generative AI in Software Engineering",
        sub_questions=[
            SubQuestion(question="How does GenAI affect productivity?", objective="Measure speed", priority_sources=["benchmarks"]),
            SubQuestion(question="Does GenAI cause security vulnerabilities?", objective="Analyze quality", priority_sources=["audits"]),
            SubQuestion(question="What are adoption trends?", objective="Analyze adoption", priority_sources=["surveys"])
        ],
        report_outline=["Intro", "Productivity", "Security", "Adoption", "Conclusion"]
    )
    research_data = ResearchReportData(
        original_topic="Generative AI in Software Engineering",
        results=[
            SubQuestionResearch(sub_question="How does GenAI affect productivity?", sources=[], findings=[]),
            SubQuestionResearch(sub_question="Does GenAI cause security vulnerabilities?", sources=[], findings=[]),
            SubQuestionResearch(sub_question="What are adoption trends?", sources=[], findings=[])
        ]
    )
    report = ResearchReport(
        markdown_content="# Report content",
        sections_metadata=[],
        cited_source_ids=[],
        revision_number=1
    )
    return plan, research_data, report

def test_critic_approval(critic_setup):
    plan, research_data, report = critic_setup
    mock_review = CriticReview(
        approved=True,
        quality_scores=QualityScores(coverage=9.0, factual_consistency=9.5, citation_validity=9.0, clarity=9.0, completeness=9.0),
        issues=[],
        summary="Excellent report meeting all criteria."
    )
    fake_llm = FakeLLMService(canned_responses=[mock_review])
    critic = CriticAgent(llm_service=fake_llm)
    review = critic.evaluate_report(plan.original_topic, plan, research_data, report, revision_number=1)
    assert review.approved is True
    assert len(review.issues) == 0

def test_critic_rejection_with_critical_issue(critic_setup):
    plan, research_data, report = critic_setup
    mock_review = CriticReview(
        approved=True, # LLM says true, but agent logic overrides if high/critical severity issue exists
        quality_scores=QualityScores(coverage=5.0, factual_consistency=4.0, citation_validity=5.0, clarity=6.0, completeness=5.0),
        issues=[
            CriticIssue(severity="critical", location="Section 2", explanation="Hallucinated fact", corrective_action="Remove unsupported statement.")
        ],
        summary="Contains critical hallucination."
    )
    fake_llm = FakeLLMService(canned_responses=[mock_review])
    critic = CriticAgent(llm_service=fake_llm)
    review = critic.evaluate_report(plan.original_topic, plan, research_data, report, revision_number=1)
    assert review.approved is False

def test_critic_revision_limit(critic_setup):
    plan, research_data, report = critic_setup
    mock_review = CriticReview(
        approved=False,
        quality_scores=QualityScores(coverage=6.0, factual_consistency=6.0, citation_validity=6.0, clarity=6.0, completeness=6.0),
        issues=[
            CriticIssue(severity="high", location="Section 1", explanation="Minor gap", corrective_action="Add context.")
        ],
        summary="Needs work."
    )
    fake_llm = FakeLLMService(canned_responses=[mock_review])
    critic = CriticAgent(llm_service=fake_llm, max_revisions=2)
    review = critic.evaluate_report(plan.original_topic, plan, research_data, report, revision_number=3)
    assert review.approved is True
    assert "Forced approval" in review.summary
