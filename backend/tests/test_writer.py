import pytest
from pydantic import ValidationError
from backend.app.schemas.plan import ResearchPlan, SubQuestion
from backend.app.schemas.research import ResearchReportData, SubQuestionResearch, ResearchSource, ResearchFinding
from backend.app.schemas.report import ResearchReport, SectionMetadata
from backend.app.agents.writer import WriterAgent
from backend.app.services.llm import FakeLLMService

@pytest.fixture
def sample_plan_and_data():
    plan = ResearchPlan(
        original_topic="Generative AI in Software Engineering",
        sub_questions=[
            SubQuestion(question="How does GenAI affect productivity?", objective="Measure speed", priority_sources=["benchmarks"]),
            SubQuestion(question="Does GenAI cause security vulnerabilities?", objective="Analyze quality", priority_sources=["audits"]),
            SubQuestion(question="What are adoption trends?", objective="Analyze adoption", priority_sources=["surveys"])
        ],
        report_outline=["Intro", "Productivity", "Security", "Adoption", "Conclusion"]
    )
    source = ResearchSource(
        url="https://example.com/ai-study",
        title="AI Study 2026",
        snippet="Productivity increased by 40%.",
        retrieval_timestamp="2026-01-01T00:00:00Z"
    )
    finding = ResearchFinding(
        sub_question="How does GenAI affect productivity?",
        finding_text="Productivity increased significantly.",
        supporting_urls=["https://example.com/ai-study"],
        source_claims=["40% faster"],
        verified=True
    )
    research_data = ResearchReportData(
        original_topic="Generative AI in Software Engineering",
        results=[
            SubQuestionResearch(
                sub_question="How does GenAI affect productivity?",
                sources=[source],
                findings=[finding]
            ),
            SubQuestionResearch(
                sub_question="Does GenAI cause security vulnerabilities?",
                sources=[],
                findings=[
                    ResearchFinding(
                        sub_question="Does GenAI cause security vulnerabilities?",
                        finding_text="Insufficient evidence available.",
                        supporting_urls=[],
                        source_claims=[],
                        verified=False,
                        limitations="No data found."
                    )
                ]
            ),
            SubQuestionResearch(
                sub_question="What are adoption trends?",
                sources=[],
                findings=[]
            )
        ]
    )
    return plan, research_data

def test_writer_successful_report(sample_plan_and_data):
    plan, research_data = sample_plan_and_data
    mock_report = ResearchReport(
        markdown_content="# Research Report\nExecutive summary... [S1]",
        sections_metadata=[SectionMetadata(title="Intro", summary="Summary...")],
        cited_source_ids=["S1"],
        revision_number=1
    )
    fake_llm = FakeLLMService(canned_responses=[mock_report])
    writer = WriterAgent(llm_service=fake_llm)
    report = writer.write_report(plan, research_data)
    assert report.revision_number == 1
    assert "S1" in report.cited_source_ids
    assert len(writer.run_traces) == 1
    assert writer.run_traces[0]["success"] is True

def test_writer_invalid_citation_id(sample_plan_and_data):
    plan, research_data = sample_plan_and_data
    mock_report = ResearchReport(
        markdown_content="# Report",
        sections_metadata=[],
        cited_source_ids=["S99"], # Invalid ID not in sources
        revision_number=1
    )
    fake_llm = FakeLLMService(canned_responses=[mock_report])
    writer = WriterAgent(llm_service=fake_llm)
    with pytest.raises(ValueError, match="Invalid cited source ID"):
        writer.write_report(plan, research_data)
    assert len(writer.run_traces) == 1
    assert writer.run_traces[0]["success"] is False

def test_writer_revision_feedback(sample_plan_and_data):
    plan, research_data = sample_plan_and_data
    mock_report = ResearchReport(
        markdown_content="# Revised Report",
        sections_metadata=[],
        cited_source_ids=["S1"],
        revision_number=2
    )
    fake_llm = FakeLLMService(canned_responses=[mock_report])
    writer = WriterAgent(llm_service=fake_llm)
    report = writer.write_report(plan, research_data, critic_feedback="Please expand security section", revision_number=2)
    assert report.revision_number == 2
    assert report.critic_feedback_addressed == "Please expand security section"
    assert writer.run_traces[0]["revision_number"] == 2
    assert writer.run_traces[0]["critic_feedback"] == "Please expand security section"
