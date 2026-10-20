import pytest
from pydantic import ValidationError
from backend.app.agents.planner import PlannerAgent
from backend.app.schemas.plan import ResearchPlan, SubQuestion
from backend.app.services.llm import FakeLLMService

@pytest.fixture
def ai_software_topic():
    return "The impact of generative AI on software development productivity and code quality"

def test_planner_valid_topic(ai_software_topic):
    valid_plan_data = {
        "original_topic": ai_software_topic,
        "sub_questions": [
            {"question": "How does gen AI affect coding speed?", "objective": "Measure productivity", "priority_sources": ["benchmarks"]},
            {"question": "Does AI generated code introduce bugs?", "objective": "Analyze code quality", "priority_sources": ["case studies"]},
            {"question": "What are developer sentiment and adoption trends?", "objective": "Assess developer experience", "priority_sources": ["surveys"]}
        ],
        "report_outline": ["Introduction", "Productivity Impact", "Code Quality", "Conclusion"]
    }
    fake_llm = FakeLLMService(canned_responses=[valid_plan_data])
    planner = PlannerAgent(llm_service=fake_llm)
    plan = planner.create_plan(ai_software_topic)
    assert plan.original_topic == ai_software_topic
    assert len(plan.sub_questions) == 3
    assert len(planner.run_traces) == 1
    assert planner.run_traces[0]["success"] is True

def test_planner_blank_input():
    planner = PlannerAgent(llm_service=FakeLLMService())
    with pytest.raises(ValueError):
        planner.create_plan("   ")

def test_planner_excessive_length():
    planner = PlannerAgent(llm_service=FakeLLMService())
    long_topic = "a" * 501
    with pytest.raises(ValueError):
        planner.create_plan(long_topic)

def test_planner_duplicate_sub_questions(ai_software_topic):
    invalid_plan_data = {
        "original_topic": ai_software_topic,
        "sub_questions": [
            {"question": "Same question?", "objective": "obj1", "priority_sources": ["src1"]},
            {"question": "same question?", "objective": "obj2", "priority_sources": ["src2"]},
            {"question": "Third question?", "objective": "obj3", "priority_sources": ["src3"]}
        ],
        "report_outline": ["Intro", "Body", "Conclusion"]
    }
    fake_llm = FakeLLMService(canned_responses=[invalid_plan_data])
    planner = PlannerAgent(llm_service=fake_llm)
    with pytest.raises((ValidationError, ValueError)):
        planner.create_plan(ai_software_topic)
    assert len(planner.run_traces) == 1
    assert planner.run_traces[0]["success"] is False

def test_planner_malformed_output(ai_software_topic):
    fake_llm = FakeLLMService(canned_responses=[{"original_topic": ai_software_topic, "sub_questions": []}])
    planner = PlannerAgent(llm_service=fake_llm)
    with pytest.raises((ValidationError, ValueError)):
        planner.create_plan(ai_software_topic)
    assert planner.run_traces[0]["success"] is False
