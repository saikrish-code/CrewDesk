import concurrent.futures
import pytest
from backend.app.services.budget import BudgetManager
from backend.app.services.llm import FakeLLMService, BudgetExceededError
from backend.app.graph.workflow import CrewDeskWorkflow
from backend.app.agents.planner import PlannerAgent
from backend.app.schemas.plan import ResearchPlan, SubQuestion

def test_budget_manager_boundaries():
    bm = BudgetManager(max_llm_calls=3, max_search_calls=2)
    assert bm.acquire_llm() is True
    assert bm.acquire_llm() is True
    assert bm.acquire_llm() is True
    assert bm.acquire_llm() is False # Exceeded

    assert bm.acquire_search() is True
    assert bm.acquire_search() is True
    assert bm.acquire_search() is False # Exceeded

def test_budget_manager_concurrency():
    bm = BudgetManager(max_llm_calls=50, max_search_calls=50)
    def worker():
        successes = 0
        for _ in range(20):
            if bm.acquire_llm():
                successes += 1
        return successes

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker) for _ in range(5)]
        results = [f.result() for f in futures]

    stats = bm.get_stats()
    assert stats["llm_usage"] == 50
    assert sum(results) == 50

def test_workflow_budget_exhaustion_preserves_partial():
    bm = BudgetManager(max_llm_calls=1, max_search_calls=1) # Extremely tight budget
    plan = ResearchPlan(
        original_topic="Test Topic",
        sub_questions=[
            SubQuestion(question="Q1?", objective="O1", priority_sources=["S1"]),
            SubQuestion(question="Q2?", objective="O2", priority_sources=["S2"]),
            SubQuestion(question="Q3?", objective="O3", priority_sources=["S3"])
        ],
        report_outline=["Intro", "Conclusion"]
    )
    planner_llm = FakeLLMService(canned_responses=[plan], budget_manager=bm)
    planner = PlannerAgent(llm_service=planner_llm)

    workflow = CrewDeskWorkflow(planner=planner, llm_budget=1, search_budget=1)
    state = workflow.run("Test Topic")

    assert state["status"] == "budget_exhausted"
    assert state["llm_usage"] <= 1
