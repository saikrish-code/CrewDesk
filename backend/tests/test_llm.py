import pytest
from pydantic import BaseModel, ValidationError
from backend.app.services.llm import FakeLLMService, LLMCallRecord

class SampleModel(BaseModel):
    title: str
    score: int

def test_fake_llm_success_text():
    fake = FakeLLMService(canned_responses=["Hello World"])
    res = fake.invoke("Say hello", agent="test_agent", purpose="greeting")
    assert res == "Hello World"
    assert len(fake.call_records) == 1
    record = fake.call_records[0]
    assert record.success is True
    assert record.agent == "test_agent"
    assert record.purpose == "greeting"
    assert record.total_tokens == 25

def test_fake_llm_success_structured():
    fake = FakeLLMService(canned_responses=[{"title": "Report", "score": 95}])
    res = fake.invoke("Generate report", agent="analyst", purpose="analysis", response_model=SampleModel)
    assert isinstance(res, SampleModel)
    assert res.title == "Report"
    assert res.score == 95
    assert len(fake.call_records) == 1
    assert fake.call_records[0].success is True

def test_fake_llm_malformed_response():
    fake = FakeLLMService(canned_responses=[{"title": "Report", "score": "invalid_score"}])
    with pytest.raises((ValidationError, ValueError)):
        fake.invoke("Generate report", response_model=SampleModel)
    assert len(fake.call_records) == 1
    assert fake.call_records[0].success is False

def test_fake_llm_failure():
    fake = FakeLLMService(should_fail=True, fail_with=TimeoutError("Connection timed out"))
    with pytest.raises(TimeoutError):
        fake.invoke("Timeout test", max_retries=0)
    assert len(fake.call_records) == 1
    assert fake.call_records[0].success is False
    assert "timed out" in fake.call_records[0].error

def test_fake_llm_explicit_retries():
    fake = FakeLLMService(should_fail=True, fail_with=Exception("API Error"))
    with pytest.raises(Exception):
        fake.invoke("Retry test", max_retries=2)
    assert len(fake.call_records) == 3
    for record in fake.call_records:
        assert record.success is False
