import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.api.router import active_runs

client = TestClient(app)

def test_create_run_valid():
    response = client.post("/api/runs", json={"topic": "Quantum Computing Impact"})
    assert response.status_code == 201
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "running"
    assert data["original_topic"] == "Quantum Computing Impact"

def test_create_run_invalid_topic():
    response = client.post("/api/runs", json={"topic": "   "})
    assert response.status_code == 422

def test_get_run_status():
    active_runs["run_sample_123"] = {
        "run_id": "run_sample_123",
        "status": "completed",
        "original_topic": "Quantum Computing"
    }
    response = client.get("/api/runs/run_sample_123")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

def test_get_run_report():
    active_runs["run_sample_123"] = {
        "run_id": "run_sample_123",
        "status": "completed",
        "current_draft": {
            "markdown_content": "# Final Quantum Computing Report",
            "sections_metadata": [],
            "cited_source_ids": ["S1"],
            "revision_number": 1
        }
    }
    response = client.get("/api/runs/run_sample_123/report")
    assert response.status_code == 200
    assert response.json()["markdown_content"] == "# Final Quantum Computing Report"

def test_get_nonexistent_run():
    response = client.get("/api/runs/run_nonexistent_999")
    assert response.status_code == 404

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
