from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas import HealthResponse
from backend.app.config import settings

app = FastAPI(title="CrewDesk API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", service="crewdesk-backend")
