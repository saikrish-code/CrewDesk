from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.schemas import HealthResponse
from backend.app.config import settings
from backend.app.api.router import router

app = FastAPI(title="CrewDesk API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", service="crewdesk-backend")
