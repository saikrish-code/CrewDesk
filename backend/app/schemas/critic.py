from typing import List
from pydantic import BaseModel, Field

class CriticIssue(BaseModel):
    severity: str = Field(..., description="low, medium, high, or critical")
    location: str
    explanation: str
    corrective_action: str

class QualityScores(BaseModel):
    coverage: float = Field(..., ge=0.0, le=10.0)
    factual_consistency: float = Field(..., ge=0.0, le=10.0)
    citation_validity: float = Field(..., ge=0.0, le=10.0)
    clarity: float = Field(..., ge=0.0, le=10.0)
    completeness: float = Field(..., ge=0.0, le=10.0)

class CriticReview(BaseModel):
    approved: bool
    quality_scores: QualityScores
    issues: List[CriticIssue]
    summary: str
