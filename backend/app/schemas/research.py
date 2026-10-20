from typing import List, Optional
from pydantic import BaseModel, Field

class ResearchSource(BaseModel):
    url: str
    title: str
    snippet: str
    published_date: Optional[str] = None
    retrieval_timestamp: str

class ResearchFinding(BaseModel):
    sub_question: str
    finding_text: str
    supporting_urls: List[str]
    source_claims: List[str]
    verified: bool = True
    limitations: Optional[str] = None

class SubQuestionResearch(BaseModel):
    sub_question: str
    sources: List[ResearchSource]
    findings: List[ResearchFinding]
    error: Optional[str] = None

class ResearchReportData(BaseModel):
    original_topic: str
    results: List[SubQuestionResearch]
