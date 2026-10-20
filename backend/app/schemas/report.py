from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SectionMetadata(BaseModel):
    title: str
    sub_question: Optional[str] = None
    summary: str

class ResearchReport(BaseModel):
    markdown_content: str
    sections_metadata: List[SectionMetadata]
    cited_source_ids: List[str]
    revision_number: int = 1
    critic_feedback_addressed: Optional[str] = None
