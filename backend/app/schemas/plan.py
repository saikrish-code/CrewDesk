from typing import List
from pydantic import BaseModel, Field, field_validator

class SubQuestion(BaseModel):
    question: str
    objective: str
    priority_sources: List[str]

class ResearchPlan(BaseModel):
    original_topic: str
    sub_questions: List[SubQuestion] = Field(..., min_length=3, max_length=4)
    report_outline: List[str]

    @field_validator("original_topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Topic cannot be blank")
        if len(v.strip()) > 500:
            raise ValueError("Topic is excessively long")
        return v.strip()

    @field_validator("sub_questions")
    @classmethod
    def validate_sub_questions(cls, v: List[SubQuestion]) -> List[SubQuestion]:
        if len(v) < 3 or len(v) > 4:
            raise ValueError("Must have exactly 3 or 4 sub-questions")
        questions_text = [sq.question.strip().lower() for sq in v]
        if len(set(questions_text)) != len(questions_text):
            raise ValueError("Sub-questions must be distinct and non-redundant")
        return v
