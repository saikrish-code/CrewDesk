import time
import uuid
from typing import Any, List, Optional, Type
from pydantic import BaseModel
from langchain_groq import ChatGroq
from backend.app.config import settings
from backend.app.logging import logger

class LLMCallRecord(BaseModel):
    call_id: str
    agent: str
    purpose: str
    start_time: float
    duration: float
    success: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None

class BaseLLMService:
    def invoke(
        self,
        prompt: str,
        agent: str = "default",
        purpose: str = "general",
        temperature: float = 0.7,
        timeout: float = 30.0,
        response_model: Optional[Type[BaseModel]] = None,
        max_retries: int = 0
    ) -> Any:
        raise NotImplementedError

class GroqLLMService(BaseLLMService):
    def __init__(self):
        self.call_records: List[LLMCallRecord] = []

    def _get_client(self, temperature: float, timeout: float):
        return ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
            temperature=temperature,
            timeout=timeout,
            max_retries=0
        )

    def invoke(
        self,
        prompt: str,
        agent: str = "default",
        purpose: str = "general",
        temperature: float = 0.7,
        timeout: float = 30.0,
        response_model: Optional[Type[BaseModel]] = None,
        max_retries: int = 0
    ) -> Any:
        attempts = max_retries + 1
        last_exception = None

        for attempt in range(attempts):
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            start_time = time.time()
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            try:
                llm = self._get_client(temperature=temperature, timeout=timeout)
                if response_model is not None:
                    structured_llm = llm.with_structured_output(response_model)
                    result = structured_llm.invoke(prompt)
                else:
                    response = llm.invoke(prompt)
                    result = response.content

                duration = time.time() - start_time

                try:
                    if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
                        usage = response.response_metadata["token_usage"]
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                except Exception:
                    pass

                self.call_records.append(
                    LLMCallRecord(
                        call_id=call_id,
                        agent=agent,
                        purpose=purpose,
                        start_time=start_time,
                        duration=duration,
                        success=True,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )
                )
                logger.info(f"LLM call {call_id} succeeded for agent {agent} purpose {purpose}")
                return result

            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                last_exception = e
                logger.error(f"LLM call {call_id} failed on attempt {attempt + 1}/{attempts}: {error_msg}")

                self.call_records.append(
                    LLMCallRecord(
                        call_id=call_id,
                        agent=agent,
                        purpose=purpose,
                        start_time=start_time,
                        duration=duration,
                        success=False,
                        error=error_msg
                    )
                )

                if attempt == attempts - 1:
                    raise last_exception

class FakeLLMService(BaseLLMService):
    def __init__(self, canned_responses: Optional[List[Any]] = None, should_fail: bool = False, fail_with: Optional[Exception] = None):
        self.call_records: List[LLMCallRecord] = []
        self.canned_responses = canned_responses or []
        self.should_fail = should_fail
        self.fail_with = fail_with or Exception("Fake LLM failure")
        self.call_index = 0

    def invoke(
        self,
        prompt: str,
        agent: str = "default",
        purpose: str = "general",
        temperature: float = 0.7,
        timeout: float = 30.0,
        response_model: Optional[Type[BaseModel]] = None,
        max_retries: int = 0
    ) -> Any:
        attempts = max_retries + 1
        last_exception = None

        for attempt in range(attempts):
            call_id = f"fake_{uuid.uuid4().hex[:8]}"
            start_time = time.time()

            try:
                if self.should_fail:
                    raise self.fail_with

                if self.call_index < len(self.canned_responses):
                    res = self.canned_responses[self.call_index]
                    self.call_index += 1
                else:
                    res = "Default fake response"

                if response_model is not None and isinstance(res, dict):
                    parsed = response_model(**res)
                elif response_model is not None and isinstance(res, response_model):
                    parsed = res
                elif response_model is not None and isinstance(res, str):
                    import json
                    parsed = response_model(**json.loads(res))
                else:
                    parsed = res

                duration = time.time() - start_time
                self.call_records.append(
                    LLMCallRecord(
                        call_id=call_id,
                        agent=agent,
                        purpose=purpose,
                        start_time=start_time,
                        duration=duration,
                        success=True,
                        prompt_tokens=10,
                        completion_tokens=15,
                        total_tokens=25
                    )
                )
                return parsed

            except Exception as e:
                duration = time.time() - start_time
                error_msg = str(e)
                last_exception = e
                self.call_records.append(
                    LLMCallRecord(
                        call_id=call_id,
                        agent=agent,
                        purpose=purpose,
                        start_time=start_time,
                        duration=duration,
                        success=False,
                        error=error_msg
                    )
                )
                if attempt == attempts - 1:
                    raise last_exception
