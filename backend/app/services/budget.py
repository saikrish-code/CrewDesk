import threading
from typing import Dict, Any

class BudgetManager:
    def __init__(self, max_llm_calls: int = 25, max_search_calls: int = 15):
        self.max_llm_calls = max_llm_calls
        self.max_search_calls = max_search_calls
        self.llm_usage = 0
        self.search_usage = 0
        self._lock = threading.Lock()

    def acquire_llm(self) -> bool:
        with self._lock:
            if self.llm_usage >= self.max_llm_calls:
                return False
            self.llm_usage += 1
            return True

    def acquire_search(self) -> bool:
        with self._lock:
            if self.search_usage >= self.max_search_calls:
                return False
            self.search_usage += 1
            return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_llm_calls": self.max_llm_calls,
                "llm_usage": self.llm_usage,
                "remaining_llm_calls": max(0, self.max_llm_calls - self.llm_usage),
                "max_search_calls": self.max_search_calls,
                "search_usage": self.search_usage,
                "remaining_search_calls": max(0, self.max_search_calls - self.search_usage)
            }
