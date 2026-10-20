from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str = ""
    tavily_api_key: str = ""
    groq_model_name: str = "llama-3.3-70b-versatile"
    max_llm_calls_per_run: int = 25
    max_search_calls_per_run: int = 15
    max_log_file_size_bytes: int = 10 * 1024 * 1024
    log_retention_days: int = 30
    port: int = 8000
    host: str = "0.0.0.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
