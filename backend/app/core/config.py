import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Settings(BaseSettings):
    PROJECT_NAME: str = "RazorPay Sentinel"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/razorpay_sentinel"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"
    BASE_DIR: str = BASE_DIR
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")

    # Phase 3: Optional LLM Reasoning Configuration
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
