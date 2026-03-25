"""
Singleton configuration manager for RAG Evaluation System.
All configuration is loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .exceptions import ConfigurationError

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration settings."""
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = 120


@dataclass(frozen=True)
class RAGConfig:
    """RAG service configuration."""
    service_url: str
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration."""
    storage_type: str = "sqlite"  # "local" or "sqlite"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    database_url: Optional[str] = None
    chunk_size: int = 8192  # For chunked file operations


@dataclass(frozen=True)
class UIConfig:
    """Gradio UI configuration."""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    max_threads: int = 40


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation configuration."""
    max_concurrent: int = 10
    timeout: int = 120
    enable_llm_evaluation: bool = True


@dataclass(frozen=True)
class Config:
    """
    Main configuration class using Singleton pattern.
    All settings are loaded from environment variables.
    """
    llm: LLMConfig
    rag: RAGConfig
    storage: StorageConfig
    ui: UIConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        # OpenAI API Key (optional for demo mode without LLM evaluation)
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # RAG Service URL (optional - can use mock adapter for testing)
        rag_service_url = os.getenv("RAG_SERVICE_URL", "")

        # Storage configuration
        storage_type = os.getenv("STORAGE_TYPE", "sqlite")
        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        database_url = os.getenv("DATABASE_URL")

        return cls(
            llm=LLMConfig(
                api_key=openai_api_key,
                api_base=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
                timeout=int(os.getenv("OPENAI_TIMEOUT", "120")),
            ),
            rag=RAGConfig(
                service_url=rag_service_url,
                timeout=int(os.getenv("RAG_TIMEOUT", "60")),
                max_retries=int(os.getenv("RAG_MAX_RETRIES", "3")),
                retry_delay=float(os.getenv("RAG_RETRY_DELAY", "1.0")),
            ),
            storage=StorageConfig(
                storage_type=storage_type,
                data_dir=data_dir,
                database_url=database_url,
                chunk_size=int(os.getenv("FILE_CHUNK_SIZE", "8192")),
            ),
            ui=UIConfig(
                server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
                share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
                max_threads=int(os.getenv("GRADIO_MAX_THREADS", "40")),
            ),
            evaluation=EvaluationConfig(
                max_concurrent=int(os.getenv("MAX_CONCURRENT_EVALUATIONS", "10")),
                timeout=int(os.getenv("EVALUATION_TIMEOUT", "120")),
                enable_llm_evaluation=os.getenv("ENABLE_LLM_EVALUATION", "true").lower() == "true",
            ),
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        # Ensure data directory exists
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        (self.storage.data_dir / "annotations").mkdir(exist_ok=True)
        (self.storage.data_dir / "results").mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get singleton configuration instance.
    Uses lru_cache to ensure only one instance exists.
    """
    config = Config.from_env()
    config.validate()
    return config


def reload_config() -> Config:
    """Reload configuration (clears cache)."""
    get_config.cache_clear()
    return get_config()