"""
Configuration Management for RAG Pipeline.

Supports:
- Environment variables for secrets
- YAML files for business config
- Hot reload with Watchdog
- Validation with Pydantic
"""

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


# === Configuration Models ===


class LLMConfig(BaseModel):
    """LLM service configuration."""

    model: str = "qwen-plus"
    temperature: float = 0.7
    max_tokens: int = 2048
    enable_thinking: bool = False
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""

    model: str = "text-embedding-v3"
    dimension: int = 1024
    batch_size: int = 20
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 0.5


class RerankConfig(BaseModel):
    """Rerank service configuration."""

    model: str = "gte-rerank"
    top_k: int = 5
    timeout: int = 30
    max_retries: int = 2
    retry_delay: float = 0.5


class StorageConfig(BaseModel):
    """Storage configuration."""

    data_dir: Path = Path("./data")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    vector_top_k: int = 20
    fulltext_top_k: int = 20
    graph_top_k: int = 10
    vector_weight: float = 0.5
    fulltext_weight: float = 0.3
    graph_weight: float = 0.2

    @field_validator("vector_weight", "fulltext_weight", "graph_weight")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v


class SessionConfig(BaseModel):
    """Session configuration."""

    max_history_turns: int = 5
    session_timeout: int = 3600  # seconds


class FAQConfig(BaseModel):
    """FAQ configuration."""

    match_threshold: float = 0.85
    enable_semantic_match: bool = True
    exact_match_boost: float = 1.5


class RefusalConfig(BaseModel):
    """Refusal configuration."""

    out_of_domain_threshold: float = 0.3
    sensitive_words_enabled: bool = True
    low_relevance_threshold: float = 0.2


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = 3
    recovery_timeout: float = 60.0


class DegradationConfig(BaseModel):
    """Degradation configuration."""

    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig
    )


class RAGConfig(BaseModel):
    """Complete RAG configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    faq: FAQConfig = Field(default_factory=FAQConfig)
    refusal: RefusalConfig = Field(default_factory=RefusalConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)


class EnvironmentConfig(BaseSettings):
    """Environment variables configuration (for secrets)."""

    # Alibaba Cloud
    dashscope_api_key: str = ""

    # LLM overrides
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # Embedding overrides
    embedding_model: str = "text-embedding-v3"
    embedding_dimension: int = 1024

    # Rerank overrides
    rerank_model: str = "gte-rerank"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Storage
    data_dir: str = "./data"

    # Rate limits
    dashscope_qpm: int = 60

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class ConfigManager:
    """
    Configuration Manager with hot-reload support.

    Features:
    - Load from YAML + environment variables
    - Hot reload with Watchdog
    - Thread-safe updates
    - Callbacks for config changes
    """

    _instance: Optional["ConfigManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config: Optional[RAGConfig] = None
        self._env_config: Optional[EnvironmentConfig] = None
        self._config_path: Optional[Path] = None
        self._reload_callbacks: list[Callable[[RAGConfig], None]] = []
        self._observer: Optional[Observer] = None
        self._initialized = True

    @property
    def config(self) -> RAGConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = RAGConfig()
        return self._config

    @property
    def env(self) -> EnvironmentConfig:
        """Get environment configuration."""
        if self._env_config is None:
            self._env_config = EnvironmentConfig()
        return self._env_config

    def initialize(self, config_path: str | Path | None = None) -> None:
        """
        Initialize configuration from file.

        Args:
            config_path: Path to YAML configuration file
        """
        if config_path:
            self._config_path = Path(config_path)
            self._config = RAGConfig.from_yaml(self._config_path)
        else:
            self._config = RAGConfig()

        self._env_config = EnvironmentConfig()
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        if not self._config or not self._env_config:
            return

        # Apply environment overrides
        if self._env_config.llm_model:
            self._config.llm.model = self._env_config.llm_model
        if self._env_config.llm_temperature:
            self._config.llm.temperature = self._env_config.llm_temperature
        if self._env_config.llm_max_tokens:
            self._config.llm.max_tokens = self._env_config.llm_max_tokens

        if self._env_config.embedding_model:
            self._config.embedding.model = self._env_config.embedding_model
        if self._env_config.embedding_dimension:
            self._config.embedding.dimension = self._env_config.embedding_dimension

        if self._env_config.rerank_model:
            self._config.rerank.model = self._env_config.rerank_model

        if self._env_config.neo4j_uri:
            self._config.storage.neo4j_uri = self._env_config.neo4j_uri
        if self._env_config.neo4j_user:
            self._config.storage.neo4j_user = self._env_config.neo4j_user
        if self._env_config.neo4j_password:
            self._config.storage.neo4j_password = self._env_config.neo4j_password

        if self._env_config.data_dir:
            self._config.storage.data_dir = Path(self._env_config.data_dir)

    def start_watching(self) -> None:
        """Start watching configuration file for changes."""
        if not self._config_path or not self._config_path.exists():
            return

        if self._observer:
            return

        class ConfigHandler(FileSystemEventHandler):
            def __init__(self, manager: "ConfigManager"):
                self.manager = manager

            def on_modified(self, event):
                if event.src_path == str(self.manager._config_path):
                    import asyncio

                    asyncio.create_task(self.manager.reload_now())

        self._observer = Observer()
        self._observer.schedule(
            ConfigHandler(self),
            str(self._config_path.parent),
            recursive=False,
        )
        self._observer.start()

    def stop_watching(self) -> None:
        """Stop watching configuration file."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def on_reload(self, callback: Callable[[RAGConfig], None]) -> None:
        """Register a callback for configuration reload."""
        self._reload_callbacks.append(callback)

    async def reload_now(self) -> None:
        """Reload configuration immediately."""
        async with self._lock:
            old_config = self._config

            try:
                self._config = RAGConfig.from_yaml(self._config_path)
                self._apply_env_overrides()

                # Notify callbacks
                for callback in self._reload_callbacks:
                    try:
                        result = callback(self._config)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        # Log but don't fail
                        print(f"Config reload callback failed: {e}")

            except Exception as e:
                # Rollback on failure
                self._config = old_config
                raise

    def get_merged_config(self) -> dict[str, Any]:
        """Get merged configuration as dictionary."""
        config = self.config.model_dump(mode="json")
        env = self.env.model_dump()

        # Add secrets from env
        config["dashscope_api_key"] = env.get("dashscope_api_key", "")
        config["dashscope_qpm"] = env.get("dashscope_qpm", 60)

        return config


@lru_cache(maxsize=1)
def get_config_manager() -> ConfigManager:
    """Get singleton ConfigManager instance."""
    return ConfigManager()


def get_config() -> RAGConfig:
    """Get current RAG configuration."""
    return get_config_manager().config


def get_env_config() -> EnvironmentConfig:
    """Get environment configuration."""
    return get_config_manager().env