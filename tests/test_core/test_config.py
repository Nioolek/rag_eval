"""Tests for core configuration and exceptions."""

import os
import pytest

from src.core.config import Config, get_config, reload_config
from src.core.exceptions import (
    RAGEvalError,
    ConfigurationError,
    StorageError,
    AnnotationError,
    EvaluationError,
    ValidationError,
    PathTraversalError,
)


class TestConfig:
    """Tests for configuration management."""

    def test_config_from_env(self):
        """Test configuration loading from environment."""
        os.environ["OPENAI_API_KEY"] = "test-key-123"
        os.environ["OPENAI_MODEL"] = "gpt-4"

        config = Config.from_env()

        assert config.llm.api_key == "test-key-123"
        assert config.llm.model == "gpt-4"

    def test_config_singleton(self):
        """Test configuration singleton pattern."""
        os.environ["OPENAI_API_KEY"] = "test-key-singleton"

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_config_reload(self):
        """Test configuration reload."""
        os.environ["OPENAI_API_KEY"] = "test-key-reload"

        reload_config()
        config = get_config()

        assert config.llm.api_key == "test-key-reload"

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        original_key = os.environ.pop("OPENAI_API_KEY", None)

        with pytest.raises(ConfigurationError):
            Config.from_env()

        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key


class TestExceptions:
    """Tests for custom exceptions."""

    def test_base_exception(self):
        """Test base exception."""
        error = RAGEvalError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_exception_with_details(self):
        """Test exception with details."""
        error = RAGEvalError("Test error", details={"key": "value"})
        assert "key" in str(error)
        assert error.details == {"key": "value"}

    def test_exception_hierarchy(self):
        """Test exception inheritance."""
        assert issubclass(ConfigurationError, RAGEvalError)
        assert issubclass(StorageError, RAGEvalError)
        assert issubclass(AnnotationError, RAGEvalError)
        assert issubclass(EvaluationError, RAGEvalError)
        assert issubclass(ValidationError, RAGEvalError)
        assert issubclass(PathTraversalError, StorageError)

    def test_exception_raise_catch(self):
        """Test raising and catching exceptions."""
        with pytest.raises(RAGEvalError):
            raise ConfigurationError("Config error")

        with pytest.raises(StorageError):
            raise PathTraversalError("Path traversal detected")