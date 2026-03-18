"""
LLM Evaluator using LangChain OpenAI for LLM-based metrics.
"""

import asyncio
from functools import lru_cache
from typing import Any, Optional

from ..core.config import get_config
from ..core.exceptions import ConfigurationError, EvaluationError
from ..core.logging import logger


class LLMEvaluator:
    """
    LLM-based evaluator using LangChain OpenAI.
    Singleton pattern for connection management.
    """

    def __init__(self):
        self._llm = None
        self._initialized = False
        self._token_usage = {"prompt": 0, "completion": 0, "total": 0}

    async def initialize(self) -> None:
        """Initialize the LLM client."""
        if self._initialized:
            return

        try:
            config = get_config()

            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                api_key=config.llm.api_key,
                base_url=config.llm.api_base,
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                request_timeout=config.llm.timeout,
            )

            self._initialized = True
            logger.info(f"Initialized LLM evaluator with model: {config.llm.model}")

        except ImportError:
            logger.warning(
                "langchain-openai not installed. "
                "Install with: pip install langchain-openai"
            )
            self._initialized = True

        except Exception as e:
            raise EvaluationError(f"Failed to initialize LLM evaluator: {e}")

    @property
    def llm(self) -> Any:
        """Get the LLM client."""
        return self._llm

    async def evaluate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Evaluate using LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            LLM response text
        """
        if not self._initialized:
            await self.initialize()

        if not self._llm:
            raise ConfigurationError("LLM client not initialized")

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.messages import SystemMessage, HumanMessage

            if system_prompt:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt),
                ]
            else:
                messages = [HumanMessage(content=prompt)]

            response = await self._llm.ainvoke(messages)

            # Track token usage
            if hasattr(response, 'usage_metadata'):
                self._token_usage["prompt"] += response.usage_metadata.get("input_tokens", 0)
                self._token_usage["completion"] += response.usage_metadata.get("output_tokens", 0)
                self._token_usage["total"] = self._token_usage["prompt"] + self._token_usage["completion"]

            return response.content

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            raise EvaluationError(f"LLM evaluation failed: {e}")

    async def batch_evaluate(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        batch_size: int = 5,
    ) -> list[str]:
        """
        Batch evaluate multiple prompts.

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            batch_size: Concurrent batch size

        Returns:
            List of responses
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tasks = [self.evaluate(p, system_prompt) for p in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Batch evaluation failed for prompt {i+j}: {result}")
                    results.append("")
                else:
                    results.append(result)

        return results

    def get_token_usage(self) -> dict[str, int]:
        """Get total token usage."""
        return self._token_usage.copy()

    def reset_token_usage(self) -> None:
        """Reset token usage counter."""
        self._token_usage = {"prompt": 0, "completion": 0, "total": 0}

    async def close(self) -> None:
        """Close the LLM client."""
        self._llm = None
        self._initialized = False


# Singleton instance
_evaluator_instance: Optional[LLMEvaluator] = None


async def get_llm_evaluator() -> LLMEvaluator:
    """Get singleton LLM evaluator instance."""
    global _evaluator_instance

    if _evaluator_instance is None:
        _evaluator_instance = LLMEvaluator()
        await _evaluator_instance.initialize()

    return _evaluator_instance