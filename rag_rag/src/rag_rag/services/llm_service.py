"""
LLM Service Implementation.

OpenAI-compatible API integration for text generation and query understanding.
Supports Alibaba Cloud Coding Plan and other OpenAI-compatible endpoints.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_rag.core.exceptions import LLMServiceError, RateLimitError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.services.llm")


@dataclass
class LLMConfig:
    """LLM service configuration."""

    api_key: str = ""
    base_url: str = "https://coding.dashscope.aliyuncs.com/v1"
    model: str = "qwen-plus"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class LLMOutput:
    """LLM generation output."""

    content: str
    thinking_process: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class LLMService:
    """
    LLM Service with OpenAI-compatible backend.

    Features:
    - Text generation with thinking mode
    - Streaming support
    - Retry with exponential backoff
    - Rate limit handling
    - Supports Alibaba Cloud Coding Plan and other OpenAI-compatible APIs
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[AsyncOpenAI] = None

    async def initialize(self) -> None:
        """Initialize the LLM client."""
        try:
            if not self.config.api_key:
                raise LLMServiceError(
                    "API key not configured. Set OPENAI_API_KEY environment variable."
                )

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            logger.info(
                f"LLM Service initialized: {self.config.model} "
                f"(base_url: {self.config.base_url})"
            )

        except Exception as e:
            raise LLMServiceError(f"Failed to initialize LLM client: {e}")

    async def close(self) -> None:
        """Close the LLM client."""
        if self._client:
            await self._client.close()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> LLMOutput:
        """
        Generate text using LLM.

        Args:
            system_prompt: System instruction
            user_prompt: User query
            temperature: Override temperature
            max_tokens: Override max tokens
            enable_thinking: Enable thinking mode

        Returns:
            LLMOutput with generated content
        """
        if self._client is None:
            await self.initialize()

        messages = []

        # Add system message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user message
        messages.append({"role": "user", "content": user_prompt})

        try:
            # Build extra body for thinking mode (provider-specific)
            extra_body = {}
            if enable_thinking:
                # Some providers support thinking mode via extra parameters
                extra_body["enable_thinking"] = True

            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                extra_body=extra_body if extra_body else None,
            )

            # Parse response
            choice = response.choices[0]
            content = choice.message.content or ""

            # Try to extract thinking process if available
            thinking = ""
            if hasattr(choice.message, "model_extra") and choice.message.model_extra:
                thinking = choice.message.model_extra.get("thinking", "")

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return LLMOutput(
                content=content,
                thinking_process=thinking,
                token_usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason,
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "429" in error_msg:
                raise RateLimitError(
                    f"Rate limit exceeded: {e}",
                    retry_after=60,
                )
            logger.error(f"LLM generation failed: {e}")
            raise LLMServiceError(f"LLM generation failed: {e}")

    async def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream text generation.

        Yields tokens as they are generated.
        """
        if self._client is None:
            await self.initialize()

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        try:
            extra_body = {}
            if enable_thinking:
                extra_body["enable_thinking"] = True

            stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=True,
                extra_body=extra_body if extra_body else None,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            raise LLMServiceError(f"LLM streaming failed: {e}")

    async def rewrite_query(
        self,
        query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        rewrite_type: str = "expansion",
    ) -> dict[str, Any]:
        """
        Rewrite a query for better retrieval.

        Args:
            query: Original query
            conversation_history: Previous conversation turns
            rewrite_type: Type of rewrite (expansion, clarification, multi_turn)

        Returns:
            Dict with rewritten_query, type, and confidence
        """
        system_prompt = """你是一个专业的查询改写助手。
你的任务是将用户的查询改写为更适合检索的形式。

改写规则：
1. 扩展查询(expansion)：添加相关关键词，保留原意
2. 澄清查询(clarification)：消除歧义，使意图更明确
3. 多轮改写(multi_turn)：结合对话历史，补全省略信息

请以JSON格式输出：
{"rewritten_query": "改写后的查询", "type": "改写类型", "confidence": 0.0-1.0}
"""

        user_prompt = f"原查询: {query}\n改写类型: {rewrite_type}"

        if conversation_history:
            history_str = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-5:]  # Last 5 turns
            ])
            user_prompt = f"对话历史:\n{history_str}\n\n{user_prompt}"

        try:
            output = await self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
            )

            # Parse JSON response
            try:
                result = json.loads(output.content)
                return {
                    "original_query": query,
                    "rewritten_query": result.get("rewritten_query", query),
                    "rewrite_type": result.get("type", rewrite_type),
                    "confidence": result.get("confidence", 0.8),
                }
            except json.JSONDecodeError:
                # Fallback: use entire response as rewritten query
                return {
                    "original_query": query,
                    "rewritten_query": output.content.strip(),
                    "rewrite_type": rewrite_type,
                    "confidence": 0.6,
                }

        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return {
                "original_query": query,
                "rewritten_query": query,
                "rewrite_type": rewrite_type,
                "confidence": 0.0,
            }

    async def check_refusal(
        self,
        query: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Check if the query should be refused.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Dict with should_refuse, reason, and type
        """
        system_prompt = """你是一个内容安全判断助手。
根据提供的上下文，判断是否应该拒绝回答用户的问题。

拒绝条件：
1. 超出知识范围(out_of_domain)：上下文与问题无关
2. 敏感问题(sensitive)：涉及敏感话题
3. 相关性过低(low_relevance)：上下文不足以回答问题

请以JSON格式输出：
{"should_refuse": true/false, "reason": "原因说明", "type": "拒绝类型"}
"""

        user_prompt = f"用户问题: {query}\n\n上下文:\n{context[:2000]}"

        try:
            output = await self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
            )

            try:
                result = json.loads(output.content)
                return {
                    "should_refuse": result.get("should_refuse", False),
                    "refusal_reason": result.get("reason", ""),
                    "refusal_type": result.get("type", ""),
                }
            except json.JSONDecodeError:
                return {
                    "should_refuse": False,
                    "refusal_reason": "",
                    "refusal_type": "",
                }

        except Exception as e:
            logger.error(f"Refusal check failed: {e}")
            return {
                "should_refuse": False,
                "refusal_reason": "",
                "refusal_type": "",
            }