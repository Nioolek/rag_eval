"""
LLM Service Implementation.

Alibaba Qwen integration for text generation and query understanding.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from rag_rag.core.exceptions import LLMServiceError, RateLimitError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.services.llm")


@dataclass
class LLMConfig:
    """LLM service configuration."""

    api_key: str = ""
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
    token_usage: dict[str, int] = None
    model: str = ""
    finish_reason: str = ""

    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {}


class LLMService:
    """
    LLM Service with Alibaba Qwen backend.

    Features:
    - Text generation with thinking mode
    - Streaming support
    - Retry with exponential backoff
    - Rate limit handling
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the LLM client."""
        try:
            import dashscope
            from dashscope import Generation

            dashscope.api_key = self.config.api_key
            self._client = Generation
            logger.info(f"LLM Service initialized: {self.config.model}")

        except ImportError:
            raise LLMServiceError(
                "dashscope not installed. Install with: pip install dashscope"
            )

    async def close(self) -> None:
        """Close the LLM client."""
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

        # Build request parameters
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "result_format": "message",
        }

        # Enable thinking mode if requested
        if enable_thinking:
            params["enable_thinking"] = True

        try:
            # Run in executor for sync API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.call(**params),
            )

            if response.status_code != 200:
                if response.code == "RateLimit":
                    raise RateLimitError(
                        f"Rate limit exceeded: {response.message}",
                        retry_after=60,
                    )
                raise LLMServiceError(
                    f"LLM API error: {response.code} - {response.message}"
                )

            # Parse response
            output = response.output
            content = output.choices[0].message.content
            thinking = output.choices[0].message.get("thinking", "")

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return LLMOutput(
                content=content,
                thinking_process=thinking,
                token_usage=usage,
                model=self.config.model,
                finish_reason=output.choices[0].finish_reason,
            )

        except RateLimitError:
            raise
        except Exception as e:
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

        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "result_format": "message",
            "stream": True,
        }

        if enable_thinking:
            params["enable_thinking"] = True

        try:
            responses = self._client.call(**params)

            for response in responses:
                if response.status_code != 200:
                    raise LLMServiceError(
                        f"LLM API error: {response.code} - {response.message}"
                    )

                # Yield content chunks
                if response.output and response.output.choices:
                    delta = response.output.choices[0].message.get("content", "")
                    if delta:
                        yield delta

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